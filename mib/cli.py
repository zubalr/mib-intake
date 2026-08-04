"""Read a directory of PDF packets and write predictions.

Contract (DOCKER_SUBMISSION.md): exactly two arguments, `<input_pdf_dir>` and
`<output_predictions_path>`. Runs offline, read-only root filesystem, scratch in
/tmp, 4 vCPU / 8 GiB, averaging 6 seconds per PDF.

The entrypoint enforces two output invariants:

  * Every input PDF produces exactly one output row. A crash, timeout, or
    an unparseable file falls back to a prior-based record rather than dropping
    the case.
  * No field is blank. `validate_submission.py` rejects records whose
    `sponsor_id` or `arrival_date` are malformed, and the evaluator scores a
    wrong value identically to a blank.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from mib.lexicon import Lexicon
from mib.policy import (UNREADABLE_PATH, Calibration, apply_reference_date,
                        corpus_revoked_sponsors, corpus_years, decide)
from mib.schema import FALLBACK_ARRIVAL_DATE, FALLBACK_SPONSOR_ID, Prediction, write_jsonl

# Per-PDF wall-clock ceiling. The budget is averaged over the set, so an
# individual packet may exceed six seconds without stalling the batch.
# On timeout we emit the best record built so far rather than dropping the case.
PER_PDF_TIMEOUT_S = 55

_LEXICON: Lexicon | None = None
_CALIBRATION: Calibration | None = None
_ADJUDICATOR = None


def _worker_init() -> None:
    """Load shared read-only resources once per worker process."""
    global _LEXICON, _CALIBRATION, _ADJUDICATOR
    _LEXICON = Lexicon()
    _CALIBRATION = Calibration()
    # The deterministic paths remain available if the model cannot be loaded.
    from mib.model import Adjudicator
    _ADJUDICATOR = Adjudicator.try_load()


def fallback_prediction(case_id: str, lexicon: Lexicon, calibration: Calibration,
                        reason: str) -> Prediction:
    """Return a schema-valid record for an unreadable packet.

    Every field takes its training-prior mode.

    Adjudication does not use the normal policy path. An
    all-unknown Record lands on `fee_unknown`, whose 94% NEEDS_REVIEW rate was
    estimated from otherwise readable packets. An unreadable packet instead
    uses the dedicated unreadable-packet distribution.
    """
    adjudication, confidence = decide(calibration.probs(UNREADABLE_PATH))
    path = UNREADABLE_PATH
    return Prediction(
        case_id=case_id,
        applicant_name=lexicon.data["applicant_name"]["prior_mode"],
        species_code=lexicon.prior_mode("species_code"),
        home_world=lexicon.prior_mode("home_world"),
        visa_class=lexicon.prior_mode("visa_class"),
        sponsor_id=FALLBACK_SPONSOR_ID,
        arrival_date=FALLBACK_ARRIVAL_DATE,
        declared_purpose=lexicon.prior_mode("declared_purpose"),
        risk_flags="none",
        # Printed defaults do not enter adjudication.
        fee_status=lexicon.prior_mode("fee_status"),
        adjudication=adjudication,
        confidence=confidence,
        debug={"path": path, "fallback": reason},
    )


def process_one(pdf_path_str: str) -> dict:
    """Process a single packet. Must never raise: the caller relies on that."""
    pdf_path = Path(pdf_path_str)
    case_id = pdf_path.stem
    assert _LEXICON is not None and _CALIBRATION is not None

    def _timeout(_signum, _frame):
        raise TimeoutError(f"exceeded {PER_PDF_TIMEOUT_S}s")

    previous = signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(PER_PDF_TIMEOUT_S)
    try:
        from mib.pipeline import extract_packet
        ex = extract_packet(pdf_path, _LEXICON)
        return {"case_id": case_id, "printed": ex.printed, "record": ex.record,
                "note": ex.note, "features": ex.features, "failed": False}
    except BaseException as exc:  # noqa: BLE001 - every packet needs an output row
        print(f"[warn] {case_id}: {type(exc).__name__}: {exc}", file=sys.stderr)
        if os.environ.get("MIB_DEBUG"):
            traceback.print_exc()
        return {"case_id": case_id, "failed": True,
                "reason": f"{type(exc).__name__}: {exc}"}
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def corpus_reference_date(records) -> str | None:
    """Approximate packet receipt date for the staleness rule.

    Packets carry an arrival date but no receipt date. The reference is derived
    from arrival dates in the corpus rather than fixed to the public dataset.
    """
    import datetime as dt
    dates = []
    for rec in records:
        value = getattr(rec, "arrival_date", None)
        if not value or value == "unknown":
            continue
        try:
            dates.append(dt.date.fromisoformat(value))
        except (ValueError, TypeError):
            continue
    if not dates:
        # Without a corpus date, staleness remains indeterminate.
        return None
    dates.sort()

    # Remove distant year clusters before selecting the upper percentile.
    # OCR errors such as 6/8 substitutions are correlated across scans, so a
    # median-centered window is more stable than an untrimmed percentile.
    median = dates[len(dates) // 2]
    deviations = sorted(abs((d - median).days) for d in dates)
    mad = deviations[len(deviations) // 2]
    window = max(730, 6 * mad)
    kept = [d for d in dates if abs((d - median).days) <= window] or dates

    index = min(len(kept) - 1, int(len(kept) * 0.98))
    return kept[index].isoformat()


def corpus_median_date(records) -> str | None:
    """Median arrival date of the corpus being scored.

    Used only for the printed output when no trusted arrival date is available.
    The value is derived from the current corpus rather than the public dataset.
    """
    import datetime as dt
    dates = []
    for rec in records:
        value = getattr(rec, "arrival_date", None)
        if not value or value == "unknown":
            continue
        try:
            dates.append(dt.date.fromisoformat(value))
        except (ValueError, TypeError):
            continue
    if not dates:
        return None
    dates.sort()
    return dates[len(dates) // 2].isoformat()


def resolve_fallback_sponsors(rows: list[dict]) -> tuple[str | None, int]:
    """Replace output-only sponsor placeholders with the current corpus mode.

    The placeholder is necessary while packets are processed independently,
    but once the corpus is available it is a dominated final guess: an observed
    sponsor has some chance of matching, while the reserved placeholder carries
    no document evidence. This pass is deliberately output-only and therefore
    cannot affect revocation policy or adjudication.
    """
    import re
    from collections import Counter

    sponsor_re = re.compile(r"SPN-\d{4}\Z")
    counts = Counter(
        str(row.get("sponsor_id", ""))
        for row in rows
        if str(row.get("sponsor_id", "")) != FALLBACK_SPONSOR_ID
        and sponsor_re.fullmatch(str(row.get("sponsor_id", "")))
    )
    if not counts:
        return None, 0

    # Stable lexical tie-break keeps predictions deterministic.
    mode = min(counts, key=lambda value: (-counts[value], value))
    replaced = 0
    for row in rows:
        if row.get("sponsor_id") == FALLBACK_SPONSOR_ID:
            row["sponsor_id"] = mode
            replaced += 1
    return mode, replaced


def available_cpus() -> int:
    """CPU count the container is actually allowed to use.

    `os.cpu_count()` reports the *host's* CPUs, which inside the scoring
    container is 14 while `--cpus 4` is enforced by the cgroup. Spawning 8 OCR
    workers onto a 4-CPU quota just adds contention and context switching. Read
    the cgroup quota and fall back to the host count only when there is none.
    """
    for quota_path, period_path in (
        ("/sys/fs/cgroup/cpu.max", None),                        # cgroup v2
        ("/sys/fs/cgroup/cpu/cpu.cfs_quota_us",
         "/sys/fs/cgroup/cpu/cpu.cfs_period_us"),                # cgroup v1
    ):
        try:
            raw = Path(quota_path).read_text().strip()
            if period_path is None:
                quota_s, period_s = raw.split()
            else:
                quota_s, period_s = raw, Path(period_path).read_text().strip()
            if quota_s in ("max", "-1"):
                continue
            quota, period = int(quota_s), int(period_s)
            if quota > 0 and period > 0:
                return max(1, quota // period)
        except (OSError, ValueError):
            continue
    return os.cpu_count() or 4


def apply_box_date(printed: dict[str, str], years: dict) -> None:
    """Settle a box-recovered arrival date against the corpus year histogram.

    Staged rather than applied at parse time because both branches need the
    corpus: the strict branch repairs an implausible year onto the observed
    distribution, and the cheaper branch is only admissible for a date already
    in the modal year.
    """
    from mib.policy import repair_year

    staged = printed.pop("_box_dates", "")
    if not staged or not years:
        return
    modal = max(years, key=years.get)
    candidates = set()
    for item in staged.split("|"):
        branch, _, value = item.partition(":")
        if branch == "A":
            candidates.add(repair_year(value, years) or value)
        elif branch == "B" and value[:4] == modal:
            candidates.add(value)
    if len(candidates) == 1:
        printed["arrival_date"] = next(iter(candidates))


def apply_box_sponsors(rows: list[dict], revoked: set[str],
                       placeholder: str | None) -> int:
    """Settle box-recovered sponsor ids once the corpus placeholder is known."""
    from mib.pipeline import BOX_SPONSOR_KEYS

    replaced = 0
    for row in rows:
        standard, seen, rescue = (row.pop(key, None)
                                  for key in BOX_SPONSOR_KEYS)
        current = str(row.get("sponsor_id", "")).strip().upper()
        on_placeholder = bool(placeholder) and current == placeholder
        if on_placeholder:
            candidate = rescue or standard
        else:
            candidate = standard if seen else None
        if not candidate or candidate == current:
            continue
        # Never trade one known-revoked sponsor for another. The packet is
        # disqualified on either reading, so the substitution buys nothing and
        # cannot be checked against anything.
        if current in revoked and candidate in revoked:
            continue
        # A rescue from the placeholder must not invent a revocation either.
        if on_placeholder and candidate in revoked:
            continue
        row["sponsor_id"] = candidate
        replaced += 1
    return replaced


def _enforce_output_schema(rows: list[dict]) -> None:
    """Last line of defence before writing.

    Every row must satisfy validate_submission.py regardless of what happened
    upstream: an invalid sponsor_id, arrival_date, fee_status or adjudication
    invalidates the record. Anything still bad is replaced with a prior value --
    a wrong value scores the same as a blank, but an *invalid* one fails the row.
    """
    from mib.policy import ADJUDICATION_VALUES_SET
    from mib.validate import FEE_VALUES, valid_date, valid_sponsor

    repaired = 0
    for row in rows:
        if not valid_sponsor(str(row.get("sponsor_id", ""))):
            row["sponsor_id"] = FALLBACK_SPONSOR_ID
            repaired += 1
        if not valid_date(str(row.get("arrival_date", ""))):
            row["arrival_date"] = FALLBACK_ARRIVAL_DATE
            repaired += 1
        if str(row.get("fee_status", "")) not in FEE_VALUES:
            row["fee_status"] = "unknown"
            repaired += 1
        if str(row.get("adjudication", "")) not in ADJUDICATION_VALUES_SET:
            row["adjudication"] = "NEEDS_REVIEW"
            repaired += 1
        try:
            conf = float(row.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        row["confidence"] = min(1.0, max(0.0, conf))
        for key, value in list(row.items()):
            if key != "confidence" and not isinstance(value, str):
                row[key] = str(value)
    if repaired:
        print(f"[warn] repaired {repaired} invalid output fields", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MIB intake packet adjudicator")
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--workers", type=int, default=None,
                        help="Default: one per available CPU (scoring gives 4).")
    parser.add_argument("--limit", type=int, default=None, help="Dev only.")
    args = parser.parse_args(argv)

    pdfs = sorted(args.input_dir.glob("*.pdf"))
    if args.limit:
        pdfs = pdfs[: args.limit]
    if not pdfs:
        print(f"[error] no PDFs under {args.input_dir}", file=sys.stderr)
        write_jsonl(args.output_path, [])
        return 1

    workers = args.workers or max(1, min(available_cpus(), 8))
    print(f"[info] {len(pdfs)} packets, {workers} workers", file=sys.stderr)

    rows: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as pool:
        futures = {pool.submit(process_one, str(p)): p for p in pdfs}
        for done, future in enumerate(as_completed(futures), 1):
            pdf = futures[future]
            try:
                row = future.result()
            except BaseException as exc:  # noqa: BLE001 -- worker died outright
                row = {"case_id": pdf.stem, "failed": True,
                       "reason": f"worker died: {type(exc).__name__}"}
                print(f"[warn] {pdf.stem}: worker died: {exc}", file=sys.stderr)
            rows[row["case_id"]] = row
            if done % 250 == 0:
                print(f"[info] {done}/{len(pdfs)}", file=sys.stderr)

    # Phase 2: adjudicate. Cheap and in-process, so it can use corpus-level
    # context that a per-packet worker could not see.
    _worker_init()
    assert _LEXICON is not None and _CALIBRATION is not None
    from mib.pipeline import (BOX_SPONSOR_KEYS, finalize,
                              resolve_printed_date)

    good = [r for r in rows.values() if not r.get("failed")]
    reference = corpus_reference_date([r["record"] for r in good])
    revoked = corpus_revoked_sponsors([r["record"] for r in good])
    median_date = corpus_median_date([r["record"] for r in good])
    years = corpus_years([r["record"] for r in good])
    print(f"[info] staleness reference date: {reference}", file=sys.stderr)
    print(f"[info] corpus arrival years: {years}", file=sys.stderr)
    print(f"[info] corpus-derived revoked sponsors: {sorted(revoked)}", file=sys.stderr)
    adjudicator_name = (
        "model " + str(_ADJUDICATOR.metadata.get("kind"))
        if _ADJUDICATOR else "hand-built paths"
    )
    print(f"[info] adjudicator: {adjudicator_name}", file=sys.stderr)

    final: dict[str, dict] = {}
    for case_id, row in rows.items():
        if row.get("failed"):
            final[case_id] = fallback_prediction(
                case_id, _LEXICON, _CALIBRATION, row.get("reason", "?")).to_row()
            continue
        record = apply_reference_date(row["record"], reference, revoked)
        resolve_printed_date(row["printed"], record, median_date, years)
        apply_box_date(row["printed"], years)
        final[case_id] = finalize(
            row["printed"], record, row["note"], _CALIBRATION,
            adjudicator=_ADJUDICATOR, features=row.get("features")).to_row()
        # Carried across `finalize`, which emits only schema fields. Both keys
        # are removed again by `apply_box_sponsors` before anything is written.
        for key in BOX_SPONSOR_KEYS:
            if key in row["printed"]:
                final[case_id][key] = row["printed"][key]
    rows = final

    # Belt and braces: assert one row per input PDF before writing.
    missing = [p.stem for p in pdfs if p.stem not in rows]
    for case_id in missing:
        _worker_init()
        assert _LEXICON is not None and _CALIBRATION is not None
        rows[case_id] = fallback_prediction(
            case_id, _LEXICON, _CALIBRATION, reason="never returned").to_row()
    if missing:
        print(f"[warn] backfilled {len(missing)} cases", file=sys.stderr)

    ordered = [rows[p.stem] for p in pdfs]
    for row in ordered:
        row.pop("_debug", None)
    _enforce_output_schema(ordered)
    sponsor_mode, sponsor_replacements = resolve_fallback_sponsors(ordered)
    box_sponsors = apply_box_sponsors(ordered, revoked, sponsor_mode)
    if box_sponsors:
        print(f"[info] box-recovered sponsor ids: {box_sponsors}",
              file=sys.stderr)
    if sponsor_replacements:
        print(
            f"[info] corpus sponsor fallback: {sponsor_mode} "
            f"({sponsor_replacements} rows)",
            file=sys.stderr,
        )

    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(output, "w") as f:
        for row in ordered:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    print(f"[info] wrote {len(ordered)} predictions to {output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
