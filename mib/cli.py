"""Entrypoint: read a directory of PDF packets, write predictions.

Contract (DOCKER_SUBMISSION.md): exactly two arguments, `<input_pdf_dir>` and
`<output_predictions_path>`. Runs offline, read-only root filesystem, scratch in
/tmp, 4 vCPU / 8 GiB, averaging 6 seconds per PDF.

Two invariants this module exists to guarantee, both worth more than any
extraction cleverness (see WORKLOG §1.1 and §1.2):

  * **Every input PDF produces exactly one output row.** A crash, a timeout, or
    an unparseable file falls back to a prior-based record rather than dropping
    the case. The evaluator charges a missing case its full classification and
    extraction denominator while the advertised "missing-case penalty" is a
    rounding error -- omitting is never correct.
  * **No field is ever blank.** `validate_submission.py` rejects records whose
    `sponsor_id` or `arrival_date` are malformed, and the evaluator scores a
    wrong value identically to a blank, so a prior-mode guess is free upside.
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
                        corpus_revoked_sponsors, decide)
from mib.schema import FALLBACK_ARRIVAL_DATE, FALLBACK_SPONSOR_ID, Prediction, write_jsonl

# Per-PDF wall-clock ceiling. The budget is 6 s/PDF *averaged* over the set, so
# a single pathological packet may overrun -- but it must not stall the run.
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
    # Optional. If the artifact is missing or unreadable the run still completes
    # on the hand-built paths rather than failing -- a model is an optimisation,
    # not a dependency.
    from mib.model import Adjudicator
    _ADJUDICATOR = Adjudicator.try_load()


def fallback_prediction(case_id: str, lexicon: Lexicon, calibration: Calibration,
                        reason: str) -> Prediction:
    """A never-blank record for a packet we could not read at all.

    Every field takes its training-prior mode.

    The adjudication deliberately does *not* run the normal policy path. An
    all-unknown Record lands on `fee_unknown`, whose 94% NEEDS_REVIEW rate was
    measured on packets we successfully read and whose fee genuinely was
    unknown. Reporting 0.94 here would be badly overconfident: we know nothing
    about this packet, so the honest probability is the unreadable-packet prior.
    Overconfidence on damaged packets is exactly what the Brier term punishes.
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
        # Prior mode like every other field here. The *adjudication* stays
        # honest (it uses the unreadable-packet prior, below) -- only the
        # printed field guesses, and a guess on a field we could not read is
        # free under the evaluator's scoring.
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
    except BaseException as exc:  # noqa: BLE001 -- a dropped case is never correct
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

    Packets carry only an arrival date -- there is no receipt date to read (the
    forensics pass found exactly one date per packet). Rather than hardcode a
    constant fitted to the public corpus, take the latest arrival date actually
    present in the set being scored: intake cannot receive a packet before the
    applicant arrives, so the newest arrival is a lower bound on "now" for this
    corpus. A private test set from a different period then still works.
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
        # No readable date anywhere in the corpus. Any constant here would be
        # fitted to the era of whatever set it was chosen on, so return None and
        # let staleness be *undeterminable* -- `_is_stale` already reports that
        # honestly, and `decision_path` has a `staleness_indeterminate` branch
        # for it. Guessing a date would silently manufacture staleness verdicts
        # out of nothing.
        return None
    dates.sort()

    # Trim outliers *before* taking the percentile.
    #
    # A high percentile alone is not enough. It survives a single misread, but
    # `6` and `8` collide often enough that a whole *cluster* of packets reads
    # "2028" where the truth is "2026" -- 33 of 1,000 once literal mining began
    # recovering dates the label parser had been dropping. At 3.3% contamination
    # the 98th percentile lands inside the bad cluster, which pushed the
    # reference two years out, marked 372 packets stale, and cost 3.3
    # classification points.
    #
    # The median cannot be moved by 3% contamination, so measure distance from
    # it. The window floor is deliberately generous -- it must not trim a corpus
    # that genuinely spans a wide range. The goal is only to drop dates that sit
    # implausibly far from every other date in the same corpus.
    median = dates[len(dates) // 2]
    deviations = sorted(abs((d - median).days) for d in dates)
    mad = deviations[len(deviations) // 2]
    window = max(730, 6 * mad)
    kept = [d for d in dates if abs((d - median).days) <= window] or dates

    index = min(len(kept) - 1, int(len(kept) * 0.98))
    return kept[index].isoformat()


def corpus_median_date(records) -> str | None:
    """Median arrival date of the corpus being scored.

    Used to fill `arrival_date` on packets where no trusted evidence supplied
    one. The evaluator scores a wrong value exactly like a blank and drops
    genuinely unrecoverable fields from the denominator, so this guess is free
    upside -- but only if it is drawn from the corpus in hand rather than from
    the era of the public training set.
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
    from mib.pipeline import finalize

    good = [r for r in rows.values() if not r.get("failed")]
    reference = corpus_reference_date([r["record"] for r in good])
    revoked = corpus_revoked_sponsors([r["record"] for r in good])
    median_date = corpus_median_date([r["record"] for r in good])
    print(f"[info] staleness reference date: {reference}", file=sys.stderr)
    print(f"[info] corpus-derived revoked sponsors: {sorted(revoked)}", file=sys.stderr)
    print(f"[info] adjudicator: "
          f"{'model ' + str(_ADJUDICATOR.metadata.get('kind')) if _ADJUDICATOR else 'hand-built paths'}",
          file=sys.stderr)

    final: dict[str, dict] = {}
    for case_id, row in rows.items():
        if row.get("failed"):
            final[case_id] = fallback_prediction(
                case_id, _LEXICON, _CALIBRATION, row.get("reason", "?")).to_row()
            continue
        record = apply_reference_date(row["record"], reference, revoked)
        if record.arrival_date == "unknown" and median_date:
            row["printed"]["arrival_date"] = median_date
        final[case_id] = finalize(
            row["printed"], record, row["note"], _CALIBRATION,
            adjudicator=_ADJUDICATOR, features=row.get("features")).to_row()
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
