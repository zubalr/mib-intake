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
from mib.policy import UNREADABLE_PATH, Calibration, decide
from mib.schema import Prediction, write_jsonl

# Per-PDF wall-clock ceiling. The budget is 6 s/PDF *averaged* over the set, so
# a single pathological packet may overrun -- but it must not stall the run.
# On timeout we emit the best record built so far rather than dropping the case.
PER_PDF_TIMEOUT_S = 55

_LEXICON: Lexicon | None = None
_CALIBRATION: Calibration | None = None


def _worker_init() -> None:
    """Load shared read-only resources once per worker process."""
    global _LEXICON, _CALIBRATION
    _LEXICON = Lexicon()
    _CALIBRATION = Calibration()


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
        sponsor_id="SPN-1000",
        arrival_date="2026-04-01",
        declared_purpose=lexicon.prior_mode("declared_purpose"),
        risk_flags="none",
        fee_status="unknown",
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
        from mib.pipeline import build_prediction  # imported late; see pipeline.py
        prediction = build_prediction(pdf_path, _LEXICON, _CALIBRATION)
    except BaseException as exc:  # noqa: BLE001 -- a dropped case is never correct
        prediction = fallback_prediction(
            case_id, _LEXICON, _CALIBRATION,
            reason=f"{type(exc).__name__}: {exc}",
        )
        print(f"[warn] {case_id}: {type(exc).__name__}: {exc}", file=sys.stderr)
        if os.environ.get("MIB_DEBUG"):
            traceback.print_exc()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)

    return prediction.to_row() | {"_debug": prediction.debug}


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

    workers = args.workers or min(os.cpu_count() or 4, 8)
    print(f"[info] {len(pdfs)} packets, {workers} workers", file=sys.stderr)

    rows: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as pool:
        futures = {pool.submit(process_one, str(p)): p for p in pdfs}
        for done, future in enumerate(as_completed(futures), 1):
            pdf = futures[future]
            try:
                row = future.result()
            except BaseException as exc:  # noqa: BLE001 -- worker died outright
                _worker_init()
                assert _LEXICON is not None and _CALIBRATION is not None
                row = fallback_prediction(
                    pdf.stem, _LEXICON, _CALIBRATION,
                    reason=f"worker died: {type(exc).__name__}",
                ).to_row()
                print(f"[warn] {pdf.stem}: worker died: {exc}", file=sys.stderr)
            rows[row["case_id"]] = row
            if done % 250 == 0:
                print(f"[info] {done}/{len(pdfs)}", file=sys.stderr)

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
