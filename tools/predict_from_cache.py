#!/usr/bin/env python3
"""Produce predictions.jsonl from a cached extraction, without touching PDFs.

Turns a ~20 minute measurement cycle into a couple of seconds, which is what
makes decision-rule and model experiments practical at all.

Deliberately reuses `mib.cli.corpus_reference_date` and `mib.pipeline.finalize`
rather than reimplementing them, so cached scoring cannot drift from what the
container actually does. The first thing this was used for was reproducing the
live pipeline's score exactly -- a cached scorer that quietly disagrees with the
shipped one is worse than no scorer.

Usage:
    PYTHONPATH=. python tools/predict_from_cache.py \\
        --cache scratch/cache_train.pkl --out /tmp/mib-train/cached.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mib.cli import _enforce_output_schema, corpus_reference_date, fallback_prediction
from mib.lexicon import Lexicon
from mib.policy import Calibration
from mib.pipeline import finalize
from tools.build_cache import load_cache


def build_rows(cache: dict, calibration: Calibration, lexicon: Lexicon,
               adjudicator=None) -> list[dict]:
    rows = cache["rows"]
    good = [r for r in rows if not r["failed"]]
    reference = corpus_reference_date([r["record"] for r in good])

    out: list[dict] = []
    for row in rows:
        if row["failed"]:
            out.append(fallback_prediction(
                row["case_id"], lexicon, calibration, row.get("reason", "?")).to_row())
            continue
        record = row["record"]
        if record.receipt_date is None:
            record.receipt_date = reference
        prediction = finalize(row["printed"], record, row["note"], calibration,
                              adjudicator=adjudicator, features=row.get("features"))
        out.append(prediction.to_row())

    _enforce_output_schema(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--calibration", type=Path, default=None)
    ap.add_argument("--model", type=Path, default=None,
                    help="Optional trained adjudicator (.joblib).")
    args = ap.parse_args()

    cache = load_cache(args.cache)
    calibration = Calibration(args.calibration) if args.calibration else Calibration()
    lexicon = Lexicon()

    adjudicator = None
    if args.model:
        from mib.model import Adjudicator
        adjudicator = Adjudicator.load(args.model)

    rows = build_rows(cache, calibration, lexicon, adjudicator)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"Wrote {len(rows)} predictions to {args.out}")


if __name__ == "__main__":
    main()
