#!/usr/bin/env python3
"""Produce predictions from cached extraction without reading PDFs.

The cached path reuses the production reference-date and finalization functions
to keep scoring behavior consistent with the container.

Usage:
    PYTHONPATH=. python tools/predict_from_cache.py \\
        --cache scratch/cache_train.pkl --out /tmp/mib-train/cached.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mib.cli import (_enforce_output_schema, corpus_median_date,
                     corpus_reference_date, fallback_prediction,
                     resolve_fallback_sponsors)
from mib.lexicon import Lexicon
from mib.policy import (Calibration, apply_reference_date, corpus_revoked_sponsors,
                        corpus_years)
from mib.pipeline import finalize, resolve_printed_date
from tools.build_cache import load_cache


def build_rows(cache: dict, calibration: Calibration, lexicon: Lexicon,
               adjudicator=None) -> list[dict]:
    rows = cache["rows"]
    good = [r for r in rows if not r["failed"]]
    reference = corpus_reference_date([r["record"] for r in good])
    revoked = corpus_revoked_sponsors([r["record"] for r in good])
    median_date = corpus_median_date([r["record"] for r in good])
    years = corpus_years([r["record"] for r in good])

    out: list[dict] = []
    for row in rows:
        if row["failed"]:
            out.append(fallback_prediction(
                row["case_id"], lexicon, calibration, row.get("reason", "?")).to_row())
            continue
        record = apply_reference_date(row["record"], reference, revoked)
        resolve_printed_date(row["printed"], record, median_date, years)
        prediction = finalize(row["printed"], record, row["note"], calibration,
                              adjudicator=adjudicator, features=row.get("features"))
        out.append(prediction.to_row())

    _enforce_output_schema(out)
    resolve_fallback_sponsors(out)
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
