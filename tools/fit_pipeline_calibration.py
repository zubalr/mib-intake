#!/usr/bin/env python3
"""Fit calibration by running the *real* pipeline over the labelled training set.

Fitting on true field values (tools/fit_calibration.py) measures the policy
ceiling. This measures the system we actually ship: paths are assigned from
extracted evidence, so the fitted probabilities absorb extraction error too.
That matters because a path's reliability depends on how well we can read the
packets that land on it, not just on the policy rule.

Two-pass by necessity: decision paths are computed first (they are a pure
function of the extracted record), then the outcome distribution per path is
fitted, then written back for the pipeline to consume.

Usage:
    PYTHONPATH=. python tools/fit_pipeline_calibration.py \
        --pdf-dir ../mib-doc-challenge/data/train \
        --labels ../mib-doc-challenge/data/train_labels.csv \
        --out policy/calibration.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from mib.extract import parse_packet
from mib.lexicon import Lexicon
from mib.cli import corpus_reference_date
from mib.pipeline import ADJUDICATOR_NOTE_PATH, extract_packet
from mib.policy import (
    APPROVED, DENIED, NEEDS_REVIEW, OUTCOMES, UNKNOWN, UNREADABLE_PATH,
    Record, apply_reference_date, corpus_revoked_sponsors, decide, decision_path,
)

PRIOR_STRENGTH = 4.0
_LEX: Lexicon | None = None


def _init() -> None:
    global _LEX
    _LEX = Lexicon()


def extract_one(pdf: str):
    """Extract one packet using the *same* code path the pipeline ships.

    Rebuilding the Record here (an earlier version did) lets the fitter drift
    from the shipped pipeline, which silently mis-calibrates every path.
    """
    assert _LEX is not None
    ex = extract_packet(Path(pdf), _LEX)
    return ex.record, ex.note


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdf-dir", type=Path,
                    help="Extract from PDFs. Ignored when --cache is given.")
    ap.add_argument("--cache", type=Path,
                    help="Read a tools/build_cache.py cache instead of re-extracting.")
    ap.add_argument("--labels", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    if not args.cache and not args.pdf_dir:
        ap.error("one of --cache or --pdf-dir is required")

    with open(args.labels, newline="") as f:
        truth = {r["case_id"]: r["adjudication"].strip() for r in csv.DictReader(f)}

    # The cache holds exactly what `extract_one` would recompute, so reading it
    # is not an approximation -- it is the same extraction, already done. Fitting
    # from PDFs spent a full extraction pass on every calibration refit, which
    # doubled the cost of every experiment.
    if args.cache:
        from tools.build_cache import load_cache
        rows = [r for r in load_cache(args.cache)["rows"] if not r["failed"]]
        case_ids = [r["case_id"] for r in rows]
        results = [(r["record"], r["note"]) for r in rows]
    else:
        pdfs = sorted(args.pdf_dir.glob("*.pdf"))
        case_ids = [p.stem for p in pdfs]
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init) as pool:
            results = list(pool.map(extract_one, [str(p) for p in pdfs], chunksize=16))

    # Same corpus-derived staleness reference the CLI uses at scoring time.
    reference = corpus_reference_date([r for r, _ in results])
    revoked = corpus_revoked_sponsors([r for r, _ in results])
    print(f"staleness reference date: {reference}")
    print(f"corpus-derived revoked sponsors: {sorted(revoked)}")

    counts: dict[str, Counter] = defaultdict(Counter)
    note_correct = note_total = 0
    for case_id, (record, note) in zip(case_ids, results):
        t = truth[case_id]
        apply_reference_date(record, reference, revoked)
        path = ADJUDICATOR_NOTE_PATH if note is not None else decision_path(record)
        counts[path][t] += 1
        if note is not None:
            note_total += 1
            note_correct += int(note == t)

    global_counts = Counter(truth.values())
    total = sum(global_counts.values())
    prior = {o: global_counts[o] / total for o in OUTCOMES}

    paths = {}
    for path, c in counts.items():
        n = sum(c.values())
        paths[path] = {
            o: round((c[o] + PRIOR_STRENGTH * prior[o]) / (n + PRIOR_STRENGTH), 6)
            for o in OUTCOMES
        }

    # See tools/fit_calibration.py: an unreadable packet is illegible by
    # definition, and FIELD_MANUAL.md makes illegible a NEEDS_REVIEW.
    paths.setdefault(UNREADABLE_PATH,
                     {"APPROVED": 0.12, "DENIED": 0.28, "NEEDS_REVIEW": 0.60})

    # Laplace-smoothed so a perfect run does not claim probability 1.0.
    accuracies = {}
    if note_total:
        accuracies[ADJUDICATOR_NOTE_PATH] = round(
            (note_correct + 1.0) / (note_total + 2.0), 6)

    payload = {
        "_source": "fitted by running the real pipeline over the labelled training set",
        "_prior_strength": PRIOR_STRENGTH,
        "_n_rows": len(case_ids),
        "_note_accuracy_raw": f"{note_correct}/{note_total}",
        "fallback": {o: round(prior[o], 6) for o in OUTCOMES},
        "paths": paths,
        "accuracies": accuracies,
        "_support": {p: sum(c.values()) for p, c in counts.items()},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Wrote {args.out}")
    print(f"adjudicator note accuracy: {note_correct}/{note_total}\n")
    print(f"{'path':30s} {'n':>5s}  {'P(A)':>6s} {'P(D)':>6s} {'P(NR)':>6s}  -> decision")
    for path in sorted(counts, key=lambda p: -sum(counts[p].values())):
        n = sum(counts[path].values())
        p = paths[path]
        d, conf = decide(p)
        marker = "  (note-driven)" if path == ADJUDICATOR_NOTE_PATH else ""
        print(f"{path:30s} {n:5d}  {p['APPROVED']:6.3f} {p['DENIED']:6.3f} "
              f"{p['NEEDS_REVIEW']:6.3f}  -> {d} @ {conf:.2f}{marker}")


if __name__ == "__main__":
    main()
