#!/usr/bin/env python3
"""Fit the decision-path -> outcome-probability table on public training labels.

The calibration table doubles as the classifier: `policy.decide` takes the
expected-value argmax over these probabilities, and reports the probability of
the chosen outcome as the confidence. Because the evaluator's calibration term
is a proper scoring rule (Brier), fitting honest empirical frequencies is
simultaneously the classification-optimal and calibration-optimal choice.

Smoothing matters. A path seen 8 times that happened to be 8/8 DENIED must not
report probability 1.0 -- that is exactly the overconfidence the Brier term
punishes. We shrink each path toward the global prior with a Dirichlet-style
pseudocount.

Usage:
    python tools/fit_calibration.py \
        --labels ../mib-doc-challenge/data/train_labels.csv \
        --out policy/calibration.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from mib.policy import OUTCOMES, Record, decision_path, decide

# Pseudocounts pulling each path toward the global prior. Chosen so a path with
# ~10 observations still moves most of the way to its own empirical rate, while
# a path with 2 observations stays close to the prior.
PRIOR_STRENGTH = 4.0


def record_from_label_row(row: dict, receipt_date: str | None) -> Record:
    """Build a Record from *true* field values.

    This measures the policy ceiling: how well the rules would do given perfect
    extraction. Document-only evidence (stamps, waivers, diplomatic notes) is
    not present in the labels, so those stay False here and the calibration
    absorbs their effect as within-path variance -- which is precisely what we
    want the confidence to reflect.
    """
    flags = frozenset(f.strip() for f in row["risk_flags"].split("|")
                      if f.strip() and f.strip() != "none")
    return Record(
        case_id=row["case_id"],
        visa_class=row["visa_class"].strip(),
        sponsor_id=row["sponsor_id"].strip(),
        fee_status=row["fee_status"].strip(),
        arrival_date=row["arrival_date"].strip(),
        risk_flags=flags,
        receipt_date=receipt_date,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--assumed-receipt-date", default="2026-07-15",
        help="Fallback packet receipt date, used only until the real per-packet "
             "receipt date is read from the PDFs.")
    args = parser.parse_args()

    with open(args.labels, newline="") as f:
        rows = list(csv.DictReader(f))

    counts: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        rec = record_from_label_row(row, args.assumed_receipt_date)
        counts[decision_path(rec)][row["adjudication"].strip()] += 1

    global_counts = Counter(r["adjudication"].strip() for r in rows)
    total = sum(global_counts.values())
    prior = {o: global_counts[o] / total for o in OUTCOMES}

    paths = {}
    for path, c in counts.items():
        n = sum(c.values())
        paths[path] = {
            o: round((c[o] + PRIOR_STRENGTH * prior[o]) / (n + PRIOR_STRENGTH), 6)
            for o in OUTCOMES
        }

    payload = {
        "_source": "fitted on public data/train_labels.csv via tools/fit_calibration.py",
        "_prior_strength": PRIOR_STRENGTH,
        "_n_rows": len(rows),
        "fallback": {o: round(prior[o], 6) for o in OUTCOMES},
        "paths": paths,
        "_support": {p: sum(c.values()) for p, c in counts.items()},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Wrote {args.out}\n")
    print(f"{'path':28s} {'n':>5s}  {'P(A)':>6s} {'P(D)':>6s} {'P(NR)':>6s}  -> decision")
    for path in sorted(counts, key=lambda p: -sum(counts[p].values())):
        n = sum(counts[path].values())
        p = paths[path]
        decision, conf = decide(p)
        print(f"{path:28s} {n:5d}  {p['APPROVED']:6.3f} {p['DENIED']:6.3f} "
              f"{p['NEEDS_REVIEW']:6.3f}  -> {decision} @ {conf:.2f}")


if __name__ == "__main__":
    main()
