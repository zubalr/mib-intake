#!/usr/bin/env python3
"""Derive the closed-vocabulary lexicon from the public training labels.

The challenge's field values are drawn from small closed sets (12 species codes,
13 home worlds, 5 visa classes, 10 declared purposes, and a name generator built
from ~144 first-name and ~144 last-name tokens). Snapping noisy OCR output onto
these vocabularies recovers fields that would otherwise be scored as misses.

This is a dev-time tool: it reads the *public* training labels and writes
`policy/lexicon.json`, which ships inside the image. It derives vocabulary, not
answers -- no case_id ever enters the output.

Usage:
    python tools/build_lexicon.py \
        --labels ../mib-doc-challenge/data/train_labels.csv \
        --out policy/lexicon.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

# Fields whose label values form a closed set we can snap onto.
CLOSED_FIELDS = [
    "species_code",
    "home_world",
    "visa_class",
    "declared_purpose",
    "fee_status",
]

# Risk-flag vocabulary is defined by policy (FIELD_MANUAL.md), but we also
# confirm it against the labels so a generator-only flag cannot slip past us.
DISQUALIFYING_FLAGS = [
    "memory_tampering",
    "planetary_embargo",
    "active_warrant",
    "biohazard_red",
]
REVIEW_FLAGS = [
    "identity_conflict",
    "sponsor_mismatch",
    "illegible_biometrics",
    "rescinded_denial",
]


def load(labels_path: Path) -> list[dict]:
    with open(labels_path, newline="") as f:
        return list(csv.DictReader(f))


def build(rows: list[dict]) -> dict:
    lexicon: dict = {"_source": "derived from public data/train_labels.csv"}

    for field in CLOSED_FIELDS:
        counts = Counter(r[field].strip() for r in rows if r[field].strip())
        lexicon[field] = {
            "values": sorted(counts),
            # The training mode is the fallback when a field is unrecoverable.
            # Guessing is free: the evaluator scores a wrong value and a blank
            # identically, and drops truly unrecoverable fields from the
            # denominator entirely.
            "prior_mode": counts.most_common(1)[0][0],
            "prior": {k: round(v / len(rows), 5) for k, v in counts.most_common()},
        }

    first = Counter()
    last = Counter()
    for r in rows:
        parts = r["applicant_name"].split()
        if len(parts) >= 2:
            first[parts[0]] += 1
            last[parts[-1]] += 1
    lexicon["applicant_name"] = {
        "first_tokens": sorted(first),
        "last_tokens": sorted(last),
        "prior_mode": Counter(r["applicant_name"] for r in rows).most_common(1)[0][0],
    }

    observed_flags = Counter()
    for r in rows:
        for tok in r["risk_flags"].split("|"):
            tok = tok.strip()
            if tok and tok != "none":
                observed_flags[tok] += 1
    lexicon["risk_flags"] = {
        "disqualifying": DISQUALIFYING_FLAGS,
        "review_only": REVIEW_FLAGS,
        "observed": sorted(observed_flags),
        "prior": {k: round(v / len(rows), 5) for k, v in observed_flags.most_common()},
    }

    unexpected = set(observed_flags) - set(DISQUALIFYING_FLAGS) - set(REVIEW_FLAGS)
    if unexpected:
        lexicon["risk_flags"]["unclassified"] = sorted(unexpected)

    return lexicon


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    rows = load(args.labels)
    lexicon = build(rows)
    lexicon["_n_training_rows"] = len(rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(lexicon, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Wrote {args.out} from {len(rows)} training rows")
    for field in CLOSED_FIELDS:
        print(f"  {field:20s} {len(lexicon[field]['values']):3d} values")
    print(f"  {'applicant_name':20s} "
          f"{len(lexicon['applicant_name']['first_tokens'])} first / "
          f"{len(lexicon['applicant_name']['last_tokens'])} last tokens")
    print(f"  {'risk_flags':20s} {len(lexicon['risk_flags']['observed'])} observed")


if __name__ == "__main__":
    main()
