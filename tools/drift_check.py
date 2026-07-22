#!/usr/bin/env python3
"""Compare two prediction sets for distribution drift.

The validation labels are private, so the validation score cannot be measured
directly. What *can* be measured is whether the system behaves the same way on
both corpora. If validation shows a very different mix of adjudications,
decision paths, unknown-field rates or confidences than training, then either
the two sets differ in composition or something in the pipeline is failing
silently on unfamiliar layouts -- and the train score stops being evidence for
anything.

This is a smoke test with judgement attached, not a hypothesis test: the two
splits are not required to be identically distributed, so read the deltas and
ask whether each one has an explanation.

Usage:
    PYTHONPATH=. python tools/drift_check.py \\
        --a /tmp/mib-train/run15-paths.jsonl --a-name train \\
        --b /tmp/mib-val-out/predictions.jsonl --b-name validation
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

FIELDS = ("applicant_name", "species_code", "home_world", "visa_class",
          "sponsor_id", "arrival_date", "declared_purpose", "risk_flags",
          "fee_status")

# What the pipeline prints when it recovered nothing for a field. A rising
# fallback rate on validation is the clearest single sign that extraction is
# failing on layouts it has not seen.
FALLBACKS = {
    "sponsor_id": "SPN-1000",
    "arrival_date": "2026-04-01",
    "fee_status": "unknown",
    "risk_flags": "none",
}


def load(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def pct(count: int, total: int) -> str:
    return f"{100.0 * count / total:5.1f}%" if total else "    -"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", required=True, type=Path)
    ap.add_argument("--b", required=True, type=Path)
    ap.add_argument("--a-name", default="A")
    ap.add_argument("--b-name", default="B")
    args = ap.parse_args()

    a, b = load(args.a), load(args.b)
    na, nb = len(a), len(b)
    print(f"{args.a_name}: {na} rows      {args.b_name}: {nb} rows\n")

    def section(title: str) -> None:
        print(f"-- {title}")
        print(f"   {'key':28} {args.a_name:>10} {args.b_name:>10}   delta")

    section("adjudication mix")
    ca, cb = Counter(r["adjudication"] for r in a), Counter(r["adjudication"] for r in b)
    for key in sorted(set(ca) | set(cb)):
        fa, fb = ca[key] / na, cb[key] / nb
        print(f"   {key:28} {pct(ca[key], na):>10} {pct(cb[key], nb):>10}   "
              f"{100*(fb-fa):+6.1f}")

    # Decision paths are only present when the caller kept debug output; the
    # shipped schema strips it, so this section is skipped on real submissions.
    pa = Counter((r.get("debug") or {}).get("path") for r in a)
    pb = Counter((r.get("debug") or {}).get("path") for r in b)
    if set(pa) | set(pb) != {None}:
        section("decision path")
        for key in sorted((set(pa) | set(pb)) - {None},
                          key=lambda k: -(pa[k] + pb[k])):
            print(f"   {str(key):28} {pct(pa[key], na):>10} {pct(pb[key], nb):>10}   "
                  f"{100*(pb[key]/nb - pa[key]/na):+6.1f}")

    section("fallback (nothing recovered) rate")
    for field, value in FALLBACKS.items():
        fa = sum(1 for r in a if r.get(field) == value)
        fb = sum(1 for r in b if r.get(field) == value)
        print(f"   {field:28} {pct(fa, na):>10} {pct(fb, nb):>10}   "
              f"{100*(fb/nb - fa/na):+6.1f}")

    section("mean confidence")
    ma = sum(r["confidence"] for r in a) / na
    mb = sum(r["confidence"] for r in b) / nb
    print(f"   {'mean':28} {ma:10.3f} {mb:10.3f}   {mb-ma:+6.3f}")
    for lo, hi in ((0.0, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)):
        fa = sum(1 for r in a if lo <= r["confidence"] < hi)
        fb = sum(1 for r in b if lo <= r["confidence"] < hi)
        print(f"   {f'[{lo:.1f},{hi:.1f})':28} {pct(fa, na):>10} {pct(fb, nb):>10}   "
              f"{100*(fb/nb - fa/na):+6.1f}")

    section("vocabulary coverage (distinct values)")
    for field in FIELDS:
        va = {r.get(field) for r in a}
        vb = {r.get(field) for r in b}
        unseen = len(vb - va)
        print(f"   {field:28} {len(va):10} {len(vb):10}   "
              f"{unseen:+6} new in {args.b_name}")


if __name__ == "__main__":
    main()
