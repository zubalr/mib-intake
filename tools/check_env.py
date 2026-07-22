#!/usr/bin/env python3
"""Assert the current environment matches requirements.txt exactly.

Why this exists as a hard check rather than a note in the README:

The learned adjudicator is a **pickle**. It is written by whatever scikit-learn
version `tools/train_adjudicator.py` runs under and read by whatever version the
Docker image installs. Those two drifted apart without anyone noticing -- the
model was trained on 1.9.0 while the image pinned 1.7.2 -- and the only symptom
was an `InconsistentVersionWarning` buried in container stderr. The run still
produced 200 well-formed predictions. Silent, plausible, wrong-in-principle
output is the worst failure mode available to a scored submission, and it is
undetectable from the score itself.

Usage (run before `docker build`, and after any dependency change):
    PYTHONPATH=. python tools/check_env.py
"""

from __future__ import annotations

import importlib.metadata as metadata
import re
import sys
from pathlib import Path

REQUIREMENTS = Path(__file__).resolve().parent.parent / "requirements.txt"
_PIN_RE = re.compile(r"^([A-Za-z0-9_.\-]+)==([A-Za-z0-9_.\-]+)\s*$")


def pinned() -> dict[str, str]:
    out = {}
    for line in REQUIREMENTS.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = _PIN_RE.match(line)
        if not match:
            raise SystemExit(f"requirements.txt line is not a hard pin: {line!r}")
        out[match.group(1)] = match.group(2)
    return out


def main() -> int:
    problems = []
    for package, want in sorted(pinned().items()):
        try:
            have = metadata.version(package)
        except metadata.PackageNotFoundError:
            problems.append(f"  {package}: pinned {want}, NOT INSTALLED")
            continue
        if have != want:
            problems.append(f"  {package}: pinned {want}, installed {have}")

    if problems:
        print("Environment does not match requirements.txt:", file=sys.stderr)
        print("\n".join(problems), file=sys.stderr)
        print("\nThe adjudicator is a pickle -- training and runtime must agree.\n"
              "Either reinstall (`pip install -r requirements.txt`) or update the\n"
              "pins to match this environment, then RETRAIN so the artifact is\n"
              "written by the same version that will read it.", file=sys.stderr)
        return 1

    print("Environment matches requirements.txt.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
