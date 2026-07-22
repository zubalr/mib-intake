#!/usr/bin/env python3
"""Assert a built Docker image matches the current working tree.

Why this is a script and not a habit:

A `docker build` that fails partway leaves the *previous* image in place under
the same tag. Nothing errors on the next `docker run` -- it simply runs older
code. That happened here: a build hit a BuildKit deadline, and the next
container run produced 40 well-formed predictions and exited 0 while missing a
policy change made twenty minutes earlier. The only reason it was caught was an
expected log line that did not appear.

Every failure mode in this project that actually cost something has been silent
(a pickle read by the wrong scikit-learn, a feature computed before the corpus
statistic existed, a stale extraction cache). This is the same shape, so it gets
the same treatment: an explicit check with a non-zero exit.

Usage:
    PYTHONPATH=. python tools/verify_image.py mib-intake:final
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Everything COPY'd into the image by the Dockerfile.
TRACKED = [*sorted((ROOT / "mib").glob("*.py")),
           *sorted((ROOT / "policy").glob("*")),
           ROOT / "run.sh"]


def digest(paths) -> str:
    sha = hashlib.sha256()
    for path in paths:
        sha.update(path.name.encode())
        sha.update(path.read_bytes())
    return sha.hexdigest()


def main() -> int:
    image = sys.argv[1] if len(sys.argv) > 1 else "mib-intake:final"
    local = digest(TRACKED)

    script = (
        "import hashlib,pathlib;"
        "p=[*sorted(pathlib.Path('/app/mib').glob('*.py')),"
        "*sorted(pathlib.Path('/app/policy').glob('*')),pathlib.Path('/app/run.sh')];"
        "s=hashlib.sha256();"
        "[ (s.update(f.name.encode()), s.update(f.read_bytes())) for f in p ];"
        "print(s.hexdigest())"
    )
    try:
        result = subprocess.run(
            ["docker", "run", "--rm", "--entrypoint", "python3", image, "-c", script],
            capture_output=True, text=True, timeout=180)
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"could not inspect image {image}: {exc}", file=sys.stderr)
        return 2
    if result.returncode != 0:
        print(f"could not inspect image {image}:\n{result.stderr}", file=sys.stderr)
        return 2

    inside = result.stdout.strip()
    if inside != local:
        print(f"IMAGE IS STALE: {image}", file=sys.stderr)
        print(f"  working tree : {local}", file=sys.stderr)
        print(f"  image        : {inside}", file=sys.stderr)
        print("\nRebuild before running anything you intend to trust:\n"
              f"  docker build -t {image} .", file=sys.stderr)
        return 1

    print(f"{image} matches the working tree ({local[:16]}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
