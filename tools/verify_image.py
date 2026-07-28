#!/usr/bin/env python3
"""Assert that a built Docker image matches the current working tree.

The check hashes every source and policy file copied into the image. It prevents
a failed build from leaving an older image under the requested tag.

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
