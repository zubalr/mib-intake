#!/usr/bin/env python3
"""Extract every packet once and cache the result.

Extraction (PDF parse + OCR) is ~99% of pipeline cost; adjudication is
microseconds. Before this, each measurement cycle re-extracted all 1,000
training packets **twice** -- once to refit calibration, once to score --
costing ~20 minutes per experiment. With a cache, extraction happens once and
every downstream experiment (calibration refit, model training, decision-rule
changes, scoring) runs in seconds against the cached evidence.

The cache is also the training set for the learned adjudicator: it stores the
feature vector alongside the record, so `tools/train_adjudicator.py` never
touches a PDF.

Rebuild the cache whenever anything in the extraction path changes
(`mib/extract.py`, `mib/ocr.py`, `mib/pdfio.py`, `mib/lexicon.py`,
`mib/features.py`). A stale cache silently measures the wrong system, so the
cache records a fingerprint of those files and consumers warn on mismatch.

Usage:
    PYTHONPATH=. python tools/build_cache.py \\
        --pdf-dir ../mib-doc-challenge/data/train \\
        --out scratch/cache_train.pkl
"""

from __future__ import annotations

import argparse
import hashlib
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from mib.lexicon import Lexicon
from mib.pipeline import extract_packet

# Files whose contents change what extraction produces.
EXTRACTION_SOURCES = ("extract.py", "ocr.py", "pdfio.py", "lexicon.py",
                      "features.py", "pipeline.py")

_LEX: Lexicon | None = None


def extraction_fingerprint() -> str:
    """Hash of the extraction code, so a stale cache can be detected."""
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parent.parent / "mib"
    for name in sorted(EXTRACTION_SOURCES):
        path = root / name
        if path.exists():
            digest.update(path.read_bytes())
    lex = Path(__file__).resolve().parent.parent / "policy" / "lexicon.json"
    if lex.exists():
        digest.update(lex.read_bytes())
    return digest.hexdigest()[:16]


def _init() -> None:
    global _LEX
    _LEX = Lexicon()


def _one(pdf: str) -> dict:
    assert _LEX is not None
    case_id = Path(pdf).stem
    try:
        ex = extract_packet(Path(pdf), _LEX)
        return {"case_id": case_id, "printed": ex.printed, "record": ex.record,
                "note": ex.note, "features": ex.features, "failed": False}
    except BaseException as exc:  # noqa: BLE001 - one bad packet must not stop the build
        return {"case_id": case_id, "failed": True,
                "reason": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdf-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    pdfs = sorted(args.pdf_dir.glob("*.pdf"))
    if args.limit:
        pdfs = pdfs[: args.limit]
    if not pdfs:
        raise SystemExit(f"No PDFs under {args.pdf_dir}")

    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init) as pool:
        rows = list(pool.map(_one, [str(p) for p in pdfs], chunksize=8))
    elapsed = time.time() - started

    failed = sum(r["failed"] for r in rows)
    payload = {
        "fingerprint": extraction_fingerprint(),
        "pdf_dir": str(args.pdf_dir),
        "n": len(rows),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Cached {len(rows)} packets to {args.out}")
    print(f"  fingerprint : {payload['fingerprint']}")
    print(f"  failures    : {failed}")
    print(f"  elapsed     : {elapsed:.0f}s ({elapsed / max(1, len(rows)):.2f}s/pdf)")


def load_cache(path: str | Path, *, warn_stale: bool = True) -> dict:
    """Load a cache, warning loudly if the extraction code has moved on."""
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if warn_stale:
        current = extraction_fingerprint()
        if payload.get("fingerprint") != current:
            print(f"[warn] cache {path} is STALE "
                  f"(built {payload.get('fingerprint')}, code {current}). "
                  f"Rebuild with tools/build_cache.py before trusting results.")
    return payload


if __name__ == "__main__":
    main()
