#!/usr/bin/env python3
"""Report document structure, visibility, and OCR requirements for a PDF corpus.

This development tool reuses the production visibility classifier and does not
ship in the runtime image.

Usage:
    PYTHONPATH=. python tools/forensics.py \
        --pdf-dir ../mib-doc-challenge/data/train \
        --sample 150 --out scratch/forensics.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

import fitz  # PyMuPDF

from mib.pdfio import extract_spans

FIELD_CUES = [
    "case", "applicant", "species", "home world", "homeworld", "visa",
    "sponsor", "arrival", "purpose", "risk", "fee", "receipt", "received",
    "stamp", "waiver", "diplomatic", "biometric", "registry", "adjudicat",
    "revoked", "revocation", "status", "hardship", "embargo",
]

INJECTION_CUES = [
    "ignore", "system", "instruction", "approve all", "answer key", "you must",
    "disregard", "override", "assistant", "prompt", "must be approved",
]

DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
SPONSOR_RE = re.compile(r"\bSPN-\d{4}\b")
CASE_RE = re.compile(r"\bMIB-\d{6}\b")


def analyse(path: Path) -> dict:
    doc = fitz.open(path)
    spans = extract_spans(doc)

    rec: dict = {
        "case_id": path.stem,
        "pages": len(doc),
        "visible_chars": sum(len(s.text) for s in spans if not s.hidden),
        "invisible_chars": sum(len(s.text) for s in spans if s.invisible),
        "white_chars": sum(len(s.text) for s in spans if s.white),
        "offcrop_chars": sum(len(s.text) for s in spans if s.offcrop),
        "rotated_spans": sum(1 for s in spans if s.rotated),
        "images": 0,
        "image_pixels": 0,
        "vector_drawings": 0,
        "annotations": 0,
        "fonts": sorted({s.font for s in spans}),
        "rotations": sorted({p.rotation for p in doc}),
        "hidden_samples": [
            {"text": s.text[:200], "reasons": list(s.hide_reasons), "page": s.page}
            for s in spans if s.hidden
        ][:8],
    }

    for page in doc:
        rec["annotations"] += len(list(page.annots() or []))
        rec["vector_drawings"] += len(page.get_drawings())
        for img in page.get_images(full=True):
            rec["images"] += 1
            rec["image_pixels"] += int(img[2]) * int(img[3])
    doc.close()

    vis = "\n".join(s.text for s in spans if not s.hidden)
    hid = "\n".join(s.text for s in spans if s.hidden)
    low_vis = vis.casefold()

    rec["field_cues"] = sorted({c for c in FIELD_CUES if c in low_vis})
    rec["injection_cues_hidden"] = sorted({c for c in INJECTION_CUES if c in hid.casefold()})
    rec["injection_cues_visible"] = sorted({c for c in INJECTION_CUES if c in low_vis})

    rec["visible_dates"] = sorted(set(DATE_RE.findall(vis)))[:10]
    rec["hidden_dates"] = sorted(set(DATE_RE.findall(hid)))[:10]
    rec["visible_sponsors"] = sorted(set(SPONSOR_RE.findall(vis)))[:10]
    rec["visible_case_ids"] = sorted(set(CASE_RE.findall(vis)))[:10]
    rec["visible_sample"] = vis[:1200]
    return rec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf-dir", required=True, type=Path)
    parser.add_argument("--sample", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    pdfs = sorted(args.pdf_dir.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs under {args.pdf_dir}")
    random.Random(args.seed).shuffle(pdfs)
    pdfs = pdfs[: args.sample]

    records = []
    for i, pdf in enumerate(pdfs, 1):
        try:
            records.append(analyse(pdf))
        except Exception as exc:  # noqa: BLE001 - forensics must never abort
            records.append({"case_id": pdf.stem, "error": repr(exc)})
        if i % 50 == 0:
            print(f"  ...{i}/{len(pdfs)}", flush=True)

    ok = [r for r in records if "error" not in r]
    err = [r for r in records if "error" in r]
    print(f"\n=== parsed {len(ok)}/{len(records)} ===")
    if err:
        print(f"errors: {err[:3]}")

    print(f"pages per packet: {Counter(r['pages'] for r in ok).most_common()}")
    print(f"page rotations:   {Counter(x for r in ok for x in r['rotations']).most_common()}")

    vis = sorted(r["visible_chars"] for r in ok)
    print(f"\nvisible chars  min/p10/median/max: {vis[0]} / {vis[len(vis)//10]} / "
          f"{vis[len(vis)//2]} / {vis[-1]}")
    print(f"packets with <50 visible chars (need OCR): "
          f"{sum(1 for r in ok if r['visible_chars'] < 50)}/{len(ok)}")

    for kind in ("invisible_chars", "white_chars", "offcrop_chars"):
        n = sum(1 for r in ok if r[kind] > 0)
        print(f"{kind:18s} present in {n}/{len(ok)} packets")
    print(f"{'rotated_spans':18s} present in "
          f"{sum(1 for r in ok if r['rotated_spans'] > 0)}/{len(ok)} packets")

    imgs = sorted(r["images"] for r in ok)
    print(f"\nimages/packet min/median/max: {imgs[0]} / {imgs[len(imgs)//2]} / {imgs[-1]}")
    print(f"packets with annotations: {sum(1 for r in ok if r['annotations'] > 0)}/{len(ok)}")
    print(f"packets with vector art:  {sum(1 for r in ok if r['vector_drawings'] > 0)}/{len(ok)}")

    visible_fields = Counter(c for r in ok for c in r["field_cues"]).most_common()
    hidden_injections = Counter(
        c for r in ok for c in r["injection_cues_hidden"]
    ).most_common()
    visible_injections = Counter(
        c for r in ok for c in r["injection_cues_visible"]
    ).most_common()
    print(f"\nfield cues (visible):     {visible_fields}")
    print(f"injection cues (hidden):  {hidden_injections}")
    print(f"injection cues (visible): {visible_injections}")

    n_multi_case = sum(1 for r in ok if len(r["visible_case_ids"]) > 1)
    print(f"\npackets naming >1 case id visibly: {n_multi_case}/{len(ok)}")
    n_multi_spn = sum(1 for r in ok if len(r["visible_sponsors"]) > 1)
    print(f"packets naming >1 sponsor visibly: {n_multi_spn}/{len(ok)}")
    ndates = Counter(len(r["visible_dates"]) for r in ok)
    print(f"visible date count per packet: {sorted(ndates.items())}")
    print(f"packets with a hidden-only date: "
          f"{sum(1 for r in ok if r['hidden_dates'] and not r['visible_dates'])}/{len(ok)}")

    print("\n--- sample hidden text ---")
    shown = 0
    for r in ok:
        for s in r["hidden_samples"]:
            reasons = "+".join(s["reasons"])
            print(f"  [{r['case_id']} p{s['page']}] "
                  f"{reasons} :: {s['text'][:110]!r}")
            shown += 1
            if shown >= 20:
                break
        if shown >= 20:
            break

    print("\n--- one packet's visible text ---")
    if ok:
        print(ok[0]["visible_sample"][:1500])

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(records, f, indent=2, default=str)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
