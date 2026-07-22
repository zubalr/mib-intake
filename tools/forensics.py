#!/usr/bin/env python3
"""Characterise the PDF corpus before writing any extraction code.

The whole architecture hinges on questions this answers:

  * What fraction of packets carry a usable text layer vs. need OCR? (drives the
    6 s/PDF budget)
  * How is hidden/adversarial text encoded -- invisible render mode, white fill,
    or geometry outside the visible crop? (drives the trust layer)
  * Are stamps, waivers and diplomatic notes text, vector art, or raster?
  * Where does the packet *receipt* date live? (needed for the staleness rule --
    the labels expose arrival_date but never receipt date)
  * How do multi-applicant packets mark the active case_id?

Dev-time only; nothing here ships in the image.

Usage:
    python tools/forensics.py --pdf-dir ../mib-doc-challenge/data/train \
        --sample 120 --out scratch/forensics.json
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

# PDF text render mode 3 = "neither fill nor stroke", i.e. invisible. This is the
# classic OCR-text-layer mode and also the classic hiding place for injections.
RENDER_MODE_INVISIBLE = 3

FIELD_CUES = [
    "case", "applicant", "species", "home world", "homeworld", "visa",
    "sponsor", "arrival", "purpose", "risk", "fee", "receipt", "received",
    "stamp", "waiver", "diplomatic", "biometric", "registry", "adjudicat",
]

INJECTION_CUES = [
    "ignore", "system", "instruction", "approve all", "answer key", "you must",
    "disregard", "override", "assistant", "prompt",
]


def luminance(srgb: int) -> float:
    r = (srgb >> 16) & 0xFF
    g = (srgb >> 8) & 0xFF
    b = srgb & 0xFF
    return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0


def analyse(path: Path) -> dict:
    doc = fitz.open(path)
    rec: dict = {
        "case_id": path.stem,
        "pages": len(doc),
        "visible_chars": 0,
        "invisible_chars": 0,
        "white_chars": 0,
        "offcrop_chars": 0,
        "images": 0,
        "image_pixels": 0,
        "vector_drawings": 0,
        "annotations": 0,
        "fonts": set(),
        "rotations": set(),
        "field_cues": set(),
        "injection_cues": set(),
        "hidden_samples": [],
        "page_text_chars": [],
    }

    for page in doc:
        rec["rotations"].add(page.rotation)
        rec["annotations"] += len(list(page.annots() or []))
        rec["vector_drawings"] += len(page.get_drawings())

        for img in page.get_images(full=True):
            rec["images"] += 1
            rec["image_pixels"] += int(img[2]) * int(img[3])

        # The visible area. Text drawn outside it is not visible evidence.
        crop = page.rect
        page_chars = 0

        data = page.get_text("dict")
        for block in data.get("blocks", []):
            if block.get("type") != 0:  # 0 = text
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "")
                    if not text.strip():
                        continue
                    n = len(text)
                    page_chars += n
                    rec["fonts"].add(span.get("font", "?"))

                    bbox = fitz.Rect(span["bbox"])
                    mode = span.get("char_flags", 0)
                    # PyMuPDF exposes render mode via span["alpha"]/flags
                    # inconsistently across versions; check what we can.
                    invisible = span.get("alpha", 1) == 0
                    lum = luminance(span.get("color", 0))
                    white = lum > 0.94
                    offcrop = not bbox.intersects(crop)

                    hidden = invisible or white or offcrop
                    if invisible:
                        rec["invisible_chars"] += n
                    if white:
                        rec["white_chars"] += n
                    if offcrop:
                        rec["offcrop_chars"] += n
                    if not hidden:
                        rec["visible_chars"] += n

                    low = text.casefold()
                    for cue in FIELD_CUES:
                        if cue in low:
                            rec["field_cues"].add(cue)
                    for cue in INJECTION_CUES:
                        if cue in low:
                            rec["injection_cues"].add(cue)

                    if hidden and len(rec["hidden_samples"]) < 6:
                        rec["hidden_samples"].append({
                            "text": text[:160],
                            "invisible": invisible,
                            "white": white,
                            "offcrop": offcrop,
                            "color": span.get("color", 0),
                        })

        rec["page_text_chars"].append(page_chars)

    doc.close()
    for key in ("fonts", "rotations", "field_cues", "injection_cues"):
        rec[key] = sorted(rec[key], key=str)
    return rec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf-dir", required=True, type=Path)
    parser.add_argument("--sample", type=int, default=100)
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
        except Exception as exc:  # noqa: BLE001 - forensics must not abort
            records.append({"case_id": pdf.stem, "error": repr(exc)})
        if i % 25 == 0:
            print(f"  ...{i}/{len(pdfs)}", flush=True)

    ok = [r for r in records if "error" not in r]
    print(f"\n=== {len(ok)}/{len(records)} parsed ===")
    print(f"pages:            {Counter(r['pages'] for r in ok).most_common()}")
    print(f"rotations seen:   {Counter(x for r in ok for x in r['rotations']).most_common()}")

    no_text = [r for r in ok if r["visible_chars"] < 50]
    print(f"visible text <50 chars (OCR needed): {len(no_text)}/{len(ok)}")
    vis = sorted(r["visible_chars"] for r in ok)
    print(f"visible chars   min/median/max: {vis[0]} / {vis[len(vis)//2]} / {vis[-1]}")

    for kind in ("invisible_chars", "white_chars", "offcrop_chars"):
        n = sum(1 for r in ok if r[kind] > 0)
        print(f"{kind:18s} present in {n}/{len(ok)} packets")

    imgs = sorted(r["images"] for r in ok)
    print(f"images per packet min/median/max: {imgs[0]} / {imgs[len(imgs)//2]} / {imgs[-1]}")
    print(f"annotations present: {sum(1 for r in ok if r['annotations'] > 0)}/{len(ok)}")

    print(f"\nfield cues seen: {Counter(c for r in ok for c in r['field_cues']).most_common()}")
    print(f"injection cues:  {Counter(c for r in ok for c in r['injection_cues']).most_common()}")
    print(f"fonts:           {Counter(f for r in ok for f in r['fonts']).most_common(12)}")

    print("\n--- sample hidden text ---")
    shown = 0
    for r in ok:
        for s in r["hidden_samples"]:
            print(f"  [{r['case_id']}] inv={s['invisible']} white={s['white']} "
                  f"off={s['offcrop']} :: {s['text']!r}")
            shown += 1
            if shown >= 15:
                break
        if shown >= 15:
            break

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(records, f, indent=2, default=str)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
