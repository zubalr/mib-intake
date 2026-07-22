"""Span-level PDF text extraction with visibility classification.

This is the foundation of the trust layer, and it exists because of one
non-obvious PyMuPDF behaviour found by probing (see WORKLOG):

    `page.get_text()` silently drops every span whose geometry falls outside the
    **CropBox**.

That default is safe but useless. Text placed outside the visible crop is one of
the injection vectors `EVALUATION.md` names explicitly, and a system that simply
never sees it cannot distinguish "this field has no trusted evidence" from "this
field was supplied by an injection" -- a distinction the scoring rewards.

So we deliberately do the opposite: record the true visible crop, widen the
CropBox to the MediaBox so hidden geometry becomes extractable, then classify
every span against the *original* crop. Nothing is thrown away, and nothing
hidden is ever mistaken for visible.

Visibility is decided by three independent signals:

  * **render mode / alpha** -- invisible text (PDF text render mode 3) is the
    classic OCR-layer trick and the classic hiding place. PyMuPDF reports
    ``alpha`` on a 0-255 scale, *not* 0-1; an early version of this code used
    ``alpha == 0`` with a default of ``1``, which happened to work only by luck.
  * **colour vs. background** -- white-on-white text.
  * **geometry** -- outside the visible crop.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fitz

# Above this relative luminance a glyph is treated as white-on-white. Kept below
# 1.0 because near-white (254,254,254) is used to evade exact-white checks.
WHITE_LUMINANCE = 0.93


@dataclass(frozen=True)
class Span:
    """One run of text plus everything needed to decide whether to trust it."""

    text: str
    page: int
    bbox: tuple[float, float, float, float]
    font: str
    size: float
    color: int
    invisible: bool      # render mode 3 / zero alpha
    white: bool          # white-on-white
    offcrop: bool        # outside the visible page crop
    rotated: bool        # page is rotated, or the span's baseline is not level

    @property
    def hidden(self) -> bool:
        return self.invisible or self.white or self.offcrop

    @property
    def hide_reasons(self) -> tuple[str, ...]:
        reasons = []
        if self.invisible:
            reasons.append("invisible")
        if self.white:
            reasons.append("white")
        if self.offcrop:
            reasons.append("offcrop")
        return tuple(reasons)


def relative_luminance(srgb: int) -> float:
    r = (srgb >> 16) & 0xFF
    g = (srgb >> 8) & 0xFF
    b = srgb & 0xFF
    return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0


def extract_spans(doc_or_path: "fitz.Document | str | Path") -> list[Span]:
    """Extract every span in the document, hidden ones included and labelled."""
    owned = False
    if isinstance(doc_or_path, (str, Path)):
        doc = fitz.open(doc_or_path)
        owned = True
    else:
        doc = doc_or_path

    spans: list[Span] = []
    try:
        for page_index, page in enumerate(doc):
            # Capture the genuine visible area *before* widening the crop.
            visible = fitz.Rect(page.rect)
            try:
                page.set_cropbox(page.mediabox)
            except (ValueError, RuntimeError):
                # A malformed MediaBox must not cost us the page's visible text.
                pass

            page_rotated = page.rotation % 360 != 0

            for block in page.get_text("dict").get("blocks", []):
                if block.get("type") != 0:  # 0 = text, 1 = image
                    continue
                for line in block.get("lines", []):
                    # dir is the writing-direction unit vector; (1,0) is level.
                    direction = line.get("dir", (1.0, 0.0))
                    skewed = abs(float(direction[1])) > 0.01

                    for raw in line.get("spans", []):
                        text = raw.get("text", "")
                        if not text.strip():
                            continue

                        bbox = fitz.Rect(raw["bbox"])
                        color = int(raw.get("color", 0))
                        # 0-255 scale; missing means fully opaque.
                        alpha = int(raw.get("alpha", 255))

                        spans.append(Span(
                            text=text,
                            page=page_index,
                            bbox=tuple(bbox),
                            font=str(raw.get("font", "")),
                            size=float(raw.get("size", 0.0)),
                            color=color,
                            invisible=alpha == 0,
                            white=relative_luminance(color) >= WHITE_LUMINANCE,
                            offcrop=not bbox.intersects(visible),
                            rotated=page_rotated or skewed,
                        ))
    finally:
        if owned:
            doc.close()

    return spans


def visible_text(spans: list[Span]) -> str:
    return "\n".join(s.text for s in spans if not s.hidden)


def hidden_text(spans: list[Span]) -> str:
    return "\n".join(s.text for s in spans if s.hidden)
