"""Span-level PDF text extraction with visibility classification.

PyMuPDF omits spans outside the CropBox from normal text extraction. The parser
records the visible crop, temporarily widens it to the MediaBox, and classifies
each extracted span against the original crop. This exposes off-crop content to
the quarantine logic without treating it as visible evidence.

Visibility is decided by three independent signals:

  * render mode and alpha;
  * foreground color relative to the page background; and
  * geometry outside the visible crop.
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
