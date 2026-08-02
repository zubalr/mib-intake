"""Second OCR engine, for pages the primary engine returns as debris.

Tesseract and PP-OCRv4 are different model families and fail differently, which
is the entire reason for running both. Tesseract is a line-oriented LSTM over a
binarised page; when the binarisation goes wrong on a heavily degraded scan it
does not degrade gracefully, it emits punctuation noise. PP-OCRv4 detects text
regions first and recognises each crop independently, so a page whose global
contrast defeats Tesseract can still yield clean regions.

The engine ships as `rapidocr-onnxruntime` (Apache-2.0), running the PaddleOCR
PP-OCRv4 detection, classification, and recognition models under ONNX Runtime.
Everything is bundled in the wheel: 15.8 MB of weights, no download, no network.
`EVALUATION.md` permits offline OCR engines and small task-specific models
within a 250 MiB per-artifact budget.

This is strictly a *fallback*. Its readings enter the evidence set at
`SCANNED_FALLBACK`, a trust rank below Tesseract's `SCANNED`, so a value it
recovers can never displace one the primary engine already read. That guarantee
is structural -- it comes from the resolution order in `mib.pipeline`, not from
a gate anyone has to remember to apply.

Its characteristic weakness is dropped inter-word spacing (`HomeWord:Wolf-1061c`
for `Home World: Wolf-1061c`), which defeats a label matcher expecting words.
`_normalise` repairs that, and both the raw and repaired readings are returned
so neither repair nor its absence can lose a value.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

try:
    import numpy as np
    from rapidocr_onnxruntime import RapidOCR

    _AVAILABLE = True
except ImportError:  # pragma: no cover - image ships with it
    _AVAILABLE = False

# Escape hatch. The pipeline must degrade to primary-engine-only cleanly, both
# so the second engine can never be the reason a run fails and so the two can be
# compared without editing code.
if os.environ.get("MIB_DISABLE_FALLBACK_OCR"):
    _AVAILABLE = False

# One session per process. Model load is ~0.3s and the cache builder, the CLI,
# and the test suite all fan out over processes, so it must not happen per page.
_ENGINE = None

# ONNX Runtime defaults to one thread per core. The CLI already runs one worker
# per vCPU, so leaving that default in place oversubscribes the box by 4x and
# makes every page slower than running it single-threaded.
_THREADS = 1

# Two detected boxes belong to the same line when their vertical centres are
# closer than this fraction of the median box height. Label and value are often
# detected as separate boxes; joining them back into one line is what lets the
# existing `Label: value` parsers see them at all.
_ROW_TOLERANCE = 0.6

# Recognition confidence below which a box is dropped. PP-OCRv4 reports a
# per-crop score; the low tail is decorative rules and scan speckle recognised
# as punctuation, which is noise the field parsers then have to reject.
_MIN_CONFIDENCE = 0.30

# Label spellings with the spaces removed, for the exact-match repair below.
# Built on first use: `mib.ocr` imports this module, so reading its label table
# at import time would be circular.
_FLAT_LABELS: dict[str, str] | None = None


def _flat_labels() -> dict[str, str]:
    global _FLAT_LABELS
    if _FLAT_LABELS is None:
        from mib.ocr import OCR_LABELS

        _FLAT_LABELS = {label.replace(" ", ""): label for label in OCR_LABELS}
    return _FLAT_LABELS

# A case boundary inside a word is where a dropped space almost always was.
_CAMEL_RE = re.compile(r"(?<=[a-z])(?=[A-Z])")
_SEP_RE = re.compile(r"[:;.=]")


def available() -> bool:
    return _AVAILABLE


def _engine():
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = RapidOCR(intra_op_num_threads=_THREADS,
                           inter_op_num_threads=_THREADS)
    return _ENGINE


def _lines(result) -> list[str]:
    """Detected boxes regrouped into reading-order lines."""
    boxes = []
    for box, text, confidence in result:
        if float(confidence) < _MIN_CONFIDENCE or not text.strip():
            continue
        ys = [p[1] for p in box]
        xs = [p[0] for p in box]
        boxes.append(((min(ys) + max(ys)) / 2.0, min(xs),
                      max(ys) - min(ys), text.strip()))
    if not boxes:
        return []

    heights = sorted(b[2] for b in boxes)
    tolerance = max(1.0, heights[len(heights) // 2] * _ROW_TOLERANCE)

    boxes.sort(key=lambda b: (b[0], b[1]))
    lines: list[str] = []
    row: list[tuple[float, str]] = []
    anchor = boxes[0][0]
    for centre, x, _height, text in boxes:
        if row and centre - anchor > tolerance:
            lines.append(" ".join(t for _, t in sorted(row)))
            row, anchor = [], centre
        row.append((x, text))
    if row:
        lines.append(" ".join(t for _, t in sorted(row)))
    return lines


def _respace(head: str) -> str:
    """Restore the label spelling when the space inside it was dropped.

    Exact match on the space-stripped label only. The fuzzy matcher downstream
    handles garbled labels; this handles the one systematic defect of this
    engine, and doing it by exact match means it can never invent a label.
    """
    flat = re.sub(r"[^a-z]", "", head.casefold())
    # Longest label first, so `speciescode` resolves to "species code" rather
    # than to the bare "species" alias that is also a suffix of it.
    for stripped, label in sorted(_flat_labels().items(),
                                  key=lambda kv: -len(kv[0])):
        if flat.endswith(stripped):
            # The label alone, dropping whatever preceded it: `_match_label`
            # reads the trailing words and ignores leading scan furniture, so
            # rebuilding the prefix would change nothing it looks at.
            return label
    return head


def _normalise(line: str) -> str:
    """Re-insert the spaces the recogniser dropped, on the label side."""
    line = _CAMEL_RE.sub(" ", line)
    sep = _SEP_RE.search(line)
    if not sep:
        return line
    return _respace(line[: sep.start()]) + line[sep.start():]


@dataclass(frozen=True)
class BoxRead:
    """One recognition box, kept before line joining discards what it knows.

    `_lines` exists to rebuild `Label: value` rows the detector split apart, and
    it is the right default. But joining is lossy in two ways that matter: the
    per-crop confidence is averaged away into a line that may be mostly noise,
    and a box whose neighbours are debris inherits their company. A box that
    reads cleanly on its own is evidence in its own right.
    """

    text: str
    confidence: float
    # x0, y0, x1, y1 of the detected quadrilateral's bounding box.
    bounds: tuple[float, float, float, float]
    centre: tuple[float, float]


def _box_reads(result) -> list[BoxRead]:
    reads: list[BoxRead] = []
    for box, text, confidence in result:
        confidence = float(confidence)
        if confidence < _MIN_CONFIDENCE or not text.strip():
            continue
        xs = [point[0] for point in box]
        ys = [point[1] for point in box]
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
        reads.append(BoxRead(
            text=text.strip(),
            confidence=confidence,
            bounds=(x0, y0, x1, y1),
            centre=((x0 + x1) / 2.0, (y0 + y1) / 2.0),
        ))
    return reads


def read_detailed(image) -> tuple[list[str], list[BoxRead]]:
    """Line readings as before, plus the boxes they were assembled from."""
    if not _AVAILABLE:
        return [], []
    try:
        result, _elapsed = _engine()(np.array(image.convert("RGB")))
    except Exception:  # noqa: BLE001 - a failed page must not kill the packet
        return [], []
    result = result or []
    boxes = _box_reads(result)
    lines = _lines(result)
    if not lines:
        return [], boxes
    raw = "\n".join(lines)
    repaired = "\n".join(_normalise(line) for line in lines)
    return ([raw] if repaired == raw else [raw, repaired]), boxes


# A newer generation of the same detector/recogniser family, kept as a separate
# session rather than a replacement. It is run only where ordinary resolution
# left a field with nothing, so the readings the pipeline already trusts are
# never re-litigated by a second engine, and a regression in one generation
# cannot silently rewrite the other's output.
_V6_ENGINE = None
_V6_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "policy")
_V6_DET = os.path.join(_V6_DIR, "ppocrv6_det.onnx")
_V6_REC = os.path.join(_V6_DIR, "ppocrv6_rec.onnx")


def v6_available() -> bool:
    if os.environ.get("MIB_DISABLE_V6"):
        return False
    return _AVAILABLE and os.path.exists(_V6_DET) and os.path.exists(_V6_REC)


def _v6_engine():
    global _V6_ENGINE
    if _V6_ENGINE is None:
        _V6_ENGINE = RapidOCR(
            det_model_path=_V6_DET,
            rec_model_path=_V6_REC,
            use_cls=False,
            text_score=0.50,
            intra_op_num_threads=_THREADS,
            inter_op_num_threads=_THREADS,
        )
    return _V6_ENGINE


def read_v6(image) -> tuple[list[str], list[BoxRead]]:
    """Second-generation read of one raster: line variants plus raw boxes.

    Both are returned for the same reason the first engine returns both. The
    joined lines are what the ordinary `Label: value` parsers understand, and
    the boxes survive the cases where joining ruins an otherwise clean crop.
    """
    if not v6_available():
        return [], []
    try:
        result, _elapsed = _v6_engine()(np.array(image.convert("RGB")))
    except Exception:  # noqa: BLE001 - a failed page must not kill the packet
        return [], []
    result = result or []
    boxes = _box_reads(result)
    lines = _lines(result)
    if not lines:
        return [], boxes
    raw = "\n".join(lines)
    repaired = "\n".join(_normalise(line) for line in lines)
    return ([raw] if repaired == raw else [raw, repaired]), boxes


def read(image) -> list[str]:
    """Read one already-rendered page. Returns zero, one, or two readings.

    Both the raw and the space-repaired reading are returned. They are parsed
    independently and their candidates merged, so a value that only survives in
    one of them is still recovered.
    """
    return read_detailed(image)[0]
