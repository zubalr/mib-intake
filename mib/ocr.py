"""OCR fallback for packet pages that carry no text layer.

About 30% of scored fields live only on full-page scans (1224x1584 rasters with
no text layer). Recovering them is the difference between ~35/50 and a
competitive extraction score, and -- more importantly -- unread scans were the
largest source of catastrophic false approvals, because a biometric slip we
cannot read looks exactly like one that says "no risk flags".

Three findings from prototyping on real pages drive the design:

  * **Pages are rotated.** Scans appear at 90/180/270 degrees. Tesseract reads
    almost nothing at the wrong orientation and reads cleanly at the right one,
    so orientation must be resolved before anything else.
  * **Contrast enhancement makes it worse.** The intuitive move -- autocontrast
    and a contrast boost on washed-out text -- measurably *destroyed* readable
    text: a page that OCR'd perfectly raw ("Home World: Europa Station |
    Species Code: KAIJU_MICRO | Arrival Date: 2026-02-04") degraded to
    "Home Word: Cwope Station". These scans are low-contrast but clean, and the
    enhancement amplifies the scan-grid background into the glyphs. So we OCR
    the raw grayscale and only fall back to enhancement if raw yields nothing.
  * **200 dpi is enough**, at ~0.2 s per attempt, which keeps several rotation
    attempts per page inside the 6 s/packet budget.

Orientation is chosen by scoring each candidate on how many *known field
labels* it produces, rather than by Tesseract's own OSD: the labels are a closed
set we already rely on, the score is meaningful on a page with only three lines
of text, and it costs nothing extra.
"""

from __future__ import annotations

import io
import re

import fitz

try:
    import pytesseract
    from PIL import Image, ImageEnhance, ImageOps
    _OCR_AVAILABLE = True
except ImportError:  # pragma: no cover - image ships with both
    _OCR_AVAILABLE = False

RENDER_DPI = 200
TESSERACT_CONFIG = "--psm 6"
# 0 first: unrotated pages are the common case and win on the early exit.
ROTATIONS = (0, 90, 270, 180)

# Field labels as they appear on scanned pages, used both to score orientation
# and to parse values. Scans use inline "Label: value" rather than the text
# layer's paired-span layout.
OCR_LABELS = {
    "home world": "home_world",
    "species code": "species_code",
    "species match": "species_code",
    "arrival date": "arrival_date",
    "visa class": "visa_class",
    "sponsor id": "sponsor_id",
    "declared purpose": "declared_purpose",
    "applicant": "applicant_name",
    "registry name": "applicant_name",
    "case id": "_case_id",
    "fee status": "fee_status",
    "waiver code": "_waiver_code",
    "registry status": "_registry_status",
    "observed flags": "_observed_flags",
    "biometric confidence": "_biometric_confidence",
}

# The label/value separator is frequently NOT a colon after OCR: a scanned
# "Fee Status: paid" comes back as "Fee Status. paig". Accepting only ':' threw
# away several hundred recoverable values.
# First label/value separator on a line. ':' is frequently misread as '.', ';'
# or '=' on a scan ("Fee Status. paig"), so all are accepted.
_SEP_RE = re.compile(r"\s*[:;.=]\s*")

# An adjudicator note recovered from a scanned page. The finding word itself is
# usually crisp (it is set in a larger face) even when the surrounding text is not.
_NOTE_RE = re.compile(r"Finding\s*[:;.]?\s*([A-Za-z_]+)\s*[.,]?\s*Reason\s*[:;.]?\s*(.+)",
                      re.I)
# "Disqualifying risk flag: biohazard_red." / "Review-only risk flag present: x."
_NOTE_FLAG_RE = re.compile(r"risk flag(?:\s+present)?\s*[:;.]\s*([a-z_]+)", re.I)

# Matching on the *label* is fragile: OCR drops the colon and garbles the word
# ("Reason Disqualifying nsk flag biohazard_red"). The flag names themselves are
# far more distinctive than the label around them -- two underscore-joined words
# that will not occur by accident -- so scan for them directly.
#
# Safe against the hidden-text injection: white-on-white text is invisible in a
# rendered raster, so OCR never sees the planted "answer key" at all.
_FLAG_LITERALS = (
    "memory_tampering", "planetary_embargo", "active_warrant", "biohazard_red",
    "identity_conflict", "sponsor_mismatch", "illegible_biometrics",
    "rescinded_denial",
)
_FLAG_LITERAL_RE = re.compile(
    "|".join(name.replace("_", r"[_\s\-]?") for name in _FLAG_LITERALS), re.I)

# Page furniture OCR sweeps into a value when it sits on the same scan line.
# Left in place it turns "Solix Solquell" into "Solix Solquell SCAN IMAGE",
# which then reads as an identity conflict against the crisp text layer.
_FURNITURE_RE = re.compile(
    r"\b(SCAN IMAGE|REGISTRY IMAGE|PASSPORT IMAGE|CASEWORK|MIB Eyes Only"
    r"|Synthetic hiring.*|Packet MIB-\d+.*)\b", re.I)


def _strip_furniture(value: str) -> str:
    value = _FURNITURE_RE.sub(" ", value)
    # Trailing single-character debris ("Tekdane Tekmora i").
    value = re.sub(r"\s+[^A-Za-z0-9]\s*$", "", value)
    value = re.sub(r"\s+[a-zA-Z]$", "", value) if len(value.split()) > 2 else value
    return " ".join(value.split()).strip(" |.,;:-_")
# Enough of a signal to stop trying further rotations.
_GOOD_ENOUGH = 2


def available() -> bool:
    return _OCR_AVAILABLE


def _score(text: str) -> int:
    """How many known field labels this OCR attempt produced."""
    low = text.casefold()
    return sum(1 for label in OCR_LABELS if label in low)


def _ocr(image: "Image.Image", rotation: int) -> str:
    if rotation:
        image = image.rotate(rotation, expand=True)
    try:
        return pytesseract.image_to_string(image, config=TESSERACT_CONFIG)
    except Exception:  # noqa: BLE001 - a failed page must not kill the packet
        return ""


def read_page(page: "fitz.Page", dpi: int = RENDER_DPI) -> tuple[str, int]:
    """OCR one page, resolving orientation. Returns (text, rotation_used)."""
    if not _OCR_AVAILABLE:
        return "", 0

    pixmap = page.get_pixmap(dpi=dpi)
    image = Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("L")

    best_text, best_score, best_rot = "", -1, 0
    for rotation in ROTATIONS:
        text = _ocr(image, rotation)
        score = _score(text)
        if score > best_score:
            best_text, best_score, best_rot = text, score, rotation
        if score >= _GOOD_ENOUGH:
            break

    # Only now, having failed on the raw image, is enhancement worth trying --
    # it helps genuinely faint scans but harms merely low-contrast ones.
    if best_score <= 0:
        enhanced = ImageEnhance.Contrast(
            ImageOps.autocontrast(image, cutoff=1)).enhance(2.0)
        for rotation in ROTATIONS:
            text = _ocr(enhanced, rotation)
            score = _score(text)
            if score > best_score:
                best_text, best_score, best_rot = text, score, rotation
            if score >= _GOOD_ENOUGH:
                break

    return best_text, best_rot


def _match_label(raw: str) -> str | None:
    """Fuzzy-match the text preceding a separator against the known labels.

    Anchoring a strict regex at the start of the line discarded a large share of
    perfectly readable values: scans routinely prefix a line with table rules or
    stray marks, producing ``| Observed flags: illegible_biometrics | | |``,
    ``A) Observed flags: biohazard_red`` and ``(Cbeerved flags: ...``. All three
    were dropped outright even though the value itself was clean.

    So strip non-letter noise, keep the trailing words, and tolerate a garbled
    label -- ``Cbeerved flags`` should still resolve to "observed flags".
    """
    cleaned = re.sub(r"[^A-Za-z ]+", " ", raw).strip().casefold()
    if not cleaned:
        return None
    words = cleaned.split()

    # Longest trailing word-group first, so "observed flags" beats the bare
    # "flags" left behind by a partly-eaten label.
    for size in (3, 2, 1):
        if len(words) >= size:
            candidate = " ".join(words[-size:])
            if candidate in OCR_LABELS:
                return OCR_LABELS[candidate]

    tail = " ".join(words[-2:]) if len(words) >= 2 else words[-1]
    best, best_dist = None, 10**9
    for label, target in OCR_LABELS.items():
        if abs(len(label) - len(tail)) > 4:
            continue
        dist = sum(1 for a, b in zip(label, tail) if a != b) + abs(len(label) - len(tail))
        if dist < best_dist:
            best, best_dist = target, dist
    return best if best is not None and best_dist <= max(2, len(tail) // 4) else None


def parse_fields(text: str) -> tuple[dict[str, str], list[str], dict[str, str]]:
    """Parse OCR text into (fields, observed_flags, extras).

    `extras` carries non-scored signals the policy layer wants (registry status,
    waiver code, and any adjudicator finding recovered from a scanned note).
    Values are returned raw; snapping onto the closed vocabulary is the caller's
    job, so OCR noise is corrected in exactly one place.
    """
    fields: dict[str, str] = {}
    flags: list[str] = []
    extras: dict[str, str] = {}

    # An adjudicator note that happens to be scanned is still an adjudicator
    # note -- the top evidence tier, and 162/162 correct wherever the text layer
    # carried one. Its stated reason also names the governing risk flag, which
    # is often the only place that flag survives on a damaged packet.
    note = _NOTE_RE.search(text)
    if note:
        finding = note.group(1).upper().strip(" .")
        if finding in ("APPROVED", "DENIED", "NEEDS_REVIEW"):
            extras["note_finding"] = finding
            extras["note_reason"] = note.group(2).strip()
    for match in _NOTE_FLAG_RE.finditer(text):
        flags.append(match.group(1).strip(" ."))
    for match in _FLAG_LITERAL_RE.finditer(text):
        flags.append(match.group(0))

    for line in text.splitlines():
        # Find the first separator rather than anchoring the label at the start.
        sep = _SEP_RE.search(line)
        if not sep:
            continue
        target = _match_label(line[: sep.start()])
        value = _strip_furniture(line[sep.end():])
        if not target or not value:
            continue

        if target == "_observed_flags":
            for part in re.split(r"[,;|]", value):
                part = part.strip()
                if part and part.lower() != "none":
                    flags.append(part)
        elif target == "_registry_status":
            extras["registry_status"] = value
        elif target == "_waiver_code":
            extras["waiver_code"] = value
        elif target.startswith("_"):
            continue
        elif target not in fields:
            fields[target] = value

    return fields, flags, extras
