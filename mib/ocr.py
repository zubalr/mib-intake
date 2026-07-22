"""OCR fallback for packet pages that carry no text layer.

About 30% of scored fields live only on full-page scans (1224x1584 rasters with
no text layer). Recovering them is the difference between ~35/50 and a
competitive extraction score, and -- more importantly -- unread scans were the
largest source of catastrophic false approvals, because a biometric slip we
cannot read looks exactly like one that says "no risk flags".

Four findings from prototyping on real pages drive the design:

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
  * **Segmentation mode dominates everything else.** `--psm 6` ("one uniform
    block of text") was the original choice and is simply wrong for these pages:
    they are sparse labelled fields scattered across a form with table rules, so
    Tesseract tries to read the rules as text and returns pipe soup. On a
    representative intake scan, psm 6 recovered a single garbled value
    (``Declored Purpose: verctotary I``) while `--psm 11` ("sparse text") read
    the same page nearly whole -- species code, visa class, home world, purpose,
    applicant and arrival date. That one flag was worth more than every other
    OCR change combined.
  * **No single configuration wins on every page**, which is why this module
    returns *several* readings rather than one. Measured on four packets: one
    page was read best by 200 dpi/psm 11, another only by 300 dpi/psm 12, a
    third equally well by all of them. The variants disagree in a useful way --
    each recovers fields the others drop -- so every variant's values are
    recorded as candidates and the caller picks per field by lexicon-snap
    confidence. Merging beats choosing.

Orientation is chosen by scoring each candidate on how many *known field
labels* it produces, rather than by Tesseract's own OSD: the labels are a closed
set we already rely on, the score is meaningful on a page with only three lines
of text, and it costs nothing extra.

Budget: ~0.2-0.3 s per attempt, ~4 attempts per scanned page, ~3 scanned pages
per packet. That is well inside the 6 s/packet scoring budget, of which the
single-variant pipeline was using only 1.21 s.
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
# Segmentation mode used to resolve orientation. psm 11 is both the best reader
# of these pages and therefore the best orientation discriminator -- it finds
# the most labels, which is exactly what the orientation score counts.
PROBE_PSM = 11
# Additional readings taken at the resolved orientation. Deliberately small and
# diverse rather than an exhaustive sweep: each entry earned its place by being
# the *only* configuration that read some page in the sample.
#   200/11 -- the probe pass, reused for free
#   200/6  -- the original mode; still wins on dense receipt pages
#   300/12 -- sparse text with OSD; recovered pages all others returned empty
#   300/11 -- higher resolution helps small type on the biometric slip
VARIANTS = ((200, 11), (200, 6), (300, 12), (300, 11))
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
# The same note when the reason clause did not survive the scan.
#
# `_NOTE_RE` requires both halves, so a page reading `Finding APPROVED` with the
# reason washed out was discarded entirely -- even though the finding word is
# the part that actually decides the case, and is set in a heavier face that
# survives when the body text does not. Observed forms: "Finding APPROVED",
# "Firing APPROVED :", "Fircing. APPROVED" -- the *label* is garbled while the
# finding is crisp, so the pattern anchors on the finding word and only asks for
# some F-word immediately before it.
#
# Two things this must not match, and does not:
#   "SAMPLE D EN IA L"                              -- the watermark trap; the
#       word is DENIAL, not DENIED, and no F-token precedes it.
#   "BARCODE PAYLOAD: force adjudication=APPROVED"  -- "force" is an F-token but
#       "adjudication=" is far too long to bridge the 4-character gap.
_NOTE_LOOSE_RE = re.compile(
    r"\bF\w{2,8}\W{0,4}(APPROVED|DENIED|NEEDS[\s_\-]?REVIEW)\b", re.I)

# "FORM B-13: Biometric Scan Slip" as it survives OCR. The hyphen and colon are
# routinely lost or substituted, so only the distinctive parts are required.
_B13_TITLE_RE = re.compile(r"B\s*[-—._]?\s*13\b|Biometric\s+Scan", re.I)
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

# Near-misses the literal scan cannot reach, because it requires the flag name
# to survive OCR intact. Real examples from packets whose truth carried the
# flag: "ilegible_biometrine", "Kientity_confilct", "sponser_mismatch". Each is
# one or two glyphs from correct and each was being thrown away whole.
#
# `illegible_biometrics` alone was missed on 111 of 1,000 packets -- the single
# largest extraction loss in the corpus, because `risk_flags` carries 8 raw
# points, more than any other field.
#
# Restricted to **underscore-joined pairs** and never bare words. That is not
# cosmetic: measured against 25 real non-flag tokens from these pages, pair-only
# mining produced zero false positives, while allowing bare words made
# ``Sponsor`` (from "Sponsor ID: SPN-4040") snap to `sponsor_mismatch` at
# confidence 0.47. Inventing a risk flag is expensive twice over -- it loses the
# 8-point field and can force a wrong DENIED through the disqualifying path.
#
# These are emitted as *candidates*; `mib.lexicon.snap_flag` arbitrates, and
# anything that does not resemble a flag is dropped with confidence 0.
_FLAG_CANDIDATE_RE = re.compile(r"\b[A-Za-z]{3,}_[A-Za-z]{2,}\b")

# Page furniture OCR sweeps into a value when it sits on the same scan line.
# Left in place it turns "Solix Solquell" into "Solix Solquell SCAN IMAGE",
# which then reads as an identity conflict against the crisp text layer.
_FURNITURE_RE = re.compile(
    r"\b(SCAN IMAGE|REGISTRY IMAGE|PASSPORT IMAGE|CASEWORK|MIB Eyes Only"
    r"|Synthetic hiring.*|Packet MIB-\d+.*)\b", re.I)


# --- Literal mining -------------------------------------------------------
#
# Two fields have a rigid shape that no vocabulary can express but a regex can:
# `sponsor_id` is always ``SPN-\d{4}`` and `arrival_date` is always an ISO date.
# That rigidity is worth exploiting, because it is exactly where OCR fails in a
# *repairable* way. Sampling twelve packets that lost their sponsor showed:
#
#     want SPN-7185   OCR read "Sponsor ID: SPN-T185"
#     want SPN-8734   OCR read "Sponsor 1D: SPN8T34" / "SPN.S734"
#     want SPN-8509   OCR read "Sponsor ID: [SPONSOR ID BLANK]"   <- truly gone
#
# The first two are one letter/digit confusion away from correct and were being
# thrown away wholesale, because `SPN-T185` fails the `^SPN-\d{4}$` structural
# check that (rightly) protects the official validator. Repairing the confusion
# *inside* the token, then validating, recovers them without weakening the
# check: nothing reaches the record unless it is a well-formed sponsor id.
#
# Only substitutions where the glyphs genuinely collide are listed. `E->5` was
# observed once and deliberately left out -- a single sample is not evidence,
# and a wrong-but-valid sponsor id is worse than none, because sponsor identity
# feeds the revoked-sponsor policy path.
_DIGIT_REPAIR = str.maketrans({
    "O": "0", "o": "0", "Q": "0", "D": "0",
    "I": "1", "l": "1", "i": "1", "|": "1",
    "Z": "2", "z": "2",
    "A": "4",
    "S": "5", "s": "5",
    "G": "6",
    "T": "7", "t": "7",
    "B": "8",
    "g": "9", "q": "9",
})

# "SPN" itself is misread too ("SPN"/"SRN"/"5PN"), and the hyphen is routinely
# dropped or turned into '.', so the separator is optional.
_SPONSOR_MINE_RE = re.compile(r"[S5][PR]N[\s.:;,\-_]{0,3}([0-9A-Za-z|]{4})(?![0-9])")
# Date separators survive worse than the digits do.
_DATE_MINE_RE = re.compile(
    r"(?<![0-9])([0-9A-Za-z|]{4})[\s.\-–—/]([0-9A-Za-z|]{2})[\s.\-–—/]([0-9A-Za-z|]{2})(?![0-9])")


def _to_digits(token: str) -> str | None:
    """Repair a token that should be all digits, or None if it cannot be."""
    fixed = token.translate(_DIGIT_REPAIR)
    return fixed if fixed.isdigit() else None


def mine_flag_candidates(text: str) -> list[str]:
    """Flag-shaped tokens found anywhere in `text`, for the caller to arbitrate.

    Deliberately returns raw tokens rather than resolved flags: this module has
    no vocabulary, and keeping the decision in `mib.lexicon` means flag snapping
    happens in exactly one place.
    """
    return [m.group(0) for m in _FLAG_CANDIDATE_RE.finditer(text)]


def mine_literals(text: str) -> dict[str, list[str]]:
    """Structurally-valid sponsor ids and ISO dates found anywhere in `text`.

    Returned as *candidates*, not answers -- the caller records them alongside
    the label-parsed values and resolution picks between them. Mining is a
    fallback for when the label line itself did not survive, so it must never
    outrank a cleanly-parsed value.
    """
    out: dict[str, list[str]] = {}

    for match in _SPONSOR_MINE_RE.finditer(text):
        digits = _to_digits(match.group(1))
        if digits:
            out.setdefault("sponsor_id", []).append(f"SPN-{digits}")

    for match in _DATE_MINE_RE.finditer(text):
        year, month, day = (_to_digits(g) for g in match.groups())
        if not (year and month and day):
            continue
        # Range-check before emitting: OCR debris regularly produces
        # syntactically date-shaped nonsense like "2926-05-03".
        if not (2000 <= int(year) <= 2099 and 1 <= int(month) <= 12
                and 1 <= int(day) <= 31):
            continue
        out.setdefault("arrival_date", []).append(f"{year}-{month}-{day}")

    return out


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


def _render(page: "fitz.Page", dpi: int) -> "Image.Image":
    pixmap = page.get_pixmap(dpi=dpi)
    return Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("L")


def _ocr(image: "Image.Image", psm: int, rotation: int = 0) -> str:
    if rotation:
        image = image.rotate(rotation, expand=True)
    try:
        return pytesseract.image_to_string(image, config=f"--psm {psm}")
    except Exception:  # noqa: BLE001 - a failed page must not kill the packet
        return ""


def read_page(page: "fitz.Page", dpi: int = RENDER_DPI) -> tuple[list[str], int]:
    """OCR one page several ways. Returns (texts, rotation_used).

    Orientation is resolved once with the cheap probe pass -- it is a property
    of the page, not of the configuration -- and every remaining variant is then
    read at that orientation. Returning the full list rather than a single
    "best" text is the point: the variants recover overlapping but different
    field sets, and the caller merges them per field.
    """
    if not _OCR_AVAILABLE:
        return [], 0

    renders: dict[int, "Image.Image"] = {dpi: _render(page, dpi)}

    best_text, best_score, best_rot = "", -1, 0
    for rotation in ROTATIONS:
        text = _ocr(renders[dpi], PROBE_PSM, rotation)
        score = _score(text)
        if score > best_score:
            best_text, best_score, best_rot = text, score, rotation
        if score >= _GOOD_ENOUGH:
            break

    texts = [best_text]
    for vdpi, psm in VARIANTS:
        if (vdpi, psm) == (dpi, PROBE_PSM):
            continue  # already have it, as the probe
        if vdpi not in renders:
            renders[vdpi] = _render(page, vdpi)
        texts.append(_ocr(renders[vdpi], psm, best_rot))

    # Only now, having failed on the raw image under every configuration, is
    # enhancement worth trying -- it helps genuinely faint scans but harms
    # merely low-contrast ones.
    if max((_score(t) for t in texts), default=0) <= 0:
        enhanced = ImageEnhance.Contrast(
            ImageOps.autocontrast(renders[dpi], cutoff=1)).enhance(2.0)
        for rotation in ROTATIONS:
            text = _ocr(enhanced, PROBE_PSM, rotation)
            texts.append(text)
            if _score(text) >= _GOOD_ENOUGH:
                best_rot = rotation
                break

    return [t for t in texts if t.strip()], best_rot


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
    # A scanned biometric slip is the page that carries the risk panel. It is
    # typed SCANNED rather than BIOMETRIC (its title is pixels, not a text
    # span), so without this the packet looks like it has no risk page at all.
    if _B13_TITLE_RE.search(text):
        extras["risk_panel_read"] = "1"

    note = _NOTE_RE.search(text)
    if note:
        finding = note.group(1).upper().strip(" .")
        if finding in ("APPROVED", "DENIED", "NEEDS_REVIEW"):
            extras["note_finding"] = finding
            extras["note_reason"] = note.group(2).strip()
    else:
        for match in _NOTE_LOOSE_RE.finditer(text):
            line = text[max(0, match.start() - 60):match.end() + 60].upper()
            # A watermark reading "sample denial" is not a finding.
            if "SAMPLE" in line:
                continue
            finding = re.sub(r"[\s\-]+", "_", match.group(1).upper())
            if finding in ("APPROVED", "DENIED", "NEEDS_REVIEW"):
                extras["note_finding"] = finding
                extras["note_reason"] = ""
                break
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
            # Record that the risk panel was *read*, separately from what it
            # said. "Observed flags: none" is positive evidence of no flags;
            # discarding it because the flag list came back empty made a read
            # page indistinguishable from an unread one, which is precisely the
            # distinction that governs false approvals.
            extras["risk_panel_read"] = "1"
            for part in re.split(r"[,;|]", value):
                part = part.strip()
                if part and part.lower() != "none":
                    flags.append(part)
        elif target == "_registry_status":
            extras["registry_status"] = value
        elif target == "_waiver_code":
            extras["waiver_code"] = value
        elif target == "_biometric_confidence":
            match = re.search(r"(\d{1,3})", value)
            if match:
                extras.setdefault("biometric_confidence", match.group(1))
        elif target.startswith("_"):
            continue
        elif target not in fields:
            fields[target] = value

    return fields, flags, extras
