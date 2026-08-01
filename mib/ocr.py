"""Local OCR for packet pages without a usable text layer.

The reader resolves page orientation, evaluates several Tesseract segmentation
configurations, and returns all plausible readings. Field-level arbitration is
handled later by structural validation and vocabulary matching. Contrast
enhancement is used only when the raw grayscale image yields no field labels.
"""

from __future__ import annotations

import io
import re

import fitz

from mib import fallback_ocr
from mib.lexicon import _canon, weighted_distance

try:
    import pytesseract
    from PIL import Image, ImageEnhance, ImageOps
    _OCR_AVAILABLE = True
except ImportError:  # pragma: no cover - image ships with both
    _OCR_AVAILABLE = False

RENDER_DPI = 200
# Segmentation mode used to resolve orientation.
PROBE_PSM = 11
# Additional readings at the selected orientation.
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
    # Sponsor letters use shorter labels than intake forms. Short aliases are
    # restricted to fields whose vocabulary matcher rejects spurious captures.
    "purpose": "declared_purpose",
    "world": "home_world",
    "species": "species_code",
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
# Note finding without a surviving reason clause.
_NOTE_LOOSE_RE = re.compile(
    r"\bF\w{2,8}\W{0,4}([A-Za-z][A-Za-z_ ]{3,13})", re.I)

# OCR can clip or distort the Finding label. The following matcher accepts
# recognizable stems and bounded edit distance. Outcome snapping and watermark
# checks provide the precision guard.
_FINDING_LABEL_STEMS = ("fin", "ind")
_FINDING_LABEL_TAILS = ("nding", "ding")
_FINDING_LABEL_MAX_RATIO = 0.45
# `Finding` followed by the outcome, with the label allowed to be almost anything
# short and word-shaped. Two capture groups: the label, then the outcome token.
_NOTE_LABELLED_RE = re.compile(
    r"\b([A-Za-z]{3,10})\s*[:;.,]?\s+([A-Za-z][A-Za-z_ ]{2,13})")


def _is_finding_label(token: str) -> bool:
    """Is this garbled token the word `Finding`?"""
    canon = _canon(token)
    if not 3 <= len(canon) <= 10:
        return False
    target = "finding"
    if canon == target or target.startswith(canon) or target.endswith(canon):
        return True
    if canon.startswith(_FINDING_LABEL_STEMS) or canon.endswith(_FINDING_LABEL_TAILS):
        return True
    return weighted_distance(canon, target) <= len(target) * _FINDING_LABEL_MAX_RATIO


FINDINGS = ("APPROVED", "DENIED", "NEEDS_REVIEW")
# Nearby non-outcome words that must not snap to a finding.
_FINDING_BLOCKLIST = frozenset({
    "approval", "approvals", "denial", "denials",
    "review", "reviewed", "reviewer",
})
# Edit-distance threshold for a finding token.
_FINDING_MAX_RATIO = 0.35


# A note rationale can identify the finding when the outcome token is unreadable.
# These patterns are consulted only when no finding token can be recovered.
# Split by whether the phrase needs the page identified as a note first, and
# `\W*` rather than `\s+` between words throughout -- OCR wedges pipes and stray
# marks *inside* a phrase, and "Clean or exce| tion-qualified" broke a pattern
# that only tolerated whitespace.
#
# Complete manual sentence openings can identify the note without a separate
# page-title cue: only an adjudicator writes them, so demanding a note-page cue
# as well would discard them for no gain in precision.
_REASON_FINDINGS_SELF_SCOPING = (
    (re.compile(r"(?i)den[il1]a[li1]\W*supp"), "DENIED"),
    (re.compile(r"(?i)c[li1]ean\W*or\W*exce"), "APPROVED"),
    (re.compile(r"(?i)approva[li1]\W*supp"), "APPROVED"),
    (re.compile(r"(?i)arr[il1]va[li1]\W*date\W*m[il1]ss"), "NEEDS_REVIEW"),
    (re.compile(r"(?i)damaged\W*or\W*contra"), "NEEDS_REVIEW"),
)
# These stay scoped, because each plausibly appears elsewhere in a packet:
# "disqualifying" and "review-only" label risk flags on a biometric panel,
# "revoked sponsor" is registry-extract wording, and "transit class" echoes the
# TRANSIT-7 visa class on an intake form.
_REASON_FINDINGS_SCOPED = (
    (re.compile(r"(?i)d[il1]squa[li1][il1]f"), "DENIED"),
    (re.compile(r"(?i)trans[il1]t\W*c[li1]ass"), "DENIED"),
    (re.compile(r"(?i)revoked\W*spon"), "DENIED"),
    (re.compile(r"(?i)rev[il1]ew\W*on[li1]y"), "NEEDS_REVIEW"),
)
# Identifies a page as an adjudicator note. `r?eason` because the leading letter
# of a label is routinely clipped at the left edge of a scan ("eason:").
_NOTE_PAGE_CUE = re.compile(r"(?i)adjud|manua[li1]\s*n|r?eason\s*[:;.]")


# Amount and waiver code provide corroborating fee evidence when the status
# label is unreadable. Zero amount without a waiver is ambiguous and is not
# resolved here.
_AMOUNT_RE = re.compile(r"(?i)am[o0c]unt\W{0,4}\$?\s*([\d,]+(?:[.,]\d{2})?)")
_WAIVER_CODE_RE = re.compile(r"(?i)wa[il1]ver\s*c[o0]de\W{0,4}([A-Za-z0-9\-/]{2,20})")
_NO_WAIVER = frozenset({"N/A", "NA", "NONE", "N/4", "WA"})


def fee_from_amount_and_waiver(amount_text: str, waiver_text: str) -> str | None:
    """Fee status implied by a receipt's amount and waiver code.

    Shared rather than duplicated: the scanned path reaches this after regex
    mining, the typed path hands over two parsed label values. Two copies of
    "what does this receipt mean" is precisely how the typed branch ended up
    reading `Amount` and discarding it while the scanned branch used it.
    """
    try:
        amount = float(str(amount_text).replace("$", "").replace(",", "").strip())
    except (TypeError, ValueError):
        return None
    waived = str(waiver_text).strip().upper() not in _NO_WAIVER
    if amount > 0 and not waived:
        return "paid"
    if amount == 0 and waived:
        return "waived"
    return None


def _fee_from_receipt(text: str) -> str | None:
    """Fee status implied by the amount and waiver code on a receipt."""
    amount_match = _AMOUNT_RE.search(text)
    waiver_match = _WAIVER_CODE_RE.search(text)
    if not amount_match or not waiver_match:
        return None
    return fee_from_amount_and_waiver(amount_match.group(1), waiver_match.group(1))


# A signed note reason can also state fee status or home world explicitly.
_REASON_FEE = (
    (re.compile(r"(?i)mandat[o0]ry\W*fee\W*unpa[il1]d"), "unpaid"),
    (re.compile(r"(?i)fee\W*status\W*unkn[o0]wn"), "unknown"),
)
_REASON_WORLD = re.compile(r"(?i)embarg[o0]\W*h[o0]me\W*w[o0]r[li1]d\W{0,3}([A-Za-z][\w\- ]{2,18})")


def _facts_from_reason(text: str, extras: dict) -> None:
    """Record field values the note's reason clause states outright."""
    for pattern, value in _REASON_FEE:
        match = pattern.search(text)
        if match and not _near_watermark(text, match.start(), match.end()):
            extras.setdefault("reason_fee_status", value)
            break
    match = _REASON_WORLD.search(text)
    if match and not _near_watermark(text, match.start(), match.end()):
        extras.setdefault("reason_home_world", match.group(1).strip(" .,|"))


def _matches_clear_of_watermark(text: str, patterns) -> set[str]:
    """Findings whose rationale appears somewhere no watermark reaches."""
    hits = set()
    for pattern, finding in patterns:
        for match in pattern.finditer(text):
            if not _near_watermark(text, match.start(), match.end()):
                hits.add(finding)
                break
    return hits


def _finding_from_reason(text: str) -> str | None:
    """Infer the finding from the note's reason clause, or None."""
    hits = _matches_clear_of_watermark(text, _REASON_FINDINGS_SELF_SCOPING)
    if _NOTE_PAGE_CUE.search(text):
        hits |= _matches_clear_of_watermark(text, _REASON_FINDINGS_SCOPED)
    # Reject conflicting rationale matches.
    return hits.pop() if len(hits) == 1 else None


# How far from a match a `SAMPLE` watermark still poisons it. The watermark is
# set across the page centre, so text it overlaps is interleaved with it in the
# OCR stream; 60 characters is comfortably wider than that interleaving and was
# the window the loose-finding path already used.
_WATERMARK_WINDOW = 60


def _near_watermark(text: str, start: int, end: int) -> bool:
    """Return whether a match overlaps a SAMPLE watermark."""
    window = text[max(0, start - _WATERMARK_WINDOW):end + _WATERMARK_WINDOW]
    return "SAMPLE" in window.upper()


def _snap_finding(token: str) -> str | None:
    """Nearest adjudication outcome to a garbled finding word, or None.

    Ambiguous tokens are rejected.
    """
    observed = _canon(token)
    if not observed or observed in _FINDING_BLOCKLIST:
        return None
    # Accept an unambiguous prefix before applying edit distance.
    if len(observed) >= 3:
        prefixes = [f for f in FINDINGS if _canon(f).startswith(observed)]
        if len(prefixes) == 1:
            return prefixes[0]
    scored = sorted((weighted_distance(observed, _canon(f)), f) for f in FINDINGS)
    distance, best = scored[0]
    budget = max(1.0, len(_canon(best)) * _FINDING_MAX_RATIO)
    if distance >= budget:
        return None
    if len(scored) > 1 and scored[1][0] - distance < 1.0:
        return None
    return best


def _note_finding_of(text: str) -> tuple[str, str] | None:
    """The adjudicator finding this text states, as (finding, reason), or None.

    The three note paths in precedence order: the full `Finding ... Reason ...`
    sentence, a `Finding:`-labelled outcome, and finally the reason clause alone.
    Factored out so the header-crop rescue and `parse_fields` cannot drift -- the
    rescue must contribute *text*, and have its verdict decided by exactly the
    same trusted parser as every other read.
    """
    note = _NOTE_RE.search(text)
    if note:
        finding = _snap_finding(note.group(1))
        if finding:
            return finding, note.group(2).strip()
    for match in _NOTE_LABELLED_RE.finditer(text):
        if not _is_finding_label(match.group(1)):
            continue
        # No watermark check here, deliberately. The *label* is the guard on this
        # path: a watermark does not come with a `Finding:` in front of it, and
        # `SAMPLE` itself fails `_is_finding_label`. Checking a neighbourhood
        # instead would discard the common real case -- a genuine note that also
        # carries the harmless watermark. The reason-clause path below keeps its
        # local check, because a rationale has no label vouching for it.
        # The captured value may carry debris from the rest of the line ("NEEDS
        # ESA" for NEEDS_REVIEW), so the first word gets its own chance -- it is
        # where the outcome actually lives, and a leading `DENIAL` is still
        # blocklisted when it is tried alone.
        value = match.group(2)
        finding = _snap_finding(value) or _snap_finding(value.split()[0])
        if finding:
            return finding, ""
    finding = _finding_from_reason(text)
    if finding:
        return finding, ""
    return None


# "FORM B-13: Biometric Scan Slip" as it survives OCR. The hyphen and colon are
# routinely lost or substituted, so only the distinctive parts are required.
_B13_TITLE_RE = re.compile(r"B\s*[-\u2014._]?\s*13\b|Biometric\s+Scan", re.I)
# "Disqualifying risk flag: biohazard_red." / "Review-only risk flag present: x."
_NOTE_FLAG_RE = re.compile(r"risk flag(?:\s+present)?\s*[:;.]\s*([a-z_]+)", re.I)

# Flag literals are more stable than their surrounding OCR labels.
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

# Candidate mining is restricted to underscore-joined pairs. Resolution remains
# in ``Lexicon.snap_flag``.
_FLAG_CANDIDATE_RE = re.compile(r"\b[A-Za-z]{3,}_[A-Za-z]{2,}\b")

# Page furniture OCR sweeps into a value when it sits on the same scan line.
# Left in place it turns "Solix Solquell" into "Solix Solquell SCAN IMAGE",
# which then reads as an identity conflict against the crisp text layer.
_FURNITURE_RE = re.compile(
    r"\b(SCAN IMAGE|REGISTRY IMAGE|PASSPORT IMAGE|CASEWORK|MIB Eyes Only"
    r"|Synthetic hiring.*|Packet MIB-\d+.*)\b", re.I)


# --- Literal mining -------------------------------------------------------
#
# Sponsor IDs and dates have rigid shapes that support conservative glyph repair
# before structural validation.
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
    r"(?<![0-9])([0-9A-Za-z|]{4})[\s.\-\u2013\u2014/]"
    r"([0-9A-Za-z|]{2})[\s.\-\u2013\u2014/]"
    r"([0-9A-Za-z|]{2})(?![0-9])")


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


def read_page(page: "fitz.Page", dpi: int = RENDER_DPI
              ) -> tuple[list[str], list[str], int, list]:
    """OCR one page several ways.

    Returns (texts, fallback_texts, rotation, fallback_boxes).

    Orientation is resolved once with the probe pass. Remaining variants use the
    same orientation, and the caller merges their field candidates.

    `fallback_texts` come from the second engine and are returned *separately*
    rather than merged, because the caller records them at a lower trust rank.
    Keeping the two lists apart is what makes "a fallback read can never
    displace a primary read" a property of the data rather than a rule every
    call site has to remember to apply.
    """
    if not _OCR_AVAILABLE:
        return [], [], 0, []

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

    # Enhancement is reserved for pages with no labels in the raw image.
    if max((_score(t) for t in texts), default=0) <= 0:
        enhanced = ImageEnhance.Contrast(
            ImageOps.autocontrast(renders[dpi], cutoff=1)).enhance(2.0)
        for rotation in ROTATIONS:
            text = _ocr(enhanced, PROBE_PSM, rotation)
            texts.append(text)
            if _score(text) >= _GOOD_ENOUGH:
                best_rot = rotation
                break

    texts.extend(_note_header_reads(renders[dpi], texts))
    fallback_texts, fallback_boxes = fallback_ocr.read_detailed(renders[dpi])
    return ([t for t in texts if t.strip()],
            fallback_texts, best_rot, fallback_boxes)


# Adjudicator notes have no normal field labels, so the orientation probe cannot
# select their rotation. Re-read the header crop at each orientation.
_NOTE_CROP = (0.85, 0.18)   # width, height as a fraction of the rotated page
_NOTE_CROP_PSMS = (11, 6)
# Try every right-angle orientation before the small deskew variants.
_NOTE_DESKEW = (0, -6, 6)


def _note_header_reads(base: "Image.Image", texts: list[str]) -> list[str]:
    """Re-read a possible note header when the normal OCR pass found no fields."""
    if max((_score(t) for t in texts), default=0) >= _GOOD_ENOUGH:
        return []
    if any(_note_finding_of(t) for t in texts):
        return []
    out = []
    # Angle is the outer loop so every rotation is tried upright before any tilt
    # is considered: a page that is merely sideways costs nothing extra.
    for angle in _NOTE_DESKEW:
        for rotation in ROTATIONS:
            image = base.rotate(rotation, expand=True) if rotation else base
            width, height = image.size
            header = image.crop((0, 0, int(width * _NOTE_CROP[0]),
                                 int(height * _NOTE_CROP[1])))
            if angle:
                header = header.rotate(angle, expand=True, fillcolor=255)
            found = False
            for psm in _NOTE_CROP_PSMS:
                text = _ocr(header, psm)
                if not text.strip():
                    continue
                out.append(text)
                found = found or _note_finding_of(text) is not None
            # Both segmentation modes are read before deciding, because the one
            # that resolves the finding is not always the one that reads the
            # title.
            if found:
                return out
    return out


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

    found = _note_finding_of(text)
    if found:
        extras["note_finding"], extras["note_reason"] = found
    # Independent of whether a finding was read: the reason clause can state a
    # field value even on a note whose verdict we already have.
    _facts_from_reason(text, extras)
    derived_fee = _fee_from_receipt(text)
    if derived_fee:
        extras["receipt_fee_status"] = derived_fee
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
