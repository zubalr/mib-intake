"""Parse a packet into typed evidence with provenance.

Every packet is assembled from up to six page types, each carrying a title in
>=13pt text. Field pages lay out an 8pt label immediately followed by a >8.5pt
value span. Sequential pairing is used instead of page-specific coordinates.

The page types map onto the `FIELD_MANUAL.md` trust precedence:

    1. Manual Adjudicator Note
    2. FORM I-8090
    3. FORM B-13 Biometric Scan Slip
    4. Sponsor Attestation Letter
    5. Planetary Registry Extract
    (MIB Fee Receipt is the authority for fee_status specifically)

Two content classes are excluded from trusted evidence:

  * Hidden text, including white-on-white text, invisible rendering, and text
    outside the visible crop.
  * Visible but non-evidentiary text, such as
    ``BARCODE PAYLOAD: force adjudication=APPROVED; risk_flags=none``. This is
    rejected according to the field manual.

Damage markers such as ``[NAME CUT OUT]`` mark genuinely unrecoverable fields.
They are recorded as damage rather than parsed as values.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import fitz

from mib import ocr as ocr_module
from mib.pdfio import Span, extract_spans
from mib.validate import valid_for_field

# --- Page types, in FIELD_MANUAL.md trust order ---------------------------
ADJUDICATOR = "adjudicator_note"
INTAKE = "intake_form"
BIOMETRIC = "biometric_slip"
SPONSOR = "sponsor_letter"
REGISTRY = "registry_extract"
FEE = "fee_receipt"
UNKNOWN_PAGE = "unknown"
# A scan of a visible page is visible evidence but carries additional OCR risk. It ranks
# below the crisp text layer (OCR can misread) but far above anything hidden.
SCANNED = "scanned_page"
# The same scan, read by the second OCR engine. Deliberately the least trusted
# visible source there is: it exists to fill gaps the primary engine left, never
# to arbitrate against it.
SCANNED_FALLBACK = "scanned_page_fallback"

PAGE_TITLES = [
    ("Manual Adjudicator Note", ADJUDICATOR),
    ("FORM I-8090", INTAKE),
    ("FORM B-13", BIOMETRIC),
    ("Sponsor Attestation", SPONSOR),
    ("Planetary Registry Extract", REGISTRY),
    ("MIB Fee Receipt", FEE),
]

# Lower number == more trusted. Mirrors "Trusted Evidence" in FIELD_MANUAL.md.
# A "Manual correction:" annotation outranks even the page it is written on: the
# manual's top precedence tier is "visible MIB adjudicator stamp *or signed
# manual note*", and these annotations are exactly that. They appear inline on a
# form page and can contradict the printed value beside it.
MANUAL_CORRECTION = "manual_correction"
TRUST_ORDER = {
    MANUAL_CORRECTION: 0,
    ADJUDICATOR: 1,
    INTAKE: 2,
    BIOMETRIC: 3,
    SPONSOR: 4,
    REGISTRY: 5,
    FEE: 2,          # authoritative for fee_status only
    SCANNED: 6,
    SCANNED_FALLBACK: 7,
    UNKNOWN_PAGE: 9,
}

# Sources whose readings may settle the policy `Record`. A fallback reading can
# always be *printed* -- a guess costs nothing an empty field does not -- but
# letting it drive an adjudication is a separate decision, made separately.
POLICY_TRUST_MAX = TRUST_ORDER[SCANNED]

LABEL_MIN, LABEL_MAX = 7.5, 8.5
TITLE_MIN = 13.0
STAMP_MIN = 18.0

# Spans that are visible but carry no evidentiary weight per policy.
NON_EVIDENTIARY = re.compile(r"^\s*(BARCODE|QR|MRZ)\b[^:]*:", re.I)

# Damage markers -> the field they destroy.
DAMAGE_MARKERS = {
    "NAME CUT OUT": "applicant_name",
    "SPECIES WHITEOUT": "species_code",
    "DATE WASHED OUT": "arrival_date",
    "SPONSOR ID BLANK": "sponsor_id",
    "VISA CLASS TORN": "visa_class",
    "PURPOSE ILLEGIBLE": "declared_purpose",
    "FEE STATUS OBSCURED": "fee_status",
    "REGISTRY LOST": "_registry",
}
DAMAGE_RE = re.compile(r"\[([A-Z0-9 _\-]{3,40})\]")
# Damage is not always bracketed: a washed-out value can render as a bare
# sentinel where the value should be. Treated as "no evidence", never as a value.
DAMAGE_SENTINELS = {"UNREADABLE", "ILLEGIBLE", "OBSCURED", "TORN", "WHITEOUT"}

# "Manual correction: <subject> is <value>." -- a signed note overriding the
# printed field. Subjects observed: sponsor, visa class, applicant, fee status.
MANUAL_CORRECTION_RE = re.compile(
    r"Manual correction:\s*(.+?)\s+is\s+(.+?)\.?\s*$", re.I)
CORRECTION_SUBJECTS = {
    "sponsor": "sponsor_id",
    "sponsor id": "sponsor_id",
    "visa class": "visa_class",
    "applicant": "applicant_name",
    "fee status": "fee_status",
    "species": "species_code",
    "species code": "species_code",
    "home world": "home_world",
    "arrival date": "arrival_date",
    "declared purpose": "declared_purpose",
}

NOTE_RE = re.compile(r"Finding:\s*([A-Z_]+)\.\s*Reason:\s*(.*)", re.I)
# `NOTE_RE` is matched against one span at a time, so a reason clause that wraps
# onto the following line is silently truncated. The same pattern over a whole
# page recovers the continuation; `re.S` is the entire difference.
NOTE_PAGE_RE = re.compile(r"Finding:\s*([A-Z_]+)\.\s*Reason:\s*(.*)", re.I | re.S)
# A reason clause can also state the rescission outright instead of naming the
# flag token, and the sentence is itself the policy fact rather than a paraphrase
# of one.
RESCINDED_RE = re.compile(r"\bprior denial stamp rescinded\b", re.I)
SPONSOR_ATTEST_RE = re.compile(
    r"Sponsor\s+(SPN-\d{4})\s+attests that\s+(.+?)\s+is expected on Earth for\s+(.+?)\.",
    re.S,
)
SPONSOR_CLASS_RE = re.compile(r"class\s+([A-Z]+-\d+)\s+compliance", re.I)
STAMP_WORDS = {"APPROVED", "DENIED", "REVIEW", "NEEDS_REVIEW"}

# Label -> canonical field name, per page type where it appears.
LABEL_FIELDS = {
    "Applicant": "applicant_name",
    "Registry Name": "applicant_name",
    "Species Code": "species_code",
    "Home World": "home_world",
    "Visa Class": "visa_class",
    "Sponsor ID": "sponsor_id",
    "Arrival Date": "arrival_date",
    "Declared Purpose": "declared_purpose",
    "Fee Status": "fee_status",
    "Waiver Code": "_waiver_code",
    "Registry Status": "_registry_status",
    "Case ID": "_case_id",
    "Amount": "_amount",
}

INLINE_FIELDS = {
    "applicant": "applicant_name",
    "species match": "species_code",
    "observed flags": "_observed_flags",
    "case id": "_case_id",
    "biometric confidence": "_biometric_confidence",
}


@dataclass
class Observation:
    """One field value together with where it came from."""

    field: str
    value: str
    source: str          # page type
    page: int
    trust: int           # lower is better

    @property
    def trusted(self) -> bool:
        return self.trust < 9


@dataclass
class PacketEvidence:
    case_id: str
    observations: list[Observation] = field(default_factory=list)
    damaged_fields: set[str] = field(default_factory=set)
    page_types: list[str] = field(default_factory=list)
    corrections: dict[str, str] = field(default_factory=dict)
    # Adjudicator note -- the single highest-precedence signal in the corpus.
    note_finding: str | None = None
    note_reason: str = ""
    stamps: list[str] = field(default_factory=list)
    sample_denial_watermark: bool = False
    # Untrusted content, captured deliberately so we can tell "no evidence"
    # apart from "an injection tried to supply this".
    hidden_texts: list[str] = field(default_factory=list)
    non_evidentiary_texts: list[str] = field(default_factory=list)
    injection_detected: bool = False
    # Cross-document facts used by the policy layer.
    registry_status: str | None = None
    waiver_code: str | None = None
    # Typed-receipt geometry: the amount as printed, and the fee status the
    # amount and waiver code imply. Output-only by construction -- see `assemble`.
    receipt_amount: str | None = None
    receipt_geometry_fee: str | None = None
    observed_flags: list[str] = field(default_factory=list)
    # Flag-shaped tokens found in free OCR text rather than after an
    # "Observed flags:" label. Kept separate because the label is what licenses
    # aggressive truncation matching -- see Lexicon.snap_flag.
    flag_candidates: list[str] = field(default_factory=list)
    # Flag tokens read from the part of a typed note's reason clause that falls
    # below the first span. Staged rather than merged on sight: whether the
    # packet carries an injection is only settled once every page has been read.
    note_continuation_flags: list[str] = field(default_factory=list)
    sponsor_letter_sponsor: str | None = None
    sponsor_letter_name: str | None = None
    sponsor_letter_class: str | None = None
    has_text_layer: bool = True
    ocr_pages: int = 0
    # Pages the second OCR engine also read.
    fallback_pages: int = 0
    # The same non-field signals, as the second OCR engine read them. Held in
    # separate slots rather than merged into the fields above because they have
    # no trust order to protect them: a flag joins a set, a note finding
    # overrides the whole adjudication. `mib.pipeline.assemble` decides which of
    # them may take effect, and each decision was measured on its own.
    fallback_observed_flags: list[str] = field(default_factory=list)
    fallback_flag_candidates: list[str] = field(default_factory=list)
    fallback_note_finding: str | None = None
    fallback_note_reason: str = ""
    fallback_risk_panel_read: bool = False
    fallback_risk_panel_missing: bool = False
    fallback_registry_status: str | None = None
    fallback_waiver_code: str | None = None
    fallback_biometric_confidence: float | None = None
    note_from_ocr: bool = False
    risk_panel_missing: bool = False
    # The risk panel was read, independently of whether it listed any flags.
    # "Observed flags: none" is evidence of no flags; a page we never read is
    # not. Conflating the two is what produced the original false approvals.
    risk_panel_read: bool = False
    # Printed on the biometric slip. Low confidence is what drives
    # `illegible_biometrics`, so it carries signal about that flag even on
    # packets where the flag line itself did not survive.
    biometric_confidence: float | None = None

    def values(self, field_name: str) -> list[Observation]:
        """Observations for a field, most trusted first."""
        return sorted((o for o in self.observations if o.field == field_name),
                      key=lambda o: (o.trust, o.page))

    def best(self, field_name: str) -> Observation | None:
        vals = self.values(field_name)
        return vals[0] if vals else None


# What an OCR engine's readings are allowed to contribute to the evidence set.
#
# The primary engine contributes everything. The fallback engine's permissions
# are the experiment: each one was added only after repeated out-of-fold
# measurement showed it paid on the full 150-point objective. `fields` is not
# the whole story, because several of these signals bypass field resolution
# entirely -- a note finding settles the case outright, and a mined flag joins a
# set rather than competing for a slot.
PRIMARY_CONTRIBUTIONS = frozenset(
    {"fields", "literals", "flags", "note", "singletons", "reason_facts"})
FALLBACK_CONTRIBUTIONS = frozenset(
    {"fields", "literals", "flags", "note", "singletons", "reason_facts"})


def _absorb(ev: "PacketEvidence", texts: list[str], page_no: int, source: str,
            seen: set[tuple[str, str]], allow: frozenset[str]) -> None:
    """Fold one engine's readings of one page into the evidence set.

    `allow` gates each *kind* of contribution rather than each field, because
    the kinds differ in how they can hurt. A field value competes for a slot and
    loses to anything more trusted; a risk flag joins a set and can only add;
    a note finding overrides the entire adjudication. They deserve separate
    verdicts, and they got them.
    """
    trust = TRUST_ORDER[source]

    def add(name: str, value: str) -> None:
        key = (name, _clean(value).casefold())
        if key in seen:
            return
        seen.add(key)
        _record(ev, name, value, source, trust, page_no)

    # Non-field signals from the fallback engine land in parallel slots, so the
    # decision to act on them stays with `mib.pipeline.assemble` and can be
    # measured. `p` prefixes every such attribute name.
    p = "fallback_" if source == SCANNED_FALLBACK else ""

    def first(attr: str, value) -> None:
        """Set a single-valued signal, if nothing has claimed it yet."""
        if value and getattr(ev, p + attr) is None:
            setattr(ev, p + attr, value)

    for text in texts:
        fields, flags, extras = ocr_module.parse_fields(text)
        if "fields" in allow:
            for name, value in fields.items():
                add(name, value)
        if "flags" in allow:
            getattr(ev, p + "observed_flags").extend(flags)
            # An explicitly missing risk panel is unread evidence, not "no
            # flags" -- the distinction that governs false approvals.
            if extras.get("risk_panel_read"):
                setattr(ev, p + "risk_panel_read", True)
            if "RISK PANEL MISSING" in text.upper():
                setattr(ev, p + "risk_panel_missing", True)
        if "singletons" in allow:
            first("registry_status", extras.get("registry_status"))
            first("waiver_code", extras.get("waiver_code"))
            if extras.get("biometric_confidence"):
                try:
                    first("biometric_confidence", min(
                        100, int(extras["biometric_confidence"])) / 100.0)
                except ValueError:
                    pass
        # A scanned adjudicator note is still the top evidence tier.
        # Text-layer notes win if both exist.
        if "note" in allow and extras.get("note_finding") \
                and getattr(ev, p + "note_finding") is None:
            setattr(ev, p + "note_finding", extras["note_finding"])
            setattr(ev, p + "note_reason", extras.get("note_reason", ""))
            if not p:
                ev.note_from_ocr = True
        # A signed note that states a field value is the top evidence tier, so
        # these are recorded at ADJUDICATOR trust rather than as another
        # candidate from the page they were read on. A *fallback* reading of
        # such a note is not: the engine that read it is the whole question, so
        # it stays at the fallback rank where it cannot displace anything.
        if "reason_facts" in allow:
            # Same page, same authority as the printed status line, so the same
            # trust tier -- `_resolve` then breaks the tie by plausibility if
            # both are present and disagree.
            receipt_src = FEE if source == SCANNED else source
            note_src = ADJUDICATOR if source == SCANNED else source
            if extras.get("receipt_fee_status"):
                _record(ev, "fee_status", extras["receipt_fee_status"],
                        receipt_src, TRUST_ORDER[receipt_src], page_no)
            if extras.get("reason_fee_status"):
                _record(ev, "fee_status", extras["reason_fee_status"],
                        note_src, TRUST_ORDER[note_src], page_no)
            if extras.get("reason_home_world"):
                _record(ev, "home_world", extras["reason_home_world"],
                        note_src, TRUST_ORDER[note_src], page_no)

    # Mined literals go in last, so a value parsed from its own label always
    # wins the page-order tiebreak against one merely found floating in the text.
    for text in texts:
        if "literals" in allow:
            for name, values in ocr_module.mine_literals(text).items():
                for value in values:
                    add(name, value)
        if "flags" in allow:
            getattr(ev, p + "flag_candidates").extend(
                ocr_module.mine_flag_candidates(text))


def _page_type(title: str) -> str:
    for prefix, kind in PAGE_TITLES:
        if title.startswith(prefix):
            return kind
    return UNKNOWN_PAGE


def _clean(text: str) -> str:
    return " ".join(text.split()).strip()


def _note_reason_facts(ev: PacketEvidence, reason: str, page_no: int) -> None:
    """Record the field values a typed note's reason clause states outright.

    Deliberately routed through the *same* `mib.ocr` parsers the scanned path
    uses, rather than a second set of patterns here: two implementations of
    "what does this reason clause assert" would drift, and the scanned one is
    the one with measured purity behind it.

    Risk flags named in a reason are **unioned** with what the panel gave, never
    substituted -- a reason cites the governing flag, not the complete set, and
    across the typed notes it is a true subset of the truth every time it
    appears. Union preserves that; replacement would discard the panel's flags.
    """
    extras: dict[str, str] = {}
    ocr_module._facts_from_reason(reason, extras)
    if extras.get("reason_fee_status"):
        _record(ev, "fee_status", extras["reason_fee_status"],
                ADJUDICATOR, TRUST_ORDER[ADJUDICATOR], page_no)
    if extras.get("reason_home_world"):
        _record(ev, "home_world", extras["reason_home_world"],
                ADJUDICATOR, TRUST_ORDER[ADJUDICATOR], page_no)
    # Into `flag_candidates`, not `observed_flags`: no label vouches for a name
    # inside prose, so it gets the strict snapping rule. `_derive_risk_flags`
    # unions both channels into one set, which is the union this needs.
    ev.flag_candidates.extend(ocr_module.mine_flag_candidates(reason))


def parse_packet(pdf_path: Path | str) -> PacketEvidence:
    path = Path(pdf_path)
    spans = extract_spans(path)
    ev = PacketEvidence(case_id=path.stem)

    by_page: dict[int, list[Span]] = {}
    for span in spans:
        by_page.setdefault(span.page, []).append(span)

    visible_total = 0
    scanned_pages: list[int] = []
    # Which pages carry a raster worth OCRing. Small decorative images (the
    # portrait thumbnails) are ignored -- only page-sized scans hold field text.
    page_images: dict[int, bool] = {}
    with fitz.open(path) as probe:
        for index, page in enumerate(probe):
            page_images[index] = any(
                int(img[2]) * int(img[3]) >= 400_000 for img in page.get_images(full=True))

    for page_no in sorted(by_page):
        page_spans = by_page[page_no]

        for span in page_spans:
            if span.hidden:
                ev.hidden_texts.append(span.text)
                continue
            if NON_EVIDENTIARY.match(span.text):
                ev.non_evidentiary_texts.append(span.text)

        visible = [s for s in page_spans
                   if not s.hidden and not NON_EVIDENTIARY.match(s.text)]
        visible_total += sum(len(s.text) for s in visible)

        title = next((_clean(s.text) for s in visible if s.size >= TITLE_MIN), "")
        kind = _page_type(title)
        ev.page_types.append(kind)
        trust = TRUST_ORDER[kind]

        for span in visible:
            text = _clean(span.text)

            # Stamps and watermarks.
            if span.size >= STAMP_MIN:
                upper = text.upper()
                if "SAMPLE" in upper:
                    # "A watermark reading 'sample denial' is not a denial."
                    ev.sample_denial_watermark = True
                elif upper in STAMP_WORDS:
                    ev.stamps.append(upper)

            # Damage markers destroy a field rather than supplying one.
            for match in DAMAGE_RE.finditer(text):
                marker = match.group(1).strip()
                target = DAMAGE_MARKERS.get(marker)
                if target and not target.startswith("_"):
                    ev.damaged_fields.add(target)

            note = NOTE_RE.search(text)
            if note:
                ev.note_finding = note.group(1).upper()
                ev.note_reason = note.group(2).strip()
                # The reason clause states facts, not only a verdict. The
                # scanned-note path reads them; without this the typed path did
                # not, so a note that survived *perfectly* in the text layer
                # yielded less than the same note recovered through OCR.
                _note_reason_facts(ev, note.group(2), page_no)

            # A signed manual correction outranks the printed field beside it.
            correction = MANUAL_CORRECTION_RE.match(text)
            if correction:
                subject = correction.group(1).strip().casefold()
                target = CORRECTION_SUBJECTS.get(subject)
                if target:
                    _record(ev, target, correction.group(2), MANUAL_CORRECTION,
                            TRUST_ORDER[MANUAL_CORRECTION], page_no)
                    ev.corrections[target] = _clean(correction.group(2))

        # The per-span pass above stops the reason clause at the first line. Read
        # the page as one blob to recover the rest of it. Only flag names are
        # taken: the verdict and the field facts were already settled from the
        # first span, and widening those was never measured.
        if kind == ADJUDICATOR:
            blob = "\n".join(_clean(s.text) for s in visible if s.text.strip())
            full = NOTE_PAGE_RE.search(blob)
            if full:
                ev.note_continuation_flags.extend(
                    ocr_module.mine_flag_candidates(full.group(2)))
                if RESCINDED_RE.search(full.group(2)):
                    ev.note_continuation_flags.append("rescinded_denial")

        before = len(ev.observations)
        _parse_labelled_fields(ev, visible, kind, trust, page_no)
        _parse_inline_fields(ev, visible, kind, trust, page_no)

        if kind == SPONSOR:
            _parse_sponsor_letter(ev, visible, page_no)

        # OCR any page that yielded no field values but carries a raster.
        #
        # Triggering on "almost no text at all" is not enough: a scanned page
        # often keeps a crisp text-layer title and header ("MIB Fee Receipt",
        # "MIB-000003 | MIB Eyes Only") while every *value* exists only as
        # pixels. Those pages have 2+ text spans, so a sparse-text trigger skips
        # them -- which silently lost ~400 fee_status values.
        produced_values = len(ev.observations) > before
        has_raster = bool(page_images.get(page_no))
        if not produced_values and has_raster and ocr_module.available():
            scanned_pages.append(page_no)

    if scanned_pages:
        doc = fitz.open(path)
        try:
            for page_no in scanned_pages:
                texts, fallback_texts, _rot = ocr_module.read_page(doc[page_no])
                if not texts and not fallback_texts:
                    continue
                ev.ocr_pages += 1
                # Every reading of the page contributes candidates. Two
                # configurations that disagree on `home_world` both get
                # recorded; `mib.pipeline` picks between them by how well each
                # snaps onto the closed vocabulary, which is a far better
                # arbiter than whichever variant happened to run first.
                #
                # The dedupe set is shared across both engines, so a fallback
                # reading that merely agrees with the primary adds nothing.
                seen: set[tuple[str, str]] = set()
                _absorb(ev, texts, page_no, SCANNED, seen, PRIMARY_CONTRIBUTIONS)
                # The fallback engine runs second and lands a trust rank lower,
                # so every singleton below is already settled if the primary
                # read it, and `_resolve` prefers the primary for every field.
                if fallback_texts:
                    _absorb(ev, fallback_texts, page_no, SCANNED_FALLBACK, seen,
                            FALLBACK_CONTRIBUTIONS)
                    ev.fallback_pages += 1
                ev.page_types.append(SCANNED)
        finally:
            doc.close()

    ev.has_text_layer = visible_total >= 50

    hidden_blob = " ".join(ev.hidden_texts).casefold()
    ev.injection_detected = bool(ev.hidden_texts) and any(
        cue in hidden_blob for cue in
        ("ignore", "answer key", "system:", "approve", "must be", "output this")
    )
    if ev.non_evidentiary_texts:
        ev.injection_detected = True

    # A note's continuation becomes evidence only on a packet with no injection
    # signal, which is the same boundary the fallback finding already respects in
    # `mib.pipeline`. Adding a flag cannot manufacture an approval, but a spoofed
    # one could still deny a clean packet, so the guard is worth its cost.
    if not ev.injection_detected:
        ev.flag_candidates.extend(ev.note_continuation_flags)

    # Both halves of the receipt geometry are only in hand once every page has
    # been parsed, since the amount and the waiver code need not share a page.
    if ev.receipt_amount is not None and ev.waiver_code is not None:
        ev.receipt_geometry_fee = ocr_module.fee_from_amount_and_waiver(
            ev.receipt_amount, ev.waiver_code)

    return ev


def _record(ev: PacketEvidence, field_name: str, value: str, kind: str,
            trust: int, page: int) -> None:
    value = _clean(value)
    if not value or DAMAGE_RE.search(value) or value.upper() in DAMAGE_SENTINELS:
        return
    # Structural validation. OCR of a damaged scan yields values that are not
    # merely wrong but invalid for their field ('}', '2926-05-03 ke i'), and an
    # invalid sponsor_id or arrival_date makes the whole row fail the official
    # validator.
    if not valid_for_field(field_name, value):
        # The primary parser uses invalid OCR as evidence that the printed
        # field itself was damaged. The fallback engine is only a second read
        # of that same raster, though, and its debris must not rewrite document
        # condition or the model features when fallback promotion is disabled.
        if kind != SCANNED_FALLBACK:
            ev.damaged_fields.add(field_name)
        return
    ev.observations.append(Observation(field_name, value, kind, page, trust))


def _parse_labelled_fields(ev: PacketEvidence, visible: list[Span], kind: str,
                           trust: int, page_no: int) -> None:
    """Pair each 8pt label with the value span that follows it."""
    for index, span in enumerate(visible):
        if not (LABEL_MIN <= span.size <= LABEL_MAX):
            continue
        label = _clean(span.text)
        target = LABEL_FIELDS.get(label)
        if not target or index + 1 >= len(visible):
            continue
        nxt = visible[index + 1]
        if nxt.size <= LABEL_MAX:
            continue
        value = _clean(nxt.text)

        if target == "_registry_status":
            ev.registry_status = value
        elif target == "_waiver_code":
            ev.waiver_code = value
        elif target == "_amount":
            # Parsed and dropped until now, while the scanned path derived a fee
            # status from the same two numbers.
            ev.receipt_amount = value
        elif target.startswith("_"):
            continue
        else:
            _record(ev, target, value, kind, trust, page_no)


def _parse_inline_fields(ev: PacketEvidence, visible: list[Span], kind: str,
                         trust: int, page_no: int) -> None:
    """The biometric slip uses inline ``Key: value`` rather than label pairs."""
    for span in visible:
        text = _clean(span.text)
        if ":" not in text or span.size >= TITLE_MIN:
            continue
        key, _, value = text.partition(":")
        target = INLINE_FIELDS.get(key.strip().casefold())
        if not target:
            continue
        value = _clean(value)
        if target == "_observed_flags":
            for part in re.split(r"[,;|]", value):
                part = part.strip()
                if part and part.lower() != "none":
                    ev.observed_flags.append(part)
        elif target == "_biometric_confidence":
            match = re.search(r"(\d{1,3})", value)
            if match and ev.biometric_confidence is None:
                ev.biometric_confidence = min(100, int(match.group(1))) / 100.0
        elif target.startswith("_"):
            continue
        else:
            _record(ev, target, value, kind, trust, page_no)


def _parse_sponsor_letter(ev: PacketEvidence, visible: list[Span], page: int = 0) -> None:
    blob = " ".join(_clean(s.text) for s in visible)
    match = SPONSOR_ATTEST_RE.search(blob)
    if match:
        ev.sponsor_letter_sponsor = match.group(1)
        name = _clean(match.group(2))
        # Only treat it as a name if it is not a damage marker -- an early
        # version appended "[NAME CUT OUT]" as an applicant_name observation.
        if not DAMAGE_RE.search(name) and name.upper() not in DAMAGE_SENTINELS:
            ev.sponsor_letter_name = name
        _record(ev, "declared_purpose", match.group(3), SPONSOR,
                TRUST_ORDER[SPONSOR], page)
        _record(ev, "sponsor_id", match.group(1), SPONSOR, TRUST_ORDER[SPONSOR], page)
        _record(ev, "applicant_name", name, SPONSOR, TRUST_ORDER[SPONSOR], page)
    cls = SPONSOR_CLASS_RE.search(blob)
    if cls:
        ev.sponsor_letter_class = cls.group(1).upper()
        _record(ev, "visa_class", ev.sponsor_letter_class, SPONSOR,
                TRUST_ORDER[SPONSOR], page)
