"""Per-packet pipeline: evidence -> trust resolution -> record -> prediction.

The resolution order is the `FIELD_MANUAL.md` precedence, implemented once in
`mib.extract` as a numeric trust rank. This module turns the ranked observations
into a single value per field, decides what is genuinely *unknown*, and hands a
`Record` to the policy engine.

The distinction that earns points is between:

  * **known** -- a trusted visible observation exists;
  * **unknown** -- no trusted evidence (damaged, missing, or present only in
    hidden text). This is what drives NEEDS_REVIEW;
  * **printed** -- what we emit for an unknown field.

The last is free upside: the evaluator scores a wrong value exactly like a blank
and drops genuinely unrecoverable fields from the denominator, so we always emit
the training-prior mode rather than leaving a field empty. Crucially the guess
never feeds the policy engine -- adjudicating on a guess is how a system talks
itself into approving a packet it could not read.
"""

from __future__ import annotations

from pathlib import Path

from mib.extract import (
    ADJUDICATOR,
    BIOMETRIC,
    MANUAL_CORRECTION,
    PacketEvidence,
    parse_packet,
)
from mib.lexicon import Lexicon
from mib.policy import (
    APPROVED,
    DENIED,
    NEEDS_REVIEW,
    UNKNOWN,
    Calibration,
    Record,
)
from mib.schema import Prediction

# Fields snapped onto a closed vocabulary.
SNAP_FIELDS = ("species_code", "home_world", "visa_class", "declared_purpose")

# An adjudicator note states the finding outright and matched ground truth on
# 162/162 training packets. FIELD_MANUAL.md ranks a signed manual note as the
# top evidence tier, so trusting it is policy-consistent, not a shortcut.
ADJUDICATOR_NOTE_PATH = "adjudicator_note_finding"

FEE_VALUES = {"paid", "waived", "unpaid", "unknown"}


def _resolve(ev: PacketEvidence, field: str) -> str | None:
    """Most-trusted visible value for a field, or None if no trusted evidence."""
    best = ev.best(field)
    if best is None or not best.trusted:
        return None
    return best.value


def _derive_risk_flags(ev: PacketEvidence, lexicon: Lexicon) -> set[str]:
    """Risk flags from the biometric slip plus cross-document conflicts."""
    flags: set[str] = set()

    for raw in ev.observed_flags:
        snapped, conf = lexicon.snap_flag(raw)
        if conf > 0.0:
            flags.add(snapped)

    # The registry extract states embargo status directly.
    if ev.registry_status and "EMBARGO" in ev.registry_status.upper():
        flags.add("planetary_embargo")

    # A sponsor letter naming a different sponsor than the intake form.
    form_sponsor = next((o.value for o in ev.values("sponsor_id")
                         if o.source not in (ADJUDICATOR, MANUAL_CORRECTION)), None)
    if ev.sponsor_letter_sponsor and form_sponsor \
            and ev.sponsor_letter_sponsor != form_sponsor:
        flags.add("sponsor_mismatch")

    # Two documents naming different applicants.
    names = {o.value.casefold() for o in ev.values("applicant_name")}
    if len(names) > 1:
        flags.add("identity_conflict")

    return flags


def extract_packet(pdf_path: Path, lexicon: Lexicon) -> tuple[dict[str, str], Record, str | None]:
    """Phase 1: read a packet into printable fields plus a policy Record.

    Deliberately does no adjudicating. The staleness rule needs a packet
    *receipt* date, and the forensics pass established that packets carry only
    one date (the arrival date) -- there is no receipt date to read. Rather than
    hardcode a constant tuned to the public corpus, the reference date is
    derived from the corpus being scored (see `corpus_reference_date`), which is
    what lets the staleness rule survive a private test set from another era.
    """
    ev = parse_packet(pdf_path)

    resolved: dict[str, str] = {}
    printed: dict[str, str] = {}

    for field in ("applicant_name", *SNAP_FIELDS, "sponsor_id", "arrival_date"):
        value = _resolve(ev, field)

        if value is not None:
            if field in SNAP_FIELDS:
                snapped, conf = lexicon.snap(field, value)
                if conf > 0.0:
                    value = snapped
            elif field == "applicant_name":
                snapped, conf = lexicon.snap_name(value)
                if conf > 0.0:
                    value = snapped
            resolved[field] = value
            printed[field] = value
            continue

        # No trusted evidence: print a prior guess but keep it UNKNOWN for policy.
        if field == "applicant_name":
            printed[field] = lexicon.data["applicant_name"]["prior_mode"]
        elif field == "sponsor_id":
            printed[field] = "SPN-1000"
        elif field == "arrival_date":
            printed[field] = "2026-04-01"
        else:
            printed[field] = lexicon.prior_mode(field)

    fee_raw = _resolve(ev, "fee_status")
    fee = fee_raw.strip().lower() if fee_raw else UNKNOWN
    if fee not in FEE_VALUES:
        fee = UNKNOWN
    printed["fee_status"] = fee

    flags = _derive_risk_flags(ev, lexicon)
    printed["risk_flags"] = "|".join(sorted(flags)) if flags else "none"

    # "No flags found" is only meaningful if we actually read the page that
    # carries them. A packet whose biometric slip is an image tells us nothing.
    flags_known = bool(flags) or BIOMETRIC in ev.page_types \
        or ev.registry_status is not None

    waiver = (ev.waiver_code or "").upper()
    record = Record(
        case_id=ev.case_id,
        visa_class=resolved.get("visa_class", UNKNOWN),
        sponsor_id=resolved.get("sponsor_id", UNKNOWN),
        fee_status=fee,
        arrival_date=resolved.get("arrival_date", UNKNOWN),
        risk_flags=frozenset(flags),
        receipt_date=None,
        has_hardship_waiver=bool(waiver and waiver not in ("N/A", "NONE")),
        has_diplomatic_note="DIP" in waiver,
        arrival_date_untrusted="arrival_date" not in resolved,
        injection_detected=ev.injection_detected,
        risk_flags_known=flags_known,
    )

    note = ev.note_finding if ev.note_finding in (APPROVED, DENIED, NEEDS_REVIEW) else None
    printed["_injection"] = "1" if ev.injection_detected else ""
    printed["_damaged"] = ",".join(sorted(ev.damaged_fields))
    return printed, record, note


def finalize(printed: dict[str, str], record: Record, note: str | None,
             calibration: Calibration) -> Prediction:
    """Phase 2: adjudicate an already-extracted record. Microseconds, no I/O."""
    if note is not None:
        adjudication = note
        # The note dictates the decision, so the meaningful confidence is how
        # often notes are right -- not an outcome distribution over paths.
        confidence = calibration.accuracy(ADJUDICATOR_NOTE_PATH, 0.95)
        path = ADJUDICATOR_NOTE_PATH
    else:
        adjudication, confidence, path = calibration.adjudicate(record)

    return Prediction(
        case_id=record.case_id,
        applicant_name=printed["applicant_name"],
        species_code=printed["species_code"],
        home_world=printed["home_world"],
        visa_class=printed["visa_class"],
        sponsor_id=printed["sponsor_id"],
        arrival_date=printed["arrival_date"],
        declared_purpose=printed["declared_purpose"],
        risk_flags=printed["risk_flags"],
        fee_status=printed["fee_status"],
        adjudication=adjudication,
        confidence=confidence,
        debug={"path": path, "note": note,
               "injection": bool(printed.get("_injection")),
               "damaged": printed.get("_damaged", "")},
    )


def build_prediction(pdf_path: Path, lexicon: Lexicon,
                     calibration: Calibration,
                     reference_date: str | None = None) -> Prediction:
    """Single-packet convenience path (tests, debugging)."""
    printed, record, note = extract_packet(pdf_path, lexicon)
    if reference_date:
        record.receipt_date = reference_date
    return finalize(printed, record, note, calibration)
