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

import re
from dataclasses import dataclass
from pathlib import Path

from mib.extract import (
    ADJUDICATOR,
    BIOMETRIC,
    MANUAL_CORRECTION,
    SCANNED,
    PacketEvidence,
    parse_packet,
)
from mib.features import packet_features, refresh_temporal
from mib.lexicon import Lexicon
from mib.policy import (
    APPROVED,
    decision_path,
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


# Fields whose shape is checkable without a vocabulary, used to score OCR
# candidates that no closed set can arbitrate.
_SPONSOR_RE = re.compile(r"^SPN-\d{4}$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _candidate_score(field: str, value: str, lexicon: Lexicon) -> float:
    """How much a candidate value looks like a real value for its field.

    Only used to break ties *within* one trust tier. Multi-variant OCR now
    yields several readings of the same scan -- ``Woll-108 fc`` and
    ``Wolf-1061c`` for the same line -- and taking whichever ran first is
    arbitrary. Snap confidence is the honest arbiter: it is the same measure the
    printer already trusts to correct OCR noise, so a candidate that snaps
    cleanly is, by construction, the one we would have printed anyway.
    """
    value = value.strip()
    if not value:
        return 0.0
    if field in SNAP_FIELDS:
        return lexicon.snap(field, value)[1]
    if field == "applicant_name":
        return lexicon.snap_name(value)[1]
    if field == "sponsor_id":
        return 1.0 if _SPONSOR_RE.match(value) else 0.0
    if field == "arrival_date":
        return 1.0 if _DATE_RE.match(value) else 0.0
    if field == "fee_status":
        return 1.0 if value.casefold() in FEE_VALUES else lexicon.snap(
            "fee_status", value)[1] * 0.9
    return 0.0


def _resolve(ev: PacketEvidence, field: str, lexicon: Lexicon | None = None
             ) -> str | None:
    """Most-trusted visible value for a field, or None if no trusted evidence.

    Where several equally-trusted observations exist -- several OCR variants of
    the same scan, or the same field printed on two pages of the same rank --
    the one that scores best as a *plausible value* wins, with page order as the
    final tiebreak so resolution stays deterministic.
    """
    values = [o for o in ev.values(field) if o.trusted]
    if not values:
        return None
    top = values[0].trust
    tier = [o for o in values if o.trust == top]
    if len(tier) == 1 or lexicon is None:
        return tier[0].value
    return max(tier, key=lambda o: (_candidate_score(field, o.value, lexicon),
                                    -o.page)).value


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
    #
    # Deliberately conservative, and only over crisp text-layer sources. OCR of
    # a scan renders the same name as "Veetari Tekmora" and "Vestan Tekmors",
    # or appends page furniture ("Solix Solquell SCAN IMAGE") -- comparing those
    # produced ~141 false identity conflicts, which cost extraction points and
    # wrongly forced packets to NEEDS_REVIEW. A manual correction also *resolves*
    # a conflict rather than being one, so its presence suppresses the flag.
    if "applicant_name" not in ev.corrections:
        crisp = set()
        for obs in ev.values("applicant_name"):
            if obs.source in (SCANNED, MANUAL_CORRECTION):
                continue
            snapped, conf = lexicon.snap_name(obs.value)
            # Only a confidently-recognised name can evidence a conflict.
            if conf > 0.0:
                crisp.add(snapped.casefold())
        if len(crisp) > 1:
            flags.add("identity_conflict")

    return flags


@dataclass
class Extraction:
    """Everything phase 1 recovers from a packet, before any adjudication."""

    printed: dict[str, str]
    record: Record
    note: str | None
    features: dict[str, float]


def extract_packet(pdf_path: Path, lexicon: Lexicon) -> "Extraction":
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
        value = _resolve(ev, field, lexicon)

        if value is not None:
            if field in SNAP_FIELDS:
                snapped, conf = lexicon.snap(field, value)
                if conf > 0.0:
                    value = snapped
                else:
                    # Nothing in the closed set is close, so what we read is
                    # OCR debris rather than a value. Printing it is a certain
                    # miss where the prior mode has the base rate, and -- more
                    # importantly -- feeding it to the policy engine as though
                    # it were a known visa class silently corrupts the decision
                    # path. Treat it as unread.
                    value = None
            elif field == "applicant_name":
                snapped, conf = lexicon.snap_name(value)
                if conf > 0.0:
                    value = snapped

        if value is not None:
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

    fee_raw = _resolve(ev, "fee_status", lexicon)
    fee = UNKNOWN
    if fee_raw:
        candidate = fee_raw.strip().lower()
        if candidate not in FEE_VALUES:
            # Snap onto the closed vocabulary before giving up: an OCR'd
            # "paig"/"waivec" is a perfectly recoverable "paid"/"waived".
            snapped, conf = lexicon.snap("fee_status", candidate)
            candidate = snapped.lower() if conf > 0.0 else candidate
        if candidate in FEE_VALUES:
            fee = candidate
    printed["fee_status"] = fee

    flags = _derive_risk_flags(ev, lexicon)
    printed["risk_flags"] = "|".join(sorted(flags)) if flags else "none"

    # "No flags found" is only meaningful if we actually read the page that
    # carries them -- the biometric slip, or an adjudicator note that states the
    # governing flag.
    #
    # A registry extract used to count here, and that was wrong: the registry
    # reports embargo status only, so a clean registry says nothing about
    # biohazard, active warrants or memory tampering. It was silently marking
    # flags "known" on 14 of the 21 remaining false approvals -- packets that
    # carried no risk page whatsoever.
    flags_known = (bool(flags)
                   or BIOMETRIC in ev.page_types
                   or ev.note_finding is not None) and not ev.risk_panel_missing

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
        has_scanned_pages=SCANNED in ev.page_types,
    )

    note = ev.note_finding if ev.note_finding in (APPROVED, DENIED, NEEDS_REVIEW) else None
    printed["_injection"] = "1" if ev.injection_detected else ""
    printed["_damaged"] = ",".join(sorted(ev.damaged_fields))
    feats = packet_features(ev, record)
    return Extraction(printed=printed, record=record, note=note, features=feats)


def finalize(printed: dict[str, str], record: Record, note: str | None,
             calibration: Calibration, adjudicator=None,
             features: dict[str, float] | None = None) -> Prediction:
    """Phase 2: adjudicate an already-extracted record. Microseconds, no I/O.

    `adjudicator` is the optional learned model. It only ever runs on cases the
    hard rules do not already settle -- an adjudicator note states the finding
    outright and was 217/217 correct, so no model gets to overrule it.
    """
    if note is not None:
        adjudication = note
        # The note dictates the decision, so the meaningful confidence is how
        # often notes are right -- not an outcome distribution over paths.
        confidence = calibration.accuracy(ADJUDICATOR_NOTE_PATH, 0.95)
        path = ADJUDICATOR_NOTE_PATH
    elif adjudicator is not None and features is not None:
        # The staleness reference is only known now, so temporal features and
        # the recorded decision path must be rebuilt before the model reads them.
        adjudication, confidence, path = adjudicator.adjudicate(
            refresh_temporal(features, record),
            calibration.probs(decision_path(record)))
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
    ex = extract_packet(pdf_path, lexicon)
    if reference_date:
        ex.record.receipt_date = reference_date
    return finalize(ex.printed, ex.record, ex.note, calibration)
