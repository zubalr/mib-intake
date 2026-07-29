"""Per-packet pipeline from evidence to prediction.

The resolution order is the `FIELD_MANUAL.md` precedence, implemented once in
`mib.extract` as a numeric trust rank. This module turns the ranked observations
into a single value per field, decides what is genuinely *unknown*, and hands a
`Record` to the policy engine.

The distinction that earns points is between:

  * known: a trusted visible observation exists;
  * unknown: no trusted evidence (damaged, missing, or present only in
    hidden text). This is what drives NEEDS_REVIEW;
  * printed: the schema-valid value emitted for an unknown field.

Printed defaults are kept separate from policy evidence. A value emitted for
schema completeness cannot justify an adjudication.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from mib.extract import (
    ADJUDICATOR,
    BIOMETRIC,
    MANUAL_CORRECTION,
    POLICY_TRUST_MAX,
    SCANNED,
    SCANNED_FALLBACK,
    SPONSOR,
    TRUST_ORDER,
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
    repair_year,
)
from mib.schema import FALLBACK_ARRIVAL_DATE, FALLBACK_SPONSOR_ID, Prediction

# Fields snapped onto a closed vocabulary.
SNAP_FIELDS = ("species_code", "home_world", "visa_class", "declared_purpose")

# The field manual gives an explicit adjudicator finding the highest trust rank.
ADJUDICATOR_NOTE_PATH = "adjudicator_note_finding"

FEE_VALUES = {"paid", "waived", "unpaid", "unknown"}


# Fields whose shape is checkable without a vocabulary, used to score OCR
# candidates that no closed set can arbitrate.
_SPONSOR_RE = re.compile(r"^SPN-\d{4}$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# An identity conflict must be evidenced by *crisp* identity documents. Any OCR
# reading of a scan is excluded, whichever engine produced it -- two engines
# disagreeing about a smudged name is a property of the smudge, not of the
# applicant.
_NOT_IDENTITY_SOURCES = (SCANNED, SCANNED_FALLBACK, MANUAL_CORRECTION, SPONSOR)


def _candidate_score(field: str, value: str, lexicon: Lexicon) -> float:
    """How much a candidate value looks like a real value for its field.

    Used only to break ties within one trust tier. Closed-vocabulary snap
    confidence resolves competing OCR readings consistently with output
    normalization.
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


def _resolve(ev: PacketEvidence, field: str, lexicon: Lexicon | None = None,
             max_trust: int | None = None) -> str | None:
    """Most-trusted visible value for a field, or None if no trusted evidence.

    Where several equally-trusted observations exist -- several OCR variants of
    the same scan, or the same field printed on two pages of the same rank --
    the one that scores best as a *plausible value* wins, with page order as the
    final tiebreak so resolution stays deterministic.

    `max_trust` excludes ranks below it. The printed output resolves over every
    source; the policy `Record` resolves over a restricted set, which is how a
    second-engine reading can fill a blank field without being allowed to swing
    an adjudication.
    """
    values = [o for o in ev.values(field) if o.trusted
              and (max_trust is None or o.trust <= max_trust)]
    if not values:
        return None
    top = values[0].trust
    tier = [o for o in values if o.trust == top]
    if len(tier) == 1 or lexicon is None:
        return tier[0].value
    return max(tier, key=lambda o: (_candidate_score(field, o.value, lexicon),
                                    -o.page)).value


def _derive_risk_flags(ev: PacketEvidence, lexicon: Lexicon,
                       with_fallback: bool = False) -> set[str]:
    """Risk flags from the biometric slip plus cross-document conflicts.

    `with_fallback` folds in what the second OCR engine read. Flags are a set,
    so a fallback reading can only ever *add* one -- there is no trust order to
    protect here, which is exactly why this needs its own verdict rather than
    riding along with field resolution.
    """
    flags: set[str] = set()

    # Two sources, deliberately snapped with different strictness.
    #
    # `observed_flags` came from the value side of an "Observed flags:" label,
    # so context already guarantees each token is a flag; truncation matching is
    # safe and recovers scans clipped to "resc" or "ifle".
    observed = list(ev.observed_flags)
    candidates = list(ev.flag_candidates)
    registry = ev.registry_status
    if with_fallback:
        observed += ev.fallback_observed_flags
        candidates += ev.fallback_flag_candidates
        registry = registry or ev.fallback_registry_status

    for raw in observed:
        snapped, conf = lexicon.snap_flag(raw, allow_truncation=True)
        if conf > 0.0:
            flags.add(snapped)

    # `flag_candidates` were mined from free text with no label vouching for
    # them, so they get the strict rule. Anything unlike a flag returns
    # confidence 0 and is dropped.
    for raw in candidates:
        snapped, conf = lexicon.snap_flag(raw)
        if conf > 0.0:
            flags.add(snapped)

    # The registry extract states embargo status directly.
    if registry and "EMBARGO" in registry.upper():
        flags.add("planetary_embargo")

    # A sponsor letter naming a different sponsor than the intake form.
    form_sponsor = next((o.value for o in ev.values("sponsor_id")
                         if o.source not in (ADJUDICATOR, MANUAL_CORRECTION)), None)
    if ev.sponsor_letter_sponsor and form_sponsor \
            and ev.sponsor_letter_sponsor != form_sponsor:
        flags.add("sponsor_mismatch")

    # A sponsor letter naming a different applicant is sponsor evidence, not an
    # identity-document conflict.
    if ev.sponsor_letter_name:
        letter_name, letter_conf = lexicon.snap_name(ev.sponsor_letter_name)
        identity = set()
        for obs in ev.values("applicant_name"):
            if obs.source in _NOT_IDENTITY_SOURCES:
                continue
            snapped, conf = lexicon.snap_name(obs.value)
            if conf > 0.0:
                identity.add(snapped.casefold())
        if letter_conf > 0.0 and identity and letter_name.casefold() not in identity:
            flags.add("sponsor_mismatch")

    # Identity conflict requires disagreement between crisp identity sources.
    # OCR scans, sponsor letters, and corrections are excluded from this test.
    if "applicant_name" not in ev.corrections:
        crisp = set()
        for obs in ev.values("applicant_name"):
            if obs.source in _NOT_IDENTITY_SOURCES:
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


def _fee_value(raw: str | None, lexicon: Lexicon) -> str | None:
    """The fee status this reading states, or None if it states none."""
    if not raw:
        return None
    candidate = raw.strip().lower()
    if candidate not in FEE_VALUES:
        # Snap onto the closed vocabulary before giving up: an OCR'd
        # "paig"/"waivec" is a perfectly recoverable "paid"/"waived".
        snapped, conf = lexicon.snap("fee_status", candidate)
        candidate = snapped.lower() if conf > 0.0 else candidate
    return candidate if candidate in FEE_VALUES else None


def _snap(field: str, value: str | None, lexicon: Lexicon) -> str | None:
    """Normalise a raw reading onto its vocabulary, or reject it as debris."""
    if value is None:
        return None
    if field in SNAP_FIELDS:
        snapped, conf = lexicon.snap(field, value)
        # Nothing in the closed set is close, so what we read is OCR debris
        # rather than a value. Printing it is a certain miss where the prior
        # mode has the base rate, and -- more importantly -- feeding it to the
        # policy engine as though it were a known visa class silently corrupts
        # the decision path. Treat it as unread.
        return snapped if conf > 0.0 else None
    if field == "applicant_name":
        snapped, conf = lexicon.snap_name(value)
        return snapped if conf > 0.0 else value
    return value


# Which fallback-engine signals may take effect. Each entry was decided by
# repeated out-of-fold measurement on the full 150-point objective, not by
# whether the reading looked plausible.
#
#   fields  -- fallback field values may settle the policy Record, not just
#              the printed output. Printing is unconditional: a guess costs
#              nothing that an empty field does not.
#   flags   -- fallback risk flags join the flag set.
#   note    -- a fallback-read adjudicator finding may settle the case.
#   panel   -- a fallback read counts as having read the risk panel, which is
#              what turns "no flags found" into evidence of no flags.
PROMOTE: frozenset[str] = frozenset()


def extract_packet(pdf_path: Path, lexicon: Lexicon,
                   promote: frozenset[str] = PROMOTE) -> "Extraction":
    """Phase 1: read a packet into printable fields plus a policy Record."""
    return assemble(parse_packet(pdf_path), lexicon, promote)


def assemble(ev: PacketEvidence, lexicon: Lexicon,
             promote: frozenset[str] = PROMOTE) -> "Extraction":
    """Turn an evidence set into printable fields plus a policy Record.

    Deliberately does no adjudicating. The staleness rule needs a packet
    *receipt* date, and the forensics pass established that packets carry only
    one date (the arrival date) -- there is no receipt date to read. Rather than
    hardcode a constant tuned to the public corpus, the reference date is
    derived from the corpus being scored (see `corpus_reference_date`), which is
    what lets the staleness rule survive a private test set from another era.

    Split out from `extract_packet` so the same evidence can be assembled under
    different trust ceilings without re-reading the PDF, which is what makes
    evaluating the fallback engine's promotion options affordable.
    """
    resolved: dict[str, str] = {}
    printed: dict[str, str] = {}
    policy_trust_max = (TRUST_ORDER[SCANNED_FALLBACK] if "fields" in promote
                        else POLICY_TRUST_MAX)

    for field in ("applicant_name", *SNAP_FIELDS, "sponsor_id", "arrival_date"):
        # Printed output resolves over every source; the policy record resolves
        # only over sources allowed to drive an adjudication.
        shown = _snap(field, _resolve(ev, field, lexicon), lexicon)
        trusted = _snap(field, _resolve(ev, field, lexicon, policy_trust_max),
                        lexicon)

        if trusted is not None:
            resolved[field] = trusted
        if shown is not None:
            printed[field] = shown
            continue

        # No trusted evidence: print a prior guess but keep it UNKNOWN for policy.
        if field == "applicant_name":
            printed[field] = lexicon.data["applicant_name"]["prior_mode"]
        elif field == "sponsor_id":
            printed[field] = FALLBACK_SPONSOR_ID
        elif field == "arrival_date":
            printed[field] = FALLBACK_ARRIVAL_DATE
        else:
            printed[field] = lexicon.prior_mode(field)

    # Did trusted evidence *state* a fee status? `unknown` is a value a receipt
    # prints, and it is also the sentinel for having read nothing, so the two
    # are indistinguishable downstream unless the distinction is captured here.
    fee = _fee_value(_resolve(ev, "fee_status", lexicon, policy_trust_max),
                     lexicon)
    fee_observed = fee is not None
    shown_fee = _fee_value(_resolve(ev, "fee_status", lexicon), lexicon)

    # Print what the document said, including a stated "unknown" -- guessing the
    # prior mode there overwrites a correct value with `paid`. The fallback is
    # only for fields nothing trustworthy stated.
    printed["fee_status"] = shown_fee or lexicon.prior_mode("fee_status")
    fee = fee or UNKNOWN

    # Flags are always *printed* from both engines -- an extra flag that the
    # second engine read is worth the same as one the first did, and there is no
    # trust order to violate. Whether they reach the policy record is separate.
    shown_flags = _derive_risk_flags(ev, lexicon, with_fallback=True)
    flags = (shown_flags if "flags" in promote
             else _derive_risk_flags(ev, lexicon))
    printed["risk_flags"] = ("|".join(sorted(shown_flags)) if shown_flags
                             else "none")

    note_finding = ev.note_finding
    panel_read = ev.risk_panel_read
    panel_missing = ev.risk_panel_missing
    if "note" in promote:
        note_finding = note_finding or ev.fallback_note_finding
    if "panel" in promote:
        panel_read = panel_read or ev.fallback_risk_panel_read
        panel_missing = panel_missing or ev.fallback_risk_panel_missing

    # "No flags found" is only meaningful if we actually read the page that
    # carries them: the biometric slip or an adjudicator note that states the
    # governing flag.
    #
    # A registry extract covers embargo status only and does not establish that
    # the complete risk panel was read.
    flags_known = (bool(flags)
                   or BIOMETRIC in ev.page_types
                   or panel_read
                   or note_finding is not None) and not panel_missing

    waiver = ev.waiver_code
    if waiver is None and "fields" in promote:
        waiver = ev.fallback_waiver_code
    waiver = (waiver or "").upper()
    record = Record(
        case_id=ev.case_id,
        visa_class=resolved.get("visa_class", UNKNOWN),
        sponsor_id=resolved.get("sponsor_id", UNKNOWN),
        fee_status=fee,
        fee_explicit_unknown=fee_observed and fee == UNKNOWN,
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

    note = note_finding if note_finding in (APPROVED, DENIED, NEEDS_REVIEW) else None
    printed["_injection"] = "1" if ev.injection_detected else ""
    printed["_damaged"] = ",".join(sorted(ev.damaged_fields))
    feats = packet_features(ev, record, promote)
    return Extraction(printed=printed, record=record, note=note, features=feats)


def resolve_printed_date(printed: dict[str, str], record: Record,
                         median_date: str | None,
                         years: dict[str, int] | None = None) -> None:
    """Settle the *printed* arrival date using corpus-level context.

    Two output-only corrections are applied without changing the policy record:

      * when no date is recovered, print the corpus median;
      * when the year is an OCR error, print the repaired
        year (`repair_year`).

    Neither value is promoted to policy evidence.

    The CLI, cached scorer, and out-of-fold writer all call this function.
    """
    if record.arrival_date == UNKNOWN:
        if median_date:
            printed["arrival_date"] = median_date
        return
    mended = repair_year(record.arrival_date, years or {})
    if mended:
        printed["arrival_date"] = mended


def finalize(printed: dict[str, str], record: Record, note: str | None,
             calibration: Calibration, adjudicator=None,
             features: dict[str, float] | None = None) -> Prediction:
    """Phase 2: adjudicate an already-extracted record. Microseconds, no I/O.

    ``adjudicator`` is optional and runs only when deterministic evidence does
    not settle the case.
    """
    if note is not None:
        adjudication = note
        # Note confidence is the fitted accuracy of explicit note findings.
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
