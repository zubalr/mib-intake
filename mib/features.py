"""Document-evidence features for the learned adjudicator.

Features describe extraction coverage, provenance, document condition, policy
state, and OCR quality. Packet identifiers, filenames, and content hashes are
excluded.
"""

from __future__ import annotations

import datetime as _dt

from mib.extract import (
    ADJUDICATOR,
    BIOMETRIC,
    FEE,
    INTAKE,
    MANUAL_CORRECTION,
    REGISTRY,
    SCANNED,
    SPONSOR,
    PacketEvidence,
)
from mib.policy import (
    DISQUALIFYING_FLAGS,
    OUTCOMES,
    REVIEW_FLAGS,
    REVOKED_SPONSORS_PUBLIC,
    UNKNOWN,
    Calibration,
    Record,
    decision_path,
)

_CALIBRATION: Calibration | None = None


def _prior() -> Calibration:
    """The fitted path table, loaded once per process."""
    global _CALIBRATION
    if _CALIBRATION is None:
        _CALIBRATION = Calibration()
    return _CALIBRATION

SCORED_FIELDS = (
    "applicant_name", "species_code", "home_world", "visa_class",
    "sponsor_id", "arrival_date", "declared_purpose",
)
PAGE_KINDS = (ADJUDICATOR, INTAKE, BIOMETRIC, SPONSOR, REGISTRY, FEE, SCANNED)
VISA_CLASSES = ("XW-1", "XW-2", "DIP-1", "MED-3", "TRANSIT-7")
FEE_STATES = ("paid", "waived", "unpaid", "unknown")

def refresh_temporal(features: dict[str, float], record: Record) -> dict[str, float]:
    """Recompute every feature that depends on corpus-level context.

    Per-packet features are created before corpus-level dates and sponsor
    frequencies are known. This function updates every dependent feature after
    that context has been computed.

    Mutates and returns `features` so callers can use it inline.
    """
    path = decision_path(record)
    for key in [k for k in features if k.startswith("path__")]:
        del features[key]
    features["path__" + path] = 1.0

    # Include the calibrated policy distribution as a compact prior.
    probs = _prior().probs(path)
    for outcome in OUTCOMES:
        features["prior_" + outcome] = float(probs.get(outcome, 0.0))

    # Sponsor-frequency context is available only after phase 1.
    features["sponsor_revoked"] = float(record.sponsor_id in REVOKED_SPONSORS_PUBLIC
                                        or record.sponsor_revoked_in_packet)

    margin, known = 0.0, 0.0
    if record.arrival_date != UNKNOWN and record.receipt_date:
        try:
            arrival = _dt.date.fromisoformat(record.arrival_date)
            receipt = _dt.date.fromisoformat(record.receipt_date)
            margin = ((receipt - arrival).days - 180) / 180.0
            known = 1.0
        except (ValueError, TypeError):
            pass
    features["stale_margin"] = margin
    features["stale_margin_known"] = known
    return features


def packet_features(ev: PacketEvidence, record: Record) -> dict[str, float]:
    """Flat numeric feature vector for one packet."""
    flags = record.flag_set()
    feats: dict[str, float] = {}

    # Policy state.
    feats["path__" + decision_path(record)] = 1.0

    # Evidence coverage.
    known = 0
    for field in SCORED_FIELDS:
        have = bool(ev.best(field) and ev.best(field).trusted)
        feats[f"known_{field}"] = float(have)
        known += have
    feats["known_field_count"] = float(known)
    feats["known_field_frac"] = known / len(SCORED_FIELDS)
    # "Did we read the fee status" is not "is it a value other than unknown".
    # A receipt that states `unknown` was read perfectly; treating those 45
    # packets as unreadable is the same sentinel collision the decision path
    # had, one layer up. They are the *most* readable packets in the bucket.
    feats["fee_known"] = float(record.fee_status != UNKNOWN
                               or record.fee_explicit_unknown)
    feats["fee_stated_unknown"] = float(record.fee_explicit_unknown)

    # Corroboration across independent sources.
    total_obs = agree = conflict = 0
    for field in SCORED_FIELDS:
        values = {o.value.casefold() for o in ev.values(field) if o.trusted}
        count = len([o for o in ev.values(field) if o.trusted])
        total_obs += count
        if count > 1:
            agree += int(len(values) == 1)
            conflict += int(len(values) > 1)
    feats["obs_total"] = float(total_obs)
    feats["fields_corroborated"] = float(agree)
    feats["fields_conflicting"] = float(conflict)

    # Risk flags.
    feats["n_flags"] = float(len(flags))
    feats["has_disqualifying"] = float(bool(flags & DISQUALIFYING_FLAGS))
    feats["n_review_flags"] = float(len(flags & REVIEW_FLAGS))
    for flag in sorted(DISQUALIFYING_FLAGS | REVIEW_FLAGS):
        feats[f"flag_{flag}"] = float(flag in flags)
    feats["risk_flags_known"] = float(record.risk_flags_known)
    feats["risk_panel_missing"] = float(ev.risk_panel_missing)
    feats["risk_panel_read"] = float(ev.risk_panel_read)
    # -1 encodes "not printed", which is distinct from a genuine low score.
    feats["biometric_confidence"] = (
        ev.biometric_confidence if ev.biometric_confidence is not None else -1.0)
    feats["biometric_confidence_known"] = float(ev.biometric_confidence is not None)

    # Categorical fields.
    for visa in VISA_CLASSES:
        feats[f"visa_{visa}"] = float(record.visa_class == visa)
    feats["visa_unknown"] = float(record.visa_class == UNKNOWN)
    for state in FEE_STATES:
        feats[f"fee_{state}"] = float(record.fee_status == state)

    # Sponsor evidence.
    feats["sponsor_known"] = float(record.sponsor_id != UNKNOWN)
    # Placeholder: recomputed by `refresh_temporal` once the corpus-derived
    # revoked set exists. Phase 1 cannot know it.
    feats["sponsor_revoked"] = float(record.sponsor_id in REVOKED_SPONSORS_PUBLIC
                                     or record.sponsor_revoked_in_packet)
    feats["sponsor_mismatch_seen"] = float(
        bool(ev.sponsor_letter_sponsor)
        and ev.sponsor_letter_sponsor != record.sponsor_id)

    # -- Dates: staleness as a continuous margin, not just a boolean -------
    margin = 0.0
    has_margin = 0.0
    if record.arrival_date != UNKNOWN and record.receipt_date:
        try:
            arrival = _dt.date.fromisoformat(record.arrival_date)
            receipt = _dt.date.fromisoformat(record.receipt_date)
            # Positive = how far past the 180-day staleness line.
            margin = ((receipt - arrival).days - 180) / 180.0
            has_margin = 1.0
        except (ValueError, TypeError):
            pass
    feats["stale_margin"] = margin
    feats["stale_margin_known"] = has_margin
    feats["arrival_untrusted"] = float(record.arrival_date_untrusted)

    # -- Waivers and notes -------------------------------------------------
    feats["has_hardship_waiver"] = float(record.has_hardship_waiver)
    feats["has_diplomatic_note"] = float(record.has_diplomatic_note)
    feats["registry_embargo"] = float(
        bool(ev.registry_status) and "EMBARGO" in (ev.registry_status or "").upper())
    feats["registry_present"] = float(ev.registry_status is not None)

    # -- Document shape ----------------------------------------------------
    for kind in PAGE_KINDS:
        feats[f"page_{kind}"] = float(kind in ev.page_types)
    feats["n_pages"] = float(len(ev.page_types))
    feats["n_scanned"] = float(ev.page_types.count(SCANNED))
    feats["scanned_frac"] = (ev.page_types.count(SCANNED) / len(ev.page_types)
                             if ev.page_types else 0.0)
    feats["ocr_pages"] = float(ev.ocr_pages)
    feats["has_text_layer"] = float(ev.has_text_layer)
    feats["n_damaged_fields"] = float(len(ev.damaged_fields))
    # *Which* field was damaged, not just how many. A torn sponsor id and a
    # torn applicant name are different situations: the first removes evidence
    # the policy engine needs, the second removes a display field. The count
    # alone cannot express that. Restricted to the three fields damaged often
    # enough to estimate -- rarer markers added noise without signal.
    for field in ("sponsor_id", "arrival_date", "applicant_name"):
        feats[f"damaged_{field}"] = float(field in ev.damaged_fields)
    feats["n_corrections"] = float(len(ev.corrections))

    # -- Adversarial content ----------------------------------------------
    # Whether an injection was *attempted* is a legitimate document property.
    # Its content is never used -- only the fact that the packet carried one.
    feats["injection_detected"] = float(ev.injection_detected)
    feats["n_hidden_spans"] = float(min(len(ev.hidden_texts), 20))
    feats["non_evidentiary_spans"] = float(min(len(ev.non_evidentiary_texts), 10))
    feats["sample_denial_watermark"] = float(ev.sample_denial_watermark)

    # -- Stamps ------------------------------------------------------------
    stamps = {s.upper() for s in ev.stamps}
    feats["stamp_approved"] = float("APPROVED" in stamps)
    feats["stamp_denied"] = float("DENIED" in stamps)
    feats["stamp_review"] = float("REVIEW" in stamps or "NEEDS_REVIEW" in stamps)
    feats["stamp_conflict"] = float(len(stamps) > 1)

    return feats
