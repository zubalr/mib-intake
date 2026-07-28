"""Document features for the learned adjudicator.

The hand-built decision paths are a 16-bucket histogram: every packet landing on
`risk_page_unreadable` gets the same probability vector, even though the packets
inside that bucket differ in ways that predict the outcome (how many fields we
recovered, whether the biometric confidence was low, how damaged the packet is,
how close to stale the arrival date is). Roughly 208 decidable cases were being
hedged to NEEDS_REVIEW because a bucket average sat near the decision boundary.

These features let a small model separate within a bucket. Design constraints:

  * **Nothing packet-identifying.** No case_id, no filename, no page hashes.
    8090 hand-reviews for leaderboard-fitting, and a model that can memorise a
    packet is indefensible even if it scores well. Every feature here is a
    property of the *evidence*, and would mean the same thing on a private test
    packet drawn from a different generator run.
  * **Few and dense.** 1,000 training rows is not much; each feature must earn
    its place or it becomes a way to overfit.
  * **The decision-path label is itself a feature**, so the model starts from
    everything the hand-built policy knows and only has to improve on it.
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

    Named for the staleness reference it originally handled; it now also covers
    the corpus-derived revoked-sponsor set, which is discovered in the same
    phase and has the same failure mode if left stale.

    Features are built in phase 1, per packet, but the staleness reference date
    is a *corpus-level* statistic only known in phase 2. Without this refresh,
    `stale_margin_known` was 0 for all 1,000 training packets -- the feature was
    dead -- and `path__*` recorded a path computed with `receipt_date=None`,
    which is a different (and wrong) branch than the pipeline actually takes.
    A model trained on those features was learning from a system that does not
    exist.

    Mutates and returns `features` so callers can use it inline.
    """
    path = decision_path(record)
    for key in [k for k in features if k.startswith("path__")]:
        del features[key]
    features["path__" + path] = 1.0

    # The hand-built path prior, handed to the model as three numbers.
    #
    # The path one-hot already tells the model *which* rule fired, but not what
    # that rule believes -- the learner has to rediscover each path's outcome
    # distribution from the handful of training rows that land on it. Giving it
    # the fitted probabilities directly turns the task from "learn the prior"
    # into "correct the prior", which is both easier and the thing we actually
    # want. Measured on the 662-packet trainable subset, 5 folds x 5 repeats:
    #
    #     hgb_isotonic  69.84 -> 73.56      hgb_tiny  69.01 -> 72.94
    #     hgb_sigmoid   70.10 -> 73.30      rf        70.40 -> 71.94
    #
    # It is corpus-derived, not label-derived: `calibration.json` is refitted
    # from whatever corpus is being scored, and a path absent from the table
    # falls back to the global prior.
    probs = _prior().probs(path)
    for outcome in OUTCOMES:
        features["prior_" + outcome] = float(probs.get(outcome, 0.0))

    # Revocation is now discovered from the corpus, so this cannot be computed
    # in phase 1 -- exactly like the staleness features below.
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

    # -- What the policy engine already concluded --------------------------
    feats["path__" + decision_path(record)] = 1.0

    # -- Evidence coverage: which fields did we actually recover? ----------
    known = 0
    for field in SCORED_FIELDS:
        have = bool(ev.best(field) and ev.best(field).trusted)
        feats[f"known_{field}"] = float(have)
        known += have
    feats["known_field_count"] = float(known)
    feats["known_field_frac"] = known / len(SCORED_FIELDS)
    feats["fee_known"] = float(record.fee_status != UNKNOWN)

    # -- Corroboration: how many independent sources agree on a field? -----
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

    # -- Risk flags --------------------------------------------------------
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

    # -- Categorical fields, one-hot --------------------------------------
    for visa in VISA_CLASSES:
        feats[f"visa_{visa}"] = float(record.visa_class == visa)
    feats["visa_unknown"] = float(record.visa_class == UNKNOWN)
    for state in FEE_STATES:
        feats[f"fee_{state}"] = float(record.fee_status == state)

    # -- Sponsor -----------------------------------------------------------
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
