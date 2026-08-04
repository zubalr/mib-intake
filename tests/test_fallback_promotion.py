"""Trust-boundary tests for the independent OCR fallback."""

from mib import pipeline, policy
from mib.cli import resolve_fallback_sponsors
from mib.extract import (
    POLICY_TRUST_MAX,
    REGISTRY,
    SCANNED,
    SCANNED_FALLBACK,
    TRUST_ORDER,
    Observation,
    PacketEvidence,
    _record,
)
from mib.features import packet_features
from mib.lexicon import Lexicon


def _evidence(*observations: Observation) -> PacketEvidence:
    ev = PacketEvidence(case_id="MIB-TEST")
    ev.observations.extend(observations)
    return ev


def _fallback(field: str, value: str) -> Observation:
    return Observation(
        field=field,
        value=value,
        source=SCANNED_FALLBACK,
        page=0,
        trust=TRUST_ORDER[SCANNED_FALLBACK],
    )


def _primary(field: str, value: str, page: int = 0) -> Observation:
    return Observation(
        field=field,
        value=value,
        source=SCANNED,
        page=page,
        trust=TRUST_ORDER[SCANNED],
    )


def test_invalid_fallback_read_does_not_invent_document_damage():
    ev = _evidence()
    _record(
        ev,
        "arrival_date",
        "2926-05-03 debris",
        SCANNED_FALLBACK,
        TRUST_ORDER[SCANNED_FALLBACK],
        0,
    )
    assert "arrival_date" not in ev.damaged_fields

    _record(
        ev,
        "arrival_date",
        "2926-05-03 debris",
        SCANNED,
        TRUST_ORDER[SCANNED],
        0,
    )
    assert "arrival_date" in ev.damaged_fields


def test_fallback_is_inert_when_no_permission_is_enabled():
    lexicon = Lexicon()
    record = policy.Record(case_id="MIB-TEST")
    primary = _evidence()
    with_fallback = _evidence(_fallback("visa_class", "XW-2"))

    assert packet_features(primary, record, frozenset()) == packet_features(
        with_fallback, record, frozenset()
    )
    assert TRUST_ORDER[SCANNED] == POLICY_TRUST_MAX


def test_printed_permission_does_not_promote_general_policy_fields():
    lexicon = Lexicon()
    ex = pipeline.assemble(
        _evidence(_fallback("visa_class", "XW-2")),
        lexicon,
        frozenset({"printed"}),
    )

    assert ex.printed["visa_class"] == "XW-2"
    assert ex.record.visa_class == policy.UNKNOWN


def test_fee_permission_is_narrow():
    lexicon = Lexicon()
    ev = _evidence(
        _fallback("fee_status", "unpaid"),
        _fallback("visa_class", "XW-2"),
    )
    ex = pipeline.assemble(ev, lexicon, frozenset({"printed", "fee"}))

    assert ex.record.fee_status == "unpaid"
    assert ex.record.visa_class == policy.UNKNOWN


def test_output_only_fallback_date_survives_corpus_finalization():
    lexicon = Lexicon()
    ex = pipeline.assemble(
        _evidence(_fallback("arrival_date", "2026-03-15")),
        lexicon,
        frozenset({"printed"}),
    )

    assert ex.record.arrival_date == policy.UNKNOWN
    assert ex.printed["arrival_date"] == "2026-03-15"
    pipeline.resolve_printed_date(
        ex.printed,
        ex.record,
        median_date="2026-06-01",
        years={},
    )
    assert ex.printed["arrival_date"] == "2026-03-15"


def test_invalid_primary_closed_value_falls_through_for_output_only():
    lexicon = Lexicon()
    ex = pipeline.assemble(
        _evidence(
            _primary("home_world", "not a world"),
            _fallback("home_world", "Titan Freeport"),
        ),
        lexicon,
        frozenset({"printed"}),
    )

    assert ex.printed["home_world"] == "Titan Freeport"
    assert ex.record.visa_class == policy.UNKNOWN


def test_closed_value_majority_wins_within_one_trust_tier():
    lexicon = Lexicon()
    ex = pipeline.assemble(
        _evidence(
            _primary("visa_class", "XW-2", page=0),
            _primary("visa_class", "DIP-1", page=1),
            _primary("visa_class", "DIP-1", page=2),
        ),
        lexicon,
        frozenset({"printed"}),
    )

    assert ex.printed["visa_class"] == "DIP-1"


def test_manual_sponsor_correction_does_not_invent_mismatch():
    lexicon = Lexicon()
    ev = _evidence(
        Observation(
            field="sponsor_id",
            value="SPN-9999",
            source="manual_correction",
            page=0,
            trust=TRUST_ORDER["manual_correction"],
        ),
        Observation(
            field="sponsor_id",
            value="SPN-1111",
            source="intake_form",
            page=0,
            trust=TRUST_ORDER["intake_form"],
        ),
    )
    ev.corrections["sponsor_id"] = "SPN-9999"
    ev.sponsor_letter_sponsor = "SPN-9999"

    ex = pipeline.assemble(ev, lexicon, frozenset())

    assert "sponsor_mismatch" not in ex.record.risk_flags


def test_manual_applicant_correction_does_not_invent_sponsor_mismatch():
    lexicon = Lexicon()
    ev = _evidence(
        Observation(
            field="applicant_name",
            value="Arikesh Solzarn",
            source="manual_correction",
            page=0,
            trust=TRUST_ORDER["manual_correction"],
        ),
        Observation(
            field="applicant_name",
            value="Xannax Qornax",
            source="intake_form",
            page=0,
            trust=TRUST_ORDER["intake_form"],
        ),
    )
    ev.corrections["applicant_name"] = "Arikesh Solzarn"
    ev.sponsor_letter_name = "Arikesh Solzarn"

    ex = pipeline.assemble(ev, lexicon, frozenset())

    assert "sponsor_mismatch" not in ex.record.risk_flags


def test_registry_embargo_is_world_specific_in_printed_output_only():
    """The fitted world list may cost a transcription, never a denial.

    `EMBARGOED_WORLDS` comes from the public labels rather than from the field
    manual, so it shapes what gets printed and not what gets adjudicated: the
    `Record` keeps `planetary_embargo` whatever the world turns out to be.
    """
    lexicon = Lexicon()
    embargoed = _evidence(_primary("home_world", "TRAPPIST-1e"))
    embargoed.registry_status = "EMBARGO REVIEW"
    review_only = _evidence(_primary("home_world", "Wolf-1061c"))
    review_only.registry_status = "EMBARGO REVIEW"

    embargoed_ex = pipeline.assemble(embargoed, lexicon, frozenset())
    review_only_ex = pipeline.assemble(review_only, lexicon, frozenset())

    assert "planetary_embargo" in embargoed_ex.printed["risk_flags"]
    assert "planetary_embargo" not in review_only_ex.printed["risk_flags"]

    assert "planetary_embargo" in embargoed_ex.record.risk_flags
    assert "planetary_embargo" in review_only_ex.record.risk_flags


def test_unreadable_world_keeps_the_stated_embargo():
    """No world evidence is not evidence that the embargo does not apply."""
    lexicon = Lexicon()
    ev = _evidence()
    ev.registry_status = "EMBARGO REVIEW"

    ex = pipeline.assemble(ev, lexicon, frozenset())

    assert "planetary_embargo" in ex.printed["risk_flags"]
    assert "planetary_embargo" in ex.record.risk_flags


def test_production_permissions_match_the_measured_pair():
    assert pipeline.PROMOTE == frozenset({"printed", "fee"})


def test_unique_registry_name_repairs_printed_identity_only():
    lexicon = Lexicon()
    ev = _evidence(
        Observation(
            field="applicant_name",
            value="Wrong Intake",
            source="intake_form",
            page=0,
            trust=TRUST_ORDER["intake_form"],
        ),
        Observation(
            field="applicant_name",
            value="Arivoss Orimora",
            source=REGISTRY,
            page=1,
            trust=TRUST_ORDER[REGISTRY],
        ),
    )

    ex = pipeline.assemble(ev, lexicon, frozenset())

    assert ex.printed["applicant_name"] == "Arivoss Orimora"


def test_manual_name_correction_still_outranks_registry_name():
    lexicon = Lexicon()
    ev = _evidence(
        Observation(
            field="applicant_name",
            value="Lutari Veemora",
            source="manual_correction",
            page=0,
            trust=TRUST_ORDER["manual_correction"],
        ),
        Observation(
            field="applicant_name",
            value="Arivoss Orimora",
            source=REGISTRY,
            page=1,
            trust=TRUST_ORDER[REGISTRY],
        ),
    )
    ev.corrections["applicant_name"] = "Lutari Veemora"

    ex = pipeline.assemble(ev, lexicon, frozenset())

    assert ex.printed["applicant_name"] == "Lutari Veemora"


def test_fallback_direct_finding_overrides_only_at_finalization():
    lexicon = Lexicon()
    ev = _evidence()
    ev.fallback_note_finding = "APPROVED"

    ex = pipeline.assemble(ev, lexicon, frozenset())
    prediction = pipeline.finalize(
        ex.printed,
        ex.record,
        ex.note,
        policy.Calibration(),
    )

    assert ex.note is None
    assert prediction.adjudication == "APPROVED"
    assert prediction.confidence == 0.95


def test_injected_packet_cannot_promote_a_fallback_finding():
    ev = _evidence()
    ev.injection_detected = True
    ev.fallback_note_finding = "APPROVED"

    ex = pipeline.assemble(ev, Lexicon(), frozenset())

    assert "_fallback_note" not in ex.printed


def test_output_only_sponsor_placeholder_uses_corpus_mode():
    rows = [
        {"sponsor_id": "SPN-0000"},
        {"sponsor_id": "SPN-4040"},
        {"sponsor_id": "SPN-4040"},
        {"sponsor_id": "SPN-7331"},
    ]

    mode, replaced = resolve_fallback_sponsors(rows)

    assert mode == "SPN-4040"
    assert replaced == 1
    assert rows[0]["sponsor_id"] == "SPN-4040"
