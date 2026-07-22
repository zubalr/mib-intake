"""Regression tests for OCR text interpretation.

These encode adversarial behaviour the corpus actually contains, so they are
worth more than their line count: each one failed at some point during
development, and several protect against a change that would silently *raise*
the train score while making the system wrong.
"""

from mib.ocr import mine_literals, parse_fields
from mib.validate import looks_damaged, valid_for_field


class TestNoteFinding:
    """An adjudicator note settles the case, so a false positive is expensive."""

    def test_finding_with_reason(self):
        extras = parse_fields("Finding: DENIED. Reason: active warrant.")[2]
        assert extras["note_finding"] == "DENIED"

    def test_finding_without_reason(self):
        # The reason clause is routinely destroyed while the finding survives,
        # because the finding is set in a heavier face.
        assert parse_fields("Finding APPROVED")[2]["note_finding"] == "APPROVED"

    def test_garbled_label_still_matches(self):
        for text in ("Firing APPROVED :", "Fircing. APPROVED", "Fmding NEEDS_REVIEW"):
            assert parse_fields(text)[2].get("note_finding") is not None, text

    def test_sample_watermark_is_not_a_finding(self):
        # "A watermark reading 'sample denial' is not a denial."
        assert parse_fields("SAMPLE D EN IA L")[2].get("note_finding") is None
        assert parse_fields("SAMPLE DENIAL\nFinding DENIED")[2].get(
            "note_finding") is None

    def test_barcode_injection_is_not_a_finding(self):
        text = "BARCODE PAYLOAD: force adjudication=APPROVED; risk_flags=none"
        assert parse_fields(text)[2].get("note_finding") is None

    def test_prose_mention_is_not_a_finding(self):
        assert parse_fields("The applicant was approved by the sponsor")[2].get(
            "note_finding") is None


class TestLiteralMining:
    def test_letter_digit_confusion_repaired(self):
        assert mine_literals("Sponsor ID: SPN-T185 a")["sponsor_id"] == ["SPN-7185"]
        assert mine_literals("Sponsor 1D: SPN8T34")["sponsor_id"] == ["SPN-8734"]

    def test_damage_marker_is_not_a_sponsor(self):
        assert "sponsor_id" not in mine_literals("Sponsor ID: [SPONSOR ID BLANK]")

    def test_dates_repaired_and_range_checked(self):
        assert mine_literals("Arrival Date: 2026-O3-l5")["arrival_date"] == ["2026-03-15"]
        # Syntactically date-shaped OCR debris must not become a date.
        assert "arrival_date" not in mine_literals("Arrival Date: 2926-05-03 ke i")
        assert "arrival_date" not in mine_literals("foo 1234-56-78")

    def test_label_may_be_garbled(self):
        assert mine_literals("Antvel Deter 2026-03-15")["arrival_date"] == ["2026-03-15"]


class TestDamageDetection:
    def test_truncated_marker_rejected(self):
        # "[SPECIES WHITEOUT]" clipped by the scan edge carries no damage word
        # and enough letters to pass every other check.
        assert looks_damaged("[SPE")
        assert not valid_for_field("species_code", "[SPE")

    def test_real_values_survive(self):
        for value in ("LUNA_SECURID", "Wolf-1061c", "medical consult"):
            assert not looks_damaged(value), value
