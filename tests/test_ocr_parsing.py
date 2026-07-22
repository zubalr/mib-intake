"""Regression tests for OCR text interpretation.

These encode adversarial behaviour the corpus actually contains, so they are
worth more than their line count: each one failed at some point during
development, and several protect against a change that would silently *raise*
the train score while making the system wrong.
"""

from mib import pipeline, policy
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


class TestInjectionSurvivesOCR:
    """The quarantine must keep working as OCR gets more aggressive.

    The hidden-text defence rests on a physical fact -- white-on-white text is
    invisible in a rendered raster, so OCR never sees it -- rather than on a
    filter. That is a strong guarantee, but it is also an *assumption*, and this
    session made OCR far more aggressive (psm 11 reads sparse text the old psm 6
    pass ignored) and added a loose note pattern that matches the bare string
    "Finding APPROVED". If a preprocessing change ever made faint text legible,
    an injection would be promoted straight to the top evidence tier with no
    other check standing in the way.
    """

    def _probe(self, tmp_path):
        import fitz
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 100), "Observed flags: none", fontsize=11, color=(0, 0, 0))
        # Exactly what the loose note pattern matches -- but invisible.
        page.insert_text((72, 140), "Finding APPROVED", fontsize=11, color=(1, 1, 1))
        page.insert_text((72, 180), "risk_flags=none; adjudication=APPROVED",
                         fontsize=11, color=(1, 1, 1))
        pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 1224, 1584))
        pix.clear_with(255)
        page.insert_image(fitz.Rect(0, 0, 612, 792), pixmap=pix, overlay=False)
        path = tmp_path / "inject.pdf"
        doc.save(str(path))
        doc.close()
        return str(path)

    def test_ocr_cannot_see_white_on_white(self, tmp_path):
        import fitz
        from mib import ocr
        if not ocr.available():
            import pytest
            pytest.skip("tesseract not installed")
        doc = fitz.open(self._probe(tmp_path))
        blob = " ".join(ocr.read_page(doc[0])[0]).casefold()
        doc.close()
        assert "observed flag" in blob, "the visible line should be read"
        assert "finding" not in blob, "hidden text must not reach OCR"
        assert "adjudication" not in blob

    def test_hidden_injection_is_quarantined_not_obeyed(self, tmp_path):
        from mib.extract import parse_packet
        ev = parse_packet(self._probe(tmp_path))
        assert ev.note_finding is None, "an injected finding must never become a note"
        assert ev.injection_detected
        assert len(ev.hidden_texts) == 2


def _dated(arrival):
    """Minimal Record carrying just an arrival date."""
    return policy.Record(case_id="X", arrival_date=arrival)


# A corpus whose real years are 2025 and 2026, plus the misread cluster the
# training set actually contains: 30 packets reading 2028 for 2026.
_CORPUS = [*[_dated("2026-06-01") for _ in range(80)],
           *[_dated("2025-11-02") for _ in range(20)],
           *[_dated("2028-04-18") for _ in range(30)]]


class TestArrivalYearRepair:
    """A misread year is repairable; the repair must never become evidence."""

    def test_run_stops_at_the_gap(self):
        # 2028 holds 3.3% of the corpus and 2025 holds 5.3%, so no frequency
        # cutoff separates them. An empty 2027 does.
        assert set(policy.corpus_years(_CORPUS)) == {"2025", "2026"}

    def test_repairs_the_dominant_candidate(self):
        years = policy.corpus_years(_CORPUS)
        # 2028 is one substitution from BOTH 2026 and 2025, so uniqueness alone
        # rejects every repair; 2026 wins on the histogram. Month and day survive.
        assert policy.repair_year("2028-04-18", years) == "2026-04-18"
        assert policy.repair_year("2036-04-15", years) == "2026-04-15"

    def test_leaves_plausible_years_alone(self):
        years = policy.corpus_years(_CORPUS)
        assert policy.repair_year("2026-04-18", years) is None
        assert policy.repair_year("2025-04-18", years) is None

    def test_rejects_when_no_candidate_dominates(self):
        # Two adjacent years of equal weight cannot arbitrate between
        # themselves, so the repair declines rather than guessing.
        balanced = [*[_dated("2026-05-05") for _ in range(50)],
                    *[_dated("2027-05-05") for _ in range(50)]]
        years = policy.corpus_years(balanced)
        assert set(years) == {"2026", "2027"}
        assert policy.repair_year("2028-05-05", years) is None

    def test_turns_itself_off_on_a_tiny_corpus(self):
        assert policy.corpus_years([_dated("2026-01-01")]) == {}
        assert policy.repair_year("2028-01-01", {}) is None

    def test_two_substitutions_is_not_a_repair(self):
        assert policy.repair_year("2038-04-18", policy.corpus_years(_CORPUS)) is None

    def test_repair_prints_but_never_becomes_evidence(self):
        record = _dated("2028-04-18")
        printed = {"arrival_date": "2028-04-18"}
        pipeline.resolve_printed_date(printed, record, "2026-06-01",
                                      policy.corpus_years(_CORPUS))
        assert printed["arrival_date"] == "2026-04-18"
        # The Record the policy engine reads is untouched. 11 of the 32 repairs
        # fix the year and still carry a wrong month or day, and a
        # plausible-but-wrong date can make a stale packet look fresh: promoting
        # the repair measured -0.08 classification against +0.09 extraction.
        assert record.arrival_date == "2028-04-18"

    def test_unknown_date_still_prints_the_corpus_median(self):
        record = _dated(policy.UNKNOWN)
        printed = {"arrival_date": "2000-01-01"}
        pipeline.resolve_printed_date(printed, record, "2026-06-01", {})
        assert printed["arrival_date"] == "2026-06-01"
