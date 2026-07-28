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
        assert parse_fields("SAMPLE DENIAL")[2].get("note_finding") is None

    def test_a_watermark_does_not_veto_a_real_labelled_finding(self):
        # This assertion used to be the opposite, under a page-wide
        # `"SAMPLE" not in text` guard. Measured on the corpus: pages carrying
        # *both* a SAMPLE watermark and a `Finding:` label agree with the truth
        # 3/3, with no disagreements -- genuine adjudicator notes routinely also
        # carry the harmless watermark, so the page-wide guard was discarding
        # real findings to protect against a watermark somewhere else entirely.
        # The label is the guard: `SAMPLE` itself fails `_is_finding_label`.
        assert parse_fields("SAMPLE DENIAL\nFinding DENIED")[2].get(
            "note_finding") == "DENIED"

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


class TestReasonScoping:
    """A rationale that is a whole sentence identifies its own page."""

    def test_intact_rationale_survives_destroyed_scope_cues(self):
        # MIB-000333, verbatim: the rationale is clean while `Adjudicator`,
        # `Manual` and `Reason` are each damaged past recognition.
        text = ("Manu... . _judicator Note Feria g2280NFn_ | Reascr, "
                "Clean or exception-qualifled packet.")
        assert parse_fields(text)[2].get("note_finding") == "APPROVED"

    def test_debris_inside_the_phrase(self):
        # MIB-000357: OCR wedged a pipe and a space into "exception".
        text = "| Manual Adjudicator Note nding: APPRO' eason: Clean or exce| tion-qualified packet."
        assert parse_fields(text)[2].get("note_finding") == "APPROVED"

    def test_rationale_with_no_note_furniture_at_all(self):
        # MIB-000748: no title, no `Reason:` label, just the rationale.
        text = "DENED Denial supported by damaged MD elton ot yy salle policy notes,"
        assert parse_fields(text)[2].get("note_finding") == "DENIED"

    def test_scoped_rationales_still_need_a_note_page(self):
        # "Disqualifying risk flag" is how the *biometric panel* labels a flag,
        # so on its own it must not be read as an adjudicator finding.
        panel = "FORM B-13 Biometric Scan Slip Observed flags: disqualifying biohazard_red"
        assert parse_fields(panel)[2].get("note_finding") is None
        # ...but it decides the case on a page that is a note.
        note = "Manual Adjudicator Note Reason: Disqualifying risk flag: biohazard_red."
        assert parse_fields(note)[2].get("note_finding") == "DENIED"

    def test_watermark_still_blocks_the_reason_path(self):
        text = "SAMPLE DENIAL Denial supported by surviving visible evidence"
        assert parse_fields(text)[2].get("note_finding") is None

    def test_two_rationales_are_rejected(self):
        text = ("Manual Adjudicator Note Reason: Clean or exception-qualified "
                "packet. Denial supported by damaged policy notes.")
        assert parse_fields(text)[2].get("note_finding") is None


class TestAbbreviatedLabels:
    """Sponsor letters abbreviate the labels the intake form spells out."""

    def test_short_labels_parse(self):
        fields = parse_fields("Purpose: cultural exchange")[0]
        assert fields.get("declared_purpose") == "cultural exchange"
        assert parse_fields("World: Wolf-1061c")[0].get("home_world") == "Wolf-1061c"
        assert parse_fields("Species: TRIANGULAN")[0].get("species_code") == "TRIANGULAN"

    def test_long_labels_still_parse(self):
        fields = parse_fields("Declared Purpose: reactor maintenance")[0]
        assert fields.get("declared_purpose") == "reactor maintenance"
        assert parse_fields("Home World: Luyten-b")[0].get("home_world") == "Luyten-b"

    def test_free_text_fields_did_not_get_short_labels(self):
        # Deliberately NOT aliased: nothing rejects a plausible-looking string
        # for these, so a short label breaks far more than it fixes.
        for line, field in (("Sponsor: SPN-1234", "sponsor_id"),
                            ("Fee: paid", "fee_status"),
                            ("Arrival: 2026-05-01", "arrival_date")):
            assert parse_fields(line)[0].get(field) is None, line


class TestReasonFacts:
    """The reason clause states facts, not only a verdict."""

    def test_fee_status_from_reason(self):
        extras = parse_fields("Adjudicator Note Finding: DENIED "
                              "Reason: Mandatory fee unpaid")[2]
        assert extras.get("reason_fee_status") == "unpaid"
        assert parse_fields("Reason: Fee status unknown.")[2] \
            .get("reason_fee_status") == "unknown"

    def test_home_world_from_reason(self):
        # The finding for this rationale is only 67% pure (6 DENIED / 3
        # NEEDS_REVIEW) so it is NOT read as a verdict -- but the world it names
        # is a fact, and that is safe to read.
        extras = parse_fields("Manual Agjudicator Note | | "
                              "Reason: Embargo home world: Wolf-1061c.")[2]
        assert extras.get("reason_home_world") == "Wolf-1061c"
        assert extras.get("note_finding") is None

    def test_watermark_blocks_reason_facts_too(self):
        extras = parse_fields("SAMPLE DENIAL Reason: Mandatory fee unpaid")[2]
        assert extras.get("reason_fee_status") is None

    def test_facts_survive_a_note_that_already_has_a_finding(self):
        # The fact extraction must not be gated on the finding being missing.
        extras = parse_fields("Finding: DENIED. Reason: Mandatory fee unpaid.")[2]
        assert extras.get("note_finding") == "DENIED"
        assert extras.get("reason_fee_status") == "unpaid"


class TestFeeFromReceipt:
    """A receipt states the fee three ways; two of them survive the third."""

    def test_positive_amount_no_waiver_is_paid(self):
        extras = parse_fields("MIB Fee Receipt Amount $809.00 Waiver Code N/A")[2]
        assert extras.get("receipt_fee_status") == "paid"

    def test_zero_amount_with_waiver_is_waived(self):
        extras = parse_fields("Fee Receipt Amount $0.00 Waiver Code DIP-WAIVER")[2]
        assert extras.get("receipt_fee_status") == "waived"

    def test_zero_amount_no_waiver_is_ambiguous(self):
        # 12 unpaid vs 10 unknown on the training corpus -- genuinely undecidable,
        # so it must yield nothing rather than guess the majority.
        extras = parse_fields("Fee Receipt Amount $0.00 Waiver Code N/A")[2]
        assert extras.get("receipt_fee_status") is None

    def test_needs_both_facts(self):
        assert parse_fields("Amount $809.00")[2].get("receipt_fee_status") is None
        assert parse_fields("Waiver Code N/A")[2].get("receipt_fee_status") is None
