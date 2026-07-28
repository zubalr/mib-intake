"""Tests for closed-vocabulary matching."""

import pytest

from mib.lexicon import Lexicon, weighted_distance


@pytest.fixture(scope="module")
def lx():
    return Lexicon()


@pytest.mark.parametrize("observed,expected", [
    ("VENUSIAN_MYCELIAL", "VENUSIAN_MYCELIAL"),   # clean
    ("V3NUSIAN_MYCEL1AL", "VENUSIAN_MYCELIAL"),   # digit/letter confusion
    ("0RION GRAYS", "ORION_GRAYS"),               # O/0 plus separator noise
    ("CENTAURl_SYNTH", "CENTAURI_SYNTH"),         # l/I confusion
    ("centauri synth", "CENTAURI_SYNTH"),         # case and separator folding
])
def test_species_snapping(lx, observed, expected):
    value, conf = lx.snap("species_code", observed)
    assert value == expected
    assert conf > 0.0


@pytest.mark.parametrize("observed,expected", [
    ("XW 2", "XW-2"),
    ("MED3", "MED-3"),
    ("TRANSlT-7", "TRANSIT-7"),
    ("dip 1", "DIP-1"),
])
def test_visa_snapping(lx, observed, expected):
    value, conf = lx.snap("visa_class", observed)
    assert value == expected
    assert conf > 0.0


def test_multichar_confusion_rn_m(lx):
    """`rn` read as `m` is the classic OCR failure and costs 2 under plain
    Levenshtein, which would push a correct match past any sane threshold."""
    assert weighted_distance("luzam", "luzarn") < weighted_distance("luzam", "lunax")
    value, conf = lx.snap_name("1xodane Luzam")
    assert value == "Ixodane Luzarn"
    assert conf > 0.0


def test_unknown_value_is_not_force_matched(lx):
    """A value with no plausible lexicon entry must survive unchanged with zero
    confidence. Forcing it onto the nearest entry converts a possible hit into a
    guaranteed miss -- and the private test set may hold unseen values."""
    value, conf = lx.snap("species_code", "ZZZ_TOTALLY_UNKNOWN")
    assert value == "ZZZ_TOTALLY_UNKNOWN"
    assert conf == 0.0


def test_out_of_vocab_name_token_is_preserved(lx):
    """One unmatched token must not be rewritten just because the other matched."""
    value, conf = lx.snap_name("Qqqqq Wwwww")
    assert value == "Qqqqq Wwwww"
    assert conf == 0.0


def test_exact_match_is_full_confidence(lx):
    assert lx.snap("home_world", "Europa Station") == ("Europa Station", 1.0)
    assert lx.snap_name("Ixodane Luzarn")[1] == 1.0


def test_closed_vocabularies_have_expected_size(lx):
    """Guards against a malformed lexicon rebuild silently widening a field."""
    assert len(lx.values("species_code")) == 12
    assert len(lx.values("home_world")) == 13
    assert len(lx.values("visa_class")) == 5
    assert len(lx.values("declared_purpose")) == 10
    assert lx.disqualifying_flags == {
        "memory_tampering", "planetary_embargo", "active_warrant", "biohazard_red",
    }
