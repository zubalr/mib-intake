"""Safety constraints on learned adjudication."""

import numpy as np

from mib.features import adjudicator_feature_allowed
from mib.model import Adjudicator
from mib.policy import APPROVED, NEEDS_REVIEW, OUTCOMES


class _ApprovingModel:
    classes_ = np.array(OUTCOMES)

    def predict_proba(self, rows):
        return np.array([[0.60, 0.25, 0.15] for _ in rows])


def test_unreadable_risk_page_cannot_be_approved():
    feature = "path__risk_page_unreadable"
    adjudicator = Adjudicator(_ApprovingModel(), [feature])

    action, confidence, path = adjudicator.adjudicate({feature: 1.0})

    assert action == NEEDS_REVIEW
    assert confidence == 0.15
    assert path == "model"


def test_same_probabilities_may_approve_a_readable_path():
    feature = "path__clean"
    adjudicator = Adjudicator(_ApprovingModel(), [feature])

    action, confidence, path = adjudicator.adjudicate({feature: 1.0})

    assert action == APPROVED
    assert confidence == 0.60
    assert path == "model"


def test_redundant_sparse_features_are_excluded_from_training():
    assert not adjudicator_feature_allowed("path__risk_page_absent")
    assert not adjudicator_feature_allowed("known_applicant_name")
    assert not adjudicator_feature_allowed("flag_active_warrant")
    assert not adjudicator_feature_allowed("n_corrections")

    assert adjudicator_feature_allowed("prior_APPROVED")
    assert adjudicator_feature_allowed("known_field_count")
    assert adjudicator_feature_allowed("n_damaged_fields")
