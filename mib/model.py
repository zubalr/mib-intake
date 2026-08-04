"""Probability model for cases not settled by deterministic policy evidence.

The model receives document-evidence features only. Case IDs, filenames, and
hidden text are excluded. It estimates class probabilities; the final action is
still selected by the expected-value rule in ``mib.policy``.
"""

from __future__ import annotations

from pathlib import Path

from mib.policy import (
    APPROVED,
    DENIED,
    NEEDS_REVIEW,
    OUTCOMES,
    PAYOFF,
    decide,
)

MODEL_PATH = Path(__file__).resolve().parent.parent / "policy" / "adjudicator.joblib"

# A model may refine an ambiguous policy bucket, but it may not turn a packet
# whose risk page exists and could not be read into an approval. Repeated
# out-of-fold evaluation improves both score and catastrophic false approvals:
# the model's approvals in these two paths are wrong more often than right.
_NO_APPROVAL_PATHS = frozenset({
    "risk_page_unreadable",
    "risk_page_unreadable_med",
})

def _expected_value(action: str, probs: dict[str, float]) -> float:
    return sum(
        PAYOFF[action][truth] * probs.get(truth, 0.0)
        for truth in OUTCOMES
    )


def decide_with_safety(
    probs: dict[str, float], path: str | None
) -> tuple[str, float]:
    """Apply the learned-model approval boundary to a probability estimate."""
    adjudication, confidence = decide(probs)
    if adjudication != APPROVED or path not in _NO_APPROVAL_PATHS:
        return adjudication, confidence

    adjudication = max(
        (DENIED, NEEDS_REVIEW),
        key=lambda action: _expected_value(action, probs),
    )
    return adjudication, probs.get(adjudication, 0.0)


class Adjudicator:
    """Wraps a fitted sklearn classifier plus its feature ordering."""

    def __init__(self, model, feature_names: list[str], metadata: dict | None = None,
                 blend_weight: float = 1.0):
        self.model = model
        self.feature_names = feature_names
        self.metadata = metadata or {}
        # Weight on the model when blending with the hand-built path prior.
        # 1.0 selects the model only. Lower values retain part of the
        # deterministic path distribution.
        self.blend_weight = blend_weight
        # sklearn orders classes_ alphabetically; map explicitly rather than
        # assuming, so a class-order change cannot silently swap outcomes.
        self.classes = list(model.classes_)

    # -- persistence -------------------------------------------------------

    def save(self, path: str | Path = MODEL_PATH) -> None:
        import joblib
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": self.model, "feature_names": self.feature_names,
             "metadata": self.metadata, "blend_weight": self.blend_weight},
            path, compress=3)

    @classmethod
    def load(cls, path: str | Path = MODEL_PATH) -> "Adjudicator":
        import joblib
        payload = joblib.load(path)
        return cls(payload["model"], payload["feature_names"],
                   payload.get("metadata"), payload.get("blend_weight", 1.0))

    @classmethod
    def try_load(cls, path: str | Path = MODEL_PATH) -> "Adjudicator | None":
        """Load the artifact when available; otherwise use policy paths."""
        try:
            return cls.load(path)
        except Exception:  # noqa: BLE001 - a missing/corrupt model must not crash a run
            return None

    # -- inference ---------------------------------------------------------

    def vectorize(self, features: dict[str, float]) -> list[float]:
        """Features in training order; anything unseen is 0.

        Sparse one-hot keys (`path__*`, `flag_*`) are absent rather than zero in
        the feature dict, so a plain lookup with a 0 default is correct.
        """
        return [float(features.get(name, 0.0)) for name in self.feature_names]

    def probabilities(self, features: dict[str, float]) -> dict[str, float]:
        row = self.vectorize(features)
        raw = self.model.predict_proba([row])[0]
        probs = {outcome: 0.0 for outcome in OUTCOMES}
        for cls, value in zip(self.classes, raw):
            probs[cls] = float(value)
        return probs

    def adjudicate(self, features: dict[str, float],
                   path_probs: dict[str, float] | None = None
                   ) -> tuple[str, float, str]:
        """Returns (adjudication, confidence, path-label) like Calibration."""
        probs = self.probabilities(features)
        if path_probs is not None and self.blend_weight < 1.0:
            w = self.blend_weight
            probs = {o: w * probs.get(o, 0.0) + (1.0 - w) * path_probs.get(o, 0.0)
                     for o in OUTCOMES}
        active_paths = [
            name.removeprefix("path__")
            for name, value in features.items()
            if name.startswith("path__") and value > 0.0
        ]
        path = active_paths[0] if len(active_paths) == 1 else None
        adjudication, confidence = decide_with_safety(probs, path)
        return adjudication, confidence, "model"
