"""Closed-vocabulary snapping.

Every scored field except `applicant_name` and `sponsor_id` is drawn from a
small closed set, and even names are generated from ~144 first-name and ~144
last-name tokens. That turns extraction from open-ended transcription into
nearest-neighbour matching: OCR that reads ``V3NUSIAN_MYCEL1AL`` still scores
full marks once snapped to ``VENUSIAN_MYCELIAL``.

Two things this must get right:

  * **Degrade gracefully.** The private test set may contain names the public
    training labels never showed. When nothing is within the distance
    threshold we keep the observed string rather than force a wrong match --
    a bad snap converts a would-be hit into a guaranteed miss.
  * **Be OCR-aware.** Confusions are not uniform. `0/O`, `1/l/I`, `5/S`, `8/B`,
    `rn/m` are near-free substitutions and should cost less than an arbitrary
    edit, otherwise the threshold has to be loosened enough to admit real errors.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

DEFAULT_LEXICON = Path(__file__).resolve().parent.parent / "policy" / "lexicon.json"

# Character pairs an OCR engine confuses at near-zero semantic cost. Substituting
# within a group is charged a fraction of a normal edit.
_CONFUSION_GROUPS = [
    set("0OQD"),
    set("1lI|"),
    set("5S"),
    set("8B"),
    set("2Z"),
    set("6G"),
    set("9g"),
    set("UV"),
    set("nh"),
    set("cC"),
]
_CONFUSABLE: dict[str, set[str]] = {}
for _group in _CONFUSION_GROUPS:
    for _ch in _group:
        _CONFUSABLE.setdefault(_ch, set()).update(_group - {_ch})

_SUB_CONFUSABLE = 0.3  # cost of an OCR-plausible substitution
_SUB_NORMAL = 1.0

# Confusions that span more than one character, which a plain per-character
# substitution table cannot express. `rn` -> `m` is the classic: it costs a full
# insert *and* a substitute under vanilla Levenshtein (distance 2), which pushes
# a genuinely-close match outside any sane threshold. Charged once, cheaply.
_MULTI_CONFUSIONS = [
    ("rn", "m"), ("m", "rn"),
    ("cl", "d"), ("d", "cl"),
    ("ii", "n"), ("n", "ii"),
    ("vv", "w"), ("w", "vv"),
    ("li", "h"), ("h", "li"),
    ("nn", "m"), ("m", "nn"),
]
_MULTI_COST = 0.4
# Grouped by the (len_a, len_b) shape so the DP can look up only what applies.
_MULTI_BY_SHAPE: dict[tuple[int, int], list[tuple[str, str]]] = {}
for _x, _y in _MULTI_CONFUSIONS:
    _MULTI_BY_SHAPE.setdefault((len(_x), len(_y)), []).append((_x, _y))
_MAX_MULTI = max(max(len(x), len(y)) for x, y in _MULTI_CONFUSIONS)


def _sub_cost(a: str, b: str) -> float:
    if a == b:
        return 0.0
    if b in _CONFUSABLE.get(a, ()):
        return _SUB_CONFUSABLE
    return _SUB_NORMAL


def weighted_distance(a: str, b: str) -> float:
    """Levenshtein distance with OCR-aware substitution and multi-char rules.

    Uses a full matrix (not two rows) because the multi-character confusions
    need to reach back up to two rows and columns.
    """
    if a == b:
        return 0.0
    if not a:
        return float(len(b))
    if not b:
        return float(len(a))

    n, m = len(a), len(b)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = float(i)
    for j in range(1, m + 1):
        dp[0][j] = float(j)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best = min(
                dp[i - 1][j] + 1.0,                              # deletion
                dp[i][j - 1] + 1.0,                              # insertion
                dp[i - 1][j - 1] + _sub_cost(a[i - 1], b[j - 1]),
            )
            for (la, lb), pairs in _MULTI_BY_SHAPE.items():
                if i >= la and j >= lb:
                    sub_a = a[i - la:i]
                    sub_b = b[j - lb:j]
                    for x, y in pairs:
                        if sub_a == x and sub_b == y:
                            cand = dp[i - la][j - lb] + _MULTI_COST
                            if cand < best:
                                best = cand
            dp[i][j] = best

    return dp[n][m]


def _canon(text: str) -> str:
    """Fold away noise that never carries meaning: case, spacing, separators."""
    return re.sub(r"[^a-z0-9]+", "", str(text or "").casefold())


class Lexicon:
    def __init__(self, path: str | Path = DEFAULT_LEXICON):
        with open(path) as f:
            self.data = json.load(f)

    # -- closed sets ------------------------------------------------------

    def values(self, field: str) -> list[str]:
        return self.data[field]["values"]

    def prior_mode(self, field: str) -> str:
        return self.data[field]["prior_mode"]

    def snap(self, field: str, observed: str, max_ratio: float = 0.34) -> tuple[str, float]:
        """Snap `observed` onto the closed set for `field`.

        Returns ``(value, confidence)``. Confidence is 1.0 for an exact canonical
        match and decays with edit distance. When nothing is close enough we
        return the observed string with 0.0 confidence -- the caller decides
        whether to fall back to the prior mode.
        """
        return self._snap_to(tuple(self.values(field)), observed, max_ratio)

    def snap_name(self, observed: str, max_ratio: float = 0.34) -> tuple[str, float]:
        """Snap a full applicant name token-by-token.

        Names are two generated tokens; snapping each independently is far more
        robust than matching the whole string, because one mangled token should
        not drag the other down.
        """
        parts = str(observed or "").split()
        if len(parts) < 2:
            return str(observed or "").strip(), 0.0

        first, fc = self._snap_to(tuple(self.data["applicant_name"]["first_tokens"]),
                                  parts[0], max_ratio)
        last, lc = self._snap_to(tuple(self.data["applicant_name"]["last_tokens"]),
                                 parts[-1], max_ratio)

        # Keep the observed token whenever its own snap failed. Snapping one
        # token must never rewrite the other: an out-of-vocabulary name (which
        # the private test set may well contain) should survive intact rather
        # than be dragged onto the nearest generated token.
        if fc <= 0.0:
            first = parts[0]
        if lc <= 0.0:
            last = parts[-1]

        return f"{first} {last}", min(fc, lc)

    @staticmethod
    @lru_cache(maxsize=100_000)
    def _snap_to(candidates: tuple[str, ...], observed: str,
                 max_ratio: float) -> tuple[str, float]:
        obs = _canon(observed)
        if not obs:
            return "", 0.0

        best, best_dist = None, float("inf")
        runner_up = float("inf")
        for cand in candidates:
            dist = weighted_distance(obs, _canon(cand))
            if dist < best_dist:
                best, best_dist, runner_up = cand, dist, best_dist
                if dist == 0.0:
                    break
            elif dist < runner_up:
                runner_up = dist

        if best is None:
            return str(observed).strip(), 0.0

        # Threshold scales with length: a 4-char code tolerates less absolute
        # error than a 20-char species name.
        budget = max(1.0, len(obs) * max_ratio)
        if best_dist > budget:
            return str(observed).strip(), 0.0

        conf = max(0.0, 1.0 - best_dist / (budget + 1e-9))

        # An ambiguous snap is worse than none: if the runner-up is nearly as
        # close, we cannot tell which value was intended, and a confident wrong
        # snap turns a possible hit into a certain miss. Exact matches are exempt.
        if best_dist > 0.0 and (runner_up - best_dist) < 0.5:
            conf *= 0.35

        return best, conf

    # -- risk flags -------------------------------------------------------

    @property
    def disqualifying_flags(self) -> set[str]:
        return set(self.data["risk_flags"]["disqualifying"])

    @property
    def review_flags(self) -> set[str]:
        return set(self.data["risk_flags"]["review_only"])

    def snap_flag(self, observed: str, max_ratio: float = 0.3) -> tuple[str, float]:
        all_flags = tuple(sorted(self.disqualifying_flags | self.review_flags))
        return self._snap_to(all_flags, observed, max_ratio)
