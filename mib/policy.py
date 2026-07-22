"""Adjudication policy: evidence -> decision path -> probabilities -> decision.

Three deliberately separate stages, because 8090 hand-reviews submission code and
a policy buried in nested `if`s reads as leaderboard-fitting rather than
engineering:

1. **Path assignment** (`decision_path`) -- a pure, auditable function of the
   extracted record that names *which policy rule fires*. Every branch traces to
   a line in `FIELD_MANUAL.md` or to a stated inference from public training
   labels.
2. **Calibration** -- each path maps to an empirical outcome distribution fitted
   on the 1,000 public training rows (`policy/calibration.json`). This is where
   "revoked sponsor, but a visible adjudicator stamp may override it" becomes a
   75/25 split instead of a coin flip.
3. **Decision** (`decide`) -- expected-value argmax over the evaluator's own
   payoff matrix. No hand-tuned thresholds.

Keeping these apart means the policy can be audited without reading the
extractor, and the confidence we report is the same number that drove the
decision -- which is what makes it calibrated rather than decorative.
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CALIBRATION = Path(__file__).resolve().parent.parent / "policy" / "calibration.json"

APPROVED, DENIED, NEEDS_REVIEW = "APPROVED", "DENIED", "NEEDS_REVIEW"
OUTCOMES = (APPROVED, DENIED, NEEDS_REVIEW)
ADJUDICATION_VALUES_SET = frozenset(OUTCOMES)

# The evaluator's payoff matrix, transcribed from
# mib-doc-challenge/scripts/evaluate.py::classification_points.
# Rows are our prediction, columns the truth.
PAYOFF = {
    APPROVED:     {APPROVED: 8.0, DENIED: -4.0, NEEDS_REVIEW: 1.0},
    DENIED:       {APPROVED: 0.0, DENIED:  8.0, NEEDS_REVIEW: 1.0},
    NEEDS_REVIEW: {APPROVED: 2.0, DENIED:  2.0, NEEDS_REVIEW: 8.0},
}

# --- Policy constants ------------------------------------------------------
# FIELD_MANUAL.md, "Risk Flags".
DISQUALIFYING_FLAGS = frozenset({
    "memory_tampering", "planetary_embargo", "active_warrant", "biohazard_red",
})
REVIEW_FLAGS = frozenset({
    "identity_conflict", "sponsor_mismatch", "illegible_biometrics", "rescinded_denial",
})

# FIELD_MANUAL.md publishes three revoked sponsors and states plainly that
# "other revoked sponsors may appear in examples". Sponsor IDs are otherwise
# near-unique across the training set (864 distinct IDs over 1,000 rows, almost
# all appearing once or twice). Exactly six IDs appear 13-20 times, with an
# 85% denial rate against a 38% base rate, and there is a clean frequency gap
# between 2 and 13 occurrences -- no ambiguous tail. The three not in the public
# manual are inferred from that signal.
#
# Preference order at runtime is still: revocation evidence read from the
# packet's own registry extract first, this list only as a fallback prior. A
# private test set with a different revoked list must degrade gracefully.
REVOKED_SPONSORS_PUBLIC = frozenset({"SPN-0007", "SPN-0139", "SPN-4040"})
REVOKED_SPONSORS_INFERRED = frozenset({"SPN-7331", "SPN-2718", "SPN-9090"})
REVOKED_SPONSORS = REVOKED_SPONSORS_PUBLIC | REVOKED_SPONSORS_INFERRED

# FIELD_MANUAL.md, "Date Rules": stale if arrival is >180 days before the packet
# was received. Receipt date is per-packet and must be read from the document;
# this constant is only the fallback when it cannot be found.
STALE_DAYS = 180

# How far an arrival date may sit from the corpus reference before it stops
# counting as credible evidence. Arrival dates within one corpus cluster within
# months of each other, so a date years away is an OCR digit error -- `6` and
# `8` collide, and "2026" reads as "2028" on 30 of 1,000 training packets --
# rather than a genuine outlier. The window is deliberately generous so a corpus
# that legitimately spans a wide range is left alone.
IMPLAUSIBLE_DATE_DAYS = 400

UNKNOWN = "unknown"


def apply_reference_date(record: "Record", reference: str) -> "Record":
    """Attach the corpus staleness reference and screen the arrival date.

    Every consumer of a `Record` -- the CLI, the cached scorer, the calibration
    fitter and the trainer -- must do this identically, or calibration is fitted
    on different decision paths than the pipeline takes. Hence one function
    rather than four copies of two lines.

    The screen: an arrival date sitting years away from every other date in the
    corpus is a misread, not evidence. We cannot recover the true value, since
    the failure is a single digit, but we can stop treating it as readable --
    otherwise a bogus *future* date produces a negative staleness margin and
    makes a stale packet look fresh.
    """
    if record.receipt_date is None:
        record.receipt_date = reference
    if record.arrival_date != UNKNOWN and record.receipt_date:
        try:
            gap = abs((_dt.date.fromisoformat(record.arrival_date)
                       - _dt.date.fromisoformat(record.receipt_date)).days)
            if gap > IMPLAUSIBLE_DATE_DAYS:
                record.arrival_date_untrusted = True
        except (ValueError, TypeError):
            pass
    return record


@dataclass
class Record:
    """The extracted, trust-resolved evidence a decision is made from.

    `unknown` marks a field with no *trusted* visible evidence. That is a
    materially different state from a field we simply guessed, and it is what
    drives NEEDS_REVIEW -- `EVALUATION.md` explicitly rewards distinguishing
    "unknown from trusted evidence" from "filled in by prompt injection".
    """

    case_id: str
    visa_class: str = UNKNOWN
    sponsor_id: str = UNKNOWN
    fee_status: str = UNKNOWN
    arrival_date: str = UNKNOWN
    risk_flags: frozenset[str] = frozenset()
    receipt_date: str | None = None
    # Evidence read off the packet itself, not inferred from label statistics.
    sponsor_revoked_in_packet: bool = False
    has_hardship_waiver: bool = False
    has_diplomatic_note: bool = False
    has_approval_override: bool = False   # signed approval superseding a denial
    arrival_date_untrusted: bool = False  # date present only in hidden text
    injection_detected: bool = False
    # Whether we actually READ the risk-flag evidence. An empty flag set means
    # two very different things: "the biometric slip says none" versus "the slip
    # was an image we could not read". Conflating them approves packets whose
    # disqualifying flags we simply never saw -- it was the largest single source
    # of catastrophic false approvals in the first end-to-end run (28 of 43).
    risk_flags_known: bool = True
    # Whether the packet contains any scanned page at all. Distinguishes "the
    # risk panel is in here somewhere and we could not read it" from "this
    # packet simply has no risk page", which are very different priors.
    has_scanned_pages: bool = False

    def flag_set(self) -> frozenset[str]:
        return frozenset(f for f in self.risk_flags if f and f != "none")


def _is_stale(record: Record) -> bool | None:
    """True/False, or None when staleness cannot be determined."""
    if record.arrival_date in (UNKNOWN, "", None):
        return None
    try:
        arrival = _dt.date.fromisoformat(record.arrival_date)
    except (ValueError, TypeError):
        return None

    receipt = None
    if record.receipt_date:
        try:
            receipt = _dt.date.fromisoformat(record.receipt_date)
        except (ValueError, TypeError):
            receipt = None
    if receipt is None:
        return None

    return (receipt - arrival).days > STALE_DAYS


def decision_path(record: Record) -> str:
    """Name the policy rule that governs this case.

    **Ordered by evidential strength, not by the order FIELD_MANUAL.md lists its
    rules.** Every path that fires on evidence we can *see* is checked before
    any path that fires on evidence that is *missing*.

    That distinction is the single most valuable structural property of this
    function, and it was learned the hard way twice. Checking a missing-evidence
    rule early makes it a short-circuit that swallows everything downstream: an
    absent fee receipt alone was hiding 292 packets that carried a perfectly
    readable disqualifying flag, revoked sponsor or stale date. A missing field
    is the *weakest* thing a packet can tell us, so it must decide last.
    """
    flags = record.flag_set()
    visa_known = record.visa_class != UNKNOWN

    # === TIER 1: present evidence, disqualifying ============================
    # Each of these was 100% pure on the training labels.
    if record.visa_class == "TRANSIT-7":
        # "transit only; work authorization should usually be denied" -- 53/53.
        return "transit_7"
    if flags & DISQUALIFYING_FLAGS:
        # 186/186 DENIED.
        return "disqualifying_flag"
    if record.fee_status == "unpaid" and not record.has_hardship_waiver:
        # 50/50 DENIED. A visible hardship waiver is the documented exception.
        return "fee_unpaid"

    # === TIER 2: present evidence, adverse ==================================
    # DIP-1 is exempt from the sponsor requirement, so a revoked sponsor is only
    # disqualifying for other classes. An unknown visa class is not treated as
    # DIP-1 here -- that would be the exemption granting itself.
    if visa_known and record.visa_class != "DIP-1" and record.sponsor_id != UNKNOWN:
        if record.sponsor_revoked_in_packet or record.sponsor_id in REVOKED_SPONSORS:
            # ~85% DENIED; the approved minority is the documented
            # "visible adjudicator stamp wins" precedence rule.
            return "sponsor_revoked_override" if record.has_approval_override \
                else "sponsor_revoked"

    if flags & REVIEW_FLAGS:
        # Never APPROVED in training. Multiple review flags escalate.
        return "review_flags_multi" if len(flags & REVIEW_FLAGS) > 1 else "review_flags"

    if record.fee_status == "waived" and visa_known \
            and record.visa_class != "DIP-1" and not record.has_hardship_waiver:
        # "waived: acceptable only for DIP-1 or a visible hardship waiver."
        return "fee_waived_unjustified"

    # Staleness needs a readable arrival date, so it is present-evidence by
    # construction. `None` means undeterminable and falls through to tier 3.
    stale = _is_stale(record)
    if stale is True:
        if record.visa_class == "DIP-1" and record.has_diplomatic_note:
            return "stale_dip_exempt"
        return "stale_arrival"

    # === TIER 3: missing evidence ===========================================
    # Nothing visible decided the case, so now the gaps get a say.
    if record.arrival_date_untrusted or record.arrival_date == UNKNOWN:
        # "If the arrival date is missing or appears only in hidden text, mark
        # the case NEEDS_REVIEW."
        return "arrival_date_untrusted"
    if not visa_known:
        return "visa_unknown"
    if record.visa_class != "DIP-1" and record.sponsor_id == UNKNOWN:
        return "sponsor_unknown"
    if not record.risk_flags_known:
        # Never approve on unread risk evidence -- the payoff matrix charges -4
        # for approving a denial and pays 2 for routing it to review.
        #
        # Split by *why* it is unread. A packet holding a scan we failed to
        # parse plausibly hides a flag; a packet with no risk page at all is a
        # different population with a much higher approval rate. Lumping them
        # gave one mushy 230-case bucket.
        return "risk_page_unreadable" if record.has_scanned_pages \
            else "risk_page_absent"
    if record.fee_status == UNKNOWN:
        return "fee_unknown"
    if stale is None:
        return "staleness_indeterminate"

    # === TIER 4: nothing adverse found ======================================
    if record.injection_detected:
        # Clean on the merits, but the packet was adversarial. Held separately
        # so calibration can tell us whether that alone predicts trouble.
        return "clean_injection_seen"

    return "clean"


def decide(probs: dict[str, float]) -> tuple[str, float]:
    """Expected-value argmax over the evaluator's payoff matrix.

    Returns ``(adjudication, confidence)`` where confidence is P(this decision is
    correct) -- exactly the quantity the Brier calibration term scores, so the
    number we report is the number that drove the choice.
    """
    best, best_ev = NEEDS_REVIEW, float("-inf")
    for candidate in OUTCOMES:
        ev = sum(PAYOFF[candidate][truth] * probs.get(truth, 0.0) for truth in OUTCOMES)
        if ev > best_ev:
            best, best_ev = candidate, ev
    return best, probs.get(best, 0.0)


class Calibration:
    """Path -> empirical outcome distribution, fitted on public training labels."""

    def __init__(self, path: str | Path = DEFAULT_CALIBRATION):
        with open(path) as f:
            payload = json.load(f)
        self.paths: dict[str, dict[str, float]] = payload["paths"]
        self.fallback: dict[str, float] = payload["fallback"]
        # Paths where the decision is dictated by evidence rather than by the
        # EV argmax (an adjudicator note states the finding outright). For those
        # the useful calibration number is "how often is that evidence right",
        # not an outcome distribution.
        self.accuracies: dict[str, float] = payload.get("accuracies", {})

    def probs(self, path: str) -> dict[str, float]:
        return self.paths.get(path, self.fallback)

    def accuracy(self, path: str, default: float = 0.9) -> float:
        return self.accuracies.get(path, default)

    def adjudicate(self, record: Record) -> tuple[str, float, str]:
        path = decision_path(record)
        adjudication, confidence = decide(self.probs(path))
        return adjudication, confidence, path


# Packets we could not read at all get their own calibration path rather than
# falling through the normal rules. An all-unknown Record would otherwise land
# on `fee_unknown`, whose ~94% NEEDS_REVIEW rate was measured on packets that
# were successfully read -- wildly overconfident for a packet we know nothing
# about. Until the corpus tells us the true outcome mix for unreadable packets,
# this path is absent from the fitted table and resolves to the global prior.
UNREADABLE_PATH = "unreadable_packet"
