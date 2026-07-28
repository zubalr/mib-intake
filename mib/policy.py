"""Adjudication policy with separate path, calibration, and decision stages.

``decision_path`` maps trusted evidence to a named policy state.
``Calibration`` maps that state to an empirical class distribution.
``decide`` selects the expected-value maximum under the published payoff matrix.
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
# Rows are predictions and columns are ground truth.
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
# "other revoked sponsors may appear in examples".
#
# Additional revoked sponsors are inferred from frequency within the corpus.
# Sponsor IDs are normally sparse, while revoked IDs recur unusually often.
REVOKED_SPONSORS_PUBLIC = frozenset({"SPN-0007", "SPN-0139", "SPN-4040"})
REVOKED_SPONSORS = REVOKED_SPONSORS_PUBLIC

# A sponsor appearing more than this multiple of the 99th-percentile frequency
# is over-represented to a degree ordinary sponsors never reach. The threshold is
# below the observed gap (the flagged IDs sit at 6 to 40 times p99 while the tail
# tops out at 1x) so the exact multiple does not matter.
REVOKED_FREQUENCY_MULTIPLE = 4
MIN_CORPUS_FOR_FREQUENCY = 400

# FIELD_MANUAL.md, "Date Rules": stale if arrival is >180 days before the packet
# was received. Receipt date is per-packet and must be read from the document;
# this constant is only the fallback when it cannot be found.
STALE_DAYS = 180

# How far an arrival date may sit from the corpus reference before it stops
# counting as credible evidence. Arrival dates within one corpus cluster within
# months of each other, so a date years away is usually an OCR digit error.
# The window is intentionally generous so a corpus
# that legitimately spans a wide range is left alone.
IMPLAUSIBLE_DATE_DAYS = 400

UNKNOWN = "unknown"


def corpus_revoked_sponsors(records) -> frozenset[str]:
    """Sponsor ids that are over-represented in the corpus being scored.

    Genuine sponsors are near-unique; a revoked one recurs because many packets
    cite it. So the signal is a frequency outlier, measured against the corpus
    itself rather than against a list copied out of the public labels.

    Returns an empty set on a corpus too small for the statistic to mean
    anything, which leaves the documented `REVOKED_SPONSORS_PUBLIC` as the only
    source when there is insufficient evidence.
    """
    counts: dict[str, int] = {}
    for record in records:
        sponsor = getattr(record, "sponsor_id", None)
        if sponsor and sponsor != UNKNOWN:
            counts[sponsor] = counts.get(sponsor, 0) + 1
    if len(counts) < MIN_CORPUS_FOR_FREQUENCY:
        return frozenset()

    frequencies = sorted(counts.values())
    p99 = frequencies[int(len(frequencies) * 0.99)]
    threshold = REVOKED_FREQUENCY_MULTIPLE * p99
    return frozenset(s for s, n in counts.items() if n > threshold)


# A repaired year must beat the runner-up by this factor before it is adopted.
# Same multiple as `REVOKED_FREQUENCY_MULTIPLE`, and for the same reason: the
# observed margin is enormous (826 against 48), so the exact value is not load
# bearing.
YEAR_DOMINANCE = 4
# Below this many dated packets the year histogram is not worth trusting, so the
# repair turns itself off rather than degrading to noise.
MIN_DATED_FOR_YEAR_REPAIR = 50


def corpus_years(records) -> dict[str, int]:
    """Return the contiguous year sequence containing the corpus mode.

    Arrival dates should form a contiguous sequence. A detached year beyond an
    empty interval is treated as an OCR outlier.
    """
    counts: dict[int, int] = {}
    for record in records:
        value = getattr(record, "arrival_date", None)
        if value and value != UNKNOWN:
            try:
                year = _dt.date.fromisoformat(value).year
            except (ValueError, TypeError):
                continue
            counts[year] = counts.get(year, 0) + 1
    if sum(counts.values()) < MIN_DATED_FOR_YEAR_REPAIR:
        return {}

    mode = max(counts, key=lambda y: counts[y])
    run = {mode: counts[mode]}
    for step in (-1, 1):
        year = mode + step
        while year in counts:
            run[year] = counts[year]
            year += step
    return {f"{y:04d}": n for y, n in run.items()}


def repair_year(value: str, years: dict[str, int]) -> str | None:
    """Adopt a corpus year one substitution from this date's, if it is decisive.

    Returns the repaired ISO date, or None to leave the value alone.

    Uniqueness alone is not usable here: 2028 is one substitution from both 2026
    and 2025, so requiring a single candidate rejects every repair. The tie
    breaks on the corpus histogram instead, requiring the winner to dominate.

    The result is an output fallback, not policy evidence. Callers must not
    promote it onto the ``Record``.
    """
    if not years:
        return None
    try:
        parsed = _dt.date.fromisoformat(value)
    except (ValueError, TypeError):
        return None
    observed = f"{parsed.year:04d}"
    if observed in years:
        return None

    hits = sorted((n, y) for y, n in years.items()
                  if len(y) == len(observed)
                  and sum(a != b for a, b in zip(y, observed)) == 1)
    if not hits:
        return None
    if len(hits) > 1 and hits[-1][0] < YEAR_DOMINANCE * hits[-2][0]:
        return None
    try:
        return parsed.replace(year=int(hits[-1][1])).isoformat()
    except ValueError:  # 29 February landing in a non-leap year
        return None


def apply_reference_date(record: "Record", reference: str,
                         revoked: frozenset[str] = frozenset()) -> "Record":
    """Attach corpus-level context to a record: staleness reference, the
    corpus-derived revoked-sponsor set, and a screen on the arrival date.

    Every consumer of a ``Record`` calls this function so corpus-level context
    is applied consistently.

    The screen: an arrival date sitting years away from every other date in the
    corpus is a misread, not evidence. We cannot recover the true value, since
    the failure is a single digit, but it should not count as readable.
    Otherwise a future OCR date produces a negative staleness margin and
    makes a stale packet look fresh.
    """
    if record.receipt_date is None:
        record.receipt_date = reference
    if record.sponsor_id in revoked:
        record.sponsor_revoked_in_packet = True
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
    drives NEEDS_REVIEW. `EVALUATION.md` explicitly rewards distinguishing
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
    # disqualifying flags are not observed.
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

    Rules based on visible evidence are evaluated before rules based on missing
    evidence. Missing fields therefore cannot suppress stronger signals.
    """
    flags = record.flag_set()
    visa_known = record.visa_class != UNKNOWN

    # Tier 1: present disqualifying evidence.
    if record.visa_class == "TRANSIT-7":
        # Transit class cannot authorize work.
        return "transit_7"
    if flags & DISQUALIFYING_FLAGS:
        return "disqualifying_flag"
    if record.fee_status == "unpaid" and not record.has_hardship_waiver:
        # A visible hardship waiver is the documented exception.
        return "fee_unpaid"

    # Tier 2: present adverse evidence.
    # DIP-1 is exempt from the sponsor requirement, so a revoked sponsor is only
    # disqualifying for other classes. An unknown visa class is not treated as
    # DIP-1 here -- that would be the exemption granting itself.
    if visa_known and record.visa_class != "DIP-1" and record.sponsor_id != UNKNOWN:
        if (record.sponsor_revoked_in_packet
                or record.sponsor_id in REVOKED_SPONSORS_PUBLIC):
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

    # Tier 3: missing evidence.
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
        # Never approve on unread risk evidence. The payoff matrix charges -4
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

    # Tier 4: no adverse evidence.
    if record.injection_detected:
        # Clean on the merits, but the packet was adversarial. Held separately
        # so calibration can tell us whether that alone predicts trouble.
        return "clean_injection_seen"

    return "clean"


def decide(probs: dict[str, float]) -> tuple[str, float]:
    """Expected-value argmax over the evaluator's payoff matrix.

    Returns ``(adjudication, confidence)`` where confidence is P(this decision is
    correct), the quantity used by the Brier calibration term.
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
# were successfully read and is overconfident for a packet with no usable data.
# Until the corpus provides an empirical outcome mix for unreadable packets,
# this path is absent from the fitted table and resolves to the global prior.
UNREADABLE_PATH = "unreadable_packet"
