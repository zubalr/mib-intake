"""Field-level validation for extracted values and emitted rows.

Validation occurs before a value becomes evidence and again before output.
Dates and sponsor IDs must satisfy the official schema even when OCR returns
damaged text.

The damage-marker check is tolerant of OCR noise. The text layer produces clean
``[DATE WASHED OUT]``, but OCR of the same marker yields ``[DATE WA._. =D OUT]``
or ``{DATE WASHED ouT}``. A strict uppercase-in-brackets pattern misses those
and lets a marker through as if it were a value.
"""

from __future__ import annotations

import datetime as _dt
import re

ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
SPONSOR_RE = re.compile(r"^SPN-\d{4}$")
NAME_RE = re.compile(r"^[A-Za-z][A-Za-z'\-]*(?: [A-Za-z][A-Za-z'\-]*)+$")

FEE_VALUES = {"paid", "waived", "unpaid", "unknown"}

# Any bracket-ish wrapper plus a damage word. OCR mangles both the brackets and
# the casing, so match on the *words*, not on the delimiters.
DAMAGE_WORDS = (
    "cut out", "whiteout", "washed", "blank", "torn", "illegible",
    "obscured", "lost", "unreadable", "redacted", "missing",
)

# Dates outside this window are OCR noise, not data. Wide enough to be
# corpus-agnostic; a private test set from another decade still passes.
MIN_YEAR, MAX_YEAR = 2000, 2099


def looks_damaged(value: str) -> bool:
    low = value.casefold()
    if any(word in low for word in DAMAGE_WORDS):
        return True
    # An opening bracket with no closing one is a damage marker the scan cut
    # off mid-word: `[SPECIES WHITEOUT]` came back as `[SPE`, which carries no
    # damage word and enough letters to pass every other check. No real field
    # value starts with a bracket, so this is safe to reject outright.
    if value[:1] in "[{(" and not any(c in value for c in "]})"):
        return True
    # A value that is mostly punctuation is OCR debris, not content.
    stripped = re.sub(r"[^A-Za-z0-9]", "", value)
    return len(stripped) < max(1, len(value) // 3)


def valid_date(value: str) -> bool:
    value = value.strip()
    if not ISO_DATE_RE.match(value):
        return False
    try:
        parsed = _dt.date.fromisoformat(value)
    except ValueError:
        return False
    return MIN_YEAR <= parsed.year <= MAX_YEAR


def valid_sponsor(value: str) -> bool:
    return bool(SPONSOR_RE.match(value.strip()))


def valid_name(value: str) -> bool:
    value = value.strip()
    return bool(NAME_RE.match(value)) and 3 <= len(value) <= 60


def valid_for_field(field: str, value: str) -> bool:
    """Whether `value` is structurally plausible for `field`."""
    value = (value or "").strip()
    if not value or looks_damaged(value):
        return False
    if field == "arrival_date":
        return valid_date(value)
    if field == "sponsor_id":
        return valid_sponsor(value)
    if field == "applicant_name":
        return valid_name(value)
    if field == "fee_status":
        # Deliberately NOT an exact membership test. OCR renders "paid" as
        # "paig"; rejecting it here would discard a value the closed-vocabulary
        # snapper recovers perfectly. Snapping happens in the pipeline, so this
        # only screens out debris.
        return len(re.sub(r"[^A-Za-z]", "", value)) >= 3
    # Closed-vocabulary fields are checked by lexicon snapping; here we only
    # reject obvious debris.
    return len(re.sub(r"[^A-Za-z0-9]", "", value)) >= 2
