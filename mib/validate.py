"""Field-level validation for extracted values and emitted rows.

Two independent guards, because OCR introduced a class of failure the text-layer
path never had:

1. **At record time** -- reject values that cannot possibly be right for their
   field before they ever become evidence. OCR on a damaged scan produced
   arrival dates of ``}``, ``f``, ``2926-05-03 ke i`` and
   ``[DATE WA._. =D OUT]``. Two of those are not merely wrong, they are
   *structurally invalid*: `validate_submission.py` rejects any record whose
   `arrival_date` is not a real ISO date or whose `sponsor_id` does not match
   ``^SPN-\\d{4}$``, so a single bad OCR read could invalidate the row.

2. **At output time** -- a final sweep over every emitted row, replacing
   anything still invalid with the prior fallback. Belt and braces: the row we
   write must satisfy the official validator no matter what the pipeline did.

The damage-marker check is deliberately fuzzy. The text layer produces clean
``[DATE WASHED OUT]``, but OCR of the same marker yields ``[DATE WA._. =D OUT]``
or ``{DATE WASHED ouT}`` -- a strict uppercase-in-brackets pattern misses those
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
        return value.casefold() in FEE_VALUES
    # Closed-vocabulary fields are checked by lexicon snapping; here we only
    # reject obvious debris.
    return len(re.sub(r"[^A-Za-z0-9]", "", value)) >= 2
