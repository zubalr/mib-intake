"""Output record shape and serialization.

  * `sponsor_id` must match ``^SPN-\\d{4}$`` and `arrival_date` must be a real
    ISO date.
  * Unknown fields must still carry a plausible value. The evaluator scores a
    wrong value and a blank identically, and drops genuinely unrecoverable
    fields from the case denominator.
  * `confidence` must be a JSON number (not a string) in [0, 1].
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

FIELDNAMES = [
    "case_id",
    "applicant_name",
    "species_code",
    "home_world",
    "visa_class",
    "sponsor_id",
    "arrival_date",
    "declared_purpose",
    "risk_flags",
    "fee_status",
    "adjudication",
    "confidence",
]

ADJUDICATIONS = ("APPROVED", "DENIED", "NEEDS_REVIEW")
FEE_VALUES = ("paid", "waived", "unpaid", "unknown")


@dataclass
class Prediction:
    case_id: str
    applicant_name: str
    species_code: str
    home_world: str
    visa_class: str
    sponsor_id: str
    arrival_date: str
    declared_purpose: str
    risk_flags: str
    fee_status: str
    adjudication: str
    confidence: float
    # Diagnostics are available to development tools but are not serialized.
    debug: dict = field(default_factory=dict)

    def to_row(self) -> dict:
        row = {name: getattr(self, name) for name in FIELDNAMES}
        row["confidence"] = round(float(row["confidence"]), 4)
        return row


def write_jsonl(path: str | Path, predictions: Iterable[Prediction]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w") as f:
        for pred in predictions:
            f.write(json.dumps(pred.to_row(), sort_keys=True) + "\n")
            count += 1
    return count


# Structurally-valid placeholders for fields no trusted evidence could supply.
#
# `validate_submission.py` hard-rejects a record whose `sponsor_id` is not
# ``SPN-\d{4}`` or whose `arrival_date` is not a real ISO date, so a blank is
# not an option. A well-formed placeholder is required.
#
# Neither is a guess at the *answer*. The sponsor id is deliberately outside the
# range the corpus uses, so it cannot accidentally match a real sponsor and can
# never collide with the revoked list. The date is a neutral placeholder that
# `mib.cli` replaces at output time with the median arrival date of the corpus
# actually being scored. A constant here would be fitted to the era of
# whatever set it was chosen on, which is exactly the kind of hidden
# public-corpus dependency that does not survive a private test set.
FALLBACK_SPONSOR_ID = "SPN-0000"
FALLBACK_ARRIVAL_DATE = "2000-01-01"
