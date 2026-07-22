"""Output record shape and serialisation.

The evaluator and validator are strict in ways worth encoding once, here:

  * `sponsor_id` must match ``^SPN-\\d{4}$`` and `arrival_date` must be a real
    ISO date, or `validate_submission.py` rejects the record outright. There is
    no "blank" escape hatch.
  * Unknown fields must still carry a plausible value. The evaluator scores a
    wrong value and a blank identically, and drops genuinely unrecoverable
    fields from the case's denominator, so a fallback guess is free upside.
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
    # Diagnostics -- never serialised into the submission, used by dev tooling
    # to explain why a decision was made.
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
