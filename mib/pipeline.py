"""Per-packet pipeline: PDF -> evidence -> trust resolution -> prediction.

STATUS: placeholder. Raises so `cli.process_one` falls back to a prior-based
record, which keeps the end-to-end plumbing (Docker contract, output format,
one-row-per-PDF invariant) testable before the extractor exists.

Real implementation lands after the corpus forensics pass -- see WORKLOG.
"""

from __future__ import annotations

from pathlib import Path

from mib.lexicon import Lexicon
from mib.policy import Calibration
from mib.schema import Prediction


def build_prediction(pdf_path: Path, lexicon: Lexicon,
                     calibration: Calibration) -> Prediction:
    raise NotImplementedError("extractor not implemented yet")
