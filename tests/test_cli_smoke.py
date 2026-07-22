"""End-to-end smoke test for the CLI.

This exists because of a specific bug class the unit tests cannot see. Phase 2
imports `finalize` and `resolve_printed_date` *inside* `main()`, and calls
`corpus_years` there too. A missing import of either is not a syntax error and
not an import error -- `import mib.cli` succeeds, every unit test passes, and the
program dies with `NameError` only when a container actually processes a packet.
That is the seventh variant of the silent failure this project keeps hitting, and
it was caught by reading the diff rather than by any test.

So: run the real entrypoint over real packets and assert the output is valid.
Skipped when the dataset is absent, since the data is gitignored.
"""

import json
from pathlib import Path

import pytest

from mib.cli import main
from mib.schema import ADJUDICATIONS, FEE_VALUES

DATA = Path(__file__).resolve().parent.parent.parent / "mib-doc-challenge" / "data" / "train"


@pytest.mark.skipif(not DATA.is_dir(), reason="training PDFs not present")
def test_cli_produces_one_valid_row_per_pdf(tmp_path):
    staged = tmp_path / "input"
    staged.mkdir()
    pdfs = sorted(DATA.glob("*.pdf"))[:4]
    assert pdfs, "no PDFs to run against"
    for pdf in pdfs:
        (staged / pdf.name).write_bytes(pdf.read_bytes())

    out = tmp_path / "predictions.jsonl"
    assert main([str(staged), str(out)]) == 0

    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == len(pdfs)
    assert {r["case_id"] for r in rows} == {p.stem for p in pdfs}
    for row in rows:
        assert row["adjudication"] in ADJUDICATIONS
        assert 0.0 <= row["confidence"] <= 1.0
        assert row["fee_status"] in FEE_VALUES
        # The two fields `validate_submission.py` hard-rejects on.
        assert len(row["sponsor_id"]) == 8 and row["sponsor_id"].startswith("SPN-")
        assert len(row["arrival_date"]) == 10 and row["arrival_date"][4] == "-"
