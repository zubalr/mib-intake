# mib-intake — MIB Doc Challenge solution

Offline pipeline for 8090's *Intergalactic Intake* challenge. Reads a directory
of adversarial PDF case packets and emits `predictions.jsonl`: ten extracted
fields plus an `APPROVED` / `DENIED` / `NEEDS_REVIEW` adjudication with a
calibrated confidence.

**Score: 123.2 / 150 out-of-fold** on the 1,000 labelled training packets —
42.6 extraction, 64.9 classification, 15.8 calibration, Brier 0.106.

See [`MEMO.md`](MEMO.md) for the technical write-up and [`WORKLOG.md`](WORKLOG.md)
for the full run-by-run history, including the regressions.

## Run it

```bash
docker build -t mib-intake .
```

```bash
docker run --rm --network none --cpus 4 --memory 8g --read-only --tmpfs /tmp \
  -v /path/to/pdfs:/input:ro -v /path/to/out:/output \
  mib-intake /input /output/predictions.jsonl
```

The entrypoint takes exactly `<input_pdf_dir> <output_predictions_path>`. No
network, no API keys, no LLM/VLM or cloud OCR — Tesseract, PyMuPDF and
scikit-learn only.

## Layout

| Path | What it does |
| --- | --- |
| `mib/pdfio.py` | Span extraction and visibility classification (hidden / white / off-crop) |
| `mib/extract.py` | Packet → typed evidence with provenance and trust ranks |
| `mib/ocr.py` | Multi-configuration Tesseract fallback for scanned pages, literal mining |
| `mib/lexicon.py` | Closed-vocabulary snapping with OCR-aware weighted edit distance |
| `mib/pipeline.py` | Trust resolution → `Record` → prediction |
| `mib/policy.py` | Decision paths, calibration, expected-value decision rule |
| `mib/features.py` | Evidence features for the learned adjudicator |
| `mib/model.py` | The learned adjudicator (falls back to hand-built paths if absent) |
| `mib/cli.py` | Two-phase parallel driver, output-schema enforcement |
| `tools/` | Training, calibration fitting, extraction cache, diagnostics |
| `policy/` | Fitted artifacts: lexicon, calibration, adjudicator (1.5 MB) |

## Development

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
```

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests -q
```

The adjudicator is a pickle, so the training environment and the image must
agree on versions. Check before building:

```bash
PYTHONPATH=. .venv/bin/python tools/check_env.py
```

A full measurement cycle runs from a cached extraction rather than re-parsing
PDFs:

```bash
PYTHONPATH=. .venv/bin/python tools/build_cache.py --pdf-dir ../mib-doc-challenge/data/train --out scratch/cache_train.pkl
```

```bash
PYTHONPATH=. .venv/bin/python tools/train_adjudicator.py --cache scratch/cache_train.pkl --labels ../mib-doc-challenge/data/train_labels.csv --out policy/adjudicator.joblib
```

Rebuild the cache after any change under `mib/` that affects extraction — it
records a fingerprint of the extraction sources and warns when stale.
