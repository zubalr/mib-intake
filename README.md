# mib-intake

Offline document extraction and adjudication pipeline for 8090's
*Intergalactic Intake* challenge.

The pipeline reads a directory of adversarial PDF case packets and writes one
JSONL prediction per packet. Each prediction contains nine extracted fields,
an `APPROVED`, `DENIED`, or `NEEDS_REVIEW` decision, and a calibrated
confidence.

On the public training set the shipped image scores **133.93 / 150**. The
repeated out-of-fold estimate is **128.53 / 150, SE 0.16**.

| Section | Training set | Out of fold |
| --- | ---: | ---: |
| Extraction | 44.16 / 50 | 44.15 / 50 |
| Classification | 72.69 / 80 | 67.88 / 80 |
| Calibration | 17.07 / 20 | 16.50 / 20 |
| Mean confidence Brier | 0.0732 | 0.0874 |

The training-set score is in-sample. The out-of-fold estimate uses repeated
stratified cross-validation over ten fold assignments, so every held-out
prediction comes from a model that did not train on that packet; it is the
figure to use for unseen packets. See [MEMO.md](MEMO.md).

The final model is trained on document-derived evidence only. It does not
receive case IDs, filenames, hidden answer text, or any packet-identifying
feature.

See [MEMO.md](MEMO.md) for the design and evaluation details.

## Run

```bash
docker build -t mib-intake .
```

```bash
docker run --rm --network none --cpus 4 --memory 8g --read-only --tmpfs /tmp \
  -v /path/to/pdfs:/input:ro \
  -v /path/to/output:/output \
  mib-intake /input /output/predictions.jsonl
```

The entrypoint accepts:

```text
<input_pdf_directory> <output_predictions_path>
```

Runtime is fully offline. The image uses PyMuPDF, Tesseract, NumPy, and
scikit-learn. It does not use an LLM, VLM, cloud OCR service, network request,
or API key.

## Design

1. PyMuPDF extracts visible text spans and identifies hidden, transparent, or
   off-crop content.
2. Tesseract processes pages without a reliable text layer using several page
   segmentation and orientation strategies.
3. Parsed values carry source provenance and trust ranks based on the field
   manual.
4. Closed-vocabulary fields use OCR-aware weighted edit distance.
5. Deterministic policy rules handle high-confidence evidence.
6. A small calibrated classifier refines probabilities for unresolved cases.
7. The final decision maximizes expected value under the challenge payoff
   matrix.

## Repository layout

| Path | Purpose |
| --- | --- |
| `mib/pdfio.py` | PDF span extraction and visibility classification |
| `mib/extract.py` | Evidence extraction with provenance |
| `mib/ocr.py` | Local OCR and scan parsing |
| `mib/lexicon.py` | OCR-aware vocabulary matching |
| `mib/pipeline.py` | Evidence resolution and prediction assembly |
| `mib/policy.py` | Policy paths, calibration, and expected-value decisions |
| `mib/features.py` | Evidence features for the learned adjudicator |
| `mib/model.py` | Model loading and inference |
| `mib/cli.py` | Parallel command-line entrypoint |
| `policy/` | Fitted lexicon, calibration, and model artifacts |
| `tools/` | Training, validation, and release checks |
| `tests/` | Unit and end-to-end tests |

## Development

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
PYTHONPATH=. .venv/bin/python -m pytest tests -q
```

Check dependency compatibility before building the image:

```bash
PYTHONPATH=. .venv/bin/python tools/check_env.py
```

Rebuild the extraction cache after changing parsing or feature code:

```bash
PYTHONPATH=. .venv/bin/python tools/build_cache.py \
  --pdf-dir ../mib-doc-challenge/data/train \
  --out scratch/cache_train.pkl
```

Train the adjudicator from the refreshed cache:

```bash
PYTHONPATH=. .venv/bin/python tools/train_adjudicator.py \
  --cache scratch/cache_train.pkl \
  --labels ../mib-doc-challenge/data/train_labels.csv \
  --out policy/adjudicator.joblib
```
