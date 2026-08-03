# mib-intake

Offline document extraction and adjudication pipeline for 8090's
*Intergalactic Intake* challenge.

The pipeline reads a directory of adversarial PDF case packets and writes one
JSONL prediction per packet. Each prediction contains nine extracted fields,
an `APPROVED`, `DENIED`, or `NEEDS_REVIEW` decision, and a calibrated
confidence.

On the public training set the image scores **137.89 / 150**.

| Section | Training set |
| --- | ---: |
| Extraction | 45.37 / 50 |
| Classification | 74.45 / 80 |
| Calibration | 18.07 / 20 |
| Mean confidence Brier | 0.0482 |
| Catastrophic false approvals | 12 |

This score is in-sample: the model was fitted on these packets, so it is a
reproducibility figure rather than an estimate of performance on unseen ones.
It is produced by the Docker image itself under the submission constraints, not
by a cached intermediate. See [MEMO.md](MEMO.md) for the held-out diagnostics
and their limits.

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

Runtime is fully offline. The image uses PyMuPDF, Tesseract, RapidOCR, NumPy,
and scikit-learn. It does not use an LLM, VLM, cloud OCR service, network
request, or API key.

## Design

1. PyMuPDF extracts visible text spans and identifies hidden, transparent, or
   off-crop content.
2. Tesseract processes pages without a reliable text layer using several page
   segmentation and orientation strategies. RapidOCR supplies a lower-trust
   second reading for fields Tesseract could not resolve.
3. Parsed values carry source provenance and trust ranks based on the field
   manual.
4. Closed-vocabulary fields use OCR-aware weighted edit distance.
5. Deterministic policy rules handle high-confidence evidence.
6. A small calibrated classifier refines probabilities for unresolved cases.
7. The final decision maximizes expected value under the challenge payoff
   matrix, with approval blocked when the packet's risk page is unreadable.

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
| `APPENDIX.md` | Extended engineering detail behind the memo |

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

## License

MIT. See [LICENSE](LICENSE).

Third-party components are used unmodified, as pinned dependencies: Tesseract
and pytesseract (Apache-2.0); `rapidocr-onnxruntime` (Apache-2.0), which bundles
the PaddleOCR PP-OCRv4 detection, classification and recognition weights
(Apache-2.0); the PaddleOCR PP-OCRv6 Small English detection and recognition
weights (Apache-2.0), redistributed by the RapidOCR project and vendored in
`policy/` so the runtime stays offline; ONNX Runtime (MIT); OpenCV (Apache-2.0);
PyMuPDF (AGPL-3.0 or Artifex commercial); Pillow (MIT-CMU); and scikit-learn,
NumPy and SciPy (BSD-3-Clause). No source, thresholds, tables, models, or
predictions from another challenge entrant are used.
