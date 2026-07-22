# Submission — zubalr

**Solution repository:** https://github.com/zubalr/mib-intake

Public, Dockerfile-based. The image entrypoint accepts exactly
`<input_pdf_dir> <output_predictions_path>` and runs fully offline.

```bash
docker build -t mib-intake .
```

```bash
docker run --rm --network none --cpus 4 --memory 8g --read-only --tmpfs /tmp \
  -v /path/to/pdfs:/input:ro -v /path/to/out:/output \
  mib-intake /input /output/predictions.jsonl
```

## Contents

| File | What it is |
| --- | --- |
| `predictions.jsonl` | Validation-set predictions (5,000 rows) |
| `MEMO.md` | Technical memo |
| `SUBMISSION.md` | This file |

## Summary

| | |
| --- | --- |
| Out-of-fold score on train | **126.17 / 150** |
| Extraction / classification / calibration | 43.64 / 66.42 / 16.12 |
| Mean Brier | 0.097 |
| Catastrophic false approvals | 16 / 1,000 |
| Runtime | 1.77 s/PDF in-container (budget 6 s) |
| Image size | 1.19 GB (limit 4 GiB) |
| Model artifact | 1.5 MB (limit 250 MiB) |

Approach: trust-ranked evidence extraction implementing `FIELD_MANUAL.md`
precedence, closed-vocabulary snapping with OCR-aware edit distance, structural
quarantine of hidden and non-evidentiary text, and an expected-value decision
rule over the evaluator's own payoff matrix. Probabilities come from a blend of
hand-built decision paths and a shallow histogram gradient-boosting classifier
trained only on evidence features — no `case_id`, filename, or packet-identifying input.

No LLMs, VLMs, or cloud OCR anywhere in the runtime: Tesseract, PyMuPDF and
scikit-learn only.

See `MEMO.md` for the reasoning, and `WORKLOG.md` in the solution repository for
the full run-by-run history including regressions and negative results.
