# Submission: zubalr

**Solution repository:** https://github.com/zubalr/mib-intake

The repository contains a Docker-based, fully offline solution. Its entrypoint
accepts an input PDF directory and an output JSONL path:

```bash
docker build -t mib-intake .
```

```bash
docker run --rm --network none --cpus 4 --memory 8g --read-only --tmpfs /tmp \
  -v /path/to/pdfs:/input:ro \
  -v /path/to/output:/output \
  mib-intake /input /output/predictions.jsonl
```

## Public-data estimate

| Section | Score |
| --- | ---: |
| Extraction | 44.01 / 50 |
| Classification | 67.09 / 80 |
| Calibration | 16.27 / 20 |
| Total | **127.38 ± 0.08 / 150** |
| Mean confidence Brier | 0.0931 |

All reported score estimates are out of fold.

## Approach

The solution combines:

- trust-ranked evidence extraction based on the field manual;
- local Tesseract OCR for scanned pages;
- OCR-aware matching for closed-vocabulary fields;
- structural quarantine of hidden and non-evidentiary text;
- deterministic rules for high-confidence policy evidence;
- a compact calibrated classifier for unresolved cases; and
- expected-value adjudication using the published payoff matrix.

The classifier uses document-evidence features only. It does not receive case
IDs, filenames, hidden answer content, or packet-identifying inputs.

No LLM, VLM, cloud OCR service, network request, or API key is used.

See `MEMO.md` for technical details and reproducibility notes.
