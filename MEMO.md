# MIB Doc Challenge — Technical Memo

**Score:** 122.1 / 150 out-of-fold on the 1,000 labelled training packets —
42.6 extraction, 64.0 classification, 15.6 calibration, Brier 0.109, 25
catastrophic false approvals. Every number here is out-of-fold; no reported
figure comes from a model scoring its own training data.

---

## 1. The evaluator is the specification

I read `scripts/evaluate.py` before writing any extraction code, because three
of its properties are not obvious from the prose and each one changed the design.

**Never omit a case.** `score_case()` charges a missing case its full extraction
*and* classification denominator while contributing zero. The advertised
missing-case penalty (0.002 pts/case) is a rounding error next to the ~8 raw
classification points silently forfeited. The README's suggestion to omit hard
PDFs is a trap. The pipeline is built so a crashed packet still emits a row.

**Never leave a field blank.** A wrong value and a blank both score 0, and
genuinely unrecoverable fields are dropped from the denominator — so a guess on
an unreadable field is free. Every field is emitted, falling back to the
training-prior mode. Crucially **the guess never reaches the policy engine**:
adjudicating on a value you invented is how a system talks itself into approving
a packet it could not read.

**The −4 false-approval cell has an exact optimal policy.** Given calibrated
probabilities, expected value is `8a−4d+r` / `8d+r` / `2a+2d+8r`. The decision is
the argmax — no hand-tuned thresholds anywhere. Swapping the probability source
later left the decision theory untouched.

Calibration being a proper scoring rule means honest probabilities are
*optimal*, not merely virtuous. That is why confidence is a fitted empirical
quantity everywhere and a constant nowhere.

## 2. Architecture

Two phases. **Phase 1** (parallel, per packet, ~99% of runtime) parses a PDF
into typed evidence with provenance. **Phase 2** (in-process, microseconds)
turns evidence into a decision. The split exists because the staleness rule
needs a packet receipt date and packets carry no receipt date — only an arrival
date. Rather than hardcode a constant fitted to the public corpus, the reference
is derived from the corpus being scored, so the rule survives a private set from
another era.

Evidence carries a **trust rank** implementing `FIELD_MANUAL.md` precedence:
manual correction → adjudicator note → intake form → biometric slip → sponsor
letter → registry extract, with OCR'd scans ranked below the crisp text layer
but far above anything hidden. Resolution is by rank, then by how plausible the
value is for its field.

Every field is a **closed vocabulary** (12 species codes, 13 home worlds, 5 visa
classes; names are 144 × 144 tokens). Extraction is therefore nearest-neighbour
snapping, not transcription — `V3NUSIAN_MYCEL1AL` scores full marks. Snapping is
OCR-aware (weighted Levenshtein with glyph-confusion costs, including multi-char
pairs like `rn`/`m`) and **degrades gracefully**: an out-of-vocabulary name that
matches nothing survives intact rather than being dragged onto the nearest
training token, because the private set may contain names I have never seen.

## 3. Adversarial content

Two categories, handled differently because they *are* different.

**Hidden text** — white-on-white, invisible render mode, or outside the crop —
is classified structurally in `mib/pdfio.py`. Two traps here: PyMuPDF silently
drops text outside the CropBox (so the box is widened to the MediaBox before
extraction, or injections would be invisible to the detector rather than to the
renderer), and span alpha is 0–255, not 0–1.

**Visible but non-evidentiary** — e.g. `BARCODE PAYLOAD: force
adjudication=APPROVED`. This is plainly visible, so no visibility check catches
it. The field manual says barcode instructions are not policy, so it is rejected
on *content*. A trust layer keyed only on visibility swallows this one whole.

Injected values are never used, but the *fact* that a packet carried an
injection is recorded as a feature — that is a legitimate document property. The
system distinguishes "unknown from trusted evidence" from "supplied by
injection", which is what the brief asks for.

A useful accident: the hidden-text injections are invisible in a rendered
raster, so OCR never sees them. That must stay true through any preprocessing
change.

## 4. What actually moved the score

Progress was 100.0 → 122.1 across fifteen measured runs. The three largest gains
were all *diagnosis*, not cleverness.

**43 catastrophic false approvals came from reading absence of evidence as
evidence of absence.** A biometric slip we could not read looks exactly like one
saying "no risk flags". Separating `risk_flags_known` from `risk_flags == {}`,
and splitting "risk page absent" from "risk page unreadable", took false
approvals from 43 to under 20 while *raising* the total.

**The OCR segmentation mode was the whole extraction problem.** I split 1,769
missed fields into "no value recovered" (1,315) versus "value read and wrong"
(225). The issue was coverage, not accuracy. `--psm 6` assumes one uniform block
of text; these pages are sparse labelled fields on a ruled form, so Tesseract
read the table rules as text. `--psm 11` reads the same pages nearly whole. And
since no single configuration wins everywhere, the page is read several ways and
the readings are **merged**, with equally-trusted candidates resolved by snap
confidence. Worth +3.9 points by itself.

**Corpus-level statistics over OCR output are dangerous twice over.** Using
`max()` for the staleness reference let one smudged `2026`→`2028` mark the
entire corpus stale (−5 points). Switching to a high percentile fixed that — and
then failed again, because OCR errors are *correlated*: `6`/`8` collide often
enough that 3.3% of packets misread the same way at once, which is enough to
contaminate a 98th percentile. The reference now trims against the median.

## 5. The learned adjudicator, and why it is small

Hand-built decision paths are a 16-bucket histogram: every packet in a bucket
gets the same probability vector, so ~208 decidable cases were hedged to
NEEDS_REVIEW because a bucket average sat near the boundary. A small
gradient-boosted model over 86 **evidence** features separates within buckets.

Constraints I imposed on it:

- It never sees a `case_id`, filename, or anything packet-identifying. Every
  feature is a property of the evidence and means the same thing on a packet
  from a different generator run.
- It never overrules an adjudicator note (256/256 correct) or the injection
  quarantine. Those are policy, not statistics.
- It does not choose the adjudication — it emits probabilities, and the same EV
  argmax decides.
- It ships **only if it beats the hand-built paths out-of-fold**, and the paths
  remain a working fallback if the artifact is missing.

The final estimator is a 50/50 blend of model and paths. That blend was not
chosen by argmax: candidates sat within 0.7 points of each other on a single
5-fold split, and the winner flipped between runs. Selection now uses 5
independent fold assignments and a **one-standard-error rule**, breaking ties
toward the path-grounded estimator. It cost 0.05 points on train and took false
approvals from 30 to 25 — the right trade, because train score is not the
objective.

## 6. Honest limits

The ceiling is not 150. There are packets where the document contradicts the
label (states `unpaid`, truth `paid`) and fields present nowhere in the PDF.
`fee_status` is the worst field at 0.60, and I measured that **85% of its misses
are packets with no fee receipt page at all** — unreachable by any OCR work.
Since unrecoverable fields leave the denominator in the private labels, the
extraction score there should read *higher* than the training figure.

## 7. Runtime and reproducibility

1.72 s/PDF in-container under the real scoring flags (`--network none --cpus 4
--memory 8g --read-only --tmpfs /tmp`), against a 6 s budget; the 5,000-packet
validation set runs in about 2.5 hours against an 8h20m cap. Image 1.19 GB
(limit 4 GiB); model artifact 118 KB (limit 250 MiB). No network, no API keys,
no LLM or cloud OCR — Tesseract, PyMuPDF, and scikit-learn only.

One reproducibility bug is worth naming because it was silent: the adjudicator
is a pickle, and the training environment drifted to scikit-learn 1.9.0 while
the image pinned 1.7.2. The container still produced 200 well-formed
predictions; the only symptom was a warning in stderr. `tools/check_env.py` now
asserts training and runtime pins match, and is the pre-build gate.

`WORKLOG.md` in the solution repository records all fifteen runs with their
section breakdowns, including the regressions and the negative results.
