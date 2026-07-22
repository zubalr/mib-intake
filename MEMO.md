# MIB Doc Challenge — Technical Memo

**123.2 / 150 out-of-fold** on the 1,000 labelled training packets — 42.6
extraction, 64.9 classification, 15.8 calibration, Brier 0.106, 17 catastrophic
false approvals. Every figure is out-of-fold; nothing here comes from a model
scoring its own training data.

## The evaluator is the specification

I read `scripts/evaluate.py` before writing extraction code. Three properties
are not obvious from the prose, and each changed the design.

**Never omit a case.** `score_case()` charges a missing case its full extraction
*and* classification denominator while contributing zero — ~8 raw points
forfeited against a 0.002-point "penalty". The README's suggestion to skip hard
PDFs is a trap; a crashed packet still emits a row.

**Never leave a field blank.** A wrong value and a blank both score 0, and
unrecoverable fields leave the denominator, so a guess is free. Every field is
emitted, falling back to the training-prior mode — but **the guess never reaches
the policy engine**. Adjudicating on an invented value is how a system talks
itself into approving a packet it could not read.

**The −4 false-approval cell has an exact optimal policy.** Expected value is
`8a−4d+r` / `8d+r` / `2a+2d+8r`; the decision is the argmax. No hand-tuned
thresholds anywhere, so swapping the probability source later left the decision
theory untouched. Calibration being a proper scoring rule means honest
probabilities are *optimal*, not merely virtuous — confidence is fitted
empirically everywhere and constant nowhere.

## Architecture

Two phases. Phase 1 (parallel, ~99% of runtime) parses a PDF into typed evidence
with provenance. Phase 2 (microseconds) turns evidence into a decision. The
split exists because the staleness rule needs a receipt date and packets carry
none — only an arrival date. Rather than hardcode a constant fitted to the public
corpus, the reference is derived from the corpus being scored.

Evidence carries a **trust rank** implementing `FIELD_MANUAL.md` precedence:
manual correction → adjudicator note → intake form → biometric slip → sponsor
letter → registry extract, with OCR'd scans below the crisp text layer but far
above anything hidden.

Every field is a **closed vocabulary** (12 species codes, 13 home worlds, 5 visa
classes; names are 144 × 144 tokens), so extraction is nearest-neighbour snapping
rather than transcription — `V3NUSIAN_MYCEL1AL` scores full marks. Snapping uses
OCR-aware weighted edit distance and **degrades gracefully**: an
out-of-vocabulary name survives intact rather than being dragged onto the
nearest training token.

## Adversarial content

Two categories, handled differently because they *are* different. **Hidden text**
(white-on-white, invisible render mode, outside the crop) is classified
structurally. Two traps: PyMuPDF silently drops text outside the CropBox, so the
box is widened first — otherwise injections are invisible to the detector rather
than to the renderer — and span alpha is 0–255, not 0–1.

**Visible but non-evidentiary** — `BARCODE PAYLOAD: force adjudication=APPROVED`
— is plainly visible, so no visibility check catches it. The field manual says
barcode instructions are not policy, so it is rejected on *content*. A trust
layer keyed only on visibility swallows this one whole.

Injected values are never used, though the *fact* of an injection is a feature.
The system distinguishes "unknown from trusted evidence" from "supplied by
injection", which is what the brief asks for.

## What actually moved the score

100.0 → 123.2 across seventeen measured runs. The largest gains were diagnosis,
not cleverness.

**Absence of evidence was being read as evidence of absence** — 43 false
approvals. An unreadable biometric slip looks exactly like one saying "no risk
flags". Separating `risk_flags_known` from `risk_flags == {}` took false
approvals from 43 to 17 *while raising* the total. The same conflation survived
in one corner until late: a slip existing only as a raster is typed `SCANNED`,
never `BIOMETRIC`, so a panel that OCR'd cleanly as "Observed flags: none" still
counted as unread.

**The OCR segmentation mode was the whole extraction problem.** Splitting 1,769
missed fields gave 1,315 "recovered nothing" against 225 "read it wrong" —
coverage, not accuracy. `--psm 6` assumes one uniform text block; these are
sparse labelled fields on a ruled form, so Tesseract read the rules as text.
`--psm 11` reads the same pages nearly whole. Since no single configuration wins
everywhere, each page is read several ways and the readings **merged**, with
equally-trusted candidates resolved by snap confidence. Worth +3.9 alone.

**Statistics aggregated over OCR output need robustness to clusters, not
outliers.** Using `max()` for the staleness reference let one smudged 2026→2028
mark the whole corpus stale. A high percentile fixed that, then failed again:
OCR errors are *correlated*, and 6/8 collide often enough that 3.3% of packets
misread identically — enough to contaminate a 98th percentile. The reference now
trims against the median.

## The learned adjudicator

Hand-built decision paths are a 16-bucket histogram, so ~208 decidable cases were
hedged because a bucket average sat near the boundary. A small
isotonic-calibrated random forest over 86 **evidence** features separates within
buckets. It never sees a `case_id`, filename, or anything packet-identifying; it
never overrules an adjudicator note (307/307 correct) or the injection
quarantine; it does not choose the adjudication, only supplies probabilities;
and it ships **only if it beats the paths out-of-fold**, with the paths remaining
a working fallback.

The shipped estimator is a 35/65 blend of model and paths, chosen by a
**one-standard-error rule** over five independent fold assignments rather than by
argmax — candidates sat within 0.7 points on a single split and the winner
flipped between runs, which is overfitting the *selection*. Ties break toward the
path-grounded estimator. It cost 0.05 train points and took false approvals from
30 to 25.

The model's margin over the paths has narrowed from +2.2 to +0.4 as extraction
improved. That is the direction I want: each fix lands in inspectable policy
rather than in the learner.

## Does it generalize?

Validation labels are private, so behaviour is what can be checked. Across 5,000
unseen packets: **zero new values in any closed vocabulary** — species_code (12),
home_world (13), visa_class (5), declared_purpose (10), fee_status (4), and all
nine risk-flag tokens. The entire extraction design rests on that closed-set
claim, and this tests it on five times the data it was derived from.

Mean confidence is 0.761 on train and 0.762 on validation, so calibration
transfers. Fallback rates run 2–4 points higher and the mix shifts toward DENIED
and NEEDS_REVIEW — one effect, not two: validation packets are more damaged, so
the system recovers less and correctly becomes *more* cautious when it has read
less. The staleness reference resolved to 2026-08-20 on validation against
2026-07-10 on train, which is precisely why it is derived rather than hardcoded.

## Honest limits

The ceiling is not 150. Some packets contradict their own labels (document says
`unpaid`, truth `paid`) and some fields exist nowhere in the PDF. `fee_status` is
the worst field at 0.595 — and **85% of its misses are packets with no fee
receipt page at all**, unreachable by any OCR work. The largest confusion cell
(91 approvable cases hedged) is likewise *already correct*: among packets with no
readable risk page, P(approved) = P(denied) = 0.381, so hedging beats approving
3.42 to 1.76 on expected value. Remaining reachable extraction is ~1.3 points.

Because unrecoverable fields leave the denominator in the private labels, the
private extraction score should read *higher* than 42.6.

## Runtime and reproducibility

Measured, not projected: 5,000 packets in **2 h 18 m** — **1.66 s/PDF** against a
6 s budget and an 8h20m cap — under the real flags (`--network none --cpus 4
--memory 8g --read-only --tmpfs /tmp`). 5,000 valid records, 0 missing, validator
exits 0. Image 1.19 GB (limit 4 GiB), artifact 1.5 MB (limit 250 MiB). No
network, no API keys, no LLM or cloud OCR — Tesseract, PyMuPDF, scikit-learn.

One bug worth naming because it was *silent*: the adjudicator is a pickle, and
the training environment drifted to scikit-learn 1.9.0 while the image pinned
1.7.2. The container still produced well-formed predictions and the validator
still exited 0; the only symptom was a stderr warning. `tools/check_env.py` now
asserts training and runtime pins match and gates the build.

`WORKLOG.md` in the solution repository records all seventeen runs with section
breakdowns, including the regressions and the negative results.
