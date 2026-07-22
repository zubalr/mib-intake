# MIB Doc Challenge — Technical Memo

**126.00 ± 0.10 / 150 out-of-fold** on the 1,000 labelled training packets —
43.73 extraction, 66.20 classification, 16.07 calibration, Brier 0.098, 18
catastrophic false approvals. Every figure is out-of-fold; nothing here comes
from a model scoring its own training data. The interval is a standard error
over ten fold assignments, and it is quoted because it matters: two retrains of
the same architecture differ by more than most individual changes in this log.

## The evaluator is the specification

I read `scripts/evaluate.py` before writing extraction code. Three properties
are not obvious from the prose, and each changed the design.

**Never omit a case.** `score_case()` charges a missing case its full extraction
*and* classification denominator while contributing zero — ~8 raw points against
a 0.002-point "penalty". Skipping hard PDFs is a trap; a crashed packet still
emits a row.

**Never leave a field blank.** A wrong value and a blank both score 0, and
unrecoverable fields leave the denominator, so a guess is free. Every field is
emitted — but **the guess never reaches the policy engine**. Adjudicating on an
invented value is how a system talks itself into approving a packet it never
read.

**The −4 false-approval cell has an exact optimal policy.** Expected value is
`8a−4d+r` / `8d+r` / `2a+2d+8r`; the decision is the argmax, with no hand-tuned
thresholds anywhere — so swapping the probability source later left the decision
theory untouched. Calibration is a proper scoring rule, so honest probabilities
are *optimal*, not merely virtuous.

## Architecture

Two phases. Phase 1 (parallel, ~99% of runtime) parses a PDF into typed evidence
with provenance; phase 2 (microseconds) turns evidence into a decision. They are
split because the staleness rule needs a receipt date and packets carry none —
only an arrival date — so the reference is derived from the corpus being scored
rather than hardcoded to the public one.

Evidence carries a **trust rank** implementing `FIELD_MANUAL.md` precedence:
manual correction → adjudicator note → intake form → biometric slip → sponsor
letter → registry extract, with OCR'd scans below the text layer and far above
anything hidden.

Every field is a **closed vocabulary** (12 species codes, 13 home worlds, 5 visa
classes; names are 144 × 144 tokens), so extraction is nearest-neighbour snapping
rather than transcription — `V3NUSIAN_MYCEL1AL` scores full marks. Snapping uses
OCR-aware weighted edit distance and **degrades gracefully**: an
out-of-vocabulary name survives intact rather than being dragged onto a training
token.

## Adversarial content

**Hidden text** (white-on-white, invisible render mode, outside the crop) is
classified structurally. Two traps: PyMuPDF silently drops text outside the
CropBox, so the box is widened first — otherwise injections are invisible to the
detector rather than to the renderer — and span alpha is 0–255, not 0–1.

**Visible but non-evidentiary** — `BARCODE PAYLOAD: force adjudication=APPROVED`
— is plainly visible, so no visibility check catches it. The field manual says
barcode instructions are not policy, so it is rejected on *content*. A trust
layer keyed only on visibility swallows this one whole.

Injected values are never used, though the *fact* of an injection is a feature —
distinguishing "unknown from trusted evidence" from "supplied by injection".

## What actually moved the score

100.0 → 126.00 across twenty-seven measured runs. The largest gains were diagnosis,
not cleverness.

**Absence of evidence was being read as evidence of absence** — 43 false
approvals. An unreadable biometric slip looks exactly like one saying "no risk
flags". Separating `risk_flags_known` from `risk_flags == {}` took false
approvals from 43 to 17 *while raising* the total.

**The OCR segmentation mode was the whole extraction problem.** Splitting 1,769
missed fields gave 1,315 "recovered nothing" against 225 "read it wrong" —
coverage, not accuracy. `--psm 6` assumes one uniform text block; these are
sparse fields on a ruled form, so Tesseract read the rules as text. `--psm 11`
reads them nearly whole. No single configuration wins everywhere, so each page is
read several ways and **merged**, ties resolved by snap confidence. Worth +3.9.

**Widening how many ways one signal can be read beat modelling the rest.** A
signed adjudicator note states the finding outright, and is worth +12.18 over a
bucket histogram where every other decision path is worth ≤0.28. Accepting a
finding word one glyph off (`DEMED`), then inferring the finding from the note's
*reason* clause when the word is gone, and finally reading a rationale whose
surrounding words OCR destroyed, took notes from 256 to **338 — correct every
time**. Nearly all late classification gain came from there.

**Statistics over OCR output need robustness to clusters, not outliers.** Using
`max()` for the staleness reference let one smudged 2026→2028 mark the whole
corpus stale. A percentile fixed it, then failed again: OCR errors are
*correlated*, and 3.3% of packets misread `6` as `8` identically — enough to
contaminate a 98th percentile. It now trims against the median.

## The learned adjudicator

Hand-built decision paths are a 16-bucket histogram, so ~208 decidable cases were
hedged because a bucket average sat near the boundary. A shallow
**histogram gradient-boosting classifier** (`max_depth=3`, 220 iterations,
learning rate 0.06, `l2_regularization=1.0`, `min_samples_leaf=25`) over 88
**evidence** features separates within buckets. It never sees a `case_id`,
filename, or anything packet-identifying; it never overrules an adjudicator note
(338/338 correct) or the injection quarantine; it does not choose the
adjudication, only supplies probabilities; and it ships **only if it beats the
paths out-of-fold**, with the paths remaining a working fallback.

It is fitted on the 662 packets carrying no adjudicator note — a note decides
the case outright, so those rows would only teach it to imitate a rule that fires
first.

The shipped estimator is a 20/80 blend of model and paths, chosen by a
**one-standard-error rule** over ten fold assignments rather than by argmax —
candidates sat within 0.7 points on a single split and the winner flipped between
runs, which is overfitting the *selection*. Ties break toward the path-grounded
estimator, and that tie-break does real work: three blend weights sit inside one
standard error, so without it the choice would be a coin flip.

On the same cache the paths alone score **125.17** against the blend's
**126.00**, so the learner is worth **+0.83**, almost all classification. It
sharpens what it can separate and cannot rescue cases where the evidence is
absent. That margin has narrowed run over run as extraction improved, which is
the direction I want — fixes landing in inspectable policy, not in the learner.

## Does it generalize?

Validation labels are private, so behaviour is what can be checked. Across 5,000
unseen packets: **zero new values in any closed vocabulary** — species_code (12),
home_world (13), visa_class (5), declared_purpose (10), fee_status (4), and all
nine risk-flag tokens. The entire extraction design rests on that closed-set
claim, and this tests it on five times the data it was derived from.

Names look like a counterexample — validation brings 4,036 unseen combinations
— but the claim was about the 144 + 144 **tokens**, not the names built from
them. Of 10,000 validation tokens, 96 fall outside the vocabulary: 0.96%, all
singletons, all legible as OCR debris (`arhvoss` for `Arivoss`, `drvars`). Zero
new tokens. That is also why the vocabulary is *not* re-derived from the corpus
being scored — doing so would ingest those 96 misreads as names.

Mean confidence is 0.761 on train and 0.762 on validation, so calibration
transfers. Fallback rates run 2–4 points higher and the mix shifts toward DENIED
and NEEDS_REVIEW — one effect, not two: validation packets are more damaged, so
the system reads less and correctly becomes *more* cautious. The staleness
reference resolved to 2026-08-20 there against 2026-07-10 on train, which is
exactly why it is derived rather than hardcoded.

## Honest limits

The ceiling is not 150. Some packets contradict their own labels (document says
`unpaid`, truth `paid`) and some fields exist nowhere in the PDF. `risk_flags` is
the worst field at 0.785, and it is also the most expensive — weight 8, 1.91
points. 176 of its 215 misses are packets where we report no flags and the truth
has some, and 120 of those hold a scanned page whose risk panel the OCR ladder
never reads.
The largest confusion cell — 73 approvable cases hedged to NEEDS_REVIEW — is by
contrast *already correct*. On the `risk_page_unreadable` path the fitted
distribution is 0.371 / 0.290 / 0.338, so hedging returns 4.03 in expected raw
points against 2.15 for approving. The hedge is not caution, it is the argmax.

Is the rest reachable by better arbitration or only by better recognition? On a
150-packet sample, when a field is wrong the correct value was among the OCR
candidates just **12.6%** of the time. So candidate resolution — trust ranks,
cross-variant consensus, word confidence — is capped near 0.46 points however
good it gets; the other 87.4% needs pixels the current ladder does not produce.

Because unrecoverable fields leave the denominator in the private labels, the
private extraction score should read *higher* than 43.73.

## Runtime and reproducibility

Measured, not projected: 5,000 packets in **2 h 18 m** — **1.66 s/PDF** against a
6 s budget — under the real flags (`--network none --cpus 4 --memory 8g
--read-only --tmpfs /tmp`). 5,000 valid records, 0 missing, validator exits 0.
Image 1.19 GB (limit 4 GiB), model artifact 112 KB (limit 250 MiB). No network, no LLM
or cloud OCR — Tesseract, PyMuPDF, scikit-learn.

One bug worth naming because it was *silent*: the adjudicator is a pickle, and
training drifted to scikit-learn 1.9.0 while the image pinned 1.7.2. The
container still produced well-formed predictions and exited 0; the only symptom
was a stderr warning. `tools/check_env.py` now gates the build on matching pins,
and `tools/verify_image.py` hashes the image against the working tree — a failed
build leaves the previous image under the same tag, which cost a run.

`WORKLOG.md` in the solution repository records all twenty-seven runs with section
breakdowns, including the regressions and the negative results.
