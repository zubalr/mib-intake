#!/usr/bin/env python3
"""Train the learned adjudicator and prove it beats the hand-built paths.

Method, and why each piece is there:

* **Out-of-fold probabilities.** Every reported number comes from
  `cross_val_predict`, so a packet's probability is produced by a model that
  never saw it. In-sample probabilities on 1,000 rows would look excellent and
  mean nothing -- and because the calibration term is a Brier score, an
  overconfident in-sample fit is actively punished on the real set.
* **Scored with the challenge's own objective**, not accuracy or log-loss. A
  model with better accuracy can easily score *worse* once the -4 false-approval
  cell and the 20-point Brier term are applied, so the selection metric is the
  actual 100-point classification+calibration total.
* **Note cases excluded from training.** An adjudicator note settles the case by
  rule and the model never runs on those, so training on them would fit a
  population the model never sees.
* **Compared head-to-head against the hand-built paths on identical folds.**
  The model only ships if it wins out-of-fold. Otherwise the paths stay.

Usage:
    PYTHONPATH=. python tools/train_adjudicator.py \\
        --cache scratch/cache_train.pkl \\
        --labels ../mib-doc-challenge/data/train_labels.csv \\
        --out policy/adjudicator.joblib
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mib.model import Adjudicator
from mib.policy import OUTCOMES, PAYOFF, Calibration, decide, decision_path
from tools.build_cache import load_cache

CLASSIFICATION_POINTS = 80.0
CALIBRATION_POINTS = 20.0
CLASSIFICATION_MAX_RAW = 8.0


def challenge_score(truths: list[str], probs: np.ndarray,
                    classes: list[str]) -> dict:
    """Score a probability matrix exactly as scripts/evaluate.py would."""
    raw = 0.0
    briers = []
    cfa = 0
    correct = 0
    for truth, row in zip(truths, probs):
        pr = {cls: float(v) for cls, v in zip(classes, row)}
        for outcome in OUTCOMES:
            pr.setdefault(outcome, 0.0)
        adjudication, confidence = decide(pr)
        raw += PAYOFF[adjudication][truth]
        is_correct = adjudication == truth
        correct += is_correct
        briers.append((confidence - (1.0 if is_correct else 0.0)) ** 2)
        cfa += int(truth == "DENIED" and adjudication == "APPROVED")

    n = len(truths)
    mean_brier = sum(briers) / n
    return {
        "classification": CLASSIFICATION_POINTS * raw / (CLASSIFICATION_MAX_RAW * n),
        "calibration": CALIBRATION_POINTS * max(0.0, 1.0 - 2.0 * mean_brier),
        "accuracy": correct / n,
        "cfa": cfa,
        "brier": mean_brier,
        "n": n,
    }


def path_baseline(records, truths, calibration: Calibration) -> dict:
    """The hand-built paths scored on the same subset, for comparison."""
    probs = []
    for record in records:
        pr = calibration.probs(decision_path(record))
        probs.append([pr.get(o, 0.0) for o in OUTCOMES])
    return challenge_score(truths, np.array(probs), list(OUTCOMES))


CANDIDATES = {
    # Few rows (~780) and 70+ features, so every candidate is regularised hard.
    "logreg": lambda: make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.3, max_iter=3000)),
    "hgb_shallow": lambda: HistGradientBoostingClassifier(
        max_depth=3, max_iter=220, learning_rate=0.06,
        min_samples_leaf=25, l2_regularization=1.0, random_state=0),
    "hgb_tiny": lambda: HistGradientBoostingClassifier(
        max_depth=2, max_iter=160, learning_rate=0.08,
        min_samples_leaf=40, l2_regularization=2.0, random_state=0),
    "rf": lambda: RandomForestClassifier(
        n_estimators=500, max_depth=8, min_samples_leaf=8,
        random_state=0, n_jobs=-1),
    # Raw tree-ensemble probabilities are poorly calibrated -- averaging votes
    # pulls them toward the middle. Uncalibrated `rf` beat the hand-built paths
    # on classification (56.63 vs 56.35) and on false approvals (15 vs 18) yet
    # lost overall purely on the Brier term (12.84 vs 13.68). Since calibration
    # is a fifth of the total score, fixing the probabilities is worth more than
    # any further accuracy. Both wrappers do their own internal CV, so wrapping
    # them in cross_val_predict keeps the evaluation properly nested.
    "rf_isotonic": lambda: CalibratedClassifierCV(
        RandomForestClassifier(n_estimators=400, max_depth=8, min_samples_leaf=8,
                               random_state=0, n_jobs=-1),
        method="isotonic", cv=4),
    "rf_sigmoid": lambda: CalibratedClassifierCV(
        RandomForestClassifier(n_estimators=400, max_depth=8, min_samples_leaf=8,
                               random_state=0, n_jobs=-1),
        method="sigmoid", cv=4),
    "hgb_isotonic": lambda: CalibratedClassifierCV(
        HistGradientBoostingClassifier(max_depth=3, max_iter=200,
                                       learning_rate=0.06, min_samples_leaf=25,
                                       l2_regularization=1.0, random_state=0),
        method="isotonic", cv=4),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", required=True, type=Path)
    ap.add_argument("--labels", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--oof-out", type=Path, default=None,
                    help="Write an honest predictions.jsonl built from "
                         "out-of-fold probabilities, for official scoring.")
    args = ap.parse_args()

    with open(args.labels, newline="") as f:
        truth = {r["case_id"]: r["adjudication"].strip() for r in csv.DictReader(f)}

    cache = load_cache(args.cache)
    rows = [r for r in cache["rows"] if not r["failed"]]

    # Apply the corpus staleness reference exactly as the pipeline does, then
    # refresh the temporal features. Skipping this trains the model on a dead
    # stale_margin and a path label computed from receipt_date=None -- i.e. on a
    # system that never runs.
    from mib.cli import corpus_reference_date
    from mib.features import refresh_temporal
    reference = corpus_reference_date([r["record"] for r in rows])
    print(f"staleness reference: {reference}")
    for row in rows:
        if row["record"].receipt_date is None:
            row["record"].receipt_date = reference
        refresh_temporal(row["features"], row["record"])

    # The model only ever runs where no note settled the case.
    trainable = [r for r in rows if r["note"] is None]
    print(f"packets: {len(rows)}  note-settled: {len(rows) - len(trainable)}  "
          f"trainable: {len(trainable)}")

    feature_names = sorted({k for r in trainable for k in r["features"]})
    X = np.array([[float(r["features"].get(n, 0.0)) for n in feature_names]
                  for r in trainable])
    y = np.array([truth[r["case_id"]] for r in trainable])
    records = [r["record"] for r in trainable]
    print(f"features: {len(feature_names)}   class mix: "
          f"{dict(zip(*np.unique(y, return_counts=True)))}")

    baseline = path_baseline(records, list(y), Calibration())
    print(f"\n{'model':14s} {'class/80':>9s} {'calib/20':>9s} {'sum/100':>8s} "
          f"{'acc':>6s} {'CFA':>4s}")
    print(f"{'paths (base)':14s} {baseline['classification']:9.2f} "
          f"{baseline['calibration']:9.2f} "
          f"{baseline['classification'] + baseline['calibration']:8.2f} "
          f"{baseline['accuracy']:6.3f} {baseline['cfa']:4d}")

    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=0)
    results = {}
    for name, factory in CANDIDATES.items():
        model = factory()
        oof = cross_val_predict(model, X, y, cv=cv, method="predict_proba", n_jobs=1)
        classes = list(np.unique(y))
        score = challenge_score(list(y), oof, classes)
        results[name] = score
        print(f"{name:14s} {score['classification']:9.2f} {score['calibration']:9.2f} "
              f"{score['classification'] + score['calibration']:8.2f} "
              f"{score['accuracy']:6.3f} {score['cfa']:4d}")

    # -- Blend the best model with the hand-built path probabilities -------
    #
    # These are two genuinely different estimators: the paths are a low-variance
    # 16-bucket histogram grounded in the field manual, the model is a
    # high-variance learner that separates *within* those buckets. Averaging two
    # such sources usually calibrates better than either alone, and it directly
    # attacks the failure mode seen here -- rf_isotonic bought +0.94 points but
    # took catastrophic false approvals from 18 to 33, because it is confident
    # in the APPROVED direction exactly where the paths are cautious.
    top = max(results, key=lambda k: results[k]["classification"] + results[k]["calibration"])
    path_probs = np.array([[Calibration().probs(decision_path(r))[o] for o in OUTCOMES]
                           for r in records])
    top_oof = cross_val_predict(CANDIDATES[top](), X, y, cv=cv,
                                method="predict_proba", n_jobs=1)
    top_classes = list(np.unique(y))
    # Re-order model columns to OUTCOMES so the two matrices are comparable.
    order = [top_classes.index(o) for o in OUTCOMES]
    top_aligned = top_oof[:, order]

    print()
    for weight in (0.2, 0.35, 0.5, 0.65, 0.8):
        blended = weight * top_aligned + (1.0 - weight) * path_probs
        score = challenge_score(list(y), blended, list(OUTCOMES))
        name = f"blend{weight:.2f}"
        results[name] = score
        results[name]["_blend"] = (top, weight)
        print(f"{name:14s} {score['classification']:9.2f} {score['calibration']:9.2f} "
              f"{score['classification'] + score['calibration']:8.2f} "
              f"{score['accuracy']:6.3f} {score['cfa']:4d}")

    best_name = max(results, key=lambda k: results[k]["classification"] + results[k]["calibration"])
    best = results[best_name]
    base_sum = baseline["classification"] + baseline["calibration"]
    best_sum = best["classification"] + best["calibration"]

    print(f"\nbest: {best_name}  ({best_sum:.2f} vs paths {base_sum:.2f}, "
          f"{best_sum - base_sum:+.2f} on the trainable subset)")

    if best_sum <= base_sum:
        print("Model does not beat the hand-built paths out-of-fold. NOT saving.")
        return

    blend_info = results[best_name].get("_blend")
    base_name, blend_weight = blend_info if blend_info else (best_name, 1.0)
    final = CANDIDATES[base_name]()
    final.fit(X, y)
    adjudicator = Adjudicator(final, feature_names, blend_weight=blend_weight,
                              metadata={
        "kind": base_name,
        "blend_weight": blend_weight,
        "oof_classification": round(best["classification"], 3),
        "oof_calibration": round(best["calibration"], 3),
        "oof_accuracy": round(best["accuracy"], 4),
        "oof_cfa": best["cfa"],
        "n_train": len(trainable),
        "folds": args.folds,
        "paths_baseline_sum": round(base_sum, 3),
    })
    adjudicator.save(args.out)
    size_kb = Path(args.out).stat().st_size / 1024
    print(f"Saved {args.out} ({size_kb:.0f} KB) -- limit is 250 MiB")

    if args.oof_out:
        _write_oof_predictions(args, cache, rows, trainable, base_name,
                               feature_names, X, y, cv, blend_weight)


def _write_oof_predictions(args, cache, rows, trainable, best_name,
                           feature_names, X, y, cv, blend_weight=1.0) -> None:
    """Write predictions using out-of-fold probabilities.

    Scoring the refit model on its own training data overstates the result --
    the first such run read 118.34 where the honest number is lower. This
    rebuilds the same submission using only probabilities from models that never
    saw the packet, so `scripts/evaluate.py` reports something believable.
    """
    import json
    from mib.cli import _enforce_output_schema, corpus_reference_date, fallback_prediction
    from mib.lexicon import Lexicon
    from mib.pipeline import finalize

    oof = cross_val_predict(CANDIDATES[best_name](), X, y, cv=cv,
                            method="predict_proba", n_jobs=1)
    classes = list(np.unique(y))
    calibration = Calibration()
    by_case = {}
    for row, probs in zip(trainable, oof):
        model_p = {c: float(v) for c, v in zip(classes, probs)}
        if blend_weight < 1.0:
            prior = calibration.probs(decision_path(row["record"]))
            model_p = {o: blend_weight * model_p.get(o, 0.0)
                          + (1.0 - blend_weight) * prior.get(o, 0.0)
                       for o in OUTCOMES}
        by_case[row["case_id"]] = model_p
    lexicon = Lexicon()
    good = [r for r in rows if not r["failed"]]
    reference = corpus_reference_date([r["record"] for r in good])

    out = []
    for row in cache["rows"]:
        if row["failed"]:
            out.append(fallback_prediction(row["case_id"], lexicon, calibration,
                                           row.get("reason", "?")).to_row())
            continue
        record = row["record"]
        if record.receipt_date is None:
            record.receipt_date = reference
        probs = by_case.get(row["case_id"])
        if probs is None:
            # Note-settled: decided by rule, no model involved.
            pred = finalize(row["printed"], record, row["note"], calibration)
        else:
            adjudication, confidence = decide(probs)
            pred = finalize(row["printed"], record, None, calibration)
            pred.adjudication = adjudication
            pred.confidence = confidence
        out.append(pred.to_row())

    _enforce_output_schema(out)
    args.oof_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.oof_out, "w") as f:
        for row in out:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"Wrote out-of-fold predictions to {args.oof_out}")


if __name__ == "__main__":
    main()
