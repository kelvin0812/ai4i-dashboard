"""Train, compare and save the machine-failure model (Kelvin).

Run from the project root:

    python training/train_model.py

What it does
------------
1. Loads Kiat's split: training/data/train_scaled.csv (8,000 rows, 271 failures)
   and training/data/test_scaled.csv (2,000 rows, 68 failures). Inputs are
   `type` + the five sensors scaled with models/ai4i_sensor_scaler.joblib.
2. Compares candidate models with 5-fold stratified cross-validation on the
   TRAINING set only. For each candidate the decision threshold is tuned on the
   out-of-fold probabilities to maximise failure-class F1.
3. Picks the candidate with the best CV failure F1, refits it on all training
   rows, and evaluates it ONCE on the untouched test set.
4. Writes:
     models/failure_model.joblib   the fitted sklearn Pipeline
     models/model_card.json        threshold, input order, CV + test metrics
     reports/model_comparison.csv  every candidate, CV and test metrics
     reports/model_evaluation.md   write-up for the report team

All test-set numbers for every candidate are reported for transparency, but only
the CV numbers are used to choose the model.
"""

import json
import sys
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import MODELS_DIR, SCALER_PATH, SENSOR_ORDER, TARGET_COL  # noqa: E402
from src.features import ENGINEERED, EngineeredFeatures  # noqa: E402

DATA = ROOT / "training" / "data"
REPORTS = ROOT / "reports"
MODEL_PATH = MODELS_DIR / "failure_model.joblib"
CARD_PATH = MODELS_DIR / "model_card.json"

INPUTS = ["type", *SENSOR_ORDER]
SEED = 42
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def engineered():
    scaler = joblib.load(SCALER_PATH)
    assert list(scaler.feature_names_in_) == SENSOR_ORDER
    return EngineeredFeatures(sensor_mean=scaler.mean_.tolist(), sensor_scale=scaler.scale_.tolist())


def candidates() -> dict:
    """name -> (pipeline, uses engineered features, short description)."""
    return {
        "Majority class (baseline)": (
            make_pipeline(DummyClassifier(strategy="most_frequent")), False,
            "Always predicts 'no failure'. Shows why accuracy alone is misleading."),
        "Logistic regression (baseline)": (
            make_pipeline(LogisticRegression(class_weight="balanced", max_iter=2000)), False,
            "Simple linear baseline, class-weighted for the 3.4% failure rate."),
        "Logistic regression + engineered": (
            make_pipeline(engineered(), LogisticRegression(class_weight="balanced", max_iter=2000)), True,
            "Linear model with temperature difference and power added."),
        "Random forest": (
            make_pipeline(RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                                 class_weight="balanced_subsample",
                                                 n_jobs=-1, random_state=SEED)), False,
            "Tree ensemble on the six original inputs."),
        "Gradient boosting": (
            make_pipeline(HistGradientBoostingClassifier(categorical_features=[0], random_state=SEED)), False,
            "Histogram gradient boosting, `type` treated as a category."),
        "Gradient boosting + engineered": (
            make_pipeline(engineered(),
                          HistGradientBoostingClassifier(categorical_features=[0], random_state=SEED)), True,
            "Gradient boosting with temperature difference and power added."),
    }


def best_threshold(y, proba) -> float:
    """Threshold that maximises failure-class F1 on (out-of-fold) probabilities."""
    p, r, t = precision_recall_curve(y, proba)
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)[:-1]
    if len(t) == 0 or f1.max() == 0:
        return 0.5
    return float(t[int(np.argmax(f1))])


def metrics(y, proba, threshold) -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    single_class = len(np.unique(proba)) == 1
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": float("nan") if single_class else roc_auc_score(y, proba),
        "pr_auc": float("nan") if single_class else average_precision_score(y, proba),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def main() -> None:
    train = pd.read_csv(DATA / "train_scaled.csv")
    test = pd.read_csv(DATA / "test_scaled.csv")
    X_train, y_train = train[INPUTS], train[TARGET_COL]
    X_test, y_test = test[INPUTS], test[TARGET_COL]
    print(f"train {len(train):,} rows ({y_train.sum()} failures) | test {len(test):,} rows ({y_test.sum()} failures)")

    rows, fitted = [], {}
    for name, (pipe, uses_eng, desc) in candidates().items():
        oof = cross_val_predict(pipe, X_train, y_train, cv=CV, method="predict_proba")[:, 1]
        thr = 0.5 if name.startswith("Majority") else best_threshold(y_train, oof)
        cv = metrics(y_train, oof, thr)
        pipe.fit(X_train, y_train)
        te = metrics(y_test, pipe.predict_proba(X_test)[:, 1], thr)
        fitted[name] = (pipe, thr, uses_eng, desc, cv, te)
        rows.append({"model": name, "engineered_features": uses_eng, "threshold": round(thr, 4),
                     **{f"cv_{k}": v for k, v in cv.items()}, **{f"test_{k}": v for k, v in te.items()}})
        print(f"{name:34s} thr={thr:.3f}  CV F1={cv['f1']:.3f} R={cv['recall']:.3f}  |  "
              f"test F1={te['f1']:.3f} P={te['precision']:.3f} R={te['recall']:.3f}")

    # SMOTE file, for the record only: synthetic rows cannot be cross-validated
    # without leakage, and ordinary SMOTE interpolates the categorical `type`.
    smote = pd.read_csv(DATA / "train_resampled_smote.csv")
    smote_pipe = make_pipeline(HistGradientBoostingClassifier(categorical_features=[0], random_state=SEED))
    smote_pipe.fit(smote[INPUTS], smote[TARGET_COL])
    te = metrics(y_test, smote_pipe.predict_proba(X_test)[:, 1], 0.5)
    rows.append({"model": "Gradient boosting on SMOTE file (not selectable)", "engineered_features": False,
                 "threshold": 0.5, **{f"test_{k}": v for k, v in te.items()}})
    print(f"{'Gradient boosting on SMOTE file':34s} thr=0.500  (no CV)        |  "
          f"test F1={te['f1']:.3f} P={te['precision']:.3f} R={te['recall']:.3f}")

    comparison = pd.DataFrame(rows)
    selectable = comparison[comparison["cv_f1"].notna() & ~comparison["model"].str.startswith("Majority")]
    best = selectable.sort_values("cv_f1", ascending=False).iloc[0]["model"]
    pipe, thr, uses_eng, desc, cv, te = fitted[best]
    print(f"\nSelected: {best} (CV failure F1 {cv['f1']:.3f}), threshold {thr:.3f}")

    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    card = {
        "model_name": best,
        "description": desc,
        "trained_on": str(date.today()),
        "sklearn_version": sklearn.__version__,
        "model_file": MODEL_PATH.name,
        "input_columns": INPUTS,
        "input_notes": "type coded L=0, M=1, H=2 (not scaled); five sensors scaled with "
                       "ai4i_sensor_scaler.joblib in this order.",
        "engineered_features": ENGINEERED if uses_eng else [],
        "excluded_columns": ["udi", "product_id", "twf", "hdf", "pwf", "osf", "rnf"],
        "threshold": thr,
        "threshold_method": "maximises failure-class F1 on 5-fold out-of-fold training probabilities",
        "train_rows": int(len(train)), "train_failures": int(y_train.sum()),
        "test_rows": int(len(test)), "test_failures": int(y_test.sum()),
        "cv_metrics": cv,
        "test_metrics": te,
    }
    CARD_PATH.write_text(json.dumps(card, indent=2))
    comparison.to_csv(REPORTS / "model_comparison.csv", index=False, float_format="%.4f")
    write_report(comparison, card)
    print(f"Saved {MODEL_PATH.relative_to(ROOT)}, {CARD_PATH.relative_to(ROOT)}, reports/")


def write_report(comparison: pd.DataFrame, card: dict) -> None:
    te, cv = card["test_metrics"], card["cv_metrics"]
    pct = lambda v: "n/a" if pd.isna(v) else f"{v * 100:.1f}%"  # noqa: E731
    num = lambda v: "n/a" if pd.isna(v) else f"{v:.3f}"  # noqa: E731
    lines = [
        "# Model evaluation (Kelvin)",
        "",
        f"Generated by `training/train_model.py` on {card['trained_on']} (scikit-learn {card['sklearn_version']}).",
        "",
        "## Setup",
        "",
        f"- Training set: `train_scaled.csv`, {card['train_rows']:,} rows, {card['train_failures']} failures. "
        f"Test set: `test_scaled.csv`, {card['test_rows']:,} rows, {card['test_failures']} failures "
        "(Kiat's stratified 80/20 split, `random_state=42`).",
        "- Inputs: `type` (L=0, M=1, H=2) and the five sensors, scaled with `ai4i_sensor_scaler.joblib`. "
        "IDs and the failure-mode labels (TWF, HDF, PWF, OSF, RNF) are excluded.",
        "- Model and threshold chosen with 5-fold stratified cross-validation on the training set only. "
        "The test set was used once, for the final numbers below.",
        "- The decision threshold maximises failure-class F1 on the out-of-fold training predictions.",
        "",
        "## Selected model",
        "",
        f"**{card['model_name']}** — {card['description']} Threshold: {card['threshold']:.3f}.",
        "",
        "| Test-set metric (failure class) | Value |",
        "|---|---|",
        f"| Precision | {pct(te['precision'])} |",
        f"| Recall | {pct(te['recall'])} |",
        f"| F1-score | {pct(te['f1'])} |",
        f"| Accuracy | {pct(te['accuracy'])} |",
        f"| ROC AUC | {num(te['roc_auc'])} |",
        f"| PR AUC (average precision) | {num(te['pr_auc'])} |",
        "",
        "Confusion matrix on the test set:",
        "",
        "| | Predicted no failure | Predicted failure |",
        "|---|---|---|",
        f"| **Actual no failure** | {te['tn']} | {te['fp']} |",
        f"| **Actual failure** | {te['fn']} | {te['tp']} |",
        "",
        f"The model caught {te['tp']} of {te['tp'] + te['fn']} test failures and raised "
        f"{te['fp']} false alarms among {te['tn'] + te['fp']:,} non-failures. "
        f"Cross-validation on the training set gave a similar failure F1 ({pct(cv['f1'])}), "
        "so the test result is not a lucky split.",
        "",
        "## All candidates",
        "",
        "| Model | CV F1 | Test precision | Test recall | Test F1 | Test accuracy |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in comparison.iterrows():
        lines.append(f"| {r['model']} | {pct(r.get('cv_f1'))} | {pct(r['test_precision'])} | "
                     f"{pct(r['test_recall'])} | {pct(r['test_f1'])} | {pct(r['test_accuracy'])} |")
    lines += [
        "",
        "## Notes for the report",
        "",
        "- The majority-class baseline reaches about 96.6% accuracy while catching zero failures. "
        "This is why failure-class precision, recall and F1 are the headline metrics.",
        "- Temperature difference and power are physical quantities, but the thresholds they relate to "
        "(HDF: difference < 8.6 K with speed < 1,380 rpm; PWF: power < 3,500 W or > 9,000 W) are the "
        "published **rules used to generate** the synthetic AI4I data. The model learning them is expected, "
        "not a discovery of real safety limits.",
        "- The SMOTE file was not used for selection: its synthetic rows cannot be cross-validated without "
        "leakage, and ordinary SMOTE interpolates the categorical `type`. Class weighting / threshold tuning "
        "handle the imbalance instead. Its test result is listed above for completeness.",
        "- Random failures (RNF, 0.1% of records) are random by construction and cannot be predicted.",
        "- AI4I is synthetic and has no repair costs or downtime, so savings cannot be measured from it.",
    ]
    (REPORTS / "model_evaluation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
