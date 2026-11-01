"""Evaluate the frozen tuning candidate once on the October final holdout."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "Data" / "ecom_debit_training_dataset.csv"
CANDIDATE_PATH = PROJECT_ROOT / "Models" / "ecom_debit_tuning_candidate.joblib"
METADATA_PATH = PROJECT_ROOT / "Models" / "ecom_debit_tuning_candidate_metadata.json"
OUTPUT_PATH = PROJECT_ROOT / "Data" / "ecom_debit_final_test_metrics.csv"
TARGET = "ecom_debit_next_30d"
TEST_SNAPSHOT = pd.Timestamp("2026-10-01")


def calculate_metrics(y_true: pd.Series, probabilities: np.ndarray, model_name: str) -> dict[str, object]:
    """Calculate final holdout metrics using the same definitions as tuning."""
    predictions = (probabilities >= 0.5).astype(int)
    prevalence = float(y_true.mean())
    top_n = max(1, int(np.ceil(len(probabilities) * 0.10)))
    top_indices = np.argsort(probabilities)[::-1][:top_n]
    constant_scores = bool(np.allclose(probabilities, probabilities[0]))
    top_precision = prevalence if constant_scores else float(y_true.iloc[top_indices].mean())
    top_capture = (top_n / len(probabilities)) if constant_scores else (
        float(y_true.iloc[top_indices].sum() / y_true.sum()) if y_true.sum() else np.nan
    )
    return {
        "split": "final_test",
        "model": model_name,
        "snapshot_date": TEST_SNAPSHOT.date().isoformat(),
        "rows": len(y_true),
        "positive_rate": prevalence,
        "accuracy_at_0_5": accuracy_score(y_true, predictions),
        "precision_at_0_5": precision_score(y_true, predictions, zero_division=0),
        "recall_at_0_5": recall_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc_average_precision": average_precision_score(y_true, probabilities),
        "brier_score": brier_score_loss(y_true, probabilities),
        "top_10pct_precision": top_precision,
        "top_10pct_lift": top_precision / prevalence if prevalence else np.nan,
        "top_10pct_positive_capture": top_capture,
    }


def main() -> None:
    """Score October with the frozen candidate; do not fit on test rows."""
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    feature_columns = metadata["features"]
    model = joblib.load(CANDIDATE_PATH)

    # Load only the reserved October rows for this final evaluation.
    data = pd.read_csv(DATA_PATH, parse_dates=["snapshot_date"])
    test = data.loc[data["snapshot_date"] == TEST_SNAPSHOT].copy()
    if test.empty or test[TARGET].nunique() < 2:
        raise ValueError("October final test must contain rows from both target classes.")

    X_test = test[feature_columns]
    y_test = test[TARGET].astype(int)
    candidate_probabilities = model.predict_proba(X_test)[:, 1]
    rows = [
        calculate_metrics(y_test, candidate_probabilities, metadata["candidate_model"])
    ]

    # Add a no-skill prior baseline fitted only from pre-test labels. This
    # provides a final-test reference without fitting/tuning on October.
    train = data.loc[data["snapshot_date"] <= "2026-08-01"]
    baseline = DummyClassifier(strategy="prior").fit(
        np.zeros((len(train), 1)), train[TARGET].astype(int)
    )
    baseline_probabilities = baseline.predict_proba(np.zeros((len(test), 1)))[:, 1]
    rows.append(calculate_metrics(y_test, baseline_probabilities, "BaselinePrior"))

    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUTPUT_PATH, index=False)
    print("Final holdout snapshot:", TEST_SNAPSHOT.date())
    print("Test rows:", len(test))
    print("Candidate parameters:", metadata["candidate_parameters"])
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("Saved final test metrics:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
