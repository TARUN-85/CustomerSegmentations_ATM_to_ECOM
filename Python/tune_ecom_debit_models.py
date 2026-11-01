"""Tune initial ECOM debit-card propensity models using train/validation only.

October is reserved as the final holdout and is intentionally excluded here.
Run the final test only after the model and campaign decision rules are frozen.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "Data" / "ecom_debit_training_dataset.csv"
MODEL_DIR = PROJECT_ROOT / "Models"
RESULTS_PATH = PROJECT_ROOT / "Data" / "ecom_debit_hyperparameter_results.csv"
TARGET = "ecom_debit_next_30d"
RANDOM_STATE = 42


def make_candidate_models() -> list[tuple[str, dict[str, Any], object]]:
    """Return baseline plus small, explicit hyperparameter candidate grids.

    Candidate ranges are intentionally compact for an initial tuning pass.
    They are not exhaustive; results on September validation data will narrow
    the settings before any final October holdout evaluation.
    """
    candidates: list[tuple[str, dict[str, Any], object]] = []

    # A feature-independent reference gives the random-selection benchmark.
    candidates.append(("BaselinePrior", {}, DummyClassifier(strategy="prior")))

    # C controls inverse regularization strength in Logistic Regression:
    # smaller C applies stronger regularization; larger C allows more flexibility.
    for c_value in (0.01, 0.1, 1.0, 10.0, 100.0):
        parameters = {"C": c_value}
        model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=c_value,
                        max_iter=2_000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
        candidates.append((f"LogisticRegression_C{c_value:g}", parameters, model))

    # Vary tree count, depth, and minimum leaf size to test the trade-off between
    # model complexity and generalization. The original 300/12/10 setup is kept
    # as an explicit reference candidate.
    forest_settings = [
        {"n_estimators": 300, "max_depth": 12, "min_samples_leaf": 10},
        {"n_estimators": 200, "max_depth": 8, "min_samples_leaf": 10},
        {"n_estimators": 300, "max_depth": 8, "min_samples_leaf": 20},
        {"n_estimators": 400, "max_depth": 12, "min_samples_leaf": 20},
        {"n_estimators": 400, "max_depth": 16, "min_samples_leaf": 10},
        {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 20},
    ]
    for index, parameters in enumerate(forest_settings, start=1):
        model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        **parameters,
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
        candidates.append((f"RandomForest_{index}", parameters, model))

    # Lower learning rates use smaller boosting updates; depth and regularization
    # control tree complexity. Row and feature sampling add further regularization.
    xgb_settings = [
        {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05,
         "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 2.0},
        {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05,
         "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 2.0},
        {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.03,
         "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 5.0},
        {"n_estimators": 500, "max_depth": 3, "learning_rate": 0.03,
         "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 2.0},
        {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05,
         "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 5.0},
        {"n_estimators": 400, "max_depth": 4, "learning_rate": 0.03,
         "subsample": 0.9, "colsample_bytree": 0.9, "reg_lambda": 5.0},
    ]
    for index, parameters in enumerate(xgb_settings, start=1):
        model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBClassifier(
                        **parameters,
                        eval_metric="logloss",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
        candidates.append((f"XGBoost_{index}", parameters, model))

    return candidates


def calculate_metrics(
    y_true: pd.Series,
    probabilities: np.ndarray,
    candidate_name: str,
    split: str,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    """Calculate accuracy, ranking, calibration, and top-decile metrics."""
    # Accuracy, precision, and recall use 0.5 only as a common comparison point.
    # A final campaign cutoff should be based on contact cost and business value.
    predictions = (probabilities >= 0.5).astype(int)
    prevalence = float(y_true.mean())

    # Simulate contacting the 10% of customers with the highest predicted scores.
    top_n = max(1, int(np.ceil(len(probabilities) * 0.10)))
    top_indices = np.argsort(probabilities)[::-1][:top_n]
    constant_scores = bool(np.allclose(probabilities, probabilities[0]))
    top_precision = prevalence if constant_scores else float(y_true.iloc[top_indices].mean())
    top_capture = (top_n / len(probabilities)) if constant_scores else (
        float(y_true.iloc[top_indices].sum() / y_true.sum()) if y_true.sum() else np.nan
    )

    return {
        "split": split,
        "candidate": candidate_name,
        "parameters": json.dumps(parameters, sort_keys=True),
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
    """Fit every candidate on older snapshots and compare on September only."""
    data = pd.read_csv(DATA_PATH, parse_dates=["snapshot_date"])

    # Explicitly discard the October final holdout before any tuning work.
    # The holdout will be used once only after the model is finalized.
    data = data.loc[data["snapshot_date"] <= "2026-09-01"].copy()
    if data.empty or data[TARGET].nunique() < 2:
        raise ValueError("Train/validation data must contain both target classes.")

    # Keep row identifiers and the future label out of the feature matrix.
    feature_columns = [
        column for column in data.columns
        if column not in {"snapshot_date", "customer_id", TARGET}
    ]

    # Train on April-August snapshots and use September as the validation month.
    train = data.loc[data["snapshot_date"] <= "2026-08-01"].copy()
    validation = data.loc[data["snapshot_date"] == "2026-09-01"].copy()
    for split_name, frame in (("train", train), ("validation", validation)):
        if frame.empty or frame[TARGET].nunique() < 2:
            raise ValueError(f"{split_name} must contain both target classes.")

    X_train = train[feature_columns]
    y_train = train[TARGET].astype(int)
    X_validation = validation[feature_columns]
    y_validation = validation[TARGET].astype(int)

    # Fit each fixed candidate on training data, then record both in-sample and
    # later-month metrics. Validation PR-AUC is the pre-declared selection rule.
    metric_rows: list[dict[str, Any]] = []
    candidate_models: dict[str, object] = {}
    candidate_parameters: dict[str, dict[str, Any]] = {}

    for name, parameters, model in make_candidate_models():
        model.fit(X_train, y_train)
        candidate_models[name] = model
        candidate_parameters[name] = parameters

        train_probabilities = model.predict_proba(X_train)[:, 1]
        metric_rows.append(
            calculate_metrics(y_train, train_probabilities, name, "train", parameters)
        )

        validation_probabilities = model.predict_proba(X_validation)[:, 1]
        metric_rows.append(
            calculate_metrics(
                y_validation,
                validation_probabilities,
                name,
                "validation",
                parameters,
            )
        )

    results = pd.DataFrame(metric_rows)

    # Select using validation PR-AUC, which evaluates ranking quality while
    # focusing on the positive adoption class. BaselinePrior is reference only.
    validation_candidates = results.loc[
        (results["split"] == "validation")
        & (results["candidate"] != "BaselinePrior")
    ].sort_values("pr_auc_average_precision", ascending=False)
    best_name = str(validation_candidates.iloc[0]["candidate"])
    best_model = candidate_models[best_name]
    best_parameters = candidate_parameters[best_name]

    # Save as a tuning candidate. Do not overwrite the previously selected model
    # or evaluate the October holdout during this tuning stage.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    candidate_model_path = MODEL_DIR / "ecom_debit_tuning_candidate.joblib"
    joblib.dump(best_model, candidate_model_path)

    metadata = {
        "candidate_model": best_name,
        "candidate_parameters": best_parameters,
        "selection_metric": "September validation PR-AUC (average precision)",
        "features": feature_columns,
        "train_snapshots": "2026-04-01 through 2026-08-01",
        "validation_snapshot": "2026-09-01",
        "final_holdout_snapshot": "2026-10-01 (not evaluated during tuning)",
        "train_rows": len(train),
        "validation_rows": len(validation),
        "candidate_count_including_baseline": len(candidate_models),
        "random_state": RANDOM_STATE,
    }
    metadata_path = MODEL_DIR / "ecom_debit_tuning_candidate_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Sort validation candidates by their selection metric for easy comparison.
    results = results.sort_values(
        ["split", "pr_auc_average_precision"],
        ascending=[True, False],
    )
    results.to_csv(RESULTS_PATH, index=False)

    print("Rows:", {"train": len(train), "validation": len(validation)})
    print("Final October holdout was not evaluated.")
    print("Best validation PR-AUC candidate:", best_name, best_parameters)
    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("Saved candidate model:", candidate_model_path)
    print("Saved tuning results:", RESULTS_PATH)


if __name__ == "__main__":
    # Prevent training from starting if another script imports this module.
    main()
