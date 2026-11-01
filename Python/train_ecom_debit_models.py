"""Train and compare models that predict 30-day ECOM debit-card adoption.

This script uses customer snapshots created by the SQL feature-engineering
query. Each row describes one customer at one snapshot date, with features from
the prior 90 days and a label for the next 30 days.

Run from the project virtual environment:
    python .\Python\train_ecom_debit_models.py
"""

from __future__ import annotations

import json
from pathlib import Path

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

# Resolve paths from this script's location so the script works regardless of
# the folder from which it is launched.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "Data" / "ecom_debit_training_dataset.csv"
MODEL_DIR = PROJECT_ROOT / "Models"
METRICS_PATH = PROJECT_ROOT / "Data" / "ecom_debit_model_metrics_with_training.csv"

# This is the outcome column created by the SQL query:
# 1 means successful debit-card ECOM transaction in the next 30 days;
# 0 means no such transaction.
TARGET = "ecom_debit_next_30d"

# Fixed seed makes randomized model procedures repeatable between runs.
RANDOM_STATE = 42


def make_models() -> dict[str, object]:
    """Build the baseline and candidate model pipelines.

    The baseline predicts the training-set class prevalence and provides a
    reference score. The candidate models learn how the customer features
    relate to future ECOM debit-card usage.

    Imputation is placed inside each pipeline so missing-value handling is
    learned from the training split only. Logistic Regression also scales
    numeric features because its coefficients are scale-sensitive. Tree models
    do not need feature scaling.
    """
    return {
        # A probability-only baseline: it does not rank customers using features.
        "BaselinePrior": DummyClassifier(strategy="prior"),
        "LogisticRegression": Pipeline(
            steps=[
                # Replace missing numeric values with the training median.
                ("imputer", SimpleImputer(strategy="median")),
                # Put numeric features on comparable scales for logistic regression.
                ("scaler", StandardScaler()),
                # A simple, interpretable linear probability model.
                ("model", LogisticRegression(max_iter=2_000, random_state=RANDOM_STATE)),
            ]
        ),
        "RandomForest": Pipeline(
            steps=[
                # Keep missing-value handling within the training pipeline.
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,  # Number of decision trees in the forest.
                        max_depth=12,  # Limit tree depth to reduce overfitting.
                        min_samples_leaf=10,  # Require enough rows in each leaf.
                        n_jobs=-1,  # Use available CPU cores for training.
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "XGBoost": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=300,  # Number of boosting rounds/trees.
                        max_depth=4,  # Control how complex each tree can become.
                        learning_rate=0.05,  # Size of each boosting update.
                        subsample=0.8,  # Train each tree on a row sample.
                        colsample_bytree=0.8,  # Train each tree on a feature sample.
                        reg_lambda=2.0,  # L2 regularization to limit complexity.
                        eval_metric="logloss",  # Probability-error metric used internally.
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def calculate_metrics(
    y_true: pd.Series,
    probabilities: np.ndarray,
    model_name: str,
    split: str,
) -> dict[str, object]:
    """Calculate classification and campaign-ranking metrics for one split.

    A 0.5 cutoff is used only for accuracy, precision, and recall. The top-10%
    metrics show how well the model ranks customers when communication capacity
    is limited. A constant-score baseline has no ranking ability, so its expected
    top-10% precision equals the overall positive rate and its expected capture
    is 10%.
    """
    # Convert predicted probabilities into yes/no predictions at the usual 0.5
    # threshold. Campaign decisions can later use a different cost-based cutoff.
    predictions = (probabilities >= 0.5).astype(int)

    # Identify the highest-scoring 10% of customers to simulate a limited
    # campaign reach. ceil ensures at least one row is included for small splits.
    top_n = max(1, int(np.ceil(len(probabilities) * 0.10)))
    top_indices = np.argsort(probabilities)[::-1][:top_n]

    # Prevalence is the share of actual adopters in this split. It is the random
    # targeting benchmark for precision and average precision.
    prevalence = float(y_true.mean())
    constant_scores = bool(np.allclose(probabilities, probabilities[0]))

    # If every row gets the same score, sorting cannot create a meaningful
    # ranking. Report the random-selection expectation for the baseline instead.
    top_precision = prevalence if constant_scores else float(y_true.iloc[top_indices].mean())
    top_capture = (top_n / len(probabilities)) if constant_scores else (
        float(y_true.iloc[top_indices].sum() / y_true.sum()) if y_true.sum() else np.nan
    )

    return {
        "split": split,  # Train, validation, or final test data.
        "model": model_name,
        "rows": len(y_true),
        "positive_rate": prevalence,
        "accuracy_at_0_5": accuracy_score(y_true, predictions),
        "precision_at_0_5": precision_score(y_true, predictions, zero_division=0),
        "recall_at_0_5": recall_score(y_true, predictions, zero_division=0),
        # ROC-AUC measures how often a positive is ranked above a negative.
        "roc_auc": roc_auc_score(y_true, probabilities),
        # Average precision summarizes precision/recall over ranking cutoffs.
        "pr_auc_average_precision": average_precision_score(y_true, probabilities),
        # Lower Brier score means predicted probabilities are closer to outcomes.
        "brier_score": brier_score_loss(y_true, probabilities),
        "top_10pct_precision": top_precision,
        # Lift compares top-decile precision with random selection prevalence.
        "top_10pct_lift": top_precision / prevalence if prevalence else np.nan,
        # Share of all adopters found in the top-decile selected group.
        "top_10pct_positive_capture": top_capture,
    }


def main() -> None:
    """Load the dataset, train models, select one, and save artifacts."""
    # Parse snapshot dates so chronological comparisons use real dates.
    data = pd.read_csv(DATA_PATH, parse_dates=["snapshot_date"])

    # Stop early if the input cannot support supervised binary classification.
    if data.empty:
        raise ValueError("Training dataset is empty.")
    if data[TARGET].nunique() < 2:
        raise ValueError("Training dataset must contain both target classes.")

    # Do not give the model the customer ID or snapshot date as predictors.
    # They identify a row/time but are not intended behavior features.
    feature_columns = [
        column
        for column in data.columns
        if column not in {"snapshot_date", "customer_id", TARGET}
    ]

    # Use chronological splits to imitate production: train on older snapshots,
    # compare models on September, and reserve October for a final holdout.
    train = data.loc[data["snapshot_date"] <= "2026-08-01"].copy()
    validation = data.loc[data["snapshot_date"] == "2026-09-01"].copy()
    test = data.loc[data["snapshot_date"] == "2026-10-01"].copy()

    # Each split must have rows and both target classes for the selected metrics.
    for split_name, frame in (("train", train), ("validation", validation), ("test", test)):
        if frame.empty or frame[TARGET].nunique() < 2:
            raise ValueError(f"{split_name} split must be non-empty and contain both target classes.")

    # Separate predictor matrix (X) from the known outcome (y) for each period.
    X_train, y_train = train[feature_columns], train[TARGET].astype(int)
    X_validation, y_validation = validation[feature_columns], validation[TARGET].astype(int)
    X_test, y_test = test[feature_columns], test[TARGET].astype(int)

    # Create models and containers for their metrics and fitted objects.
    models = make_models()
    training_rows: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []
    fitted_models: dict[str, object] = {}

    # Fit each candidate only on the training dates. Record training metrics to
    # show how well it fits known data and validation metrics to compare how well
    # it generalizes to a later month.
    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted_models[name] = model

        training_probabilities = model.predict_proba(X_train)[:, 1]
        training_rows.append(calculate_metrics(y_train, training_probabilities, name, "train"))

        validation_probabilities = model.predict_proba(X_validation)[:, 1]
        validation_rows.append(
            calculate_metrics(y_validation, validation_probabilities, name, "validation")
        )

    # Combine train and validation results. The validation data—not training
    # performance—selects the best candidate, using PR-AUC as the ranking metric.
    metrics = pd.concat(
        [pd.DataFrame(training_rows), pd.DataFrame(validation_rows)],
        ignore_index=True,
    )
    metrics = metrics.sort_values(
        ["split", "pr_auc_average_precision"],
        ascending=[True, False],
    )
    candidate_rows = metrics.loc[
        (metrics["split"] == "validation") & (metrics["model"] != "BaselinePrior")
    ]
    best_name = str(candidate_rows.iloc[0]["model"])
    best_model = fitted_models[best_name]

    # Evaluate only the selected model on October. This is a final holdout and
    # must not be used to tune or choose a different candidate model.
    test_probabilities = best_model.predict_proba(X_test)[:, 1]
    test_row = calculate_metrics(y_test, test_probabilities, best_name, "test")
    metrics = pd.concat([metrics, pd.DataFrame([test_row])], ignore_index=True)

    # Save the selected, fitted pipeline. The pipeline includes imputation and
    # scaling where needed, so inference uses the same transformations as training.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "ecom_debit_propensity.joblib"
    joblib.dump(best_model, model_path)

    # Save the target, feature list, date splits, and selected model alongside
    # the binary model artifact to support traceability and future API loading.
    metadata = {
        "target": TARGET,
        "selected_model": best_name,
        "selection_metric": "validation PR-AUC (average precision)",
        "features": feature_columns,
        "train_snapshots": "2026-04-01 through 2026-08-01",
        "validation_snapshot": "2026-09-01",
        "test_snapshot": "2026-10-01",
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "random_state": RANDOM_STATE,
        "note": "Temporal evaluation; customers can appear in multiple snapshots. Test metrics are final holdout only.",
    }
    metadata_path = MODEL_DIR / "ecom_debit_propensity_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Export all model/split metrics for review and display the same summary in
    # the terminal so users can immediately compare train/validation/test rows.
    metrics.to_csv(METRICS_PATH, index=False)
    print("Rows:", {"train": len(train), "validation": len(validation), "test": len(test)})
    print("Selected by validation PR-AUC:", best_name)
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("Saved model:", model_path)
    print("Saved metrics:", METRICS_PATH)


if __name__ == "__main__":
    # Run main only when invoked as a script; importing this file will not
    # automatically load data or train models.
    main()
