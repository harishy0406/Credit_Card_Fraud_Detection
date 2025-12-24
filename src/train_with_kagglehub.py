"""
End-to-end credit card fraud detection using KaggleHub to fetch the dataset.

InternShip Details:
- Name: M HARISH GAUTHAM
- Project ID: #CC69844
- Project Title: Credit Card Fraud Detection
- Internship Domain: Data Science Intern
- Project Level: Intermediate Level
- Assigned By: CodeClause Internship

What this script does:
- Downloads the latest credit card fraud dataset from Kaggle via kagglehub
- Saves it to data/creditcard.csv
- Uses the existing data pipeline, models, and evaluation utilities
- Trains several models (log_reg, decision_tree, random_forest, grad_boost)
- Trains an additional SMOTE + Logistic Regression pipeline
- Compares models using imbalance-aware metrics (F1, ROC-AUC, PR-AUC, etc.)

Requirements (in addition to requirements.txt):
    pip install kagglehub[pandas-datasets]
"""
from __future__ import annotations

import os
from typing import Dict

import numpy as np

import kagglehub
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

from data_utils import load_creditcard_data
from evaluation import evaluate_predictions, print_detailed_report
from models import build_models


def download_dataset_to_csv(csv_path: str = "data/creditcard.csv") -> None:
    """Download the Kaggle credit card fraud dataset and save as a local CSV."""
    import pandas as pd

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    try:
        # Try using kagglehub
        print("Attempting to download with kagglehub...")
        path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
        csv_file = os.path.join(path, "creditcard.csv")
        df = pd.read_csv(csv_file)
        df.to_csv(csv_path, index=False)
        print(f"Dataset downloaded and saved to {csv_path}")
    except Exception as e:
        print(f"KaggleHub download failed: {e}")
        print("Please download creditcard.csv manually from https://www.kaggle.com/mlg-ulb/creditcardfraud")
        print(f"and place it at {csv_path}")
        raise


def train_and_evaluate_with_resampling() -> None:
    # 1) Ensure dataset is present (download if needed)
    csv_path = "data/creditcard.csv"
    if not os.path.exists(csv_path):
        print("Dataset not found locally. Downloading with kagglehub...")
        download_dataset_to_csv(csv_path)

    # 2) Load and split data using existing utility
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        _,
    ) = load_creditcard_data(csv_path)

    # 3) Build baseline models (with class_weight where supported)
    models = build_models()

    # 4) Add an advanced imbalance-handling model: SMOTE + Logistic Regression
    smote_logreg = Pipeline(
        steps=[
            ("smote", SMOTE(random_state=42)),
            (
                "log_reg",
                # No class_weight here because SMOTE balances the classes
                models["log_reg"].__class__(
                    max_iter=500,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    models["smote_log_reg"] = smote_logreg

    val_results: Dict[str, Dict[str, float]] = {}
    print("Training models (baseline + SMOTE pipeline)...")

    for name, model in models.items():
        print(f"\n=== Model: {name} ===")
        model.fit(X_train, y_train)

        y_val_pred = model.predict(X_val)
        y_val_proba: np.ndarray | None = None
        if hasattr(model, "predict_proba"):
            y_val_proba = model.predict_proba(X_val)[:, 1]
        elif hasattr(model, "decision_function"):
            y_val_proba = model.decision_function(X_val)

        metrics = evaluate_predictions(y_val, y_val_pred, y_val_proba)
        val_results[name] = metrics

        print("Validation metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    # 5) Pick best model based on validation F1 score
    best_model_name = max(val_results, key=lambda m: val_results[m].get("f1", 0.0))
    best_model = models[best_model_name]

    print(f"\nBest model on validation set: {best_model_name}")
    print("Re-fitting best model on train+val data and evaluating on test set...")

    # Refit on combined train+val for final evaluation
    X_train_val = np.vstack([X_train, X_val])
    y_train_val = np.concatenate([y_train, y_val])
    best_model.fit(X_train_val, y_train_val)

    y_test_pred = best_model.predict(X_test)
    y_test_proba: np.ndarray | None = None
    if hasattr(best_model, "predict_proba"):
        y_test_proba = best_model.predict_proba(X_test)[:, 1]
    elif hasattr(best_model, "decision_function"):
        y_test_proba = best_model.decision_function(X_test)

    test_metrics = evaluate_predictions(y_test, y_test_pred, y_test_proba)
    print("\nTest metrics (best model):")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    print_detailed_report(y_test, y_test_pred)


if __name__ == "__main__":
    train_and_evaluate_with_resampling()
