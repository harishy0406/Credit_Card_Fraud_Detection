"""
End-to-end script for credit card fraud detection.

Steps:
- Load and split the data
- Train several models with basic imbalance handling
- Evaluate on validation and test sets
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from data_utils import load_creditcard_data
from evaluation import evaluate_predictions, print_detailed_report
from models import build_models


def train_and_evaluate() -> None:
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        _,
    ) = load_creditcard_data()

    models = build_models()

    val_results: Dict[str, Dict[str, float]] = {}
    print("Training models...")

    for name, model in models.items():
        print(f"\n=== Model: {name} ===")
        model.fit(X_train, y_train)

        # Validation evaluation
        y_val_pred = model.predict(X_val)
        # Get probabilities for positive class if available
        y_val_proba: np.ndarray | None = None
        if hasattr(model, "predict_proba"):
            y_val_proba = model.predict_proba(X_val)[:, 1]
        elif hasattr(model, "decision_function"):
            # Some models expose decision_function instead
            y_val_proba = model.decision_function(X_val)

        metrics = evaluate_predictions(y_val, y_val_pred, y_val_proba)
        val_results[name] = metrics

        print("Validation metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    # Pick best model based on validation F1 score
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
    print("\nTest metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    print_detailed_report(y_test, y_test_pred)


if __name__ == "__main__":
    train_and_evaluate()


