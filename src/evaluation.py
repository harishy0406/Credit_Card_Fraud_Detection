"""
Evaluation utilities for credit card fraud detection models.

InternShip Details:
- Name: M HARISH GAUTHAM
- Project ID: #CC69844
- Project Title: Credit Card Fraud Detection
- Internship Domain: Data Science Intern
- Project Level: Intermediate Level
- Assigned By: CodeClause Internship
"""

from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

def evaluate_predictions(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None = None
) -> Dict[str, float]:
    """Compute key metrics for fraud detection."""
    metrics: Dict[str, float] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    if y_proba is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
            precision, recall, _ = precision_recall_curve(y_true, y_proba)
            # Simple PR-AUC approximation via trapezoidal rule
            pr_auc = np.trapz(precision, recall)
            metrics["pr_auc"] = pr_auc
        except ValueError:
            # E.g., when only one class present in y_true
            pass

    return metrics


def print_detailed_report(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Print confusion matrix and classification report."""
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))



