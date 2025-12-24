"""
Machine learning models for credit card fraud detection.

InternShip Details:
- Name: M HARISH GAUTHAM
- Project ID: #CC69844
- Project Title: Credit Card Fraud Detection
- Internship Domain: Data Science Intern
- Project Level: Intermediate Level
- Assigned By: CodeClause Internship
"""

from typing import Dict

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

def build_models() -> Dict[str, object]:
    """
    Create a dictionary of candidate models.

    Uses class_weight='balanced' where supported to help with imbalance.
    """
    models = {
        "log_reg": LogisticRegression(
            max_iter=500,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "decision_tree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            n_jobs=-1,
            class_weight="balanced",
            random_state=42,
        ),
        "grad_boost": GradientBoostingClassifier(
            random_state=42,
        ),
    }
    return models
