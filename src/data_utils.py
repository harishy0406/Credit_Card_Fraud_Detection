import os
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_creditcard_data(
    csv_path: str = "data/creditcard.csv",
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Load the Kaggle/ML credit card fraud dataset and return standardized splits.

    Assumes the CSV has:
    - feature columns named V1..V28 plus 'Time' and 'Amount'
    - target column named 'Class' (0 = normal, 1 = fraud)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset not found at '{csv_path}'. "
            "Please download the credit card fraud dataset "
            "and place it under 'data/creditcard.csv'."
        )

    df = pd.read_csv(csv_path)

    if "Class" not in df.columns:
        raise ValueError("Expected target column 'Class' not found in dataset.")

    X = df.drop(columns=["Class"])
    y = df["Class"].values

    # Check if we have enough samples for stratification
    y_int = y.astype(int)
    min_class_count = min(np.bincount(y_int))
    use_stratification = min_class_count >= 2

    stratify_param = y if use_stratification else None
    if not use_stratification:
        print("Warning: Not enough samples for stratification, using random split")

    # First split into train+val and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=stratify_param,
        random_state=random_state,
    )

    # Then split train and validation
    val_relative_size = val_size / (1.0 - test_size)
    stratify_train_val = y_train_val if use_stratification else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_relative_size,
        stratify=stratify_train_val,
        random_state=random_state,
    )

    # Standardize numeric features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        y_train,
        y_val,
        y_test,
        scaler,
    )
