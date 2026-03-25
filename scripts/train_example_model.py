#!/usr/bin/env python3
"""Train example sklearn and PyTorch models for demonstration.

Creates two model versions:
- v1: sklearn LogisticRegression on Iris dataset
- v2: sklearn RandomForestClassifier on Iris dataset

Models are saved to ../models/v1/ and ../models/v2/ with metadata.

Usage::

    python scripts/train_example_model.py
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def train_and_save(model, model_name: str, version: str, X_train, X_test, y_train, y_test):
    """Train a model and save it with metadata."""
    version_dir = MODELS_DIR / version
    version_dir.mkdir(parents=True, exist_ok=True)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    model_path = version_dir / "model.pkl"
    joblib.dump(model, model_path)

    metadata = {
        "version": version,
        "framework": "sklearn",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metrics": {
            "accuracy": round(accuracy, 4),
            "f1_score": round(f1, 4),
        },
        "description": f"{model_name} trained on Iris dataset",
    }
    with open(version_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  [{version}] {model_name}: accuracy={accuracy:.4f}, f1={f1:.4f}")
    print(f"  Saved to {version_dir}")
    return accuracy


def main():
    print("Loading Iris dataset...")
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )
    print(f"  Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")
    print(f"  Features: {X_train.shape[1]}")
    print()

    print("Training models...")
    # V1: LogisticRegression
    train_and_save(
        LogisticRegression(max_iter=200, random_state=42),
        "LogisticRegression",
        "v1",
        X_train, X_test, y_train, y_test,
    )

    # V2: RandomForestClassifier
    train_and_save(
        RandomForestClassifier(n_estimators=100, random_state=42),
        "RandomForestClassifier",
        "v2",
        X_train, X_test, y_train, y_test,
    )

    # Write active_version.txt
    active_file = MODELS_DIR / "active_version.txt"
    active_file.write_text("v1")
    print(f"\nActive version set to 'v1'")
    print(f"\nModels directory: {MODELS_DIR}")
    print("Done! You can now start the model server.")


if __name__ == "__main__":
    main()
