#!/usr/bin/env python3
"""Example: Save a PyTorch model, register it, and serve predictions.

This creates a simple PyTorch classifier for the Iris dataset,
saves it as a versioned model, and runs predictions through the API.

Prerequisites:
    1. Start server: docker compose up

Usage::

    python examples/deploy_torch_model.py
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
import torch
import torch.nn as nn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_URL = "http://localhost:8000"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


class IrisNet(nn.Module):
    """Simple feedforward classifier for Iris dataset."""

    def __init__(self, input_dim: int = 4, hidden_dim: int = 16, output_dim: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def train_torch_model():
    """Train a PyTorch model on Iris and save it."""
    print("1. Training PyTorch model on Iris dataset...")

    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    X_train_t = torch.tensor(X_train_s, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test_s, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    model = IrisNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        output = model(X_train_t)
        loss = criterion(output, y_train_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(X_test_t).argmax(dim=1)
        accuracy = (preds == y_test_t).float().mean().item()

    print(f"   Accuracy: {accuracy:.4f}")

    version = "v3"
    version_dir = MODELS_DIR / version
    version_dir.mkdir(parents=True, exist_ok=True)

    model_path = version_dir / "model.pt"
    torch.save(model, model_path)

    print(f"   Saved to {version_dir}\n")
    return version, model_path, round(accuracy, 4)


def main():
    print("=== Deploy PyTorch Model Example ===\n")

    version, model_path, accuracy = train_torch_model()

    print("2. Checking server health...")
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=5)
        resp.raise_for_status()
        print(f"   {resp.json()}\n")
    except requests.ConnectionError:
        print("   Server not running. Start with: docker compose up")
        print(f"   Model saved at {MODELS_DIR / version} — register it once the server is up.\n")
        return

    # Register the model via the API so the in-memory registry knows about it
    print(f"3. Registering {version} via API...")
    resp = requests.post(
        f"{BASE_URL}/model/register",
        json={
            "version": version,
            "model_path": str(model_path),
            "framework": "pytorch",
            "metrics": {"accuracy": accuracy},
            "description": "PyTorch IrisNet classifier",
        },
    )
    if resp.status_code == 200:
        print(f"   {resp.json()}\n")
    else:
        print(f"   Registration failed: {resp.json().get('detail')}\n")
        return

    # Activate the registered version
    print(f"4. Activating {version}...")
    resp = requests.post(f"{BASE_URL}/model/activate", json={"version": version})
    if resp.status_code == 200:
        print(f"   {resp.json()}\n")
    else:
        print(f"   Activation failed: {resp.json().get('detail')}\n")
        return

    # Run predictions
    print("5. Running predictions with PyTorch model...")
    test_samples = [
        [5.1, 3.5, 1.4, 0.2],
        [7.0, 3.2, 4.7, 1.4],
        [6.3, 3.3, 6.0, 2.5],
    ]
    iris_classes = ["setosa", "versicolor", "virginica"]

    for features in test_samples:
        resp = requests.post(f"{BASE_URL}/predict", json={"features": features})
        resp.raise_for_status()
        result = resp.json()
        predicted_class = iris_classes[int(result["prediction"])]
        print(f"   {features} -> {predicted_class} (conf={result['confidence']:.4f}, latency={result['latency_ms']:.2f}ms)")
    print()

    print("Done! Check Grafana at http://localhost:3000")


if __name__ == "__main__":
    main()
