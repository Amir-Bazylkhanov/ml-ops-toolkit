#!/usr/bin/env python3
"""Example: Train a sklearn model, register it, and run predictions via the API.

Prerequisites:
    1. Train example models: python scripts/train_example_model.py
    2. Start server: docker compose up

Usage::

    python examples/deploy_sklearn_model.py
"""

import requests

BASE_URL = "http://localhost:8000"


def main():
    print("=== Deploy sklearn Model Example ===\n")

    # 1. Check health
    print("1. Checking server health...")
    resp = requests.get(f"{BASE_URL}/health")
    resp.raise_for_status()
    print(f"   {resp.json()}\n")

    # 2. Check registered models
    print("2. Listing registered models...")
    resp = requests.get(f"{BASE_URL}/model/info")
    resp.raise_for_status()
    info = resp.json()
    print(f"   Active version: {info['active_version']}")
    print(f"   Total versions: {info['total_versions']}")
    for v in info["registered_versions"]:
        print(f"   - {v['version']}: {v['description']} (accuracy={v['metrics'].get('accuracy', 'N/A')})")
    print()

    # 3. Activate v1 if not active
    print("3. Activating model v1...")
    resp = requests.post(f"{BASE_URL}/model/activate", json={"version": "v1"})
    resp.raise_for_status()
    print(f"   {resp.json()}\n")

    # 4. Run predictions with Iris-like features
    print("4. Running predictions...")
    test_samples = [
        [5.1, 3.5, 1.4, 0.2],  # Iris setosa
        [7.0, 3.2, 4.7, 1.4],  # Iris versicolor
        [6.3, 3.3, 6.0, 2.5],  # Iris virginica
    ]
    iris_classes = ["setosa", "versicolor", "virginica"]

    for features in test_samples:
        resp = requests.post(f"{BASE_URL}/predict", json={"features": features})
        resp.raise_for_status()
        result = resp.json()
        predicted_class = iris_classes[int(result["prediction"])]
        print(f"   Features: {features}")
        print(f"   Prediction: {result['prediction']} ({predicted_class})")
        print(f"   Confidence: {result['confidence']:.4f}")
        print(f"   Latency: {result['latency_ms']:.2f}ms")
        print(f"   Model: {result['model_version']}")
        print()

    # 5. Switch to v2
    print("5. Switching to model v2...")
    resp = requests.post(f"{BASE_URL}/model/activate", json={"version": "v2"})
    resp.raise_for_status()
    print(f"   {resp.json()}\n")

    # 6. Run same predictions with v2
    print("6. Running predictions with v2...")
    for features in test_samples:
        resp = requests.post(f"{BASE_URL}/predict", json={"features": features})
        resp.raise_for_status()
        result = resp.json()
        predicted_class = iris_classes[int(result["prediction"])]
        print(f"   Features: {features} -> {predicted_class} (conf={result['confidence']:.4f}, model={result['model_version']})")
    print()

    # 7. Rollback to v1
    print("7. Rolling back to previous version...")
    resp = requests.post(f"{BASE_URL}/model/rollback")
    resp.raise_for_status()
    print(f"   {resp.json()}\n")

    print("Done! Check Grafana at http://localhost:3000 to see metrics.")


if __name__ == "__main__":
    main()
