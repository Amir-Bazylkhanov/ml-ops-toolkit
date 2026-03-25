#!/usr/bin/env python3
"""Example: Deploy 2 model versions, configure A/B split, collect comparison metrics.

Prerequisites:
    1. Train models: python scripts/train_example_model.py
    2. Start server: docker compose up

Usage::

    python examples/run_ab_test.py
"""

import random
import time

import requests

BASE_URL = "http://localhost:8000"


def main():
    print("=== A/B Testing Example ===\n")

    # 1. Check available models
    print("1. Checking available models...")
    resp = requests.get(f"{BASE_URL}/model/info")
    resp.raise_for_status()
    info = resp.json()
    versions = [v["version"] for v in info["registered_versions"]]
    print(f"   Available: {versions}")

    if len(versions) < 2:
        print("   ERROR: Need at least 2 model versions. Run: python scripts/train_example_model.py")
        return

    model_a = versions[0]
    model_b = versions[1]
    print(f"   Model A (control): {model_a}")
    print(f"   Model B (treatment): {model_b}\n")

    # 2. Configure A/B test with 70/30 split
    print("2. Configuring A/B test (70% A / 30% B)...")
    resp = requests.post(f"{BASE_URL}/ab-test/configure", json={
        "model_a_version": model_a,
        "model_b_version": model_b,
        "traffic_split": 0.7,
        "enabled": True,
    })
    resp.raise_for_status()
    print(f"   {resp.json()}\n")

    # 3. Send 100 prediction requests
    print("3. Sending 100 prediction requests through A/B router...")
    random.seed(42)

    for i in range(100):
        features = [
            round(random.uniform(4.0, 8.0), 1),
            round(random.uniform(2.0, 4.5), 1),
            round(random.uniform(1.0, 7.0), 1),
            round(random.uniform(0.1, 2.5), 1),
        ]
        resp = requests.post(f"{BASE_URL}/predict", json={"features": features})
        resp.raise_for_status()

        if (i + 1) % 25 == 0:
            print(f"   Sent {i + 1}/100 requests...")

        time.sleep(0.02)

    print()

    # 4. Get A/B test results
    print("4. A/B Test Results:")
    resp = requests.get(f"{BASE_URL}/ab-test/results")
    resp.raise_for_status()
    results = resp.json()

    print(f"   Total requests:      {results['total_requests']}")
    print(f"   Model A ({model_a}):")
    print(f"     Count:             {results['model_a_count']}")
    print(f"     Avg latency:       {results['model_a_avg_latency']:.2f}ms")
    print(f"     Avg confidence:    {results['model_a_avg_confidence']:.4f}")
    print(f"   Model B ({model_b}):")
    print(f"     Count:             {results['model_b_count']}")
    print(f"     Avg latency:       {results['model_b_avg_latency']:.2f}ms")
    print(f"     Avg confidence:    {results['model_b_avg_confidence']:.4f}")
    print()

    # 5. Analyze split ratio
    actual_split = results["model_a_count"] / results["total_requests"] if results["total_requests"] > 0 else 0
    print(f"5. Traffic split analysis:")
    print(f"   Configured: 70% A / 30% B")
    print(f"   Actual:     {actual_split:.0%} A / {1 - actual_split:.0%} B")
    print()

    # 6. Compare models
    print("6. Model comparison:")
    if results["model_a_avg_confidence"] > results["model_b_avg_confidence"]:
        print(f"   Model A has higher confidence ({results['model_a_avg_confidence']:.4f} vs {results['model_b_avg_confidence']:.4f})")
    else:
        print(f"   Model B has higher confidence ({results['model_b_avg_confidence']:.4f} vs {results['model_a_avg_confidence']:.4f})")

    if results["model_a_avg_latency"] < results["model_b_avg_latency"]:
        print(f"   Model A is faster ({results['model_a_avg_latency']:.2f}ms vs {results['model_b_avg_latency']:.2f}ms)")
    else:
        print(f"   Model B is faster ({results['model_b_avg_latency']:.2f}ms vs {results['model_a_avg_latency']:.2f}ms)")
    print()

    # 7. Reset results
    print("7. Resetting A/B test results...")
    resp = requests.post(f"{BASE_URL}/ab-test/reset")
    resp.raise_for_status()
    print(f"   {resp.json()}")
    print()

    print("Done! Check Grafana at http://localhost:3000 for visual comparison.")


if __name__ == "__main__":
    main()
