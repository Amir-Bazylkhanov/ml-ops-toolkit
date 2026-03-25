#!/usr/bin/env bash
# Load test script: sends requests to generate visible metrics on Grafana dashboard.
#
# Usage:
#   bash scripts/load_test.sh [base_url] [num_requests]
#
# Defaults: http://localhost:8000, 200 requests

set -euo pipefail

BASE_URL="${1:-http://localhost:8000}"
NUM_REQUESTS="${2:-200}"
PREDICT_URL="${BASE_URL}/predict"

echo "=== ML Model Server Load Test ==="
echo "Target: ${PREDICT_URL}"
echo "Requests: ${NUM_REQUESTS}"
echo ""

# Check server health first
echo "Checking server health..."
HEALTH=$(curl -s -o /dev/null -w "%{http_code}" "${BASE_URL}/health" 2>/dev/null || echo "000")
if [ "$HEALTH" != "200" ]; then
    echo "ERROR: Server not responding at ${BASE_URL}/health (HTTP ${HEALTH})"
    echo "Make sure the model server is running: docker compose up"
    exit 1
fi
echo "Server is healthy."
echo ""

# Iris dataset has 4 features: sepal_length, sepal_width, petal_length, petal_width
# Generate varied feature vectors to simulate real traffic
echo "Sending ${NUM_REQUESTS} prediction requests..."
echo ""

SUCCESS=0
ERRORS=0

for i in $(seq 1 "$NUM_REQUESTS"); do
    # Generate random Iris-like features
    SL=$(awk "BEGIN{printf \"%.1f\", 4.0 + rand() * 4.0}")
    SW=$(awk "BEGIN{printf \"%.1f\", 2.0 + rand() * 2.5}")
    PL=$(awk "BEGIN{printf \"%.1f\", 1.0 + rand() * 6.0}")
    PW=$(awk "BEGIN{printf \"%.1f\", 0.1 + rand() * 2.4}")

    RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "${PREDICT_URL}" \
        -H "Content-Type: application/json" \
        -d "{\"features\": [${SL}, ${SW}, ${PL}, ${PW}]}" 2>/dev/null)

    HTTP_CODE=$(echo "$RESPONSE" | tail -1)

    if [ "$HTTP_CODE" = "200" ]; then
        SUCCESS=$((SUCCESS + 1))
    else
        ERRORS=$((ERRORS + 1))
    fi

    # Progress indicator every 20 requests
    if [ $((i % 20)) -eq 0 ]; then
        echo "  Progress: ${i}/${NUM_REQUESTS} (success: ${SUCCESS}, errors: ${ERRORS})"
    fi

    # Small delay to spread requests over time for better visualization
    sleep 0.05
done

echo ""
echo "=== Load Test Complete ==="
echo "Total:   ${NUM_REQUESTS}"
echo "Success: ${SUCCESS}"
echo "Errors:  ${ERRORS}"
echo ""
echo "Check your Grafana dashboard at http://localhost:3000"
echo "  - Request rate should show ~${NUM_REQUESTS} total requests"
echo "  - Latency histograms should be populated"
echo "  - Confidence distribution should be visible"
