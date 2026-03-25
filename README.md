# MLOps Toolkit

Production ML deployment toolkit demonstrating Docker serving, Prometheus/Grafana monitoring, model versioning, and A/B testing infrastructure.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Docker Compose                           │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │              │    │              │    │              │      │
│  │  Model       │───>│  Prometheus  │───>│   Grafana    │      │
│  │  Server      │    │              │    │              │      │
│  │  (FastAPI)   │    │  :9090       │    │  :3000       │      │
│  │  :8000       │    │              │    │              │      │
│  │              │    └──────────────┘    └──────────────┘      │
│  │  ┌────────┐  │                                              │
│  │  │Registry│  │    ┌──────────────┐                          │
│  │  │A/B Test│  │    │              │                          │
│  │  │Metrics │  │───>│    Redis     │                          │
│  │  │Drift   │  │    │    :6379     │                          │
│  │  └────────┘  │    │              │                          │
│  └──────────────┘    └──────────────┘                          │
└─────────────────────────────────────────────────────────────────┘
```

**Components:**

| Service | Port | Description |
|---------|------|-------------|
| **Model Server** | 8000 | FastAPI server with prediction, model management, A/B testing, and monitoring endpoints |
| **Prometheus** | 9090 | Metrics collection and time-series storage, scrapes model server every 15s |
| **Grafana** | 3000 | Pre-provisioned dashboard with request rate, latency, confidence, drift monitoring |
| **Redis** | 6379 | Caching layer for the model server |

## Quick Start

### 1. Train Example Models

```bash
pip install scikit-learn joblib numpy
python scripts/train_example_model.py
```

This trains two sklearn models (LogisticRegression v1, RandomForest v2) on the Iris dataset and saves them to `models/`.

### 2. Start the Stack

```bash
cp .env.example .env
docker compose up --build
```

### 3. Verify

- **Model Server**: http://localhost:8000/health
- **API Docs**: http://localhost:8000/docs
- **Prometheus**: http://localhost:9090
- **Grafana Dashboard**: http://localhost:3000 (auto-provisioned, no login needed)

### 4. Generate Traffic

```bash
# Run example workflow
python examples/deploy_sklearn_model.py

# Or load test for Grafana visualization
bash scripts/load_test.sh
```

## API Documentation

### Prediction

#### `POST /predict`

Run inference on input features. Routes through A/B testing if configured.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

Response:
```json
{
  "prediction": 0,
  "confidence": 0.9714,
  "model_version": "v1",
  "latency_ms": 1.23,
  "timestamp": "2024-01-01T00:00:00+00:00"
}
```

Optional: specify `"model_version": "v2"` to bypass A/B routing and use a specific version.

#### `GET /health`

```json
{"status": "healthy", "active_model": "v1", "registered_models": 2}
```

#### `GET /metrics`

Prometheus-formatted metrics endpoint (scraped by Prometheus automatically).

#### `GET /model/info`

List all registered model versions with metadata and metrics.

### Model Management

#### `POST /model/register`

Register a new model version. The model file must exist at the specified path.

```bash
curl -X POST http://localhost:8000/model/register \
  -H "Content-Type: application/json" \
  -d '{
    "version": "v3",
    "model_path": "/app/models/v3/model.pkl",
    "framework": "sklearn",
    "metrics": {"accuracy": 0.96},
    "description": "Improved model with feature engineering"
  }'
```

#### `POST /model/activate`

Switch the active model version (zero-downtime).

```bash
curl -X POST http://localhost:8000/model/activate \
  -H "Content-Type: application/json" \
  -d '{"version": "v2"}'
```

#### `POST /model/rollback`

Rollback to the previously active model version.

```bash
curl -X POST http://localhost:8000/model/rollback
```

### A/B Testing

#### `POST /ab-test/configure`

Configure traffic splitting between two model versions.

```bash
curl -X POST http://localhost:8000/ab-test/configure \
  -H "Content-Type: application/json" \
  -d '{
    "model_a_version": "v1",
    "model_b_version": "v2",
    "traffic_split": 0.8,
    "enabled": true
  }'
```

- `traffic_split: 0.8` means 80% of requests go to model A, 20% to model B
- Set `enabled: false` to route all traffic to model A without removing config

#### `GET /ab-test/results`

Get aggregated A/B test statistics.

```json
{
  "total_requests": 100,
  "model_a_count": 79,
  "model_b_count": 21,
  "model_a_avg_latency": 1.45,
  "model_b_avg_latency": 2.31,
  "model_a_avg_confidence": 0.9234,
  "model_b_avg_confidence": 0.9567
}
```

#### `POST /ab-test/reset`

Clear accumulated A/B test results.

### Data Drift

#### `GET /drift/status`

Get current data drift detection status (KL divergence).

```json
{
  "kl_divergence": 0.034,
  "is_drifting": false,
  "feature_scores": [0.012, 0.045, 0.028, 0.051],
  "threshold": 0.1,
  "window_size": 150
}
```

## Model Registration Guide

### Directory Structure

Models are stored in versioned directories:

```
models/
├── v1/
│   ├── model.pkl          # sklearn model (joblib)
│   └── metadata.json
├── v2/
│   ├── model.pkl
│   └── metadata.json
├── v3/
│   ├── model.pt           # PyTorch model
│   └── metadata.json
└── active_version.txt     # Tracks currently active version
```

### Supported Frameworks

| Framework | File Format | Extension |
|-----------|-------------|-----------|
| scikit-learn | joblib | `.pkl` |
| PyTorch | torch.save | `.pt` |

### Register via Script

```python
from sklearn.linear_model import LogisticRegression
import joblib

# Train
model = LogisticRegression()
model.fit(X_train, y_train)

# Save
joblib.dump(model, "models/v3/model.pkl")

# Register via API
import requests
requests.post("http://localhost:8000/model/register", json={
    "version": "v3",
    "model_path": "/app/models/v3/model.pkl",
    "framework": "sklearn",
    "metrics": {"accuracy": 0.96},
    "description": "My new model"
})
```

### Register via Pre-populated Directory

1. Save model + `metadata.json` to `models/v3/`
2. Restart the server — it auto-discovers models on startup

`metadata.json` format:
```json
{
  "version": "v3",
  "framework": "sklearn",
  "created_at": "2024-01-01T00:00:00+00:00",
  "metrics": {"accuracy": 0.96, "f1_score": 0.95},
  "description": "Model description"
}
```

## A/B Testing Guide

### Setup

1. Register at least two model versions
2. Configure A/B test:

```bash
curl -X POST http://localhost:8000/ab-test/configure \
  -d '{"model_a_version":"v1","model_b_version":"v2","traffic_split":0.8,"enabled":true}'
```

3. Send predictions normally — routing is automatic:

```bash
curl -X POST http://localhost:8000/predict \
  -d '{"features":[5.1,3.5,1.4,0.2]}'
```

### Monitor Results

```bash
# Check A/B stats
curl http://localhost:8000/ab-test/results

# View in Grafana — metrics are split by model_version
open http://localhost:3000
```

### Promote Winner

```bash
# After collecting enough data, promote the better model
curl -X POST http://localhost:8000/model/activate -d '{"version":"v2"}'

# Disable A/B test
curl -X POST http://localhost:8000/ab-test/configure \
  -d '{"model_a_version":"v2","model_b_version":"v2","traffic_split":1.0,"enabled":false}'
```

## Grafana Dashboard

The dashboard is pre-provisioned and available immediately at http://localhost:3000.

**Panels:**

| Panel | Description |
|-------|-------------|
| Request Rate | Requests per second by model version |
| Latency Percentiles | p50, p95, p99 latency histograms |
| Error Rate | Errors per second by error type |
| Prediction Confidence | Confidence score distribution |
| Active Model Version | Currently serving model |
| Data Drift Score | KL divergence gauge with threshold alerts |
| Total Requests | Cumulative request counter |
| Avg Latency | Mean request latency |

## Prometheus Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `model_request_total` | Counter | model_version, status | Total prediction requests |
| `model_request_latency_seconds` | Histogram | model_version | Request latency in seconds |
| `model_prediction_confidence` | Histogram | model_version | Prediction confidence scores |
| `model_errors_total` | Counter | error_type | Total errors by category |
| `model_active_version` | Gauge | version | Active model version indicator |
| `model_data_drift_score` | Gauge | — | KL divergence drift score |
| `model_server_info` | Info | — | Server version and framework info |

## Running Tests

```bash
cd model_server
pip install -r requirements.txt pytest pytest-asyncio
pytest tests/ -v
```

## Project Structure

```
mlops-toolkit/
├── docker-compose.yml           # Full stack orchestration
├── requirements.txt             # Development dependencies
├── .env.example                 # Environment variable template
├── model_server/
│   ├── Dockerfile
│   ├── requirements.txt         # Production dependencies
│   ├── app/
│   │   ├── main.py              # FastAPI application
│   │   ├── config.py            # Pydantic settings
│   │   ├── serving/
│   │   │   ├── model_registry.py   # Versioned model management
│   │   │   ├── predictor.py        # Framework-agnostic inference
│   │   │   └── ab_testing.py       # Traffic splitting & comparison
│   │   ├── monitoring/
│   │   │   ├── metrics.py          # Prometheus metric definitions
│   │   │   └── data_drift.py       # KL-divergence drift detection
│   │   └── middleware/
│   │       └── logging_middleware.py  # Structured JSON logging
│   └── tests/
├── monitoring/
│   ├── prometheus/prometheus.yml
│   └── grafana/                 # Pre-provisioned dashboards
├── examples/                    # Deployment examples
├── scripts/                     # Training & load testing
├── notebooks/demo.ipynb         # Interactive walkthrough
└── models/                      # Model artifacts (git-ignored)
```

## Tech Stack

- **Python 3.11** / **FastAPI** — Model serving API
- **prometheus-client** — Metrics instrumentation
- **Prometheus** — Metrics collection & storage
- **Grafana** — Monitoring dashboards
- **Redis** — Caching layer
- **Docker / Docker Compose** — Container orchestration
- **scikit-learn / PyTorch** — ML frameworks
- **scipy** — Statistical drift detection
