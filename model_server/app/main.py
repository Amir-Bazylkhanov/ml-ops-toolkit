"""FastAPI model server with prediction, health, metrics, and A/B testing endpoints.

Start with::

    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel, Field
from starlette.responses import Response

from app.config import get_settings
from app.middleware.logging_middleware import LoggingMiddleware
from app.monitoring.data_drift import DriftDetector
from app.monitoring.metrics import MetricsCollector
from app.serving.ab_testing import ABTestConfig, ABTestRouter
from app.serving.model_registry import ModelRegistry, ModelRegistryError, VersionNotFoundError
from app.serving.predictor import Predictor, PredictorError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global singletons (initialised during lifespan)
# ---------------------------------------------------------------------------
registry: ModelRegistry | None = None
predictor: Predictor | None = None
ab_router: ABTestRouter | None = None
metrics: MetricsCollector | None = None
drift_detector: DriftDetector | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise shared resources on startup."""
    global registry, predictor, ab_router, metrics, drift_detector

    settings = get_settings()

    registry = ModelRegistry(settings.model_dir)
    predictor = Predictor(registry)
    ab_router = ABTestRouter(predictor)
    metrics = MetricsCollector()
    drift_detector = DriftDetector(
        threshold=settings.drift_threshold,
        window_size=1000,
    )

    # If models are already registered and an active version exists, set up
    # default A/B config pointing active version to itself (no split).
    try:
        active = registry.get_active_version()
        metrics.set_active_model(active)
        versions = registry.list_versions()
        if len(versions) >= 2:
            ab_router.configure(ABTestConfig(
                model_a_version=versions[-2].version,
                model_b_version=versions[-1].version,
                traffic_split=settings.ab_test_split,
                enabled=settings.ab_test_enabled,
            ))
    except ModelRegistryError:
        logger.info("No active model version on startup — register a model first.")

    logger.info("Model server started. models_dir=%s", settings.model_dir)
    yield
    logger.info("Model server shutting down.")


app = FastAPI(
    title="ML Model Server",
    description="Production ML model serving with A/B testing and monitoring",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(LoggingMiddleware)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    features: list[float] = Field(..., min_length=1, description="Input feature vector")
    model_version: str | None = Field(None, description="Specific model version (uses active if omitted)")

class PredictResponse(BaseModel):
    prediction: Any
    confidence: float
    model_version: str
    latency_ms: float
    timestamp: str

class ABTestConfigRequest(BaseModel):
    model_a_version: str
    model_b_version: str
    traffic_split: float = Field(0.5, ge=0.0, le=1.0)
    enabled: bool = True

class RegisterModelRequest(BaseModel):
    version: str
    model_path: str
    framework: str = Field(..., pattern="^(sklearn|pytorch)$")
    metrics: dict[str, float] = Field(default_factory=dict)
    description: str = ""

class SetActiveRequest(BaseModel):
    version: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    """Health check endpoint."""
    active = None
    try:
        active = registry.get_active_version() if registry else None
    except ModelRegistryError:
        pass
    return {
        "status": "healthy",
        "active_model": active,
        "registered_models": len(registry) if registry else 0,
    }


@app.get("/metrics")
async def prometheus_metrics():
    """Expose Prometheus metrics."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.get("/model/info")
async def model_info():
    """Return information about all registered models and current active version."""
    if not registry:
        raise HTTPException(status_code=503, detail="Registry not initialised")

    active = None
    try:
        active = registry.get_active_version()
    except ModelRegistryError:
        pass

    versions = [asdict(m) for m in registry.list_versions()]
    return {
        "active_version": active,
        "registered_versions": versions,
        "total_versions": len(versions),
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    """Run inference on input features.

    If A/B testing is configured and enabled, the request is routed through
    the A/B test router.  Otherwise the active (or specified) model version
    is used directly.
    """
    if not predictor or not metrics:
        raise HTTPException(status_code=503, detail="Server not ready")

    try:
        # Route through A/B testing if configured
        if ab_router and ab_router._config and ab_router._config.enabled and not req.model_version:
            result = ab_router.route_prediction(req.features)
        else:
            result = predictor.predict(req.features, model_version=req.model_version)

        # Record metrics
        metrics.record_prediction(
            model_version=result.model_version,
            latency=result.latency_ms / 1000.0,  # Convert to seconds for Prometheus
            confidence=result.confidence,
        )

        # Track drift
        if drift_detector:
            try:
                drift_detector.add_observation(req.features)
                if len(drift_detector._window) >= 10:
                    drift_result = drift_detector.compute_drift()
                    metrics.record_drift_score(drift_result.kl_divergence)
            except (RuntimeError, ValueError):
                pass

        return PredictResponse(
            prediction=result.prediction,
            confidence=result.confidence,
            model_version=result.model_version,
            latency_ms=result.latency_ms,
            timestamp=result.timestamp,
        )

    except PredictorError as exc:
        metrics.record_error("prediction_error")
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        metrics.record_error("internal_error")
        logger.exception("Unexpected error during prediction")
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Model management endpoints
# ---------------------------------------------------------------------------

@app.post("/model/register")
async def register_model(req: RegisterModelRequest):
    """Register a new model version in the registry."""
    if not registry:
        raise HTTPException(status_code=503, detail="Registry not initialised")
    try:
        meta = registry.register_model(
            version=req.version,
            model_path=req.model_path,
            framework=req.framework,
            metrics=req.metrics,
            description=req.description,
        )
        return {"status": "registered", "metadata": asdict(meta)}
    except (ModelRegistryError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/model/activate")
async def activate_model(req: SetActiveRequest):
    """Set the active model version."""
    if not registry or not metrics:
        raise HTTPException(status_code=503, detail="Registry not initialised")
    try:
        registry.set_active_version(req.version)
        metrics.set_active_model(req.version)
        meta = registry.list_versions()
        active_meta = next((m for m in meta if m.version == req.version), None)
        if active_meta:
            metrics.set_model_info(req.version, active_meta.framework)
        return {"status": "activated", "version": req.version}
    except VersionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.post("/model/rollback")
async def rollback_model():
    """Rollback to the previous active model version."""
    if not registry or not metrics:
        raise HTTPException(status_code=503, detail="Registry not initialised")
    try:
        restored = registry.rollback()
        metrics.set_active_model(restored)
        return {"status": "rolled_back", "version": restored}
    except ModelRegistryError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# ---------------------------------------------------------------------------
# A/B testing endpoints
# ---------------------------------------------------------------------------

@app.post("/ab-test/configure")
async def configure_ab_test(req: ABTestConfigRequest):
    """Configure A/B test parameters."""
    if not ab_router:
        raise HTTPException(status_code=503, detail="A/B router not initialised")
    ab_router.configure(ABTestConfig(
        model_a_version=req.model_a_version,
        model_b_version=req.model_b_version,
        traffic_split=req.traffic_split,
        enabled=req.enabled,
    ))
    return {
        "status": "configured",
        "model_a": req.model_a_version,
        "model_b": req.model_b_version,
        "traffic_split": req.traffic_split,
        "enabled": req.enabled,
    }


@app.get("/ab-test/results")
async def ab_test_results():
    """Get A/B test results."""
    if not ab_router:
        raise HTTPException(status_code=503, detail="A/B router not initialised")
    try:
        results = ab_router.get_results()
        return asdict(results)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/ab-test/reset")
async def reset_ab_test():
    """Reset A/B test accumulated results."""
    if not ab_router:
        raise HTTPException(status_code=503, detail="A/B router not initialised")
    ab_router.reset_results()
    return {"status": "reset"}


# ---------------------------------------------------------------------------
# Drift endpoint
# ---------------------------------------------------------------------------

@app.get("/drift/status")
async def drift_status():
    """Get current data drift status."""
    if not drift_detector:
        raise HTTPException(status_code=503, detail="Drift detector not initialised")
    try:
        result = drift_detector.compute_drift()
        return {
            "kl_divergence": result.kl_divergence,
            "is_drifting": result.is_drifting,
            "feature_scores": result.feature_scores,
            "threshold": drift_detector.threshold,
            "window_size": len(drift_detector._window),
        }
    except RuntimeError as exc:
        return {
            "status": "insufficient_data",
            "message": str(exc),
            "window_size": len(drift_detector._window),
        }
