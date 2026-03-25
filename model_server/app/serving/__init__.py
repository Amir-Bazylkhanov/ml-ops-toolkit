"""
Serving layer for the MLOps toolkit.

Provides model registry, prediction, and A/B testing capabilities.

Example usage::

    from app.serving import ModelRegistry, Predictor, ABTestRouter
    from app.serving.ab_testing import ABTestConfig

    registry = ModelRegistry(models_dir="/models")
    predictor = Predictor(registry)
    router = ABTestRouter(predictor)
"""

from app.serving.model_registry import ModelRegistry, ModelMetadata
from app.serving.predictor import Predictor, PredictionResult
from app.serving.ab_testing import ABTestRouter, ABTestConfig, ABTestResult

__all__ = [
    "ModelRegistry",
    "ModelMetadata",
    "Predictor",
    "PredictionResult",
    "ABTestRouter",
    "ABTestConfig",
    "ABTestResult",
]
