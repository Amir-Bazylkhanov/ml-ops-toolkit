"""Tests for model registry and predictor."""

import json
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

from app.serving.model_registry import (
    ModelMetadata,
    ModelRegistry,
    ModelRegistryError,
    VersionNotFoundError,
)
from app.serving.predictor import Predictor, PredictorError, PredictionResult


@pytest.fixture
def models_dir(tmp_path):
    return tmp_path / "models"


@pytest.fixture
def trained_model(tmp_path):
    """Train a simple model and save it."""
    iris = load_iris()
    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(iris.data, iris.target)
    model_path = tmp_path / "trained_model.pkl"
    joblib.dump(model, model_path)
    return model_path


@pytest.fixture
def registry_with_model(models_dir, trained_model):
    """Registry with one registered model."""
    registry = ModelRegistry(str(models_dir))
    registry.register_model(
        version="v1",
        model_path=str(trained_model),
        framework="sklearn",
        metrics={"accuracy": 0.95},
        description="Test model",
    )
    registry.set_active_version("v1")
    return registry


class TestModelRegistry:
    def test_create_empty_registry(self, models_dir):
        registry = ModelRegistry(str(models_dir))
        assert len(registry) == 0
        assert registry.list_versions() == []

    def test_register_model(self, models_dir, trained_model):
        registry = ModelRegistry(str(models_dir))
        meta = registry.register_model(
            version="v1",
            model_path=str(trained_model),
            framework="sklearn",
            metrics={"accuracy": 0.95},
            description="Test model",
        )
        assert meta.version == "v1"
        assert meta.framework == "sklearn"
        assert meta.metrics["accuracy"] == 0.95
        assert len(registry) == 1

    def test_register_duplicate_fails(self, registry_with_model, trained_model):
        with pytest.raises(ModelRegistryError, match="already registered"):
            registry_with_model.register_model(
                version="v1",
                model_path=str(trained_model),
                framework="sklearn",
                metrics={},
            )

    def test_register_invalid_framework(self, models_dir, trained_model):
        registry = ModelRegistry(str(models_dir))
        with pytest.raises(Exception):
            registry.register_model(
                version="v1",
                model_path=str(trained_model),
                framework="tensorflow",
                metrics={},
            )

    def test_load_model(self, registry_with_model):
        model, meta = registry_with_model.load_model("v1")
        assert model is not None
        assert meta.version == "v1"
        # Should be a sklearn model that can predict
        prediction = model.predict([[5.1, 3.5, 1.4, 0.2]])
        assert prediction is not None

    def test_load_nonexistent_version(self, registry_with_model):
        with pytest.raises(VersionNotFoundError):
            registry_with_model.load_model("v999")

    def test_list_versions(self, registry_with_model, trained_model):
        registry_with_model.register_model(
            version="v2",
            model_path=str(trained_model),
            framework="sklearn",
            metrics={"accuracy": 0.97},
        )
        versions = registry_with_model.list_versions()
        assert len(versions) == 2
        assert versions[0].version == "v1"
        assert versions[1].version == "v2"

    def test_active_version(self, registry_with_model):
        assert registry_with_model.get_active_version() == "v1"

    def test_set_active_version(self, registry_with_model, trained_model):
        registry_with_model.register_model(
            version="v2",
            model_path=str(trained_model),
            framework="sklearn",
            metrics={},
        )
        registry_with_model.set_active_version("v2")
        assert registry_with_model.get_active_version() == "v2"

    def test_rollback(self, registry_with_model, trained_model):
        registry_with_model.register_model(
            version="v2",
            model_path=str(trained_model),
            framework="sklearn",
            metrics={},
        )
        registry_with_model.set_active_version("v2")
        restored = registry_with_model.rollback()
        assert restored == "v1"
        assert registry_with_model.get_active_version() == "v1"

    def test_rollback_without_previous_fails(self, models_dir, trained_model):
        registry = ModelRegistry(str(models_dir))
        registry.register_model(
            version="v1",
            model_path=str(trained_model),
            framework="sklearn",
            metrics={},
        )
        registry.set_active_version("v1")
        with pytest.raises(ModelRegistryError, match="No previous version"):
            registry.rollback()

    def test_scan_existing_models(self, models_dir, trained_model):
        # Create a registry, register a model, then create a new registry from same dir
        registry1 = ModelRegistry(str(models_dir))
        registry1.register_model(
            version="v1",
            model_path=str(trained_model),
            framework="sklearn",
            metrics={"accuracy": 0.9},
        )
        registry1.set_active_version("v1")

        # New registry should discover existing models
        registry2 = ModelRegistry(str(models_dir))
        assert len(registry2) == 1
        assert registry2.get_active_version() == "v1"

    def test_metadata_serialization(self):
        meta = ModelMetadata(
            version="v1",
            framework="sklearn",
            created_at="2024-01-01T00:00:00",
            metrics={"accuracy": 0.95},
            description="test",
        )
        d = meta.to_dict()
        restored = ModelMetadata.from_dict(d)
        assert restored.version == meta.version
        assert restored.metrics == meta.metrics


class TestPredictor:
    def test_predict(self, registry_with_model):
        predictor = Predictor(registry_with_model)
        result = predictor.predict([5.1, 3.5, 1.4, 0.2])
        assert isinstance(result, PredictionResult)
        assert result.model_version == "v1"
        assert result.latency_ms >= 0
        assert 0 <= result.confidence <= 1
        assert result.prediction in [0, 1, 2]  # Iris classes

    def test_predict_specific_version(self, registry_with_model):
        predictor = Predictor(registry_with_model)
        result = predictor.predict([5.1, 3.5, 1.4, 0.2], model_version="v1")
        assert result.model_version == "v1"

    def test_predict_nonexistent_version(self, registry_with_model):
        predictor = Predictor(registry_with_model)
        with pytest.raises(PredictorError):
            predictor.predict([5.1, 3.5, 1.4, 0.2], model_version="v999")

    def test_predict_without_active_version(self, models_dir, trained_model):
        registry = ModelRegistry(str(models_dir))
        registry.register_model(
            version="v1",
            model_path=str(trained_model),
            framework="sklearn",
            metrics={},
        )
        predictor = Predictor(registry)
        with pytest.raises(PredictorError, match="No model_version"):
            predictor.predict([5.1, 3.5, 1.4, 0.2])

    def test_cache_eviction(self, registry_with_model):
        predictor = Predictor(registry_with_model)
        predictor.predict([5.1, 3.5, 1.4, 0.2])
        assert "v1" in predictor._model_cache
        predictor.evict_cache("v1")
        assert "v1" not in predictor._model_cache

    def test_multiple_predictions_consistent(self, registry_with_model):
        predictor = Predictor(registry_with_model)
        results = [predictor.predict([5.1, 3.5, 1.4, 0.2]) for _ in range(5)]
        predictions = [r.prediction for r in results]
        assert len(set(predictions)) == 1  # Same input = same prediction
