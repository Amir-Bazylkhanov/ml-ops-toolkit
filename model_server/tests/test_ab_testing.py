"""Tests for A/B testing router."""

import pytest
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

from app.serving.model_registry import ModelRegistry
from app.serving.predictor import Predictor
from app.serving.ab_testing import ABTestConfig, ABTestRouter, ABTestRouterError


@pytest.fixture
def registry_with_two_models(tmp_path):
    """Registry with two model versions."""
    iris = load_iris()

    # Train and save model v1
    model1 = LogisticRegression(max_iter=200, random_state=42)
    model1.fit(iris.data, iris.target)
    path1 = tmp_path / "model1.pkl"
    joblib.dump(model1, path1)

    # Train and save model v2 (different params)
    model2 = LogisticRegression(max_iter=200, random_state=0, C=0.1)
    model2.fit(iris.data, iris.target)
    path2 = tmp_path / "model2.pkl"
    joblib.dump(model2, path2)

    models_dir = tmp_path / "models"
    registry = ModelRegistry(str(models_dir))
    registry.register_model("v1", str(path1), "sklearn", {"accuracy": 0.95})
    registry.register_model("v2", str(path2), "sklearn", {"accuracy": 0.93})
    registry.set_active_version("v1")
    return registry


@pytest.fixture
def ab_router(registry_with_two_models):
    predictor = Predictor(registry_with_two_models)
    return ABTestRouter(predictor)


class TestABTestConfig:
    def test_valid_config(self):
        config = ABTestConfig("v1", "v2", traffic_split=0.7, enabled=True)
        assert config.traffic_split == 0.7

    def test_invalid_split_too_high(self):
        with pytest.raises(ValueError, match="traffic_split"):
            ABTestConfig("v1", "v2", traffic_split=1.5)

    def test_invalid_split_negative(self):
        with pytest.raises(ValueError, match="traffic_split"):
            ABTestConfig("v1", "v2", traffic_split=-0.1)

    def test_boundary_splits(self):
        ABTestConfig("v1", "v2", traffic_split=0.0)
        ABTestConfig("v1", "v2", traffic_split=1.0)


class TestABTestRouter:
    def test_route_without_config_fails(self, ab_router):
        with pytest.raises(ABTestRouterError):
            ab_router.route_prediction([5.1, 3.5, 1.4, 0.2])

    def test_configure_and_route(self, ab_router):
        ab_router.configure(ABTestConfig("v1", "v2", traffic_split=0.5, enabled=True))
        result = ab_router.route_prediction([5.1, 3.5, 1.4, 0.2])
        assert result.model_version in ("v1", "v2")

    def test_disabled_routes_to_model_a(self, ab_router):
        ab_router.configure(ABTestConfig("v1", "v2", traffic_split=0.5, enabled=False))
        results = [ab_router.route_prediction([5.1, 3.5, 1.4, 0.2]) for _ in range(20)]
        versions = {r.model_version for r in results}
        assert versions == {"v1"}, "Disabled A/B test should only route to model A"

    def test_traffic_split_approximate(self, ab_router):
        ab_router.configure(ABTestConfig("v1", "v2", traffic_split=0.8, enabled=True))
        results = [ab_router.route_prediction([5.1, 3.5, 1.4, 0.2]) for _ in range(500)]
        a_count = sum(1 for r in results if r.model_version == "v1")
        ratio = a_count / len(results)
        # Should be approximately 0.8 with some tolerance
        assert 0.65 < ratio < 0.95, f"Expected ~80% to model A, got {ratio:.2%}"

    def test_get_results(self, ab_router):
        ab_router.configure(ABTestConfig("v1", "v2", traffic_split=0.5, enabled=True))
        for _ in range(50):
            ab_router.route_prediction([5.1, 3.5, 1.4, 0.2])

        results = ab_router.get_results()
        assert results.total_requests == 50
        assert results.model_a_count + results.model_b_count == 50
        assert results.model_a_avg_latency >= 0
        assert results.model_b_avg_latency >= 0

    def test_reset_results(self, ab_router):
        ab_router.configure(ABTestConfig("v1", "v2", traffic_split=0.5, enabled=True))
        for _ in range(10):
            ab_router.route_prediction([5.1, 3.5, 1.4, 0.2])

        ab_router.reset_results()
        results = ab_router.get_results()
        assert results.total_requests == 0
        assert results.model_a_count == 0
        assert results.model_b_count == 0

    def test_configure_resets_results(self, ab_router):
        ab_router.configure(ABTestConfig("v1", "v2", traffic_split=0.5, enabled=True))
        for _ in range(10):
            ab_router.route_prediction([5.1, 3.5, 1.4, 0.2])

        # Reconfigure should reset
        ab_router.configure(ABTestConfig("v1", "v2", traffic_split=0.7, enabled=True))
        results = ab_router.get_results()
        assert results.total_requests == 0

    def test_all_traffic_to_a(self, ab_router):
        ab_router.configure(ABTestConfig("v1", "v2", traffic_split=1.0, enabled=True))
        results = [ab_router.route_prediction([5.1, 3.5, 1.4, 0.2]) for _ in range(20)]
        versions = {r.model_version for r in results}
        assert versions == {"v1"}

    def test_all_traffic_to_b(self, ab_router):
        ab_router.configure(ABTestConfig("v1", "v2", traffic_split=0.0, enabled=True))
        results = [ab_router.route_prediction([5.1, 3.5, 1.4, 0.2]) for _ in range(20)]
        versions = {r.model_version for r in results}
        assert versions == {"v2"}
