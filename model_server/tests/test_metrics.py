"""Tests for Prometheus metrics and data drift detection."""

import numpy as np
import pytest
from prometheus_client import CollectorRegistry, REGISTRY

from app.monitoring.metrics import MetricsCollector
from app.monitoring.data_drift import DriftDetector, DriftResult


class TestMetricsCollector:
    def test_record_prediction(self):
        collector = MetricsCollector()
        # Should not raise
        collector.record_prediction(
            model_version="v1",
            latency=0.05,
            confidence=0.92,
            status="success",
        )

    def test_record_error(self):
        collector = MetricsCollector()
        collector.record_error("validation_error")

    def test_set_model_info(self):
        collector = MetricsCollector()
        collector.set_model_info(version="v1", framework="sklearn")

    def test_set_active_model(self):
        collector = MetricsCollector()
        collector.set_active_model("v1")

    def test_record_drift_score(self):
        collector = MetricsCollector()
        collector.record_drift_score(0.05)

    def test_multiple_collectors_share_metrics(self):
        """Class-level metrics should be shared across instances."""
        c1 = MetricsCollector()
        c2 = MetricsCollector()
        assert c1.REQUEST_COUNT is c2.REQUEST_COUNT
        assert c1.REQUEST_LATENCY is c2.REQUEST_LATENCY


class TestDriftDetector:
    def test_set_reference(self):
        data = np.random.randn(100, 4)
        detector = DriftDetector(threshold=0.1, window_size=50)
        detector.set_reference(data)
        assert detector._n_features == 4

    def test_set_reference_invalid_shape(self):
        detector = DriftDetector()
        with pytest.raises(ValueError, match="2-D"):
            detector.set_reference(np.array([1, 2, 3]))

    def test_set_reference_too_few_samples(self):
        detector = DriftDetector()
        with pytest.raises(ValueError, match="at least 2"):
            detector.set_reference(np.array([[1, 2, 3]]))

    def test_add_observation(self):
        data = np.random.randn(50, 3)
        detector = DriftDetector(reference_data=data, window_size=100)
        detector.add_observation([1.0, 2.0, 3.0])
        assert len(detector._window) == 1

    def test_add_observation_wrong_features(self):
        data = np.random.randn(50, 3)
        detector = DriftDetector(reference_data=data, window_size=100)
        with pytest.raises(ValueError, match="Expected 3 features"):
            detector.add_observation([1.0, 2.0])

    def test_compute_drift_no_reference(self):
        detector = DriftDetector()
        with pytest.raises(RuntimeError, match="No reference"):
            detector.compute_drift()

    def test_compute_drift_insufficient_window(self):
        data = np.random.randn(50, 3)
        detector = DriftDetector(reference_data=data)
        detector.add_observation([1.0, 2.0, 3.0])
        with pytest.raises(RuntimeError, match="at least 2"):
            detector.compute_drift()

    def test_no_drift_similar_distribution(self):
        """Same distribution should show low drift."""
        rng = np.random.default_rng(42)
        reference = rng.standard_normal((500, 4))
        detector = DriftDetector(reference_data=reference, threshold=0.5, window_size=200)

        # Add observations from the same distribution
        for _ in range(200):
            detector.add_observation(rng.standard_normal(4).tolist())

        result = detector.compute_drift()
        assert isinstance(result, DriftResult)
        assert not result.is_drifting
        assert result.kl_divergence < 0.5
        assert len(result.feature_scores) == 4

    def test_drift_detected_shifted_distribution(self):
        """Shifted distribution should be detected as drift."""
        rng = np.random.default_rng(42)
        reference = rng.standard_normal((500, 4))
        detector = DriftDetector(reference_data=reference, threshold=0.1, window_size=200)

        # Add observations from a shifted distribution
        for _ in range(200):
            obs = (rng.standard_normal(4) + 5.0).tolist()  # Large shift
            detector.add_observation(obs)

        result = detector.compute_drift()
        assert result.is_drifting
        assert result.kl_divergence > 0.1

    def test_sliding_window_eviction(self):
        data = np.random.randn(50, 2)
        detector = DriftDetector(reference_data=data, window_size=10)

        for i in range(20):
            detector.add_observation([float(i), float(i)])

        assert len(detector._window) == 10

    def test_constructor_with_reference(self):
        data = np.random.randn(50, 3)
        detector = DriftDetector(reference_data=data, threshold=0.2, window_size=500)
        assert detector.threshold == 0.2
        assert detector.window_size == 500
        assert detector._n_features == 3
