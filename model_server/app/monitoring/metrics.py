"""Prometheus metrics collection for the ML model server.

All metrics are defined as class-level attributes so they are registered
exactly once in the default Prometheus registry, regardless of how many
times MetricsCollector is instantiated.

Usage::

    collector = MetricsCollector()
    collector.set_model_info(version="v1.2", framework="sklearn")
    collector.set_active_model("v1.2")

    with collector.REQUEST_LATENCY.labels(model_version="v1.2").time():
        prediction = model.predict(features)

    collector.record_prediction(
        model_version="v1.2",
        latency=0.034,
        confidence=0.91,
        status="success",
    )
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, Info


class MetricsCollector:
    """Centralised Prometheus metrics for a single model server instance.

    All metric objects are class-level descriptors registered once in the
    default Prometheus CollectorRegistry.  Instantiating multiple
    MetricsCollector objects is safe and shares the same underlying
    metrics.
    """

    # ------------------------------------------------------------------
    # Metric definitions
    # ------------------------------------------------------------------

    REQUEST_COUNT: Counter = Counter(
        "model_request_total",
        "Total prediction requests",
        ["model_version", "status"],
    )

    REQUEST_LATENCY: Histogram = Histogram(
        "model_request_latency_seconds",
        "Request latency",
        ["model_version"],
        buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
    )

    PREDICTION_CONFIDENCE: Histogram = Histogram(
        "model_prediction_confidence",
        "Prediction confidence scores",
        ["model_version"],
        buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
    )

    MODEL_INFO: Info = Info(
        "model_server",
        "Model server information",
    )

    ERRORS_TOTAL: Counter = Counter(
        "model_errors_total",
        "Total errors",
        ["error_type"],
    )

    ACTIVE_MODEL: Gauge = Gauge(
        "model_active_version",
        "Currently active model version indicator",
        ["version"],
    )

    DRIFT_SCORE: Gauge = Gauge(
        "model_data_drift_score",
        "Current data drift KL divergence score",
    )

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def record_prediction(
        self,
        model_version: str,
        latency: float,
        confidence: float,
        status: str = "success",
    ) -> None:
        """Record a single prediction event.

        Args:
            model_version: Version label for the model that produced the
                prediction (e.g. ``"v1.2"``).
            latency: Wall-clock time taken for the request in seconds.
            confidence: Confidence score of the prediction in ``[0, 1]``.
            status: Outcome label, typically ``"success"`` or ``"error"``.
        """
        self.REQUEST_COUNT.labels(
            model_version=model_version, status=status
        ).inc()
        self.REQUEST_LATENCY.labels(model_version=model_version).observe(latency)
        self.PREDICTION_CONFIDENCE.labels(model_version=model_version).observe(
            confidence
        )

    def record_error(self, error_type: str) -> None:
        """Increment the error counter for a given error category.

        Args:
            error_type: Short descriptor for the error class, e.g.
                ``"validation_error"`` or ``"model_load_failure"``.
        """
        self.ERRORS_TOTAL.labels(error_type=error_type).inc()

    def set_model_info(self, version: str, framework: str) -> None:
        """Publish static model metadata to the info metric.

        Args:
            version: Human-readable version string (e.g. ``"v1.2"``).
            framework: ML framework used (e.g. ``"sklearn"``, ``"torch"``).
        """
        self.MODEL_INFO.info({"version": version, "framework": framework})

    _current_active_version: str | None = None

    def set_active_model(self, version: str) -> None:
        """Mark *version* as the currently active model.

        Sets the gauge for *version* to ``1`` and resets the previously
        active version label to ``0`` so that only one label is active
        at any time.

        Args:
            version: Version string of the model now serving traffic.
        """
        prev = MetricsCollector._current_active_version
        if prev is not None and prev != version:
            self.ACTIVE_MODEL.labels(version=prev).set(0)
        self.ACTIVE_MODEL.labels(version=version).set(1)
        MetricsCollector._current_active_version = version

    def record_drift_score(self, score: float) -> None:
        """Update the current data-drift KL-divergence gauge.

        Args:
            score: Non-negative KL divergence value.  Higher values
                indicate greater divergence from the reference distribution.
        """
        self.DRIFT_SCORE.set(score)
