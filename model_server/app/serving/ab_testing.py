"""
A/B testing router for comparing two model versions in production traffic.

Traffic is split stochastically: each request is independently assigned to
model A with probability ``config.traffic_split`` and to model B otherwise.

All results are stored in thread-safe lists so that aggregate statistics can
be queried at any time via :meth:`ABTestRouter.get_results`.

Example::

    from app.serving import Predictor, ABTestRouter
    from app.serving.ab_testing import ABTestConfig

    router = ABTestRouter(predictor)
    router.configure(ABTestConfig(
        model_a_version="v1",
        model_b_version="v2",
        traffic_split=0.5,
        enabled=True,
    ))

    result = router.route_prediction([1.0, 2.3, 0.7])
    stats = router.get_results()
"""

from __future__ import annotations

import logging
import random
import threading
from dataclasses import dataclass, field

from app.serving.predictor import PredictionResult, Predictor

logger = logging.getLogger(__name__)


@dataclass
class ABTestConfig:
    """Configuration for an A/B test.

    Attributes:
        model_a_version: Registry version string for model A (control).
        model_b_version: Registry version string for model B (treatment).
        traffic_split: Fraction of requests routed to model A (0.0–1.0).
            For example ``0.9`` means 90 % of traffic goes to A and 10 % to B.
        enabled: When ``False`` all traffic is forwarded to model A (safe
            default that disables the experiment without removing config).
    """

    model_a_version: str
    model_b_version: str
    traffic_split: float = 0.5
    enabled: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.traffic_split <= 1.0:
            raise ValueError(
                f"traffic_split must be in [0, 1], got {self.traffic_split}"
            )


@dataclass
class ABTestResult:
    """Aggregate statistics collected during an A/B test.

    Attributes:
        total_requests: Total number of routed requests since last reset.
        model_a_count: Number of requests served by model A.
        model_b_count: Number of requests served by model B.
        model_a_avg_latency: Mean latency (ms) for model A requests.
        model_b_avg_latency: Mean latency (ms) for model B requests.
        model_a_avg_confidence: Mean prediction confidence for model A.
        model_b_avg_confidence: Mean prediction confidence for model B.
    """

    total_requests: int
    model_a_count: int
    model_b_count: int
    model_a_avg_latency: float
    model_b_avg_latency: float
    model_a_avg_confidence: float
    model_b_avg_confidence: float


class ABTestRouterError(Exception):
    """Raised when the router is in an invalid state."""


class ABTestRouter:
    """Routes prediction requests to one of two model versions.

    The routing decision is made independently for every request using
    Python's :func:`random.random`, ensuring that the observed traffic split
    converges to ``config.traffic_split`` in expectation.

    Parameters:
        predictor: The :class:`~app.serving.predictor.Predictor` used to
            execute inference for both model variants.

    Thread safety:
        The internal result lists are guarded by a ``threading.Lock`` so the
        router is safe to use from multiple threads simultaneously.
    """

    def __init__(self, predictor: Predictor) -> None:
        self._predictor = predictor
        self._config: ABTestConfig | None = None
        self._lock = threading.Lock()

        # Per-model result accumulators.  Separate lists per metric allow
        # O(1) appends and a single pass to compute averages.
        self._a_latencies: list[float] = []
        self._b_latencies: list[float] = []
        self._a_confidences: list[float] = []
        self._b_confidences: list[float] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def configure(self, config: ABTestConfig) -> None:
        """Install a new :class:`ABTestConfig` and reset collected data.

        Calling this mid-experiment implicitly resets the accumulated results
        so that statistics always correspond to the current configuration.

        Parameters:
            config: The A/B test configuration to apply.
        """
        with self._lock:
            self._config = config
            self._clear_results_locked()

        logger.info(
            "A/B test configured: A=%s B=%s split=%.2f enabled=%s",
            config.model_a_version,
            config.model_b_version,
            config.traffic_split,
            config.enabled,
        )

    def route_prediction(self, features: list[float]) -> PredictionResult:
        """Route *features* to model A or B and return the result.

        The routing decision is:
        - If the test is *disabled*, always use model A.
        - Otherwise draw a uniform random float; use model A when it is
          strictly less than ``traffic_split``, model B otherwise.

        Parameters:
            features: Input feature vector passed directly to the predictor.

        Returns:
            A :class:`~app.serving.predictor.PredictionResult` from whichever
            model was selected.

        Raises:
            ABTestRouterError: If no configuration has been set.
        """
        config = self._require_config()

        if not config.enabled or random.random() < config.traffic_split:
            version = config.model_a_version
            bucket = "A"
        else:
            version = config.model_b_version
            bucket = "B"

        result = self._predictor.predict(features, model_version=version)

        with self._lock:
            if bucket == "A":
                self._a_latencies.append(result.latency_ms)
                self._a_confidences.append(result.confidence)
            else:
                self._b_latencies.append(result.latency_ms)
                self._b_confidences.append(result.confidence)

        logger.debug(
            "A/B router: bucket=%s version=%s latency=%.2f ms",
            bucket,
            version,
            result.latency_ms,
        )
        return result

    def get_results(self) -> ABTestResult:
        """Return a snapshot of the aggregated A/B test statistics.

        Returns:
            An :class:`ABTestResult` dataclass.  Averages are ``0.0`` when no
            requests have been recorded for that bucket yet.
        """
        with self._lock:
            a_count = len(self._a_latencies)
            b_count = len(self._b_latencies)
            return ABTestResult(
                total_requests=a_count + b_count,
                model_a_count=a_count,
                model_b_count=b_count,
                model_a_avg_latency=_safe_mean(self._a_latencies),
                model_b_avg_latency=_safe_mean(self._b_latencies),
                model_a_avg_confidence=_safe_mean(self._a_confidences),
                model_b_avg_confidence=_safe_mean(self._b_confidences),
            )

    def reset_results(self) -> None:
        """Clear all accumulated A/B test data.

        Useful between experiments or after a configuration change outside of
        :meth:`configure`.
        """
        with self._lock:
            self._clear_results_locked()
        logger.info("A/B test results reset")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_config(self) -> ABTestConfig:
        """Return the current config or raise if none has been set."""
        with self._lock:
            if self._config is None:
                raise ABTestRouterError(
                    "No A/B test configuration has been set. "
                    "Call configure() before routing predictions."
                )
            return self._config

    def _clear_results_locked(self) -> None:
        """Reset result lists.  Must be called while holding ``self._lock``."""
        self._a_latencies = []
        self._b_latencies = []
        self._a_confidences = []
        self._b_confidences = []

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        config = self._config
        if config is None:
            return "ABTestRouter(config=None)"
        return (
            f"ABTestRouter("
            f"model_a={config.model_a_version!r}, "
            f"model_b={config.model_b_version!r}, "
            f"traffic_split={config.traffic_split}, "
            f"enabled={config.enabled})"
        )


# ---------------------------------------------------------------------------
# Module-level utility
# ---------------------------------------------------------------------------


def _safe_mean(values: list[float]) -> float:
    """Return the arithmetic mean of *values*, or ``0.0`` for an empty list."""
    return sum(values) / len(values) if values else 0.0
