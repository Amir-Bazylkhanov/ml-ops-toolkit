"""
Generic predictor that dispatches inference to the correct framework backend.

Supports:
- **sklearn**: calls ``model.predict`` / ``model.predict_proba``.
- **pytorch**: runs a forward pass inside ``torch.no_grad()`` and applies
  softmax to obtain per-class confidence scores.

Latency is measured with :func:`time.perf_counter` and stored on every
:class:`PredictionResult`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.serving.model_registry import (
    InvalidFrameworkError,
    ModelMetadata,
    ModelRegistry,
)

logger = logging.getLogger(__name__)

_CACHE_TTL = 300  # seconds


def _make_cache_key(version: str, features: list[float]) -> str:
    payload = json.dumps([version, features], separators=(",", ":"))
    return "pred:" + hashlib.sha256(payload.encode()).hexdigest()


@dataclass
class PredictionResult:
    """Value object returned by every inference call.

    Attributes:
        prediction: Raw model output.  For classification this is the
            predicted class label; for regression it is the predicted scalar.
        confidence: Probability of the top predicted class (0.0–1.0).
            For regression models where probability is not available this
            defaults to ``0.0``.
        model_version: The registry version that produced this prediction.
        latency_ms: End-to-end inference latency in milliseconds.
        timestamp: ISO-8601 UTC timestamp of when the prediction was made.
    """

    prediction: Any
    confidence: float
    model_version: str
    latency_ms: float
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class PredictorError(Exception):
    """Raised when the predictor cannot fulfil an inference request."""


class Predictor:
    """Framework-agnostic inference engine backed by a :class:`ModelRegistry`.

    The predictor caches loaded models in memory.  When *model_version* is
    ``None`` the registry's active version is used automatically.

    Parameters:
        registry: The model registry to source models from.

    Example::

        registry = ModelRegistry("/opt/models")
        predictor = Predictor(registry)
        result = predictor.predict([1.0, 2.3, 0.7])
        print(result.prediction, result.confidence, result.latency_ms)
    """

    def __init__(self, registry: ModelRegistry, redis_url: str | None = None) -> None:
        self._registry = registry
        # Simple LRU-like in-process model cache: version -> (model, metadata)
        self._model_cache: dict[str, tuple[Any, ModelMetadata]] = {}

        self._redis = None
        if redis_url:
            try:
                import redis as redis_lib
                self._redis = redis_lib.from_url(redis_url, decode_responses=True)
                self._redis.ping()
                logger.info("Predictor: Redis cache connected at %s", redis_url)
            except Exception as exc:
                logger.warning("Predictor: Redis unavailable, cache disabled: %s", exc)
                self._redis = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        features: list[float],
        model_version: str | None = None,
    ) -> PredictionResult:
        """Run inference on *features* and return a :class:`PredictionResult`.

        Parameters:
            features: Input feature vector as a flat list of floats.
            model_version: Registry version to use.  Defaults to the active
                version when ``None``.

        Returns:
            A :class:`PredictionResult` populated with prediction, confidence,
            version, latency and timestamp.

        Raises:
            PredictorError: If the model cannot be loaded or inference fails.
        """
        version = self._resolve_version(model_version)
        model, metadata = self._get_or_load_model(version)

        cache_key = _make_cache_key(version, features)
        if self._redis is not None:
            try:
                cached = self._redis.get(cache_key)
                if cached is not None:
                    return PredictionResult(**json.loads(cached))
            except Exception as exc:
                logger.warning("Predictor: Redis read failed, skipping cache: %s", exc)

        t_start = time.perf_counter()
        try:
            prediction, confidence = self._run_inference(
                model, metadata.framework, features
            )
        except Exception as exc:
            raise PredictorError(
                f"Inference failed for version '{version}': {exc}"
            ) from exc
        latency_ms = (time.perf_counter() - t_start) * 1_000

        result = PredictionResult(
            prediction=prediction,
            confidence=round(confidence, 6),
            model_version=version,
            latency_ms=round(latency_ms, 3),
        )

        if self._redis is not None:
            try:
                self._redis.setex(cache_key, _CACHE_TTL, json.dumps(result.__dict__))
            except Exception as exc:
                logger.warning("Predictor: Redis write failed, skipping cache: %s", exc)
        logger.debug(
            "Prediction: version=%s prediction=%s confidence=%.4f latency=%.2f ms",
            version,
            prediction,
            confidence,
            latency_ms,
        )
        return result

    def evict_cache(self, version: str | None = None) -> None:
        """Remove one or all entries from the in-process model cache.

        Parameters:
            version: The specific version to evict.  When ``None`` the entire
                cache is cleared.
        """
        if version is None:
            self._model_cache.clear()
            logger.info("Evicted all entries from predictor cache")
        else:
            self._model_cache.pop(version, None)
            logger.info("Evicted version '%s' from predictor cache", version)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_version(self, version: str | None) -> str:
        """Return *version* if provided, otherwise the registry active version."""
        if version is not None:
            return version
        try:
            return self._registry.get_active_version()
        except Exception as exc:
            raise PredictorError(
                "No model_version supplied and no active version is set "
                f"in the registry: {exc}"
            ) from exc

    def _get_or_load_model(self, version: str) -> tuple[Any, ModelMetadata]:
        """Return a cached model or load it from the registry on first access."""
        if version not in self._model_cache:
            try:
                model, metadata = self._registry.load_model(version)
            except Exception as exc:
                raise PredictorError(
                    f"Failed to load model version '{version}': {exc}"
                ) from exc
            self._model_cache[version] = (model, metadata)
            logger.info("Cached model version '%s' in predictor", version)
        return self._model_cache[version]

    @staticmethod
    def _run_inference(
        model: Any,
        framework: str,
        features: list[float],
    ) -> tuple[Any, float]:
        """Dispatch inference to the appropriate framework backend.

        Returns:
            A ``(prediction, confidence)`` tuple.
        """
        if framework == "sklearn":
            return Predictor._sklearn_inference(model, features)
        if framework == "pytorch":
            return Predictor._pytorch_inference(model, features)
        raise InvalidFrameworkError(
            f"No inference backend for framework '{framework}'"
        )

    @staticmethod
    def _sklearn_inference(
        model: Any, features: list[float]
    ) -> tuple[Any, float]:
        """Run inference using a scikit-learn compatible model.

        Tries ``predict_proba`` first for classification confidence.  Falls
        back gracefully to plain ``predict`` for regressors.
        """
        import numpy as np  # type: ignore[import]

        X = np.array(features, dtype=float).reshape(1, -1)
        prediction = model.predict(X)[0]

        confidence = 0.0
        if hasattr(model, "predict_proba"):
            try:
                probas = model.predict_proba(X)[0]
                confidence = float(np.max(probas))
            except Exception:
                # Some estimators expose predict_proba but may raise for
                # certain configurations — fall back silently.
                pass

        return prediction, confidence

    @staticmethod
    def _pytorch_inference(
        model: Any, features: list[float]
    ) -> tuple[Any, float]:
        """Run inference using a PyTorch model.

        Applies softmax to logit outputs to derive per-class probabilities.
        If the model output is a scalar (regression), confidence is set to
        ``0.0``.
        """
        import torch  # type: ignore[import]
        import torch.nn.functional as F

        model.eval()
        tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            output: torch.Tensor = model(tensor)

        # Squeeze batch dimension.
        output = output.squeeze(0)

        if output.dim() == 0 or output.numel() == 1:
            # Regression: single scalar output.
            prediction = output.item()
            confidence = 0.0
        else:
            # Classification: apply softmax and take argmax.
            probas = F.softmax(output, dim=0)
            predicted_class = int(torch.argmax(probas).item())
            confidence = float(probas[predicted_class].item())
            prediction = predicted_class

        return prediction, confidence

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        cached = list(self._model_cache)
        return (
            f"Predictor(registry={self._registry!r}, "
            f"cached_versions={cached})"
        )
