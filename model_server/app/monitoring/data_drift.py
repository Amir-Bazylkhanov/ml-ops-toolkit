"""Sliding-window data drift detection using KL divergence.

The detector maintains a fixed-size window of recent observations and
computes per-feature KL divergence against a reference distribution
captured at deployment time.

Usage::

    import numpy as np
    from app.monitoring.data_drift import DriftDetector

    rng = np.random.default_rng(42)
    reference = rng.standard_normal((500, 4))

    detector = DriftDetector(threshold=0.1, window_size=200)
    detector.set_reference(reference)

    for _ in range(250):
        obs = rng.standard_normal(4).tolist()
        detector.add_observation(obs)

    result = detector.compute_drift()
    print(result.kl_divergence, result.is_drifting)
"""

from __future__ import annotations

import collections
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from scipy.stats import entropy

logger = logging.getLogger(__name__)

# Number of histogram bins used when estimating feature distributions.
_N_BINS: int = 20

# Small epsilon added to histogram counts to avoid log(0) in KL divergence.
_EPSILON: float = 1e-10


@dataclass
class DriftResult:
    """Result produced by a single call to :meth:`DriftDetector.compute_drift`.

    Attributes:
        kl_divergence: Mean KL divergence across all features.
        is_drifting: ``True`` when *kl_divergence* exceeds the configured
            threshold.
        feature_scores: Per-feature KL divergence values, ordered to match
            the columns of the reference array.
        timestamp: UTC time at which drift was computed.
    """

    kl_divergence: float
    is_drifting: bool
    feature_scores: list[float]
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )


class DriftDetector:
    """Detect input data drift relative to a captured reference distribution.

    The detector bins each feature into a fixed histogram estimated from the
    reference dataset and compares the same binning applied to the sliding
    window of recent observations using KL divergence (``scipy.stats.entropy``
    with a ``qk`` argument corresponds to D_KL(P || Q)).

    Args:
        reference_data: Optional initial reference dataset.  Rows are
            observations; columns are features.  Can also be set later via
            :meth:`set_reference`.
        threshold: KL divergence value above which the mean feature drift is
            considered significant.  Defaults to ``0.1``.
        window_size: Maximum number of observations retained in the sliding
            window.  Older observations are evicted as new ones are added.
            Defaults to ``1000``.
    """

    def __init__(
        self,
        reference_data: Optional[np.ndarray] = None,
        threshold: float = 0.1,
        window_size: int = 1000,
    ) -> None:
        self.threshold = threshold
        self.window_size = window_size

        # Sliding window: each element is a 1-D array of feature values.
        self._window: collections.deque[np.ndarray] = collections.deque(
            maxlen=window_size
        )

        # Reference distribution stored as per-feature (bin_edges, counts).
        self._reference_bins: Optional[list[tuple[np.ndarray, np.ndarray]]] = None
        self._n_features: int = 0

        if reference_data is not None:
            self.set_reference(reference_data)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def set_reference(self, data: np.ndarray) -> None:
        """Capture the reference distribution from *data*.

        Args:
            data: 2-D array of shape ``(n_samples, n_features)``.  At least
                two samples are required to estimate a meaningful distribution.

        Raises:
            ValueError: If *data* has fewer than 2 rows or is not 2-D.
        """
        data = np.asarray(data, dtype=float)
        if data.ndim != 2:
            raise ValueError(
                f"reference_data must be 2-D, got shape {data.shape}"
            )
        if data.shape[0] < 2:
            raise ValueError(
                "reference_data must contain at least 2 observations"
            )

        self._n_features = data.shape[1]
        self._reference_bins = []

        for col_idx in range(self._n_features):
            col = data[:, col_idx]
            counts, edges = np.histogram(col, bins=_N_BINS)
            self._reference_bins.append((edges, counts.astype(float)))

        logger.info(
            "DriftDetector: reference distribution set from %d observations "
            "with %d features.",
            data.shape[0],
            self._n_features,
        )

    def add_observation(self, features: list[float]) -> None:
        """Append a single observation to the sliding window.

        Args:
            features: Feature vector whose length must match the number of
                columns in the reference dataset.

        Raises:
            ValueError: If the length of *features* does not match the
                reference feature count (when a reference has been set).
        """
        obs = np.asarray(features, dtype=float)
        if self._n_features and obs.shape[0] != self._n_features:
            raise ValueError(
                f"Expected {self._n_features} features, got {obs.shape[0]}"
            )
        self._window.append(obs)

    def compute_drift(self) -> DriftResult:
        """Compute KL divergence between the reference and the current window.

        The window observations are binned using the *same* bin edges as the
        reference distribution so that the two histograms are directly
        comparable.

        Returns:
            A :class:`DriftResult` with per-feature and mean KL divergence
            values.

        Raises:
            RuntimeError: If no reference distribution has been set or the
                window contains fewer than 2 observations.
        """
        if self._reference_bins is None:
            raise RuntimeError(
                "No reference distribution set.  Call set_reference() first."
            )
        if len(self._window) < 2:
            raise RuntimeError(
                "The sliding window must contain at least 2 observations "
                "before drift can be computed."
            )

        window_array = np.stack(list(self._window))  # (n_obs, n_features)
        feature_scores: list[float] = []

        for col_idx, (edges, ref_counts) in enumerate(self._reference_bins):
            window_col = window_array[:, col_idx]
            # Bin window data using the reference edges; clip to avoid
            # observations outside the reference range being lost.
            window_counts, _ = np.histogram(window_col, bins=edges)
            window_counts = window_counts.astype(float)

            # Normalise to probability distributions, guarding against zeros.
            ref_prob = (ref_counts + _EPSILON) / (
                ref_counts.sum() + _EPSILON * len(ref_counts)
            )
            win_prob = (window_counts + _EPSILON) / (
                window_counts.sum() + _EPSILON * len(window_counts)
            )

            # D_KL(window || reference)
            kl = float(entropy(win_prob, ref_prob))
            feature_scores.append(kl)

        mean_kl = float(np.mean(feature_scores))
        is_drifting = mean_kl > self.threshold

        if is_drifting:
            logger.warning(
                "DriftDetector: data drift detected.  "
                "mean_kl=%.4f threshold=%.4f",
                mean_kl,
                self.threshold,
            )

        return DriftResult(
            kl_divergence=mean_kl,
            is_drifting=is_drifting,
            feature_scores=feature_scores,
        )
