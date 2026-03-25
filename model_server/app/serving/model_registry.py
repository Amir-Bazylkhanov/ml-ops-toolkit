"""
Local model registry for versioned model storage and lifecycle management.

Models are stored in a directory tree::

    models_dir/
        v1/
            model.pkl        # sklearn model (joblib)
            metadata.json
        v2/
            model.pt         # pytorch model (torch.save)
            metadata.json
        active_version.txt   # tracks the currently active version

Thread-safe: all mutations are protected by a ``threading.Lock``.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Supported frameworks and the file name used for each.
_FRAMEWORK_FILENAME: dict[str, str] = {
    "sklearn": "model.pkl",
    "pytorch": "model.pt",
}

_ACTIVE_VERSION_FILE = "active_version.txt"
_METADATA_FILE = "metadata.json"


@dataclass
class ModelMetadata:
    """Immutable snapshot of the metadata stored alongside a model artifact.

    Attributes:
        version: Version string (e.g. ``"v1"``).
        framework: ML framework — either ``"sklearn"`` or ``"pytorch"``.
        created_at: ISO-8601 UTC timestamp of when the model was registered.
        metrics: Arbitrary evaluation metrics (e.g. ``{"accuracy": 0.95}``).
        description: Human-readable description of the model.
    """

    version: str
    framework: str
    created_at: str
    metrics: dict[str, float] = field(default_factory=dict)
    description: str = ""

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelMetadata":
        """Reconstruct a :class:`ModelMetadata` from a plain dictionary."""
        return cls(
            version=data["version"],
            framework=data["framework"],
            created_at=data["created_at"],
            metrics=data.get("metrics", {}),
            description=data.get("description", ""),
        )

    @classmethod
    def from_json_file(cls, path: Path) -> "ModelMetadata":
        """Load metadata from a ``metadata.json`` file on disk."""
        with path.open("r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def save_to_dir(self, directory: Path) -> None:
        """Persist metadata as ``metadata.json`` inside *directory*."""
        dest = directory / _METADATA_FILE
        with dest.open("w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        logger.debug("Saved metadata for version %s to %s", self.version, dest)


class ModelRegistryError(Exception):
    """Base exception for all registry-related errors."""


class VersionNotFoundError(ModelRegistryError):
    """Raised when a requested model version does not exist in the registry."""


class InvalidFrameworkError(ModelRegistryError):
    """Raised when an unsupported framework is specified."""


class ModelRegistry:
    """Manages versioned model artefacts stored on the local filesystem.

    Parameters:
        models_dir: Root directory that contains the versioned sub-directories.
            Created automatically if it does not exist.

    Example::

        registry = ModelRegistry("/opt/models")
        meta = registry.register_model(
            version="v1",
            model_path="/tmp/my_model.pkl",
            framework="sklearn",
            metrics={"accuracy": 0.93},
            description="Logistic regression baseline",
        )
        model, meta = registry.load_model("v1")
    """

    def __init__(self, models_dir: str) -> None:
        self._root = Path(models_dir)
        self._root.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()

        # In-memory cache: version string -> ModelMetadata
        self._metadata_cache: dict[str, ModelMetadata] = {}

        self._active_version: str | None = None
        self._previous_version: str | None = None

        self._scan_models()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_model(
        self,
        version: str,
        model_path: str,
        framework: str,
        metrics: dict[str, float],
        description: str = "",
    ) -> ModelMetadata:
        """Copy a model artefact into the registry and record its metadata.

        Parameters:
            version: Unique version identifier (e.g. ``"v3"``).
            model_path: Absolute or relative path to the model file to register.
            framework: ``"sklearn"`` or ``"pytorch"``.
            metrics: Evaluation metrics to store alongside the model.
            description: Optional free-text description.

        Returns:
            The newly created :class:`ModelMetadata`.

        Raises:
            InvalidFrameworkError: If *framework* is not supported.
            FileNotFoundError: If *model_path* does not exist.
            ModelRegistryError: If *version* is already registered.
        """
        if framework not in _FRAMEWORK_FILENAME:
            raise InvalidFrameworkError(
                f"Unsupported framework '{framework}'. "
                f"Choose one of: {list(_FRAMEWORK_FILENAME)}"
            )

        source = Path(model_path).resolve()
        if not source.exists():
            raise FileNotFoundError(f"Model file not found: {source}")

        # Validate version string to prevent path traversal.
        if not version or "/" in version or "\\" in version or version.startswith("."):
            raise ModelRegistryError(
                f"Invalid version string '{version}'. "
                "Version must not contain path separators or start with '.'"
            )

        with self._lock:
            if version in self._metadata_cache:
                raise ModelRegistryError(
                    f"Version '{version}' is already registered. "
                    "Unregister it first or use a different version string."
                )

            version_dir = self._root / version
            # Ensure resolved path is inside the registry root.
            if not version_dir.resolve().is_relative_to(self._root.resolve()):
                raise ModelRegistryError(
                    f"Version directory escapes the registry root: {version_dir}"
                )
            version_dir.mkdir(parents=True, exist_ok=True)

            dest_filename = _FRAMEWORK_FILENAME[framework]
            dest = version_dir / dest_filename
            shutil.copy2(source, dest)
            logger.info("Copied model artefact %s -> %s", source, dest)

            metadata = ModelMetadata(
                version=version,
                framework=framework,
                created_at=datetime.now(timezone.utc).isoformat(),
                metrics=metrics,
                description=description,
            )
            metadata.save_to_dir(version_dir)

            self._metadata_cache[version] = metadata

        logger.info("Registered model version '%s' (framework=%s)", version, framework)
        return metadata

    def load_model(self, version: str) -> tuple[Any, ModelMetadata]:
        """Load a model artefact from disk and return it with its metadata.

        For sklearn models the artefact is loaded with :func:`joblib.load`.
        For pytorch models the artefact is loaded with :func:`torch.load`.

        Parameters:
            version: The version to load.

        Returns:
            A ``(model, metadata)`` tuple.

        Raises:
            VersionNotFoundError: If *version* is not in the registry.
        """
        with self._lock:
            metadata = self._require_version(version)
            artefact_path = self._artefact_path(version, metadata.framework)

        model = self._load_artefact(artefact_path, metadata.framework)
        logger.info(
            "Loaded model version '%s' from %s", version, artefact_path
        )
        return model, metadata

    def list_versions(self) -> list[ModelMetadata]:
        """Return metadata for all registered versions, sorted by *created_at*.

        Returns:
            A list of :class:`ModelMetadata` instances, oldest first.
        """
        with self._lock:
            return sorted(
                self._metadata_cache.values(),
                key=lambda m: m.created_at,
            )

    def get_active_version(self) -> str:
        """Return the currently active version string.

        Raises:
            ModelRegistryError: If no active version has been set.
        """
        with self._lock:
            if self._active_version is None:
                raise ModelRegistryError(
                    "No active version is set. Call set_active_version() first."
                )
            return self._active_version

    def set_active_version(self, version: str) -> None:
        """Promote *version* to the active model without restarting the server.

        The previous active version is remembered so that :meth:`rollback` can
        revert to it instantly.

        Parameters:
            version: The version to promote.

        Raises:
            VersionNotFoundError: If *version* is not registered.
        """
        with self._lock:
            self._require_version(version)
            self._previous_version = self._active_version
            self._active_version = version
            self._persist_active_version(version)

        logger.info(
            "Active version changed: %s -> %s",
            self._previous_version,
            version,
        )

    def rollback(self) -> str:
        """Revert the active version to the one that was active before the
        last call to :meth:`set_active_version`.

        Returns:
            The version string that has been restored as active.

        Raises:
            ModelRegistryError: If there is no previous version to roll back to.
        """
        with self._lock:
            if self._previous_version is None:
                raise ModelRegistryError(
                    "No previous version available for rollback."
                )
            self._require_version(self._previous_version)

            current = self._active_version
            self._active_version = self._previous_version
            self._previous_version = current
            self._persist_active_version(self._active_version)
            restored = self._active_version

        logger.info("Rolled back active version to '%s'", restored)
        return restored

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _scan_models(self) -> None:
        """Populate the in-memory cache by scanning *models_dir* on startup."""
        loaded: dict[str, ModelMetadata] = {}

        for candidate in sorted(self._root.iterdir()):
            if not candidate.is_dir():
                continue
            metadata_file = candidate / _METADATA_FILE
            if not metadata_file.exists():
                logger.debug(
                    "Skipping directory %s — no %s found",
                    candidate,
                    _METADATA_FILE,
                )
                continue
            try:
                meta = ModelMetadata.from_json_file(metadata_file)
                loaded[meta.version] = meta
                logger.debug("Discovered model version '%s'", meta.version)
            except (KeyError, json.JSONDecodeError) as exc:
                logger.warning(
                    "Could not parse metadata in %s: %s", metadata_file, exc
                )

        self._metadata_cache = loaded
        logger.info(
            "Registry scan complete — %d version(s) found", len(loaded)
        )

        # Restore active version from persistent marker file if present.
        active_file = self._root / _ACTIVE_VERSION_FILE
        if active_file.exists():
            persisted = active_file.read_text(encoding="utf-8").strip()
            if persisted in self._metadata_cache:
                self._active_version = persisted
                logger.info("Restored active version '%s' from disk", persisted)
            else:
                logger.warning(
                    "Persisted active version '%s' is not in the registry; "
                    "ignoring.",
                    persisted,
                )

    def _persist_active_version(self, version: str) -> None:
        """Write the active version string to ``active_version.txt``."""
        active_file = self._root / _ACTIVE_VERSION_FILE
        active_file.write_text(version, encoding="utf-8")

    def _require_version(self, version: str) -> ModelMetadata:
        """Return metadata for *version*, raising if it is unknown.

        Must be called while holding ``self._lock``.
        """
        try:
            return self._metadata_cache[version]
        except KeyError:
            raise VersionNotFoundError(
                f"Version '{version}' is not registered. "
                f"Available versions: {list(self._metadata_cache)}"
            ) from None

    def _artefact_path(self, version: str, framework: str) -> Path:
        """Return the full path to a model artefact file.

        Must be called while holding ``self._lock``.
        """
        filename = _FRAMEWORK_FILENAME[framework]
        path = self._root / version / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Artefact file missing for version '{version}': {path}"
            )
        return path

    @staticmethod
    def _load_artefact(path: Path, framework: str) -> Any:
        """Deserialise a model artefact from disk.

        Imports heavy dependencies lazily to keep startup time fast when
        only one framework is actually used.
        """
        if framework == "sklearn":
            import joblib  # type: ignore[import]

            return joblib.load(path)

        if framework == "pytorch":
            import torch  # type: ignore[import]

            # weights_only=False is required when the checkpoint contains
            # full model objects rather than just a state_dict.
            return torch.load(path, map_location="cpu", weights_only=False)

        raise InvalidFrameworkError(f"Cannot load artefact for framework '{framework}'")

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        versions = list(self._metadata_cache)
        return (
            f"ModelRegistry(models_dir={str(self._root)!r}, "
            f"versions={versions}, "
            f"active_version={self._active_version!r})"
        )

    def __len__(self) -> int:
        with self._lock:
            return len(self._metadata_cache)
