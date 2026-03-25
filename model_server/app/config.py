"""Application configuration loaded from environment variables or a .env file.

All settings can be overridden at runtime via environment variables whose
names match the field names uppercased (standard Pydantic Settings behaviour).

Usage::

    from app.config import get_settings

    settings = get_settings()
    print(settings.server_host, settings.server_port)

The :func:`get_settings` factory is wrapped with :func:`functools.lru_cache`
so that the ``.env`` file is parsed only once per process.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """ML model server configuration.

    Attributes:
        model_dir: Directory from which serialised model artefacts are loaded.
        redis_url: Connection URL for the Redis instance used by the A/B
            test router and request cache.
        log_level: Minimum log level forwarded to all loggers
            (``"DEBUG"``, ``"INFO"``, ``"WARNING"``, ``"ERROR"``).
        ab_test_enabled: When ``True``, the A/B test router is active and
            splits traffic between model versions.
        ab_test_split: Fraction of traffic routed to the *primary* (A)
            model version.  Must be in ``(0, 1)``.
        drift_threshold: Mean KL divergence above which input data is
            considered to have drifted from the reference distribution.
        server_host: Host address on which Uvicorn binds.
        server_port: TCP port on which Uvicorn listens.
    """

    model_dir: str = "./models"
    redis_url: str = "redis://redis:6379/0"
    log_level: str = "INFO"
    ab_test_enabled: bool = True
    ab_test_split: float = 0.8
    drift_threshold: float = 0.1
    server_host: str = "0.0.0.0"
    server_port: int = 8000

    model_config = {"env_file": ".env"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the application settings singleton.

    The first call parses the ``.env`` file and environment variables.
    Subsequent calls return the cached instance without re-reading the file.

    Returns:
        A fully populated :class:`Settings` instance.
    """
    return Settings()
