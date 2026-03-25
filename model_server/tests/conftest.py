"""Shared test fixtures."""

import sys
from pathlib import Path

# Ensure the model_server package is importable in tests
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
