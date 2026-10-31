"""PyTorch models (optional dependency: ``pip install torch``)."""

from __future__ import annotations

import importlib.util

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

__all__ = ["TORCH_AVAILABLE"]
