from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Proposal"]


@dataclass(slots=True)
class Proposal:
    selection: np.ndarray
    mask: np.ndarray
    key: bytes
    source: str
    violation: float
    score: float
