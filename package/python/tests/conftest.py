from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

REPO_ROOT = _PKG_ROOT.parents[1]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def models_dir() -> Path:
    return REPO_ROOT / "models"


@pytest.fixture(scope="session")
def instances_dir() -> Path:
    return REPO_ROOT / "instances"
