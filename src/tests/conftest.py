from __future__ import annotations

import json
from pathlib import Path

import pytest

BASELINES_PATH = Path(__file__).parent / "baselines" / "metrics.json"


@pytest.fixture(scope="session")
def baselines() -> dict:
    return json.loads(BASELINES_PATH.read_text())
