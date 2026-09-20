"""Shared test fixtures for CodeAtlas tests."""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_PROJECT = FIXTURES_DIR / "sample_project"


@pytest.fixture
def sample_project_path() -> Path:
    """Return the path to the sample project fixture."""
    return SAMPLE_PROJECT


@pytest.fixture
def sample_auth_middleware() -> Path:
    """Return the path to the sample auth middleware file."""
    return SAMPLE_PROJECT / "src" / "auth" / "middleware.py"


@pytest.fixture
def sample_recommendation_service() -> Path:
    """Return the path to the sample recommendation service file."""
    return SAMPLE_PROJECT / "src" / "recommendation" / "service.py"
