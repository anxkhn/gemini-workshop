"""Pytest fixtures and mock mode support."""

import os
import sys

import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def api_key():
    """Return API key from env or a fake one for mock tests."""
    return os.environ.get("GEMINI_API_KEY", "")


@pytest.fixture
def has_api_key(api_key):
    """Whether a real API key is available."""
    return bool(api_key)
