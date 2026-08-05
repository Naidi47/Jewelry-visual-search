"""
End-to-end integration tests.
Requires running services (MongoDB, model loaded).
Skipped in CI if services unavailable.
"""

import os

import pytest

# Skip if no MongoDB URI configured
pytestmark = pytest.mark.skipif(
    not os.getenv("MONGODB_URI"),
    reason="MongoDB not configured for integration tests"
)


def test_full_pipeline():
    """
    Complete flow: generate embedding -> index product -> search.
    This test requires a real MongoDB Atlas connection.
    """
    pass  # Implementation depends on live services
