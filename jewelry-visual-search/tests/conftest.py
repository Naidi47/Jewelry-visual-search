"""
Pytest configuration and shared fixtures.
"""

import os
from pathlib import Path

import pytest
import torch
from PIL import Image


@pytest.fixture(scope="session")
def test_image_dir():
    """Directory containing test images."""
    return Path(__file__).parent / "fixtures" / "images"


@pytest.fixture(scope="session")
def create_test_image():
    """Factory to create synthetic test images."""
    def _create(width=224, height=224, color=(128, 128, 128)):
        import io
        img = Image.new("RGB", (width, height), color=color)
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)
        return buf.getvalue()
    return _create


@pytest.fixture
def sample_jpeg(create_test_image):
    """A sample JPEG image for testing."""
    return create_test_image(color=(200, 150, 100))


@pytest.fixture
def mock_embedding():
    """A normalized 512-dim vector for testing."""
    import numpy as np
    vec = np.random.randn(512).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    return vec.tolist()
