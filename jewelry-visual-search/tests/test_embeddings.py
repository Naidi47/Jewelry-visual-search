"""
Tests for the /v1/embeddings/generate endpoint.
"""

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_check():
    """Verify health endpoint returns expected structure."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "model_loaded" in data
    assert "mongodb_connected" in data


def test_generate_embedding_with_jpeg():
    """Test embedding generation with valid JPEG."""
    # Create a small test image
    from PIL import Image
    import io
    
    img = Image.new("RGB", (224, 224), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    
    response = client.post(
        "/v1/embeddings/generate",
        files={"image": ("test.jpg", buf, "image/jpeg")},
        data={"normalize": "true", "return_metadata": "true"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert len(data["embedding"]) == 512
    assert data["dimension"] == 512
    assert "metadata" in data
    assert data["metadata"]["normalized"] is True


def test_generate_embedding_invalid_format():
    """Test rejection of non-image files."""
    response = client.post(
        "/v1/embeddings/generate",
        files={"image": ("test.txt", io.BytesIO(b"not an image"), "text/plain")}
    )
    assert response.status_code == 400


def test_generate_embedding_oversized():
    """Test rejection of files exceeding size limit."""
    # Create a large dummy file
    large_content = b"x" * (11 * 1024 * 1024)  # 11 MB
    
    response = client.post(
        "/v1/embeddings/generate",
        files={"image": ("huge.jpg", io.BytesIO(large_content), "image/jpeg")}
    )
    assert response.status_code == 400
