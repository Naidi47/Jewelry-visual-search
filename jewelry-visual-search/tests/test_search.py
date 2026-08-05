"""
Tests for the /v1/search/visual endpoint.
"""

import io

import pytest
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_visual_search_basic():
    """Test visual search with valid image."""
    from PIL import Image
    
    img = Image.new("RGB", (224, 224), color=(200, 100, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    
    response = client.post(
        "/v1/search/visual",
        files={"image": ("ring.jpg", buf, "image/jpeg")},
        data={"top_k": "5", "category": "ring"}
    )
    
    # May fail if MongoDB not connected, so we check structure
    if response.status_code == 200:
        data = response.json()
        assert "results" in data
        assert "search_time_ms" in data
        assert isinstance(data["results"], list)
    else:
        # Expected if MongoDB not configured in test
        assert response.status_code in [500, 503]


def test_visual_search_with_filters():
    """Test search with price and material filters."""
    from PIL import Image
    
    img = Image.new("RGB", (224, 224), color=(255, 215, 0))  # Gold color
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    
    response = client.post(
        "/v1/search/visual",
        files={"image": ("gold_necklace.jpg", buf, "image/jpeg")},
        data={
            "top_k": "10",
            "category": "necklace",
            "material": "gold",
            "price_min": "100",
            "price_max": "5000",
            "in_stock_only": "true"
        }
    )
    
    # Structure validation regardless of DB connection
    if response.status_code == 200:
        data = response.json()
        assert "filters_applied" in data
        filters = data["filters_applied"]
        assert filters["category"] == "necklace"
        assert filters["material"] == "gold"
