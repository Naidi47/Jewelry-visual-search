"""
Pydantic models for API request/response validation and documentation.
All models use strict typing for automatic OpenAPI generation.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class EmbeddingRequest(BaseModel):
    """
    Query parameters for embedding generation endpoint.
    
    Note: Image file comes as multipart upload, not in this body.
    """
    normalize: bool = Field(
        default=True,
        description="L2-normalize embedding to unit sphere"
    )
    return_metadata: bool = Field(
        default=False,
        description="Include processing timing and config"
    )


class EmbeddingMetadata(BaseModel):
    """Technical metadata about embedding generation."""
    model_name: str
    device: str
    inference_time_ms: float = Field(..., ge=0)
    input_image_size: tuple[int, int]
    normalized: bool


class EmbeddingResponse(BaseModel):
    """
    Standardized embedding response.
    
    Vector returned as JSON array for cross-platform compatibility.
    For storage efficiency in production, consider binary serialization.
    """
    embedding: List[float] = Field(
        ...,
        min_length=512,
        max_length=512,
        description="Dense vector representation [512]"
    )
    dimension: int = Field(default=512, ge=512, le=512)
    metadata: Optional[EmbeddingMetadata] = None


class SearchFilter(BaseModel):
    """
    Post-retrieval and vector search filters.
    
    Maps to MongoDB document fields with 'filter' type in search index.
    """
    category: Optional[str] = Field(
        default=None,
        description="Jewelry category for pre-filter",
        examples=["necklace", "ring", "bracelet", "earring", "pendant"]
    )
    material: Optional[str] = Field(
        default=None,
        description="Material for post-filter",
        examples=["gold", "silver", "platinum", "rose_gold"]
    )
    price_min: Optional[float] = Field(
        default=None,
        ge=0,
        description="Minimum price in USD"
    )
    price_max: Optional[float] = Field(
        default=None,
        ge=0,
        description="Maximum price in USD"
    )
    in_stock_only: bool = Field(
        default=True,
        description="Only return in-stock items"
    )
    
    @field_validator("price_max")
    @classmethod
    def validate_price_range(cls, v: Optional[float], info) -> Optional[float]:
        """Ensure price_max >= price_min if both provided."""
        values = info.data
        if v is not None and values.get("price_min") is not None:
            if v < values["price_min"]:
                raise ValueError("price_max must be >= price_min")
        return v


class SearchRequest(BaseModel):
    """Complete visual search request."""
    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of results to return"
    )
    filters: SearchFilter = Field(default_factory=SearchFilter)
    include_embedding: bool = Field(
        default=False,
        description="Return query embedding in response"
    )


class SearchResultItem(BaseModel):
    """
    Single search result with normalized confidence score.
    
    similarity_score is normalized to [0, 100] percentage for UX.
    raw_score preserves original cosine similarity for debugging.
    """
    product_id: str = Field(..., alias="_id")
    name: str
    category: str
    price: float = Field(..., ge=0)
    image_url: str
    similarity_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Confidence percentage [0-100]"
    )
    raw_score: float = Field(
        ...,
        description="Raw cosine similarity [-1, 1]"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        populate_by_name = True


class SearchResponse(BaseModel):
    """Paginated search results with timing."""
    query_embedding: Optional[List[float]] = Field(
        default=None,
        max_length=512
    )
    results: List[SearchResultItem]
    total_results: int = Field(..., ge=0)
    search_time_ms: float = Field(..., ge=0)
    filters_applied: SearchFilter


class HealthCheckResponse(BaseModel):
    """Service health and readiness status."""
    status: str = Field(..., pattern=r"^(healthy|degraded|unhealthy)$")
    model_loaded: bool
    model_name: str
    device: str
    mongodb_connected: bool
    average_latency_ms: Optional[float] = Field(
        default=None,
        description="Rolling average inference latency"
    )
