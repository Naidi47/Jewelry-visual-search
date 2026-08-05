"""
Configuration management using Pydantic Settings.
Handles environment-specific configuration with validation.
"""

from functools import lru_cache
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    All sensitive values must be injected via environment.
    No hardcoded secrets.
    """
    
    # Application identity
    APP_NAME: str = "Jewelry Visual Search API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = Field(default=False)
    
    # Server binding
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1
    """Uvicorn workers. Use 1 per container; scale via replicas."""
    
    # CORS
    CORS_ORIGINS: str = "*"
    
    # Model configuration
    MODEL_NAME: str = "openai/clip-vit-base-patch32"
    """
    CLIP variant determines embedding dimension:
    - openai/clip-vit-base-patch32: 512-dim (faster, smaller)
    - openai/clip-vit-large-patch14: 768-dim (more accurate)
    """
    
    MODEL_CACHE_DIR: Optional[str] = "/app/model_cache"
    """Persistent volume for model weights."""
    
    # Compute device
    DEVICE: str = Field(default="auto")
    """
    Compute device selection:
    - 'cuda': NVIDIA GPU (recommended for production)
    - 'cpu': CPU only (slower, no GPU needed)
    - 'auto': Use CUDA if available, else CPU
    """
    
    # Image processing constraints
    IMAGE_SIZE: int = 224  # CLIP's fixed input size
    MAX_IMAGE_SIZE_MB: int = 10
    SUPPORTED_FORMATS: List[str] = ["image/jpeg", "image/png", "image/webp"]
    
    # MongoDB Atlas
    MONGODB_URI: str = Field(...)
    """
    Atlas connection string format:
    mongodb+srv://username:password@cluster.mongodb.net/database?options
    
    Must include database name and retryWrites=true.
    """
    
    MONGODB_DB_NAME: str = "jewelry_inventory"
    MONGODB_COLLECTION_NAME: str = "products"
    
    # Vector search configuration
    VECTOR_INDEX_NAME: str = "jewelry_vector_index"
    VECTOR_DIMENSIONS: int = 512  # Must match MODEL_NAME output
    TOP_K_DEFAULT: int = 10
    TOP_K_MAX: int = 100
    
    # Feature flags
    ENABLE_CATEGORY_FILTER: bool = True
    
    @field_validator("MONGODB_URI")
    @classmethod
    def validate_mongodb_uri(cls, v: str) -> str:
        """Ensure MongoDB URI has required format."""
        if not v.startswith(("mongodb://", "mongodb+srv://")):
            raise ValueError("MONGODB_URI must start with mongodb:// or mongodb+srv://")
        return v
    
    @field_validator("VECTOR_DIMENSIONS")
    @classmethod
    def validate_dimensions(cls, v: int, info) -> int:
        """Validate dimensions match model."""
        model = info.data.get("MODEL_NAME", "")
        expected = {
            "openai/clip-vit-base-patch32": 512,
            "openai/clip-vit-base-patch16": 512,
            "openai/clip-vit-large-patch14": 768,
        }
        if model in expected and v != expected[model]:
            raise ValueError(
                f"VECTOR_DIMENSIONS={v} doesn't match {model} "
                f"expected={expected[model]}"
            )
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"  # Allow extra env vars without error


@lru_cache()
def get_settings() -> Settings:
    """
    Cached settings singleton.
    
    lru_cache ensures single parse and validation across imports.
    Critical for production to avoid repeated environment access.
    """
    return Settings()


# Module-level export for convenience
settings = get_settings()
