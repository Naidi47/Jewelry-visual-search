"""
Jewelry Visual Search API - FastAPI Application.

Production-ready microservice for visual similarity search using
CLIP embeddings and MongoDB Atlas Vector Search.
"""

import os
from contextlib import asynccontextmanager
from typing import Optional

import structlog
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import Settings, get_settings
from app.models.schemas import (
    EmbeddingResponse,
    EmbeddingMetadata,
    HealthCheckResponse,
    SearchFilter,
    SearchRequest,
    SearchResponse,
)
from app.services.embedding_service import EmbeddingService
from app.services.search_service import SearchService
from app.utils.image_utils import ImagePreprocessor, ImageValidationError

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


class ApplicationState:
    """
    Encapsulated application state for lifespan management.
    Avoids global variables and enables test mocking.
    """
    def __init__(self):
        self.settings: Settings = get_settings()
        self.embedding_service: Optional[EmbeddingService] = None
        self.search_service: Optional[SearchService] = None
        self.image_preprocessor: Optional[ImagePreprocessor] = None
    
    async def initialize(self) -> None:
        """Initialize all services."""
        logger.info("Initializing application services...")
        
        # Image preprocessor (lightweight, no model loading)
        self.image_preprocessor = ImagePreprocessor(
            target_size=self.settings.IMAGE_SIZE,
            max_file_size_mb=self.settings.MAX_IMAGE_SIZE_MB
        )
        
        # Embedding service (heavy: loads CLIP model)
        self.embedding_service = EmbeddingService(self.settings)
        self.embedding_service.load_model()
        
        # Search service (connects to MongoDB)
        self.search_service = SearchService(self.settings)
        await self.search_service.connect()
        
        logger.info(
            "Services initialized",
            model=self.settings.MODEL_NAME,
            device=str(self.embedding_service.device),
            mongodb_connected=True
        )
    
    async def shutdown(self) -> None:
        """Graceful cleanup of all services."""
        logger.info("Shutting down services...")
        
        if self.search_service:
            await self.search_service.disconnect()
        
        if self.embedding_service:
            self.embedding_service.cleanup()
        
        logger.info("Shutdown complete")


# Global state managed by lifespan
app_state = ApplicationState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan: handles startup and shutdown.
    """
    try:
        await app_state.initialize()
        yield
    finally:
        await app_state.shutdown()


# FastAPI application
app = FastAPI(
    title="Jewelry Visual Search API",
    version="1.0.0",
    description="""
    AI-powered visual similarity search for jewelry e-commerce.
    
    ## Features
    - CLIP-based image embeddings (512-dim)
    - MongoDB Atlas Vector Search ($vectorSearch)
    - Category/material/price filtering
    - GPU-accelerated inference
    
    ## Endpoints
    - `POST /v1/embeddings/generate` — Generate image embedding
    - `POST /v1/search/visual` — Visual similarity search
    """,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Robust absolute path calculation for static catalog image serving
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_DIR = os.path.join(BASE_DIR, "data", "jewelry_catalog", "catalog")

# Ensure image directory exists before mounting to prevent startup errors
os.makedirs(IMAGE_DIR, exist_ok=True)

# Mount static files endpoint for local image rendering
app.mount("/images", StaticFiles(directory=IMAGE_DIR), name="images")

# CORS middleware for frontend integration (allows all standard methods including OPTIONS preflight)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=600,
)


@app.exception_handler(ImageValidationError)
async def image_validation_handler(request, exc):
    """Handle image validation errors with clear messages."""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc), "type": "image_validation_error"}
    )


@app.get("/health", response_model=HealthCheckResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint for load balancers and monitoring.
    
    Returns:
        - model_loaded: Whether CLIP model is initialized
        - mongodb_connected: Whether database is reachable
        - average_latency: Rolling average inference time
    """
    embedding_ready = (
        app_state.embedding_service is not None
        and app_state.embedding_service._model_loaded
    )
    
    mongo_ready = (
        app_state.search_service is not None
        and app_state.search_service.client is not None
    )
    
    return HealthCheckResponse(
        status="healthy" if (embedding_ready and mongo_ready) else "degraded",
        model_loaded=embedding_ready,
        model_name=app_state.settings.MODEL_NAME if embedding_ready else "unknown",
        device=str(app_state.embedding_service.device) if embedding_ready else "unknown",
        mongodb_connected=mongo_ready,
        average_latency_ms=(
            app_state.embedding_service.get_average_latency()
            if embedding_ready else None
        )
    )


@app.post(
    "/v1/embeddings/generate",
    response_model=EmbeddingResponse,
    status_code=status.HTTP_200_OK,
    tags=["Embeddings"],
    summary="Generate image embedding",
    response_description="512-dimensional CLIP embedding vector"
)
async def generate_embedding(
    image: UploadFile = File(
        ...,
        description="Jewelry image (JPEG, PNG, WebP). Max 10MB.",
        media_type="image/*"
    ),
    normalize: bool = Form(
        default=True,
        description="L2-normalize output (recommended for search)"
    ),
    return_metadata: bool = Form(
        default=False,
        description="Include processing metadata in response"
    )
):
    """
    Generate CLIP embedding for a single image.
    
    The returned vector can be stored in MongoDB or used directly
    for similarity comparison via dot product.
    """
    if app_state.embedding_service is None or app_state.image_preprocessor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service initializing, please retry"
        )
    
    try:
        # Read and validate upload
        content = await image.read()
        app_state.image_preprocessor.validate_upload(
            content,
            image.content_type or "unknown"
        )
        
        # Preprocess to tensor
        pixel_values, orig_size = app_state.image_preprocessor.preprocess(content)
        pixel_values = pixel_values.to(app_state.embedding_service.device)
        
        # Generate embedding
        embedding_tensor, inference_time = app_state.embedding_service.embed_image(
            pixel_values,
            normalize=normalize
        )
        
        # Convert to JSON-serializable format
        embedding_list = embedding_tensor.squeeze(0).cpu().tolist()
        
        # Build response
        response_data = {
            "embedding": embedding_list,
            "dimension": len(embedding_list)
        }
        
        if return_metadata:
            response_data["metadata"] = EmbeddingMetadata(
                model_name=app_state.settings.MODEL_NAME,
                device=str(app_state.embedding_service.device),
                inference_time_ms=round(inference_time, 2),
                input_image_size=orig_size,
                normalized=normalize
            )
        
        return EmbeddingResponse(**response_data)
        
    except ImageValidationError:
        raise  # Handled by exception handler
    except Exception as e:
        logger.error("Embedding generation failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal processing error"
        )


@app.post(
    "/v1/search/visual",
    response_model=SearchResponse,
    status_code=status.HTTP_200_OK,
    tags=["Search"],
    summary="Visual similarity search",
    response_description="Ranked similar products with confidence scores"
)
async def visual_search(
    image: UploadFile = File(
        ...,
        description="Query jewelry photo to search for similar items"
    ),
    top_k: int = Form(
        default=10,
        ge=1,
        le=100,
        description="Number of results to return"
    ),
    category: Optional[str] = Form(
        default=None,
        description="Filter by category: necklace, ring, bracelet, earring, pendant"
    ),
    material: Optional[str] = Form(
        default=None,
        description="Filter by material: gold, silver, platinum, etc."
    ),
    price_min: Optional[float] = Form(
        default=None,
        ge=0,
        description="Minimum price filter"
    ),
    price_max: Optional[float] = Form(
        default=None,
        ge=0,
        description="Maximum price filter"
    ),
    in_stock_only: bool = Form(
        default=True,
        description="Only show in-stock items"
    ),
    include_embedding: bool = Form(
        default=False,
        description="Include query embedding in response"
    )
):
    """
    Complete visual search: upload a photo, find similar jewelry.
    
    Pipeline:
    1. Generate CLIP embedding from uploaded image
    2. Execute MongoDB $vectorSearch (ANN)
    3. Apply post-filters (price, stock, material)
    4. Return ranked results with normalized confidence scores
    """
    if app_state.embedding_service is None or app_state.search_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service initializing, please retry"
        )
    
    try:
        # Step 1: Generate query embedding
        content = await image.read()
        app_state.image_preprocessor.validate_upload(
            content,
            image.content_type or "unknown"
        )
        
        pixel_values, _ = app_state.image_preprocessor.preprocess(content)
        pixel_values = pixel_values.to(app_state.embedding_service.device)
        
        query_embedding_tensor, _ = app_state.embedding_service.embed_image(
            pixel_values,
            normalize=True  # Must normalize for cosine search
        )
        query_embedding = query_embedding_tensor.squeeze(0).cpu().tolist()
        
        # Step 2: Build search request
        filters = SearchFilter(
            category=category,
            material=material,
            price_min=price_min,
            price_max=price_max,
            in_stock_only=in_stock_only
        )
        
        search_request = SearchRequest(
            top_k=min(top_k, app_state.settings.TOP_K_MAX),
            filters=filters,
            include_embedding=include_embedding
        )
        
        # Step 3: Execute search
        results = await app_state.search_service.visual_search(
            query_embedding,
            search_request
        )
        
        return results
        
    except ImageValidationError:
        raise
    except Exception as e:
        logger.error("Visual search failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Search processing error"
        )


if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        workers=settings.WORKERS,
        reload=settings.DEBUG,
        access_log=True,
        proxy_headers=True,
        forwarded_allow_ips="*"
    )