"""
MongoDB Atlas Vector Search service with aggregation pipeline construction
and pre-retrieval filtering.
"""

import time
from typing import Any, Dict, List, Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from app.config import Settings, get_settings
from app.models.schemas import SearchFilter, SearchRequest, SearchResponse, SearchResultItem


class SearchService:
    """
    Handles MongoDB Atlas $vectorSearch aggregation pipeline construction
    and execution with pre-retrieval filtering for optimal recall.
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.client: Optional[AsyncIOMotorClient] = None
        self.db: Optional[AsyncIOMotorDatabase] = None
    
    async def connect(self) -> None:
        """Initialize async MongoDB connection with connection pooling."""
        self.client = AsyncIOMotorClient(
            self.settings.MONGODB_URI,
            maxPoolSize=50,
            minPoolSize=10,
            maxIdleTimeMS=45000,
            serverSelectionTimeoutMS=5000,
            retryWrites=True,
            w="majority"
        )
        self.db = self.client[self.settings.MONGODB_DB_NAME]
        
        # Verify connection with ping
        await self.client.admin.command("ping")
        print(f"Connected to MongoDB Atlas: {self.settings.MONGODB_DB_NAME}")
    
    async def disconnect(self) -> None:
        """Graceful connection cleanup."""
        if self.client:
            self.client.close()
            print("MongoDB connection closed")
    
    def _build_vector_search_stage(
        self,
        query_embedding: List[float],
        top_k: int,
        filters: SearchFilter
    ) -> Dict[str, Any]:
        """
        Construct MongoDB $vectorSearch aggregation stage with dynamic pre-filtering.
        
        Pre-filtering ensures ANN search only considers documents matching the UI
        constraints (price, category, etc.), guaranteeing accurate top_k recall.
        """
        # 1. Build the dynamic filter dictionary
        query_filter = {}

        if filters.category and self.settings.ENABLE_CATEGORY_FILTER:
            query_filter["category"] = {"$eq": filters.category}
            
        if filters.material:
            query_filter["material"] = {"$eq": filters.material}
            
        if filters.in_stock_only:
            query_filter["in_stock"] = {"$eq": True}
            
        if filters.price_min is not None or filters.price_max is not None:
            query_filter["price"] = {}
            if filters.price_min is not None:
                query_filter["price"]["$gte"] = float(filters.price_min)
            if filters.price_max is not None:
                query_filter["price"]["$lte"] = float(filters.price_max)

        # 2. Construct the vector search configuration
        vector_search_config: Dict[str, Any] = {
            "index": self.settings.VECTOR_INDEX_NAME,
            "path": "embedding",
            "queryVector": query_embedding,
            "numCandidates": top_k * 10,  # Oversample for accuracy
            "limit": top_k
        }
        
        # Inject filters if any exist
        if query_filter:
            vector_search_config["filter"] = query_filter

        return {"$vectorSearch": vector_search_config}
    
    def _build_projection_stage(self, include_embedding: bool) -> Dict[str, Any]:
        """
        Select fields to return and extract similarity score.
        """
        projection: Dict[str, Any] = {
            "_id": 1,
            "product_id": 1,
            "name": 1,
            "category": 1,
            "price": 1,
            "image_url": 1,
            "material": 1,
            "in_stock": 1,
            "description": 1,
            "similarity_score": {"$meta": "vectorSearchScore"}
        }
        
        if include_embedding:
            projection["embedding"] = 1
            
        return {"$project": projection}
    
    def _normalize_score(self, raw_score: float) -> float:
        """
        Convert Atlas vectorSearchScore to confidence percentage.
        """
        clamped = max(-1.0, min(1.0, raw_score))
        
        if clamped < 0:
            return 0.0
            
        return round(clamped * 100, 2)
    
    async def visual_search(
        self,
        query_embedding: List[float],
        request: SearchRequest
    ) -> SearchResponse:
        """
        Execute complete visual search pipeline.
        """
        start_time = time.perf_counter()
        
        # Build pipeline
        pipeline: List[Dict[str, Any]] = []
        
        # Stage 1: Vector search with integrated filtering
        pipeline.append(
            self._build_vector_search_stage(
                query_embedding,
                request.top_k,
                request.filters
            )
        )
        
        # Stage 2: Project and score extraction
        pipeline.append(
            self._build_projection_stage(request.include_embedding)
        )
        
        # Execute aggregation
        collection = self.db[self.settings.MONGODB_COLLECTION_NAME]
        cursor = collection.aggregate(pipeline)
        results = await cursor.to_list(length=request.top_k)
        
        # Transform to response model
        search_results = []
        for doc in results:
            raw_score = doc.get("similarity_score", 0.0)
            
            result = SearchResultItem(
                product_id=str(doc.get("product_id", doc["_id"])),
                name=doc.get("name", "Unknown"),
                category=doc.get("category", "unknown"),
                price=float(doc.get("price", 0)),
                image_url=doc.get("image_url", ""),
                similarity_score=self._normalize_score(raw_score),
                raw_score=raw_score,
                metadata={
                    "material": doc.get("material"),
                    "in_stock": doc.get("in_stock", True),
                    "description": doc.get("description", "")
                }
            )
            search_results.append(result)
        
        search_time = (time.perf_counter() - start_time) * 1000
        
        return SearchResponse(
            query_embedding=query_embedding if request.include_embedding else None,
            results=search_results,
            total_results=len(search_results),
            search_time_ms=round(search_time, 2),
            filters_applied=request.filters
        )
    
    async def index_product(
        self,
        product_id: str,
        embedding: List[float],
        metadata: Dict[str, Any]
    ) -> None:
        """
        Index or update a product with its embedding.
        """
        document = {
            "_id": product_id,
            "product_id": product_id,
            "embedding": embedding,
            "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            **metadata
        }
        
        await self.db[self.settings.MONGODB_COLLECTION_NAME].update_one(
            {"_id": product_id},
            {"$set": document},
            upsert=True
        )
    
    async def get_stats(self) -> Dict[str, Any]:
        """Return collection statistics."""
        collection = self.db[self.settings.MONGODB_COLLECTION_NAME]
        return {
            "total_products": await collection.estimated_document_count(),
            "categories": await collection.distinct("category"),
            "materials": await collection.distinct("material")
        }