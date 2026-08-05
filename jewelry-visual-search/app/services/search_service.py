"""
MongoDB Atlas Vector Search service with aggregation pipeline construction
and post-retrieval filtering.
"""

import time
from typing import Any, Dict, List, Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from app.config import Settings, get_settings
from app.models.schemas import SearchFilter, SearchRequest, SearchResponse, SearchResultItem


class SearchService:
    """
    Handles MongoDB Atlas $vectorSearch aggregation pipeline construction
    and execution with post-retrieval filtering.
    
    Atlas Vector Search uses Approximate Nearest Neighbor (ANN) with
    HNSW (Hierarchical Navigable Small World) algorithm for sub-100ms
    retrieval on millions of vectors.
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
        Construct MongoDB $vectorSearch aggregation stage.
        
        CRITICAL: $vectorSearch MUST be the FIRST stage in the pipeline.
        It cannot be preceded by $match, $limit, etc.
        
        numCandidates = top_k * 10 provides 10x oversampling for ANN
        accuracy before filtering reduces results.
        """
        stage = {
            "$vectorSearch": {
                "index": self.settings.VECTOR_INDEX_NAME,
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": top_k * 10,  # Oversample for accuracy
                "limit": top_k
            }
        }
        
        # Pre-filter using indexed filter fields (more efficient than post-filter)
        # Requires 'category' to be defined as filter type in search index
        if filters.category and self.settings.ENABLE_CATEGORY_FILTER:
            stage["$vectorSearch"]["filter"] = {
                "category": {"$eq": filters.category}
            }
        
        return stage
    
    def _build_post_filter_stage(self, filters: SearchFilter) -> Optional[Dict[str, Any]]:
        """
        Construct $match stage for post-retrieval filtering.
        
        Applied AFTER vector search to refine results for fields not in
        the vector index or complex range conditions.
        """
        match_conditions: Dict[str, Any] = {}
        
        if filters.material:
            match_conditions["material"] = {"$eq": filters.material}
        
        if filters.price_min is not None or filters.price_max is not None:
            price_range: Dict[str, Any] = {}
            if filters.price_min is not None:
                price_range["$gte"] = filters.price_min
            if filters.price_max is not None:
                price_range["$lte"] = filters.price_max
            match_conditions["price"] = price_range
        
        if filters.in_stock_only:
            match_conditions["in_stock"] = True
        
        return {"$match": match_conditions} if match_conditions else None
    
    def _build_projection_stage(self, include_embedding: bool) -> Dict[str, Any]:
        """
        Select fields to return and extract similarity score.
        
        The $meta: "vectorSearchScore" returns the raw similarity score
        from the ANN search. For cosine similarity with normalized vectors,
        this is effectively the dot product (since ||a|| = ||b|| = 1).
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
            # Atlas returns similarity as score metadata
            "similarity_score": {"$meta": "vectorSearchScore"}
        }
        
        if include_embedding:
            projection["embedding"] = 1
        
        return {"$project": projection}
    
    def _normalize_score(self, raw_score: float) -> float:
        """
        Convert Atlas vectorSearchScore to confidence percentage.
        
        Mathematical basis:
        - For L2-normalized vectors, cosine similarity = dot product
        - Range: [-1, 1] theoretically, [0, 1] practically for similar items
        - We clamp to [0, 1] and scale to percentage
        
        Args:
            raw_score: MongoDB $vectorSearch score (cosine similarity)
            
        Returns:
            Confidence percentage [0.0, 100.0]
        """
        # Clamp to valid cosine range
        clamped = max(-1.0, min(1.0, raw_score))
        
        # For normalized vectors, negative similarity is semantically impossible
        # for visually similar items. Treat as 0 confidence.
        if clamped < 0:
            return 0.0
        
        # Scale to percentage with 2 decimal precision
        return round(clamped * 100, 2)
    
    async def visual_search(
        self,
        query_embedding: List[float],
        request: SearchRequest
    ) -> SearchResponse:
        """
        Execute complete visual search pipeline.
        
        Aggregation pipeline stages:
        1. $vectorSearch: ANN retrieval from vector index
        2. $match: Post-filtering (price, material, stock)
        3. $project: Field selection + score extraction
        4. $limit: Final result cap
        
        Args:
            query_embedding: L2-normalized CLIP vector [512]
            request: Search parameters and filters
            
        Returns:
            SearchResponse with ranked, filtered results
        """
        start_time = time.perf_counter()
        
        # Build pipeline
        pipeline: List[Dict[str, Any]] = []
        
        # Stage 1: Vector search (REQUIRED first stage)
        pipeline.append(
            self._build_vector_search_stage(
                query_embedding,
                request.top_k,
                request.filters
            )
        )
        
        # Stage 2: Post-filter (optional)
        post_filter = self._build_post_filter_stage(request.filters)
        if post_filter:
            pipeline.append(post_filter)
        
        # Stage 3: Project and score
        pipeline.append(
            self._build_projection_stage(request.include_embedding)
        )
        
        # Stage 4: Safety limit
        pipeline.append({"$limit": request.top_k})
        
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
        
        Used by catalog ingestion pipeline. Upsert ensures idempotency.
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
