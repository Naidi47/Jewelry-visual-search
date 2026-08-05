#!/usr/bin/env python3
"""
Batch ingestion pipeline for jewelry catalog.

Processes a directory of jewelry images, generates CLIP embeddings,
and indexes them into MongoDB Atlas.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image
from pymongo import MongoClient, UpdateOne
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


class CatalogIngester:
    """
    Batch processor for indexing jewelry inventory.
    
    Pipeline:
    1. Load product metadata (name, category, price, etc.)
    2. For each product, find its image file
    3. Generate CLIP embedding
    4. Bulk write to MongoDB with upsert
    """
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = "auto",
        mongo_uri: str = None,
        db_name: str = "jewelry_inventory",
        collection_name: str = "products"
    ):
        if not mongo_uri:
            raise ValueError("MongoDB URI is required. Please set MONGODB_URI environment variable or pass --mongo-uri.")
            
        # Device selection
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Loading CLIP model on {self.device}...")
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        # MongoDB connection
        print("Connecting to MongoDB Atlas...")
        self.client = MongoClient(mongo_uri, maxPoolSize=50)
        self.collection = self.client[db_name][collection_name]
        
        # Verify connection
        self.client.admin.command("ping")
        print("Connected successfully!")
    
    def embed_image(self, image_path: Path) -> List[float]:
        """Generate normalized CLIP embedding for image."""
        image = Image.open(image_path).convert("RGB")
        
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        
        with torch.no_grad():
            features = self.model.get_image_features(pixel_values=pixel_values)
            # L2 normalize
            features = features / features.norm(dim=-1, keepdim=True)
        
        return features.cpu().squeeze().tolist()
    
    def ingest(
        self,
        catalog_dir: Path,
        metadata_file: Path = None,
        batch_size: int = 32
    ):
        """
        Process catalog directory and index all products.
        """
        catalog_dir = Path(catalog_dir)
        if not catalog_dir.exists():
            raise ValueError(f"Catalog directory not found: {catalog_dir}")
        
        # Load metadata if provided
        metadata = {}
        if metadata_file and metadata_file.exists():
            with open(metadata_file) as f:
                raw_meta = json.load(f)
                # FIX: Check for both "product_id" and "_id" to match your JSON structure
                for item in raw_meta:
                    item_id = item.get("product_id") or item.get("_id")
                    if item_id:
                        metadata[item_id] = item
        
        # Find all product directories
        product_dirs = [d for d in catalog_dir.iterdir() if d.is_dir()]
        print(f"Found {len(product_dirs)} product directories")
        
        # Process in batches for efficiency
        bulk_operations = []
        
        for product_dir in tqdm(product_dirs, desc="Processing products"):
            product_id = product_dir.name
            
            # Find image file
            image_files = list(product_dir.glob("*.jpg")) + list(product_dir.glob("*.png"))
            if not image_files:
                print(f"Warning: No image found for {product_id}")
                continue
            
            main_image = image_files[0]
            
            # Generate embedding
            try:
                embedding = self.embed_image(main_image)
            except Exception as e:
                print(f"Error processing {product_id}: {e}")
                continue
            
            # Build document
            doc = {
                "product_id": product_id,
                "image_path": str(main_image.relative_to(catalog_dir)),
                "embedding": embedding,
                "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ")
            }
            
            # FIX: Properly extract nested "metadata" from your sample_products.json
            meta = metadata.get(product_id, {})
            nested_meta = meta.get("metadata", {})
            
            doc.update({
                "name": meta.get("name", product_id),
                "category": meta.get("category", "unknown"),
                "material": meta.get("material") or nested_meta.get("material", "unknown"),
                "price": meta.get("price", 0.0),
                "in_stock": meta.get("in_stock") if "in_stock" in meta else nested_meta.get("in_stock", True),
                "image_url": meta.get("image_url", ""),
                "description": meta.get("description") or nested_meta.get("description", "")
            })
            
            # Create upsert operation
            bulk_operations.append(
                UpdateOne(
                    {"_id": product_id},
                    {"$set": doc},
                    upsert=True
                )
            )
            
            # Execute batch
            if len(bulk_operations) >= batch_size:
                self._execute_bulk(bulk_operations)
                bulk_operations = []
        
        # Final batch
        if bulk_operations:
            self._execute_bulk(bulk_operations)
        
        print(f"\nIngestion complete! Total products in collection: {self.collection.estimated_document_count()}")
    
    def _execute_bulk(self, operations: List[UpdateOne]):
        """Execute bulk write with error handling."""
        try:
            result = self.collection.bulk_write(operations, ordered=False)
            print(f"  Bulk write: {result.upserted_count} inserted, {result.modified_count} updated")
        except Exception as e:
            print(f"  Bulk write error: {e}")
    
    def close(self):
        """Cleanup resources."""
        self.client.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    # Setup robust default paths for your new jewelry_catalog folder
    BASE_DIR = Path(__file__).parent.parent
    DEFAULT_CATALOG = BASE_DIR / "data" / "jewelry_catalog" / "catalog"
    DEFAULT_METADATA = BASE_DIR / "data" / "jewelry_catalog" / "sample_products.json"
    
    # Try to load from .env automatically if python-dotenv is installed
    try:
        from dotenv import load_dotenv
        load_dotenv(BASE_DIR / ".env")
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description="Ingest jewelry catalog into Atlas")
    # Changed required=True to default paths so you don't have to type them
    parser.add_argument("--catalog-dir", default=str(DEFAULT_CATALOG), help="Directory containing product images")
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA), help="JSON file with product metadata")
    parser.add_argument("--mongo-uri", default=os.getenv("MONGODB_URI"), help="MongoDB Atlas connection string")
    parser.add_argument("--batch-size", type=int, default=32, help="Bulk write batch size")
    parser.add_argument("--device", default="auto", help="Compute device")
    
    args = parser.parse_args()
    
    ingester = CatalogIngester(
        mongo_uri=args.mongo_uri,
        device=args.device
    )
    
    try:
        ingester.ingest(
            catalog_dir=Path(args.catalog_dir),
            metadata_file=Path(args.metadata) if args.metadata else None,
            batch_size=args.batch_size
        )
    finally:
        ingester.close()


if __name__ == "__main__":
    main()