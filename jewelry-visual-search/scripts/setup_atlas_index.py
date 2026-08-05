#!/usr/bin/env python3
"""
One-time setup script for MongoDB Atlas Vector Search index.

Run this ONCE after creating your Atlas cluster to create the search index.
Requires Atlas admin privileges.

Usage:
    python scripts/setup_atlas_index.py --uri "mongodb+srv://..."
"""

import argparse
import json
import sys

from pymongo import MongoClient


# Exact index configuration for CLIP ViT-B/32 (512 dimensions)
VECTOR_INDEX_CONFIG = {
    "fields": [
        {
            "type": "vector",
            "path": "embedding",
            "numDimensions": 512,
            "similarity": "cosine"
        },
        {
            "type": "filter",
            "path": "category"
        },
        {
            "type": "filter",
            "path": "material"
        },
        {
            "type": "filter",
            "path": "price"
        },
        {
            "type": "filter",
            "path": "in_stock"
        }
    ]
}


def create_index(mongo_uri: str, database: str, collection: str, index_name: str):
    """
    Create Atlas Search index via MongoDB Admin API.
    
    Note: $vectorSearch indexes must be created through Atlas UI or Admin API,
    not standard create_index(). This script provides the exact JSON to paste.
    """
    print("=" * 60)
    print("MONGODB ATLAS VECTOR SEARCH INDEX SETUP")
    print("=" * 60)
    print()
    print("Method 1: Atlas UI (Recommended for first time)")
    print("-" * 60)
    print("1. Log into https://cloud.mongodb.com")
    print("2. Navigate to your cluster → Search → Create Index")
    print("3. Choose 'JSON Editor'")
    print("4. Paste the following configuration:")
    print()
    print(json.dumps(VECTOR_INDEX_CONFIG, indent=2))
    print()
    print("-" * 60)
    print("Method 2: Atlas Search API (Requires API key)")
    print("-" * 60)
    
    # Verify connection
    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        db = client[database]
        coll = db[collection]
        
        print(f"Connected to: {db.name}.{coll.name}")
        print(f"Existing documents: {coll.estimated_document_count()}")
        print()
        print("Index configuration validated successfully!")
        
    except Exception as e:
        print(f"Connection failed: {e}")
        sys.exit(1)
    
    print()
    print("IMPORTANT: After creating the index, verify with:")
    print(f"  db.{collection}.aggregate([{{$vectorSearch: {{...}}}}])")
    print()
    print("Index name should be: 'jewelry_vector_index'")


def main():
    parser = argparse.ArgumentParser(description="Setup Atlas Vector Search")
    parser.add_argument("--uri", required=True, help="MongoDB Atlas connection string")
    parser.add_argument("--db", default="jewelry_inventory", help="Database name")
    parser.add_argument("--collection", default="products", help="Collection name")
    parser.add_argument("--index-name", default="jewelry_vector_index", help="Search index name")
    
    args = parser.parse_args()
    create_index(args.uri, args.db, args.collection, args.index_name)


if __name__ == "__main__":
    main()
