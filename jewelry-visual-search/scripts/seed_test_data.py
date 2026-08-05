#!/usr/bin/env python3
"""
Generate synthetic jewelry test data for development.

Creates sample images and metadata without requiring real product photos.
Useful for testing the pipeline before real data is available.

Usage:
    python scripts/seed_test_data.py --output-dir ./data/test_catalog --count 50
"""

import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


# Sample jewelry catalog data
CATEGORIES = ["necklace", "ring", "bracelet", "earring", "pendant"]
MATERIALS = ["gold", "silver", "platinum", "rose_gold", "white_gold"]
GEMSTONES = ["diamond", "ruby", "sapphire", "emerald", "pearl", "none"]


def generate_synthetic_jewelry_image(category: str, material: str, size=(224, 224)):
    """
    Create a synthetic jewelry image with distinctive visual features.
    
    Uses color and shape to simulate different jewelry types,
    making them visually distinguishable for CLIP.
    """
    # Base color by material
    color_map = {
        "gold": (255, 215, 0),
        "silver": (192, 192, 192),
        "platinum": (229, 228, 226),
        "rose_gold": (183, 110, 121),
        "white_gold": (245, 245, 245)
    }
    base_color = color_map.get(material, (200, 200, 200))
    
    img = Image.new("RGB", size, color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    
    center_x, center_y = size[0] // 2, size[1] // 2
    
    if category == "ring":
        # Draw ring shape
        draw.ellipse(
            [center_x - 60, center_y - 60, center_x + 60, center_y + 60],
            outline=base_color,
            width=8
        )
        # Add gemstone
        draw.ellipse(
            [center_x - 15, center_y - 80, center_x + 15, center_y - 50],
            fill=(255, 255, 255)
        )
        
    elif category == "necklace":
        # Draw chain
        for i in range(5):
            y = 40 + i * 30
            draw.arc(
                [center_x - 80 + i*10, y, center_x + 80 - i*10, y + 40],
                start=0, end=180,
                fill=base_color,
                width=3
            )
        # Pendant
        draw.ellipse(
            [center_x - 20, center_y + 40, center_x + 20, center_y + 80],
            fill=base_color
        )
        
    elif category == "bracelet":
        # Draw bracelet band
        draw.arc(
            [center_x - 70, center_y - 20, center_x + 70, center_y + 20],
            start=0, end=360,
            fill=base_color,
            width=6
        )
        # Links
        for x in range(center_x - 60, center_x + 61, 20):
            draw.rectangle([x - 3, center_y - 15, x + 3, center_y + 15], fill=base_color)
            
    elif category == "earring":
        # Pair of earrings
        for offset in [-40, 40]:
            # Hook
            draw.arc(
                [center_x + offset - 5, 30, center_x + offset + 5, 50],
                start=0, end=180,
                fill=base_color,
                width=3
            )
            # Drop
            draw.ellipse(
                [center_x + offset - 15, 60, center_x + offset + 15, 90],
                fill=base_color
            )
            
    else:  # pendant or default
        # Simple pendant shape
        draw.polygon(
            [(center_x, 40), (center_x - 30, 100), (center_x + 30, 100)],
            fill=base_color
        )
        draw.line(
            [(center_x, 40), (center_x, 20)],
            fill=base_color,
            width=3
        )
    
    # Add some texture/noise for realism
    pixels = img.load()
    for i in range(size[0]):
        for j in range(size[1]):
            r, g, b = pixels[i, j]
            noise = random.randint(-10, 10)
            pixels[i, j] = (
                max(0, min(255, r + noise)),
                max(0, min(255, g + noise)),
                max(0, min(255, b + noise))
            )
    
    return img


def generate_metadata(product_id: str, category: str, material: str):
    """Generate realistic product metadata."""
    gemstone = random.choice(GEMSTONES)
    has_gemstone = gemstone != "none"
    
    # Price based on material and gemstone
    base_prices = {
        "gold": 500, "silver": 150, "platinum": 800,
        "rose_gold": 450, "white_gold": 550
    }
    gemstone_multiplier = 2.0 if has_gemstone else 1.0
    
    price = base_prices.get(material, 300) * gemstone_multiplier
    price = round(price * random.uniform(0.8, 1.5), 2)
    
    name_parts = [material.capitalize()]
    if has_gemstone:
        name_parts.append(gemstone.capitalize())
    name_parts.append(category.capitalize())
    
    return {
        "product_id": product_id,
        "name": " ".join(name_parts),
        "category": category,
        "material": material,
        "gemstone": gemstone,
        "price": price,
        "currency": "USD",
        "in_stock": random.random() > 0.2,  # 80% in stock
        "description": (f"Beautiful {material} {category}" + (f" with {gemstone}" if has_gemstone else "")),
        "image_url": f"https://cdn.example.com/jewelry/{product_id}.jpg"
    }


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic jewelry test data")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--count", type=int, default=50, help="Number of products to generate")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create catalog directory structure
    catalog_dir = output_dir / "catalog"
    catalog_dir.mkdir(exist_ok=True)
    
    all_metadata = []
    
    print(f"Generating {args.count} synthetic jewelry products...")
    
    for i in range(args.count):
        product_id = f"JEW{1000 + i:04d}"
        category = random.choice(CATEGORIES)
        material = random.choice(MATERIALS)
        
        # Create product directory
        product_dir = catalog_dir / product_id
        product_dir.mkdir(exist_ok=True)
        
        # Generate and save image
        img = generate_synthetic_jewelry_image(category, material)
        image_path = product_dir / "main.jpg"
        img.save(image_path, quality=90)
        
        # Generate metadata
        meta = generate_metadata(product_id, category, material)
        all_metadata.append(meta)
        
        # Save individual metadata
        with open(product_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
        
        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{args.count} products...")
    
    # Save combined metadata
    metadata_file = output_dir / "products.json"
    with open(metadata_file, "w") as f:
        json.dump(all_metadata, f, indent=2)
    
    print(f"\nComplete! Files saved to: {output_dir}")
    print(f"  Catalog: {catalog_dir}")
    print(f"  Metadata: {metadata_file}")
    print(f"\nTo ingest into MongoDB:")
    print(f"  python scripts/ingest_catalog.py --catalog-dir {catalog_dir} --metadata {metadata_file} --mongo-uri $MONGODB_URI")


if __name__ == "__main__":
    main()
