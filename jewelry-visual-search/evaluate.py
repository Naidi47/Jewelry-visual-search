"""Offline retrieval evaluation utilities for Jewelry Visual Search."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List


def recall_at_k(retrieved: List[str], relevant: Iterable[str], k: int) -> float:
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    hits = len(set(retrieved[:k]) & relevant_set)
    return hits / len(relevant_set)


def reciprocal_rank(retrieved: List[str], relevant: Iterable[str]) -> float:
    relevant_set = set(relevant)
    for rank, product_id in enumerate(retrieved, start=1):
        if product_id in relevant_set:
            return 1.0 / rank
    return 0.0


def evaluate(records: List[Dict], ks=(1, 5, 10)) -> Dict[str, float]:
    if not records:
        return {"queries": 0, "mrr": 0.0, **{f"recall@{k}": 0.0 for k in ks}}
    output = {"queries": len(records)}
    for k in ks:
        output[f"recall@{k}"] = round(
            mean(recall_at_k(r["retrieved"], r["relevant"], k) for r in records), 4
        )
    output["mrr"] = round(
        mean(reciprocal_rank(r["retrieved"], r["relevant"]) for r in records), 4
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate visual retrieval results")
    parser.add_argument(
        "--input",
        type=Path,
        help="JSON list with query_id, relevant[], and retrieved[] fields",
    )
    args = parser.parse_args()

    if args.input:
        records = json.loads(args.input.read_text(encoding="utf-8"))
    else:
        records = [
            {
                "query_id": "demo-ring",
                "relevant": ["JEW1001"],
                "retrieved": ["JEW1001", "JEW1000", "JEW1002"],
            },
            {
                "query_id": "demo-necklace",
                "relevant": ["JEW1000"],
                "retrieved": ["JEW1002", "JEW1000", "JEW1001"],
            },
        ]

    print(json.dumps(evaluate(records), indent=2))


if __name__ == "__main__":
    main()
