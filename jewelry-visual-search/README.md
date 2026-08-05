# 💎 Jewelry Visual Search Engine

Complete, production-ready AI microservice for visual similarity search in jewelry e-commerce.

## Quick Start (5 minutes)

### Prerequisites
- Docker and Docker Compose
- MongoDB Atlas account (free tier works)
- 8GB+ RAM, 10GB disk space

### 1. Clone and Configure


## Retrieval Evaluation

Run the included offline evaluator with the bundled example:

```bash
python evaluate.py --input data/sample_evaluation.json
```

It reports Recall@1, Recall@5, Recall@10, and Mean Reciprocal Rank (MRR).

## Project Assembly Rule

The `app/` package uses the updated production implementations. Setup, scripts,
tests, deployment manifests, and project tooling use the infrastructure set.
Missing root files were completed to keep the repository runnable as one project.
