.PHONY: help install install-dev test lint format seed ingest build run docker-up docker-down k8s-deploy k8s-delete clean

# Default target
help:
	@echo "Jewelry Visual Search - Available Commands"
	@echo "============================================"
	@echo "  make install        Install production dependencies"
	@echo "  make install-dev    Install development dependencies"
	@echo "  make test           Run test suite"
	@echo "  make lint           Run code linting"
	@echo "  make format         Format code with black"
	@echo "  make seed           Generate synthetic test data"
	@echo "  make ingest         Ingest catalog into MongoDB"
	@echo "  make build          Build Docker image"
	@echo "  make run            Run locally with uvicorn"
	@echo "  make docker-up      Start with docker-compose"
	@echo "  make docker-down    Stop docker-compose"
	@echo "  make k8s-deploy     Deploy to Kubernetes"
	@echo "  make k8s-delete     Remove from Kubernetes"
	@echo "  make clean          Remove build artifacts"

# Development setup
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

# Testing
test:
	pytest tests/ -v --cov=app --cov-report=term-missing

lint:
	black --check app/ tests/ scripts/
	mypy app/

format:
	black app/ tests/ scripts/

# Data generation and ingestion
seed:
	python scripts/seed_test_data.py --output-dir ./data/test_catalog --count 50

ingest:
	@if [ -z "$(MONGODB_URI)" ]; then \
		echo "Error: MONGODB_URI environment variable not set"; \
		exit 1; \
	fi
	python scripts/ingest_catalog.py \
		--catalog-dir ./data/test_catalog/catalog \
		--metadata ./data/test_catalog/products.json \
		--mongo-uri "$(MONGODB_URI)"

# Docker
build:
	docker build -t jewelry-visual-search:latest .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

# Kubernetes
k8s-deploy:
	kubectl apply -f k8s/namespace.yaml
	kubectl apply -f k8s/configmap.yaml
	@echo "Remember to create secrets: kubectl apply -f k8s/secret.yaml"
	kubectl apply -f k8s/deployment.yaml
	kubectl apply -f k8s/service.yaml
	kubectl apply -f k8s/ingress.yaml
	kubectl apply -f k8s/hpa.yaml

k8s-delete:
	kubectl delete -f k8s/ --ignore-not-found=true

# Cleanup
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf build/ dist/ *.egg-info/
