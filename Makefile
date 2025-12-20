.PHONY: help install dev-install test lint format clean docker-up docker-down migrate

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install production dependencies
	pip install -e .

dev-install: ## Install development dependencies
	pip install -e ".[dev,test]"
	playwright install
	pre-commit install

test: ## Run all tests
	pytest

test-unit: ## Run unit tests only
	pytest tests/unit/

test-property: ## Run property-based tests only
	pytest tests/property/ -m property

test-integration: ## Run integration tests only
	pytest tests/integration/

test-coverage: ## Run tests with coverage report
	pytest --cov=braf --cov-report=html --cov-report=term

lint: ## Run linting checks
	flake8 src/ tests/
	mypy src/

format: ## Format code with black and isort
	black src/ tests/
	isort src/ tests/

format-check: ## Check code formatting
	black --check src/ tests/
	isort --check-only src/ tests/

clean: ## Clean up build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

docker-up: ## Start development infrastructure
	docker-compose up -d postgres redis elasticsearch kibana prometheus grafana vault

docker-down: ## Stop development infrastructure
	docker-compose down

docker-logs: ## Show infrastructure logs
	docker-compose logs -f

migrate: ## Run database migrations
	alembic upgrade head

migrate-create: ## Create new migration (usage: make migrate-create MESSAGE="description")
	alembic revision --autogenerate -m "$(MESSAGE)"

migrate-downgrade: ## Downgrade database by one revision
	alembic downgrade -1

dev-setup: dev-install docker-up ## Complete development setup
	@echo "Waiting for services to start..."
	@sleep 10
	@make migrate
	@echo "Development environment ready!"

dev-run-c2: ## Run C2 dashboard in development mode
	python -m braf.c2.main --config config/development.yaml

dev-run-worker: ## Run worker node in development mode
	python -m braf.worker.main --config config/development.yaml

security-scan: ## Run security vulnerability scan
	safety check
	bandit -r src/

pre-commit: ## Run pre-commit hooks on all files
	pre-commit run --all-files