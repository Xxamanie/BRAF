# BRAF Monetization System Makefile

.PHONY: help setup install install-dev clean test lint format docker-build docker-up docker-down migrate

# Default target
help:
	@echo "BRAF Monetization System - Available Commands:"
	@echo ""
	@echo "Setup and Installation:"
	@echo "  setup          - Complete setup (venv, dependencies, directories)"
	@echo "  install        - Install production dependencies"
	@echo "  install-dev    - Install development dependencies"
	@echo ""
	@echo "Development:"
	@echo "  test           - Run tests"
	@echo "  lint           - Run linting checks"
	@echo "  format         - Format code with black and isort"
	@echo "  clean          - Clean up temporary files"
	@echo ""
	@echo "Database:"
	@echo "  migrate        - Run database migrations"
	@echo "  migrate-create - Create new migration"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build   - Build Docker images"
	@echo "  docker-up      - Start all services"
	@echo "  docker-down    - Stop all services"
	@echo "  docker-logs    - View logs"
	@echo ""
	@echo "Monitoring:"
	@echo "  logs           - View application logs"
	@echo "  health         - Check service health"

# Setup and Installation
setup:
	@echo "🚀 Setting up BRAF Monetization System..."
	python setup.py

install:
	@echo "📦 Installing production dependencies..."
	pip install -r requirements.txt

install-dev:
	@echo "📦 Installing development dependencies..."
	pip install -r requirements-dev.txt

# Development
test:
	@echo "🧪 Running tests..."
	pytest tests/ -v --cov=. --cov-report=html --cov-report=term

lint:
	@echo "🔍 Running linting checks..."
	flake8 .
	mypy .
	bandit -r . -x tests/

format:
	@echo "🎨 Formatting code..."
	black .
	isort .

clean:
	@echo "🧹 Cleaning up..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf build/

# Database
migrate:
	@echo "🗄️  Running database migrations..."
	alembic upgrade head

migrate-create:
	@echo "🗄️  Creating new migration..."
	@read -p "Enter migration message: " message; \
	alembic revision --autogenerate -m "$$message"

# Docker
docker-build:
	@echo "🐳 Building Docker images..."
	docker-compose build --no-cache

docker-up:
	@echo "🐳 Starting all services..."
	docker-compose up -d

docker-down:
	@echo "🐳 Stopping all services..."
	docker-compose down

docker-logs:
	@echo "📋 Viewing Docker logs..."
	docker-compose logs -f

# Monitoring
logs:
	@echo "📋 Viewing application logs..."
	tail -f logs/*.log

health:
	@echo "🏥 Checking service health..."
	@curl -s http://localhost:8000/health | python -m json.tool || echo "❌ API not responding"
	@docker-compose ps

# Development server
dev:
	@echo "🚀 Starting development server..."
	uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Production server
prod:
	@echo "🚀 Starting production server..."
	uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4

# Security scan
security:
	@echo "🔒 Running security scan..."
	bandit -r . -f json -o security-report.json
	safety check

# Performance test
perf-test:
	@echo "⚡ Running performance tests..."
	locust -f tests/performance/locustfile.py --host=http://localhost:8000

# Backup database
backup-db:
	@echo "💾 Backing up database..."
	docker-compose exec db pg_dump -U user braf_monetization > backups/db_backup_$(shell date +%Y%m%d_%H%M%S).sql

# Restore database
restore-db:
	@echo "🔄 Restoring database..."
	@read -p "Enter backup file path: " backup_file; \
	docker-compose exec -T db psql -U user -d braf_monetization < "$$backup_file"

# Generate API documentation
docs:
	@echo "📚 Generating API documentation..."
	mkdocs build

# Install pre-commit hooks
install-hooks:
	@echo "🪝 Installing pre-commit hooks..."
	pre-commit install

# Update dependencies
update-deps:
	@echo "📦 Updating dependencies..."
	pip-compile requirements.in
	pip-compile requirements-dev.in