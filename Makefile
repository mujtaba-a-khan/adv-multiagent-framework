.PHONY: help install dev lint format typecheck test test-cov test-unit test-integration test-e2e \
       run serve build docker-build docker-up docker-down clean

# ── Variables ────────────────────────────────────────────────────────────────
PYTHON := uv run python
PYTEST := uv run pytest
RUFF := uv run ruff
MYPY := uv run mypy

# ── Help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
	awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Setup 
install: ## Install production dependencies
	uv sync --frozen --no-dev

dev: ## Install all dependencies (including dev)
	uv sync --frozen
	uv run pre-commit install || true

# ── Quality
lint: ## Run linter (ruff check + format check)
	$(RUFF) check src/ tests/
	$(RUFF) format --check src/ tests/

format: ## Auto-format code
	$(RUFF) format src/ tests/
	$(RUFF) check --fix src/ tests/

typecheck: ## Run mypy type checking
	$(MYPY) src/adversarial_framework/

check: lint typecheck ## Run all quality checks

# ── Testing 
test: ## Run all tests
	$(PYTEST) tests/ -v

test-cov: ## Run tests with coverage report
	$(PYTEST) tests/ -v --cov=adversarial_framework --cov-report=term-missing --cov-report=html

test-unit: ## Run unit tests only
	$(PYTEST) tests/unit/ -v

test-integration: ## Run integration tests only
	$(PYTEST) tests/integration/ -v

test-e2e: ## Run end-to-end tests only
	$(PYTEST) tests/e2e/ -v

# ── Run 
run: ## Run CLI with default settings
	uv run adversarial --target llama3:8b --strategy pair --objective "Test objective"

serve: ## Start FastAPI development server
	uv run uvicorn adversarial_framework.api.app:create_app --factory \
		--host 0.0.0.0 --port 8000 --reload --reload-dir src

serve-ui: ## Start Next.js development server
	cd ui && npm run dev

# ── Docker 
docker-build: ## Build all Docker images
	docker compose build

docker-up: ## Start development stack
	docker compose up -d

docker-down: ## Stop development stack
	docker compose down

docker-prod: ## Start production stack
	docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

docker-logs: ## Tail container logs
	docker compose logs -f

# ── Database 
db-migrate: ## Run database migrations
	uv run alembic upgrade head

db-revision: ## Create a new migration revision
	uv run alembic revision --autogenerate -m "$(msg)"

db-downgrade: ## Rollback last migration
	uv run alembic downgrade -1

# ── Utility 
clean: ## Remove build artifacts and caches
	rm -rf dist/ build/ .mypy_cache/ .pytest_cache/ .ruff_cache/
	rm -rf htmlcov/ .coverage coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

report: ## Generate test coverage HTML report
	$(PYTEST) tests/ --cov=adversarial_framework --cov-report=html
	@echo "Coverage report: htmlcov/index.html"
