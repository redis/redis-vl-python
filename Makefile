.PHONY: install format lint test test-all test-notebooks clean redis-start redis-stop check-types docs-build docs-serve check help
.DEFAULT_GOAL := help

# Allow passing arguments to make targets (e.g., make test ARGS="--run-api-tests")
ARGS ?=

install: ## Install the project and all dependencies
	@echo "🚀 Installing project dependencies with uv"
	uv sync --all-extras

redis-start: ## Start Redis Stack in Docker
	@echo "🐳 Starting Redis Stack"
	docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest

redis-stop: ## Stop Redis Stack Docker container
	@echo "🛑 Stopping Redis Stack"
	docker stop redis-stack || true
	docker rm redis-stack || true

format: ## Format code with isort and black
	@echo "🎨 Formatting code"
	uv run isort ./redisvl ./tests/ --profile black
	uv run black ./redisvl ./tests/

check-format: ## Check code formatting
	@echo "🔍 Checking code formatting"
	uv run black --check ./redisvl

sort-imports: ## Sort imports with isort
	@echo "📦 Sorting imports"
	uv run isort ./redisvl ./tests/ --profile black

check-sort-imports: ## Check import sorting
	@echo "🔍 Checking import sorting"
	uv run isort ./redisvl --check-only --profile black

check-lint: ## Run pylint
	@echo "🔍 Running pylint"
	uv run pylint --rcfile=.pylintrc ./redisvl

check-types: ## Run mypy type checking
	@echo "🔍 Running mypy type checking"
	uv run python -m mypy ./redisvl

lint: format check-types ## Run all linting (format + type check)
	
test: ## Run tests (pass extra args with ARGS="...")
	@echo "🧪 Running tests"
	uv run python -m pytest -n auto --log-level=CRITICAL $(ARGS)
	
test-verbose: ## Run tests with verbose output
	@echo "🧪 Running tests (verbose)"
	uv run python -m pytest -n auto -vv -s --log-level=CRITICAL $(ARGS)
	
test-all: ## Run all tests including API tests
	@echo "🧪 Running all tests including API tests"
	uv run python -m pytest -n auto --log-level=CRITICAL --run-api-tests $(ARGS)

test-notebooks: ## Run notebook tests
	@echo "📓 Running notebook tests"
	uv run python -m pytest --nbval-lax ./docs/user_guide -vvv $(ARGS)

check: lint test ## Run all checks (lint + test)

docs-build: ## Build documentation
	@echo "📚 Building documentation"
	uv run make -C docs html

docs-serve: ## Serve documentation locally
	@echo "🌐 Serving documentation at http://localhost:8000"
	@echo "📁 Make sure docs are built first with 'make docs-build'"
	uv run python -m http.server --directory docs/_build/html

build: ## Build wheel and source distribution
	@echo "🏗️ Building distribution packages"
	uv build

clean: ## Clean up build artifacts and caches
	@echo "🧹 Cleaning up directory"
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".coverage" -delete
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name "dist" -exec rm -rf {} +
	find . -type d -name "build" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "_build" -exec rm -rf {} +
	find . -type f -name "*.log" -exec rm -rf {} +

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
