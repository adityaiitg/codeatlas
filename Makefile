.PHONY: help install test lint format clean index export

help:
	@echo "Available commands:"
	@echo "  make install  - Install development dependencies in editable mode"
	@echo "  make test     - Run pytest test suite"
	@echo "  make lint     - Run ruff linter checks"
	@echo "  make format   - Run ruff code formatter"
	@echo "  make clean    - Remove build artifacts and temporary files"
	@echo "  make index    - Index the current repository with CodeAtlas"

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

clean:
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .codeatlas/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

index:
	codeatlas index .
