.PHONY: help install test lint format clean index export bench bench-compare

help:
	@echo "Available commands:"
	@echo "  make install  - Install development dependencies in editable mode"
	@echo "  make test     - Run pytest test suite"
	@echo "  make lint     - Run ruff linter checks"
	@echo "  make format   - Run ruff code formatter"
	@echo "  make bench    - Run CodeAtlas performance benchmark suite"
	@echo "  make clean    - Remove build artifacts and temporary files"
	@echo "  make index    - Index the current repository with CodeAtlas"

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/ benchmarks/

format:
	ruff format src/ tests/ benchmarks/
	ruff check --fix src/ tests/ benchmarks/

bench:
	python benchmarks/benchmark_suite.py

bench-compare:
	python benchmarks/compare_semble.py

clean:
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .codeatlas/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

index:
	codeatlas index .
