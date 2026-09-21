# Contributing to CodeAtlas

Thank you for your interest in contributing to CodeAtlas!

## Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/adityaiitg/codeatlas.git
   cd codeatlas
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install editable development dependencies:**
   ```bash
   pip install -e ".[dev]"
   ```

## Running Tests & Checks

- **Run unit & integration tests:**
  ```bash
  pytest
  ```

- **Run linter:**
  ```bash
  ruff check src/ tests/
  ```

- **Format code:**
  ```bash
  ruff format src/ tests/
  ```

## Making Changes

1. Fork the repo and create a feature branch (`git checkout -b feature/my-feature`).
2. Implement your changes with accompanying tests.
3. Verify that `pytest` and `ruff` pass cleanly.
4. Submit a Pull Request.
