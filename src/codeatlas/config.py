"""CodeAtlas configuration."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Global configuration for CodeAtlas."""

    # Paths
    repo_path: Path = Path(".")
    data_dir: Path = Path(".codeatlas")

    # Embedding
    embedding_model: str = "minishlab/potion-code-16M-v2"
    embedding_dim: int = 256

    # LLM
    llm_provider: str = "ollama"
    llm_model: str = "qwen3:8b"

    # Search
    bm25_weight: float = 0.4
    semantic_weight: float = 0.4
    identifier_weight: float = 0.2
    rrf_k: int = 60
    search_limit: int = 20
    graph_expansion_depth: int = 1

    # Indexing
    ignore_patterns: list[str] = [
        "node_modules",
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        ".codeatlas",
        ".tox",
        "dist",
        "build",
        "*.pyc",
    ]

    model_config = {"env_prefix": "CODEATLAS_"}

    @property
    def db_path(self) -> Path:
        """Path to the main SQLite database."""
        return self.data_dir / "codeatlas.db"

    @property
    def wiki_dir(self) -> Path:
        """Path to the generated wiki output."""
        return self.data_dir / "wiki"
