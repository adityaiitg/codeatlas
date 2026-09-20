"""Code and wiki chunks for the search index."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel


class ChunkType(StrEnum):
    """Content type of a chunk, used for query routing."""

    CODE = "code"
    DOCS = "docs"
    CONFIG = "config"
    TEST = "test"
    GIT = "git"
    WIKI = "wiki"


class CodeChunk(BaseModel):
    """A searchable chunk of code or documentation."""

    chunk_id: str
    symbol_id: str | None = None
    file_path: str
    chunk_type: ChunkType
    content: str
    start_line: int
    end_line: int
    language: str = "python"
    content_hash: str
    is_definition: bool = False

    # Identifier tokens for BM25 boosting
    identifiers: list[str] = []
    identifier_tokens: list[str] = []
