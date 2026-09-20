"""Abstract base for language-specific parsers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from codeatlas.models.chunks import CodeChunk
from codeatlas.models.relationships import Edge
from codeatlas.models.symbols import Symbol


class LanguageParser(ABC):
    """Base class for language-specific AST parsers."""

    @abstractmethod
    def parse_file(self, file_path: Path) -> tuple[list[Symbol], list[CodeChunk], list[Edge]]:
        """Parse a source file and extract symbols, chunks, and edges."""
        ...
