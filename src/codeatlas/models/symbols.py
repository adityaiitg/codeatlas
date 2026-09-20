"""Code symbol data models — the Canonical Code IR."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel


class SymbolKind(StrEnum):
    """Types of code symbols."""

    FILE = "file"
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    INTERFACE = "interface"
    VARIABLE = "variable"
    CONFIG_KEY = "config_key"
    ENDPOINT = "endpoint"


class Symbol(BaseModel):
    """A single code symbol extracted from AST parsing."""

    node_id: str
    file_path: str
    kind: SymbolKind
    name: str
    parent_symbol: str | None = None
    signature: str | None = None
    docstring: str | None = None
    start_line: int
    end_line: int
    source_code: str
    content_hash: str
    language: str = "python"
    imports: list[str] = []
    calls: list[str] = []
    referenced_types: list[str] = []
