"""Code relationship/edge types for the knowledge graph."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel


class EdgeType(StrEnum):
    """Types of relationships between code symbols."""

    IMPORTS = "IMPORTS"
    CALLS = "CALLS"
    DEFINES = "DEFINES"
    INHERITS = "INHERITS"
    IMPLEMENTS = "IMPLEMENTS"
    OVERRIDES = "OVERRIDES"
    TESTED_BY = "TESTED_BY"
    READS_CONFIG = "READS_CONFIG"
    EXPOSED_BY = "EXPOSED_BY"
    SIMILAR_TO = "SIMILAR_TO"


class Edge(BaseModel):
    """A directed edge in the code knowledge graph."""

    source_id: str
    target_id: str
    edge_type: EdgeType
    metadata: dict | None = None
