"""Unit tests for the knowledge graph builder and queries."""

from pathlib import Path

from codeatlas.graph.builder import GraphBuilder
from codeatlas.graph.queries import GraphQueries
from codeatlas.models.chunks import ChunkType, CodeChunk
from codeatlas.models.relationships import Edge, EdgeType
from codeatlas.models.symbols import Symbol, SymbolKind


def test_graph_builder_and_queries(tmp_path: Path):
    db_path = tmp_path / "test.db"
    builder = GraphBuilder(db_path)

    # Add symbols
    sym1 = Symbol(
        node_id="auth.middleware:AuthMiddleware",
        file_path="src/auth/middleware.py",
        kind=SymbolKind.CLASS,
        name="AuthMiddleware",
        start_line=1,
        end_line=20,
        source_code="class AuthMiddleware: pass",
        content_hash="h1",
    )
    sym2 = Symbol(
        node_id="auth.middleware:AuthMiddleware.authenticate",
        file_path="src/auth/middleware.py",
        kind=SymbolKind.METHOD,
        name="authenticate",
        parent_symbol="auth.middleware:AuthMiddleware",
        start_line=10,
        end_line=18,
        source_code="def authenticate(self): pass",
        content_hash="h2",
    )
    sym3 = Symbol(
        node_id="auth.token:parse_jwt",
        file_path="src/auth/token.py",
        kind=SymbolKind.FUNCTION,
        name="parse_jwt",
        start_line=5,
        end_line=15,
        source_code="def parse_jwt(): pass",
        content_hash="h3",
    )

    builder.add_symbols([sym1, sym2, sym3])

    # Add edges
    edge1 = Edge(
        source_id=sym1.node_id,
        target_id=sym2.node_id,
        edge_type=EdgeType.DEFINES,
    )
    edge2 = Edge(
        source_id=sym2.node_id,
        target_id=sym3.node_id,
        edge_type=EdgeType.CALLS,
    )

    builder.add_edges([edge1, edge2])

    # Add chunks
    chunk1 = CodeChunk(
        chunk_id="c1",
        symbol_id=sym2.node_id,
        file_path="src/auth/middleware.py",
        chunk_type=ChunkType.CODE,
        content="def authenticate(self): pass",
        start_line=10,
        end_line=18,
        content_hash="h2",
        is_definition=True,
        identifiers=["authenticate", "AuthMiddleware"],
        identifier_tokens=["authenticate", "auth", "middleware"],
    )
    builder.add_chunks([chunk1])

    stats = builder.get_stats()
    assert stats["symbols"] == 3
    assert stats["edges"] == 2
    assert stats["chunks"] == 1
    builder.close()

    # Query graph
    gq = GraphQueries(db_path)
    assert gq.get_callers(sym3.node_id) == [sym2.node_id]
    assert gq.get_callees(sym2.node_id) == [sym3.node_id]

    neighborhood = gq.expand_neighborhood([sym2.node_id], depth=1)
    assert sym1.node_id in neighborhood
    assert sym3.node_id in neighborhood

    # Impact analysis: if sym3 changes, sym2 might be affected
    impact = gq.impact_analysis(sym1.node_id)
    assert sym2.node_id in impact["direct_dependents"]
