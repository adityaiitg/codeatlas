"""Unit tests for graph export in JSON, GraphML, and DOT formats."""

import json
from pathlib import Path

from codeatlas.graph.builder import GraphBuilder
from codeatlas.graph.queries import GraphQueries
from codeatlas.models.relationships import Edge, EdgeType
from codeatlas.models.symbols import Symbol, SymbolKind


def test_export_graph_formats(tmp_path: Path):
    db_path = tmp_path / "test.db"
    with GraphBuilder(db_path) as gb:
        s1 = Symbol(
            node_id="sym:A",
            file_path="src/a.py",
            kind=SymbolKind.FUNCTION,
            name="fn_a",
            start_line=1,
            end_line=5,
            source_code="def fn_a(): pass",
            content_hash="h1",
        )
        s2 = Symbol(
            node_id="sym:B",
            file_path="src/b.py",
            kind=SymbolKind.FUNCTION,
            name="fn_b",
            start_line=1,
            end_line=5,
            source_code="def fn_b(): pass",
            content_hash="h2",
        )
        gb.add_symbols([s1, s2])
        gb.add_edges([Edge(source_id=s1.node_id, target_id=s2.node_id, edge_type=EdgeType.CALLS)])

    gq = GraphQueries(db_path)

    # 1. JSON export
    json_path = tmp_path / "graph.json"
    gq.export_graph(json_path, format="json")
    assert json_path.exists()
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert "nodes" in data or "edges" in data

    # 2. GraphML export
    graphml_path = tmp_path / "graph.graphml"
    gq.export_graph(graphml_path, format="graphml")
    assert graphml_path.exists()
    assert "<graphml" in graphml_path.read_text(encoding="utf-8")

    # 3. DOT export
    dot_path = tmp_path / "graph.dot"
    gq.export_graph(dot_path, format="dot")
    assert dot_path.exists()
    dot_content = dot_path.read_text(encoding="utf-8")
    assert "digraph CodeAtlas" in dot_content
    assert "sym:A" in dot_content
    assert "sym:B" in dot_content
