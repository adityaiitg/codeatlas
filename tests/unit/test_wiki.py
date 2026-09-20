"""Unit tests for the living wiki generator."""

from pathlib import Path

from codeatlas.config import Settings
from codeatlas.graph.builder import GraphBuilder
from codeatlas.models.relationships import Edge, EdgeType
from codeatlas.models.symbols import Symbol, SymbolKind
from codeatlas.wiki.generator import WikiGenerator


def test_wiki_generation(tmp_path: Path):
    data_dir = tmp_path / ".codeatlas"
    settings = Settings(repo_path=tmp_path, data_dir=data_dir)
    builder = GraphBuilder(settings.db_path)

    sym = Symbol(
        node_id="rec.service:RecommendationService",
        file_path=str(tmp_path / "service.py"),
        kind=SymbolKind.CLASS,
        name="RecommendationService",
        docstring="Online recommendation engine.",
        start_line=1,
        end_line=25,
        source_code="class RecommendationService: pass",
        content_hash="h1",
    )
    builder.add_symbols([sym])

    edge = Edge(
        source_id=sym.node_id,
        target_id="call:Ranker.rank",
        edge_type=EdgeType.CALLS,
    )
    builder.add_edges([edge])
    builder.close()

    generator = WikiGenerator(settings)
    generated = generator.generate()

    assert len(generated) >= 3
    file_names = [f.name for f in generated]
    assert "index.md" in file_names
    assert "architecture.md" in file_names
    assert "workflows.md" in file_names

    arch_text = (settings.wiki_dir / "architecture.md").read_text()
    assert "System Architecture" in arch_text
    assert "RecommendationService" in arch_text
