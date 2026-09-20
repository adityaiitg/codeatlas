"""End-to-end integration tests for CodeAtlas pipeline."""

from pathlib import Path

from codeatlas.config import Settings
from codeatlas.graph.queries import GraphQueries
from codeatlas.index.indexer import Indexer
from codeatlas.retrieval.retriever import Retriever
from codeatlas.wiki.generator import WikiGenerator


def test_full_codeatlas_pipeline(tmp_path: Path):
    fixture_dir = Path(__file__).parent.parent / "fixtures" / "sample_project"
    data_dir = tmp_path / ".codeatlas"
    settings = Settings(repo_path=fixture_dir, data_dir=data_dir)

    # 1. Run Indexer (no vectors for ultra-fast CI test)
    indexer = Indexer(settings, embed_vectors=False)
    stats = indexer.index_repository(force=True)
    indexer.close()

    assert stats["files"] > 0
    assert stats["symbols"] > 0
    assert stats["edges"] > 0
    assert stats["chunks"] > 0

    # 2. Run Hybrid Search
    retriever = Retriever(settings)
    results = retriever.search("AuthMiddleware", limit=3, mode="lexical")
    assert len(results) > 0
    top_result = results[0]
    assert "AuthMiddleware" in top_result.content
    assert top_result.is_definition

    # Search for jwt
    jwt_results = retriever.search("parse_jwt", limit=3, mode="lexical")
    assert len(jwt_results) > 0
    assert any("jwt" in r.content.lower() for r in jwt_results)

    # 3. Test Graph Queries
    gq = GraphQueries(settings.db_path)
    auth_matches = [n for n in gq.G.nodes if "AuthMiddleware.authenticate" in n]
    assert len(auth_matches) > 0
    auth_node = auth_matches[0]

    callees = gq.get_callees(auth_node)
    assert any(
        "parse_jwt" in c or "_extract_token" in c or "verify_signature" in c for c in callees
    )

    # 4. Impact Analysis
    impact = gq.impact_analysis(auth_node)
    assert isinstance(impact["direct_dependents"], list)

    # 5. Living Wiki Generation
    generator = WikiGenerator(settings)
    wiki_files = generator.generate()
    assert len(wiki_files) >= 3

    arch_file = settings.wiki_dir / "architecture.md"
    assert arch_file.exists()
    assert "System Architecture" in arch_file.read_text(encoding="utf-8")
