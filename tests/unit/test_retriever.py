"""Unit tests for the hybrid retriever and query classification."""

from pathlib import Path

from codeatlas.config import Settings
from codeatlas.graph.builder import GraphBuilder
from codeatlas.models.chunks import ChunkType, CodeChunk
from codeatlas.retrieval.retriever import Retriever


def test_query_classification(tmp_path: Path):
    settings = Settings(data_dir=tmp_path)
    retriever = Retriever(settings)

    # Identifiers should trigger high lexical weight
    w_lex, w_sem = retriever._classify_query("AuthMiddleware.authenticate")
    assert w_lex > w_sem

    w_lex, w_sem = retriever._classify_query("get_token")
    assert w_lex > w_sem

    # Acronyms (JWT, HTTP) should also trigger high lexical weight
    w_lex_acronym, w_sem_acronym = retriever._classify_query("JWT")
    assert w_lex_acronym > w_sem_acronym

    # Questions should trigger high semantic weight
    w_lex, w_sem = retriever._classify_query("how does user authentication and validation work?")
    assert w_sem > w_lex


def test_lexical_search(tmp_path: Path):
    settings = Settings(data_dir=tmp_path)
    with GraphBuilder(settings.db_path) as builder:
        chunk1 = CodeChunk(
            chunk_id="chunk:jwt_auth",
            symbol_id="auth:jwt",
            file_path="src/auth.py",
            chunk_type=ChunkType.CODE,
            content="def verify_signature(token, secret):\n    return hmac(token, secret)",
            start_line=1,
            end_line=2,
            content_hash="h1",
            is_definition=True,
            identifiers=["verify_signature", "token", "secret"],
            identifier_tokens=["verify", "signature", "token", "secret"],
        )
        builder.add_chunks([chunk1])

    with Retriever(settings) as retriever:
        matches = retriever._search_lexical("verify_signature")
        assert len(matches) > 0
        assert matches[0][0] == "chunk:jwt_auth"


def test_semantic_search_with_vectors(tmp_path: Path):
    import numpy as np

    settings = Settings(data_dir=tmp_path)
    with GraphBuilder(settings.db_path) as builder:
        # Create 2 vector embeddings
        dim = settings.embedding_dim
        v1 = np.zeros(dim, dtype=np.float32)
        v1[0] = 1.0
        v2 = np.zeros(dim, dtype=np.float32)
        v2[1] = 1.0

        builder.add_embeddings([("chunk:1", v1), ("chunk:2", v2)])

    with Retriever(settings) as retriever:
        # Mock embed_query on retriever embedder to return v1
        retriever.embedder.embed_query = lambda q: v1.copy()
        results = retriever._search_semantic("test query", limit=2)
        assert len(results) == 2
        # chunk:1 should have top similarity score
        assert results[0][0] == "chunk:1"
        assert results[0][1] > results[1][1]


def test_batched_neighborhood_expansion(tmp_path: Path):
    from codeatlas.models.relationships import Edge, EdgeType
    from codeatlas.models.symbols import Symbol, SymbolKind
    from codeatlas.retrieval.retriever import SearchResult

    settings = Settings(data_dir=tmp_path)
    with GraphBuilder(settings.db_path) as builder:
        sym_main = Symbol(
            node_id="sym:main",
            file_path="src/main.py",
            kind=SymbolKind.FUNCTION,
            name="main",
            start_line=1,
            end_line=10,
            source_code="def main(): pass",
            content_hash="h_main",
        )
        sym_helper = Symbol(
            node_id="sym:helper",
            file_path="src/utils.py",
            kind=SymbolKind.FUNCTION,
            name="helper",
            signature="def helper(): pass",
            start_line=5,
            end_line=8,
            source_code="def helper(): pass",
            content_hash="h_helper",
        )
        builder.add_symbols([sym_main, sym_helper])
        builder.add_edges(
            [
                Edge(
                    source_id=sym_main.node_id,
                    target_id=sym_helper.node_id,
                    edge_type=EdgeType.CALLS,
                )
            ]
        )

    with Retriever(settings) as retriever:
        sr = SearchResult(
            chunk_id="c_main",
            symbol_id="sym:main",
            file_path="src/main.py",
            start_line=1,
            end_line=10,
            content="def main(): helper()",
            chunk_type="code",
            is_definition=True,
            score=1.0,
        )
        retriever._attach_neighborhoods([sr])
        assert len(sr.neighbors) == 1
        assert sr.neighbors[0]["node_id"] == "sym:helper"
        assert sr.neighbors[0]["name"] == "helper"


def test_context_managers(tmp_path: Path):
    from codeatlas.graph.schema import get_db_connection
    from codeatlas.index.indexer import Indexer

    settings = Settings(data_dir=tmp_path)

    with get_db_connection(settings.db_path) as conn:
        assert conn is not None
        conn.execute("SELECT 1")

    with GraphBuilder(settings.db_path) as gb:
        assert gb.conn is not None

    with Indexer(settings, embed_vectors=False) as idx:
        assert idx.graph_builder is not None

    with Retriever(settings) as ret:
        assert ret.db_path == settings.db_path


def test_retriever_metadata_sync_fast_mode(tmp_path: Path):
    settings = Settings(data_dir=tmp_path)
    with GraphBuilder(settings.db_path) as gb:
        gb.set_metadata("embedding_model", "minishlab/potion-code-16M-v2")
        gb.set_metadata("embedding_dim", "256")

    # When Retriever starts with default settings, it should automatically detect the fast model from the DB!
    retriever = Retriever(settings)
    assert retriever.settings.embedding_model == "minishlab/potion-code-16M-v2"
    assert retriever.settings.embedding_dim == 256

