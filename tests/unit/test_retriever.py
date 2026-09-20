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

    # Questions should trigger high semantic weight
    w_lex, w_sem = retriever._classify_query("how does user authentication and validation work?")
    assert w_sem > w_lex


def test_lexical_search(tmp_path: Path):
    settings = Settings(data_dir=tmp_path)
    builder = GraphBuilder(settings.db_path)

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
    builder.close()

    retriever = Retriever(settings)
    matches = retriever._search_lexical("verify_signature")
    assert len(matches) > 0
    assert matches[0][0] == "chunk:jwt_auth"
