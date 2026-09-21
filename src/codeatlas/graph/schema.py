import logging
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
-- Symbols table (graph nodes)
CREATE TABLE IF NOT EXISTS symbols (
    node_id    TEXT PRIMARY KEY,
    file_path  TEXT NOT NULL,
    kind       TEXT NOT NULL,
    name       TEXT NOT NULL,
    parent_id  TEXT,
    signature  TEXT,
    docstring  TEXT,
    source_code TEXT,
    start_line INTEGER,
    end_line   INTEGER,
    source_hash TEXT,
    language   TEXT DEFAULT 'python'
);

-- Edges table (graph relationships)
CREATE TABLE IF NOT EXISTS edges (
    source_id  TEXT NOT NULL,
    target_id  TEXT NOT NULL,
    edge_type  TEXT NOT NULL,
    metadata   TEXT,
    PRIMARY KEY (source_id, target_id, edge_type)
);

-- Chunks table (searchable content)
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id     TEXT PRIMARY KEY,
    symbol_id    TEXT,
    file_path    TEXT NOT NULL,
    chunk_type   TEXT NOT NULL,
    content      TEXT NOT NULL,
    start_line   INTEGER,
    end_line     INTEGER,
    language     TEXT DEFAULT 'python',
    content_hash TEXT,
    is_definition INTEGER DEFAULT 0,
    identifiers  TEXT,
    identifier_tokens TEXT
);

-- Full-Text Search (FTS5) table for BM25 lexical retrieval
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED,
    file_path,
    content,
    identifiers,
    identifier_tokens,
    tokenize = 'porter unicode61'
);

-- Dense vector embeddings table (portable blob storage)
CREATE TABLE IF NOT EXISTS chunk_vectors (
    chunk_id     TEXT PRIMARY KEY,
    embedding    BLOB NOT NULL,
    dim          INTEGER NOT NULL
);

-- Index manifest for incremental indexing
CREATE TABLE IF NOT EXISTS file_manifest (
    file_path   TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    mtime       REAL,
    size        INTEGER,
    language    TEXT,
    indexed_at  TEXT
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);
CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(edge_type);
CREATE INDEX IF NOT EXISTS idx_symbols_file ON symbols(file_path);
CREATE INDEX IF NOT EXISTS idx_symbols_kind ON symbols(kind);
CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name);
CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_path);
CREATE INDEX IF NOT EXISTS idx_chunks_symbol ON chunks(symbol_id);
"""


def connect_db(db_path: Path) -> sqlite3.Connection:
    """Connect to SQLite database with portable PRAGMAs for APFS, exFAT, and network drives."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=10.0)
    conn.execute("PRAGMA journal_mode = MEMORY;")
    conn.execute("PRAGMA busy_timeout = 5000;")
    conn.execute("PRAGMA synchronous = NORMAL;")

    # Attempt to load sqlite-vec extension for native vector acceleration
    try:
        import sqlite_vec

        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
    except Exception as exc:
        logger.debug("sqlite-vec extension not loaded: %s", exc)

    return conn


@contextmanager
def get_db_connection(db_path: Path) -> Generator[sqlite3.Connection, None, None]:
    """Context manager for SQLite connections that guarantees closing."""
    conn = connect_db(db_path)
    try:
        yield conn
    finally:
        conn.close()


def init_vec_table(conn: sqlite3.Connection, dim: int = 384) -> bool:
    """Attempt to initialize sqlite-vec vec0 virtual table for native ANN search."""
    try:
        conn.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vectors_vec USING vec0(
                chunk_id TEXT PRIMARY KEY,
                embedding FLOAT[{dim}] DISTANCE_METRIC=cosine
            );
            """
        )
        return True
    except Exception as exc:
        logger.debug("Could not create vec0 virtual table: %s", exc)
        return False


def has_vec_table(conn: sqlite3.Connection) -> bool:
    """Check if the chunk_vectors_vec virtual table is available."""
    try:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='chunk_vectors_vec'"
        ).fetchone()
        return row is not None
    except Exception:
        return False
