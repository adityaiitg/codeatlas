"""Build the code knowledge graph from parsed symbols, edges, chunks, and embeddings."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType

import numpy as np

from codeatlas.graph.schema import (
    SCHEMA_SQL,
    connect_db,
    get_index_metadata,
    has_vec_table,
    init_vec_table,
    set_index_metadata,
)
from codeatlas.models.chunks import CodeChunk
from codeatlas.models.relationships import Edge
from codeatlas.models.symbols import Symbol

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Persists symbols, edges, chunks, FTS5 indexes, and vector embeddings to SQLite."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = connect_db(db_path)
        self.conn.executescript(SCHEMA_SQL)
        self._has_vec = has_vec_table(self.conn)

    def __enter__(self) -> GraphBuilder:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    def remove_file(self, file_path: str):
        """Remove all symbols, edges, chunks, and FTS5 records for a specific file."""
        # Find all symbol IDs from this file
        cursor = self.conn.execute("SELECT node_id FROM symbols WHERE file_path = ?", (file_path,))
        symbol_ids = [row[0] for row in cursor.fetchall()]

        # Find all chunk IDs
        cursor = self.conn.execute("SELECT chunk_id FROM chunks WHERE file_path = ?", (file_path,))
        chunk_ids = [row[0] for row in cursor.fetchall()]

        # Delete from edges
        if symbol_ids:
            placeholders = ",".join("?" * len(symbol_ids))
            self.conn.execute(
                f"DELETE FROM edges WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders})",
                symbol_ids + symbol_ids,
            )

        # Delete from FTS & vectors
        if chunk_ids:
            placeholders = ",".join("?" * len(chunk_ids))
            self.conn.execute(
                f"DELETE FROM chunks_fts WHERE chunk_id IN ({placeholders})", chunk_ids
            )
            self.conn.execute(
                f"DELETE FROM chunk_vectors WHERE chunk_id IN ({placeholders})", chunk_ids
            )
            if self._has_vec:
                try:
                    self.conn.execute(
                        f"DELETE FROM chunk_vectors_vec WHERE chunk_id IN ({placeholders})",
                        chunk_ids,
                    )
                except Exception as exc:
                    logger.debug("Could not delete from chunk_vectors_vec: %s", exc)

        self.conn.execute("DELETE FROM symbols WHERE file_path = ?", (file_path,))
        self.conn.execute("DELETE FROM chunks WHERE file_path = ?", (file_path,))
        self.conn.execute("DELETE FROM file_manifest WHERE file_path = ?", (file_path,))
        self.conn.commit()

    def add_symbols(self, symbols: list[Symbol]):
        """Insert or replace symbols in the database using batch execution."""
        if not symbols:
            return
        rows = [
            (
                sym.node_id,
                sym.file_path,
                sym.kind.value,
                sym.name,
                sym.parent_symbol,
                sym.signature,
                sym.docstring,
                sym.source_code,
                sym.start_line,
                sym.end_line,
                sym.content_hash,
                sym.language,
            )
            for sym in symbols
        ]
        self.conn.executemany(
            """INSERT OR REPLACE INTO symbols
               (node_id, file_path, kind, name, parent_id, signature,
                docstring, source_code, start_line, end_line, source_hash, language)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        self.conn.commit()

    def add_edges(self, edges: list[Edge]):
        """Insert edges into the graph using batch execution."""
        if not edges:
            return
        rows = [
            (
                edge.source_id,
                edge.target_id,
                edge.edge_type.value,
                json.dumps(edge.metadata) if edge.metadata else None,
            )
            for edge in edges
        ]
        self.conn.executemany(
            """INSERT OR IGNORE INTO edges
               (source_id, target_id, edge_type, metadata)
               VALUES (?, ?, ?, ?)""",
            rows,
        )
        self.conn.commit()

    def add_chunks(self, chunks: list[CodeChunk]):
        """Insert chunks into the relational table and FTS5 search index using batch execution."""
        if not chunks:
            return
        chunk_rows = [
            (
                chunk.chunk_id,
                chunk.symbol_id,
                chunk.file_path,
                chunk.chunk_type.value,
                chunk.content,
                chunk.start_line,
                chunk.end_line,
                chunk.language,
                chunk.content_hash,
                int(chunk.is_definition),
                json.dumps(chunk.identifiers),
                json.dumps(chunk.identifier_tokens),
            )
            for chunk in chunks
        ]
        self.conn.executemany(
            """INSERT OR REPLACE INTO chunks
               (chunk_id, symbol_id, file_path, chunk_type, content,
                start_line, end_line, language, content_hash,
                is_definition, identifiers, identifier_tokens)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            chunk_rows,
        )

        fts_del_rows = [(c.chunk_id,) for c in chunks]
        self.conn.executemany("DELETE FROM chunks_fts WHERE chunk_id = ?", fts_del_rows)
        fts_rows = [
            (
                c.chunk_id,
                c.file_path,
                c.content,
                " ".join(c.identifiers),
                " ".join(c.identifier_tokens),
            )
            for c in chunks
        ]
        self.conn.executemany(
            """INSERT INTO chunks_fts (chunk_id, file_path, content, identifiers, identifier_tokens)
               VALUES (?, ?, ?, ?, ?)""",
            fts_rows,
        )
        self.conn.commit()

    def add_embeddings(self, chunk_id_vectors: list[tuple[str, np.ndarray]]):
        """Insert dense vector embeddings for chunks using batch execution."""
        if not chunk_id_vectors:
            return

        if not self._has_vec:
            dim = len(chunk_id_vectors[0][1])
            self._has_vec = init_vec_table(self.conn, dim=dim) or has_vec_table(self.conn)

        vec_rows = []
        vec_plugin_rows = []
        for chunk_id, vec in chunk_id_vectors:
            vec_f32 = vec.astype(np.float32)
            vec_bytes = vec_f32.tobytes()
            vec_rows.append((chunk_id, vec_bytes, len(vec)))
            if self._has_vec:
                vec_plugin_rows.append((chunk_id, vec_f32))

        self.conn.executemany(
            """INSERT OR REPLACE INTO chunk_vectors (chunk_id, embedding, dim)
               VALUES (?, ?, ?)""",
            vec_rows,
        )
        if self._has_vec and vec_plugin_rows:
            try:
                del_rows = [(cid,) for cid, _ in vec_plugin_rows]
                self.conn.executemany("DELETE FROM chunk_vectors_vec WHERE chunk_id = ?", del_rows)
                self.conn.executemany(
                    "INSERT INTO chunk_vectors_vec (chunk_id, embedding) VALUES (?, ?)",
                    vec_plugin_rows,
                )
            except Exception as exc:
                logger.debug("Failed inserting into chunk_vectors_vec: %s", exc)
        self.conn.commit()

    def update_file_manifest(
        self, file_path: str, content_hash: str, mtime: float, size: int, language: str
    ):
        """Update file manifest with current indexing metadata."""
        now = datetime.now(UTC).isoformat()
        self.conn.execute(
            """INSERT OR REPLACE INTO file_manifest
               (file_path, content_hash, mtime, size, language, indexed_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (file_path, content_hash, mtime, size, language, now),
        )
        self.conn.commit()

    def get_manifest(self) -> dict[str, str]:
        """Return a mapping of file_path -> content_hash for all indexed files."""
        cursor = self.conn.execute("SELECT file_path, content_hash FROM file_manifest")
        return {row[0]: row[1] for row in cursor.fetchall()}

    def get_manifest_details(self) -> dict[str, dict]:
        """Return a mapping of file_path -> {content_hash, mtime, size} for fast change detection."""
        cursor = self.conn.execute("SELECT file_path, content_hash, mtime, size FROM file_manifest")
        return {
            row[0]: {"content_hash": row[1], "mtime": row[2], "size": row[3]}
            for row in cursor.fetchall()
        }

    def set_metadata(self, key: str, value: str) -> None:
        """Store index metadata key-value pair."""
        set_index_metadata(self.conn, key, value)
        self.conn.commit()

    def get_metadata(self) -> dict[str, str]:
        """Retrieve all index metadata key-value pairs."""
        return get_index_metadata(self.conn)

    def get_stats(self) -> dict:
        """Get summary statistics of the indexed knowledge graph."""
        symbols_count = self.conn.execute("SELECT COUNT(*) FROM symbols").fetchone()[0]
        edges_count = self.conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        chunks_count = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        vectors_count = self.conn.execute("SELECT COUNT(*) FROM chunk_vectors").fetchone()[0]
        files_count = self.conn.execute("SELECT COUNT(*) FROM file_manifest").fetchone()[0]
        return {
            "files": files_count,
            "symbols": symbols_count,
            "edges": edges_count,
            "chunks": chunks_count,
            "vectors": vectors_count,
        }

    def close(self):
        """Close the database connection."""
        self.conn.close()
