"""Main repository indexing pipeline."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from codeatlas.config import Settings
from codeatlas.graph.builder import GraphBuilder
from codeatlas.index.embedder import Embedder
from codeatlas.parser.file_scanner import FileScanner
from codeatlas.parser.python_parser import PythonParser

logger = logging.getLogger(__name__)


class Indexer:
    """Coordinates parsing, graph persistence, FTS indexing, and vector embedding."""

    def __init__(self, settings: Settings, embed_vectors: bool = True):
        self.settings = settings
        self.embed_vectors = embed_vectors
        self.scanner = FileScanner(settings)
        self.graph_builder = GraphBuilder(settings.db_path)
        self.python_parser = PythonParser()
        self.embedder = Embedder(model_name=settings.embedding_model) if embed_vectors else None

    def index_repository(self, force: bool = False) -> dict:
        """Scan and index all files in the repository."""
        scanned_files = self.scanner.scan()
        manifest = {} if force else self.graph_builder.get_manifest()

        current_files = {str(p): p for p, _ in scanned_files}
        deleted_files = set(manifest.keys()) - set(current_files.keys())

        # Remove deleted files from index
        for del_path in deleted_files:
            self.graph_builder.remove_file(del_path)

        files_to_index: list[tuple[Path, str]] = []
        for path, lang in scanned_files:
            path_str = str(path)
            try:
                content = path.read_bytes()
                curr_hash = hashlib.sha256(content).hexdigest()
            except OSError as exc:
                logger.warning("Skipping unreadable file %s: %s", path, exc)
                continue

            if force or manifest.get(path_str) != curr_hash:
                files_to_index.append((path, lang))

        indexed_count = 0
        all_new_chunks = []

        for path, lang in files_to_index:
            path_str = str(path)
            content = path.read_bytes()
            curr_hash = hashlib.sha256(content).hexdigest()
            stat = path.stat()

            # Clean previous state for this file
            self.graph_builder.remove_file(path_str)

            if lang == "python":
                symbols, chunks, edges = self.python_parser.parse_file(path)
                self.graph_builder.add_symbols(symbols)
                self.graph_builder.add_edges(edges)
                self.graph_builder.add_chunks(chunks)
                all_new_chunks.extend(chunks)

            self.graph_builder.update_file_manifest(
                file_path=path_str,
                content_hash=curr_hash,
                mtime=stat.st_mtime,
                size=stat.st_size,
                language=lang,
            )
            indexed_count += 1

        # Generate and store dense vector embeddings
        if self.embed_vectors and self.embedder and all_new_chunks:
            texts = [c.content for c in all_new_chunks]
            chunk_ids = [c.chunk_id for c in all_new_chunks]
            vectors = self.embedder.embed_texts(texts)
            id_vectors = list(zip(chunk_ids, vectors, strict=True))
            self.graph_builder.add_embeddings(id_vectors)

        stats = self.graph_builder.get_stats()
        stats["indexed_files_this_run"] = indexed_count
        stats["deleted_files_this_run"] = len(deleted_files)
        return stats

    def close(self):
        """Close graph builder database connection."""
        self.graph_builder.close()
