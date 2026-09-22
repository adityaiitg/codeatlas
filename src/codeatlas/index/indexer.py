"""Main repository indexing pipeline."""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from types import TracebackType

from codeatlas.config import Settings
from codeatlas.graph.builder import GraphBuilder
from codeatlas.index.embedder import Embedder
from codeatlas.parser.base import LanguageParser
from codeatlas.parser.file_scanner import FileScanner
from codeatlas.parser.generic_parser import GenericCodeParser
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
        self.parsers: dict[str, LanguageParser] = {
            "python": self.python_parser,
            "typescript": GenericCodeParser("typescript"),
            "javascript": GenericCodeParser("javascript"),
            "go": GenericCodeParser("go"),
            "rust": GenericCodeParser("rust"),
            "java": GenericCodeParser("java"),
            "c": GenericCodeParser("c"),
            "cpp": GenericCodeParser("cpp"),
        }
        self.embedder = Embedder(model_name=settings.embedding_model) if embed_vectors else None

    def __enter__(self) -> Indexer:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    def index_repository(self, force: bool = False) -> dict:
        """Scan and index all files in the repository."""
        scanned_files = self.scanner.scan()
        manifest_details = {} if force else self.graph_builder.get_manifest_details()

        current_files = {str(p): p for p, _ in scanned_files}
        deleted_files = set(manifest_details.keys()) - set(current_files.keys())

        # Remove deleted files from index
        for del_path in deleted_files:
            self.graph_builder.remove_file(del_path)

        files_to_index: list[tuple[Path, str, str, os.stat_result]] = []
        for path, lang in scanned_files:
            path_str = str(path)
            cached = manifest_details.get(path_str)
            try:
                st = path.stat()
            except OSError as exc:
                logger.warning("Skipping unreadable file %s: %s", path, exc)
                continue

            # Stat-first check: if size and mtime match, skip reading disk and computing SHA-256
            if not force and cached is not None:
                cached_size = cached.get("size")
                cached_mtime = cached.get("mtime")
                if cached_size is not None and cached_mtime is not None:
                    if st.st_size == cached_size and abs(st.st_mtime - cached_mtime) < 1e-4:
                        continue

            try:
                content = path.read_bytes()
                curr_hash = hashlib.sha256(content).hexdigest()
            except OSError as exc:
                logger.warning("Skipping unreadable file %s: %s", path, exc)
                continue

            if force or cached is None or cached.get("content_hash") != curr_hash:
                files_to_index.append((path, lang, curr_hash, st))

        indexed_count = 0
        all_new_chunks = []

        for path, lang, curr_hash, stat in files_to_index:
            path_str = str(path)
            # Clean previous state for this file
            self.graph_builder.remove_file(path_str)

            parser = self.parsers.get(lang)
            if parser:
                symbols, chunks, edges = parser.parse_file(path)
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

        # Generate and store dense vector embeddings in bounded batches
        if self.embed_vectors and self.embedder and all_new_chunks:
            batch_size = 256
            for i in range(0, len(all_new_chunks), batch_size):
                chunk_batch = all_new_chunks[i : i + batch_size]
                texts = [c.content for c in chunk_batch]
                chunk_ids = [c.chunk_id for c in chunk_batch]
                vectors = self.embedder.embed_texts(texts)
                id_vectors = list(zip(chunk_ids, vectors, strict=True))
                self.graph_builder.add_embeddings(id_vectors)

        # Persist index configuration metadata
        self.graph_builder.set_metadata("embedding_model", self.settings.embedding_model)
        self.graph_builder.set_metadata("embedding_dim", str(self.settings.embedding_dim))
        self.graph_builder.set_metadata("embed_vectors", str(int(self.embed_vectors)))

        # Resolve symbolic call and import edges into canonical symbol IDs
        from codeatlas.graph.linker import GraphLinker

        linker = GraphLinker(self.settings.db_path)
        resolved_count = linker.link()
        logger.debug("Resolved %d call and symbol edges in knowledge graph", resolved_count)

        stats = self.graph_builder.get_stats()
        stats["indexed_files_this_run"] = indexed_count
        stats["deleted_files_this_run"] = len(deleted_files)
        return stats

    def close(self):
        """Close graph builder database connection."""
        self.graph_builder.close()
