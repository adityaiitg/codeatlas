from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass, field
from types import TracebackType

import numpy as np

from codeatlas.config import Settings
from codeatlas.graph.queries import GraphQueries
from codeatlas.graph.schema import get_db_connection, has_vec_table
from codeatlas.index.embedder import Embedder

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single retrieved and reranked code/doc chunk."""

    chunk_id: str
    symbol_id: str | None
    file_path: str
    start_line: int
    end_line: int
    content: str
    chunk_type: str
    is_definition: bool
    score: float
    lexical_rank: int | None = None
    semantic_rank: int | None = None
    neighbors: list[dict] = field(default_factory=list)


class Retriever:
    """Performs hybrid lexical-semantic search with graph neighborhood expansion."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.db_path = settings.db_path
        self.embedder = Embedder(model_name=settings.embedding_model)
        self.graph_queries = GraphQueries(settings.db_path)

    def __enter__(self) -> Retriever:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self):
        """Release any held resources."""
        pass

    def _classify_query(self, query: str) -> tuple[float, float]:
        """Classify query to dynamically adjust lexical vs semantic weights."""
        q = query.strip()
        # If query has code identifiers (dots, underscores, parentheses, camelCase)
        has_identifier_syntax = bool(re.search(r"[_.:()]|[a-z][A-Z]", q))
        word_count = len(q.split())

        if has_identifier_syntax or word_count <= 2:
            # Code/symbol lookup: prioritize lexical matching
            return (0.7, 0.3)
        # Conceptual / question: prioritize dense semantic vectors
        return (0.3, 0.7)

    def _search_lexical(self, query: str, limit: int = 50) -> list[tuple[str, float]]:
        """Query SQLite FTS5 for BM25 ranking with syntax sanitization and ranked fallback."""
        from codeatlas.parser.python_parser import split_identifier

        clean_q = re.sub(r'["*^~:()./\-_]', " ", query)
        raw_terms = [t for t in clean_q.split() if t]
        terms_set = set(raw_terms)
        for t in raw_terms:
            terms_set.update(split_identifier(t))
        terms = [re.sub(r"[^a-zA-Z0-9]", "", t) for t in terms_set if len(t) >= 2]
        terms = [t for t in terms if t]
        if not terms:
            return []

        match_query = " OR ".join(f'"{t}"*' for t in terms)

        with get_db_connection(self.db_path) as conn:
            try:
                cursor = conn.execute(
                    """SELECT chunk_id, bm25(chunks_fts) as rank
                       FROM chunks_fts
                       WHERE chunks_fts MATCH ?
                       ORDER BY rank ASC
                       LIMIT ?""",
                    (match_query, limit),
                )
                rows = cursor.fetchall()
                if rows:
                    return [(r[0], float(r[1])) for r in rows]
            except sqlite3.OperationalError as exc:
                logger.debug("FTS5 MATCH failed: %s; using ranked LIKE fallback", exc)

            # Fallback: ranked search scoring by term frequency in content and identifiers
            scored_rows: list[tuple[str, float]] = []
            for term in terms[:5]:
                like_pat = f"%{term}%"
                cursor = conn.execute(
                    """SELECT chunk_id, content, identifiers FROM chunks
                       WHERE content LIKE ? OR identifiers LIKE ?
                       LIMIT ?""",
                    (like_pat, like_pat, limit),
                )
                for cid, content, idents in cursor.fetchall():
                    freq = (content or "").count(term) + (idents or "").count(term) * 2
                    scored_rows.append((cid, float(freq)))

            if not scored_rows:
                return []

            agg_scores: dict[str, float] = {}
            for cid, score in scored_rows:
                agg_scores[cid] = agg_scores.get(cid, 0.0) + score

            sorted_results = sorted(agg_scores.items(), key=lambda x: x[1], reverse=True)[:limit]
            # Convert to ascending rank order (lower rank = better match)
            return [(cid, -score) for cid, score in sorted_results]

    def _search_semantic(self, query: str, limit: int = 50) -> list[tuple[str, float]]:
        """Query vector index using native sqlite-vec ANN if available, else cosine similarity."""
        query_vec = self.embedder.embed_query(query)
        query_f32 = query_vec.astype(np.float32)

        with get_db_connection(self.db_path) as conn:
            # 1. Native sqlite-vec vec0 index search
            if has_vec_table(conn):
                try:
                    cursor = conn.execute(
                        """SELECT chunk_id, distance
                           FROM chunk_vectors_vec
                           WHERE embedding MATCH ? AND k = ?
                           ORDER BY distance ASC""",
                        (query_f32, limit),
                    )
                    rows = cursor.fetchall()
                    if rows:
                        return [(r[0], max(0.0, 1.0 - float(r[1]))) for r in rows]
                except Exception as exc:
                    logger.debug("Native sqlite-vec search failed: %s; falling back", exc)

            # 2. Fallback in-memory cosine similarity
            cursor = conn.execute("SELECT chunk_id, embedding, dim FROM chunk_vectors")
            rows = cursor.fetchall()
            if not rows:
                return []

            chunk_ids = [r[0] for r in rows]
            dim = rows[0][2]
            if query_vec.shape[0] != dim:
                return []

            doc_vecs = np.array(
                [np.frombuffer(r[1], dtype=np.float32) for r in rows], dtype=np.float32
            )
            scores = self.embedder.cosine_similarity(query_vec, doc_vecs)
            top_indices = np.argsort(scores)[::-1][:limit]
            return [(chunk_ids[i], float(scores[i])) for i in top_indices]

    def search(
        self,
        query: str,
        limit: int = 10,
        mode: str = "hybrid",
        expand_graph: bool = True,
    ) -> list[SearchResult]:
        """Perform search using specified mode ('hybrid', 'lexical', 'semantic')."""
        if not self.db_path.exists():
            return []

        bm25_weight, sem_weight = self._classify_query(query)
        k = self.settings.rrf_k

        lex_results = []
        sem_results = []

        if mode in ("hybrid", "lexical"):
            lex_results = self._search_lexical(query, limit=50)

        if mode in ("hybrid", "semantic"):
            sem_results = self._search_semantic(query, limit=50)

        lex_ranks = {cid: rank for rank, (cid, _) in enumerate(lex_results, start=1)}
        sem_ranks = {cid: rank for rank, (cid, _) in enumerate(sem_results, start=1)}

        all_cids = set(lex_ranks.keys()) | set(sem_ranks.keys())
        if not all_cids:
            return []

        # Load chunk metadata in a single query
        with get_db_connection(self.db_path) as conn:
            placeholders = ",".join("?" * len(all_cids))
            cursor = conn.execute(
                f"""SELECT chunk_id, symbol_id, file_path, start_line, end_line,
                          content, chunk_type, is_definition
                   FROM chunks WHERE chunk_id IN ({placeholders})""",
                list(all_cids),
            )
            chunk_map = {
                row[0]: {
                    "chunk_id": row[0],
                    "symbol_id": row[1],
                    "file_path": row[2],
                    "start_line": row[3],
                    "end_line": row[4],
                    "content": row[5],
                    "chunk_type": row[6],
                    "is_definition": bool(row[7]),
                }
                for row in cursor.fetchall()
            }

        # Reciprocal Rank Fusion (RRF)
        scored_chunks: list[SearchResult] = []
        for cid in all_cids:
            chunk_data = chunk_map.get(cid)
            if not chunk_data:
                continue

            rrf_score = 0.0
            if cid in lex_ranks:
                rrf_score += bm25_weight / (k + lex_ranks[cid])
            if cid in sem_ranks:
                rrf_score += sem_weight / (k + sem_ranks[cid])

            # Definition boost (boost definitions by 25%)
            if chunk_data["is_definition"]:
                rrf_score *= 1.25

            res = SearchResult(
                chunk_id=cid,
                symbol_id=chunk_data["symbol_id"],
                file_path=chunk_data["file_path"],
                start_line=chunk_data["start_line"],
                end_line=chunk_data["end_line"],
                content=chunk_data["content"],
                chunk_type=chunk_data["chunk_type"],
                is_definition=chunk_data["is_definition"],
                score=rrf_score,
                lexical_rank=lex_ranks.get(cid),
                semantic_rank=sem_ranks.get(cid),
            )
            scored_chunks.append(res)

        scored_chunks.sort(key=lambda x: x.score, reverse=True)
        top_results = scored_chunks[:limit]

        # Graph neighborhood expansion
        if expand_graph:
            self._attach_neighborhoods(top_results)

        return top_results

    def _attach_neighborhoods(self, results: list[SearchResult]):
        """Expand graph neighbors for top search results in a single batched query."""
        results_with_symbols = [r for r in results if r.symbol_id]
        if not results_with_symbols:
            return

        result_neighbor_ids: dict[str, list[str]] = {}
        all_neighbor_ids: set[str] = set()

        for r in results_with_symbols:
            assert r.symbol_id is not None
            neighbors = self.graph_queries.expand_neighborhood(
                [r.symbol_id], depth=self.settings.graph_expansion_depth
            )
            neighbor_ids = [nid for nid in neighbors if nid != r.symbol_id]
            result_neighbor_ids[r.chunk_id] = neighbor_ids
            all_neighbor_ids.update(neighbor_ids)

        if not all_neighbor_ids:
            return

        neighbor_id_list = list(all_neighbor_ids)
        placeholders = ",".join("?" * len(neighbor_id_list))
        with get_db_connection(self.db_path) as conn:
            cursor = conn.execute(
                f"""SELECT node_id, kind, name, signature, file_path, start_line
                   FROM symbols WHERE node_id IN ({placeholders})""",
                neighbor_id_list,
            )
            symbols_by_id = {
                row[0]: {
                    "node_id": row[0],
                    "kind": row[1],
                    "name": row[2],
                    "signature": row[3],
                    "file_path": row[4],
                    "start_line": row[5],
                }
                for row in cursor.fetchall()
            }

        for r in results_with_symbols:
            nids = result_neighbor_ids.get(r.chunk_id, [])
            r.neighbors = [symbols_by_id[nid] for nid in nids if nid in symbols_by_id]
