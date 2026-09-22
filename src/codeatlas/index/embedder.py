"""Dense vector embedding generation using FastEmbed."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

logger = logging.getLogger(__name__)


class Embedder:
    """Generates dense vector embeddings using FastEmbed or Model2Vec."""

    def __init__(self, model_name: str = "minishlab/potion-code-16M-v2"):
        self.model_name = model_name
        self._model = None
        self._is_static = "potion" in model_name.lower() or "model2vec" in model_name.lower()

    def _get_model(self):
        if self._model is None:
            if self._is_static:
                from model2vec import StaticModel

                self._model = StaticModel.from_pretrained(self.model_name)
            else:
                from fastembed import TextEmbedding

                self._model = TextEmbedding(model_name=self.model_name)
        return self._model

    def embed_texts(self, texts: Sequence[str], batch_size: int = 64) -> list[np.ndarray]:
        """Generate normalized dense embeddings for a sequence of text strings."""
        if not texts:
            return []

        model = self._get_model()
        if self._is_static:
            embeddings = model.encode(list(texts), use_multiprocessing=False)
            return [np.array(emb, dtype=np.float32) for emb in embeddings]

        embeddings_gen = model.embed(list(texts), batch_size=batch_size)
        return [np.array(emb, dtype=np.float32) for emb in embeddings_gen]

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single search query."""
        model = self._get_model()
        if self._is_static:
            embs = model.encode([query], use_multiprocessing=False)
            return np.array(embs[0], dtype=np.float32)

        embs = list(model.embed([query]))
        return np.array(embs[0], dtype=np.float32)

    @staticmethod
    def cosine_similarity(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between 1D query vector and 2D matrix of doc vectors."""
        # Normalize
        q_norm = np.linalg.norm(query_vec)
        if q_norm > 0:
            query_vec = query_vec / q_norm

        d_norm = np.linalg.norm(doc_vecs, axis=1, keepdims=True)
        d_norm[d_norm == 0] = 1e-9
        normalized_docs = doc_vecs / d_norm

        return np.dot(normalized_docs, query_vec)
