"""Thin wrapper around sentence-transformers with an LRU-style text cache.

GuardLens only needs a fixed embedding function — it does not care which
model produced the vectors, as long as they're L2-normalized so cosine
similarity reduces to a dot product (and Euclidean distance in HDBSCAN
becomes monotonic with cosine distance).
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 max_cache_size: int = 50_000):
        self.model_name = model_name
        self.max_cache_size = max_cache_size
        self._model = SentenceTransformer(model_name)
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        todo = [t for t in dict.fromkeys(texts) if t not in self._cache]
        for i in range(0, len(todo), batch_size):
            batch = todo[i:i + batch_size]
            vecs = self._model.encode(batch, batch_size=batch_size,
                                      show_progress_bar=False,
                                      normalize_embeddings=True)
            for t, v in zip(batch, vecs):
                self._cache[t] = v
                self._cache.move_to_end(t)
                if len(self._cache) > self.max_cache_size:
                    self._cache.popitem(last=False)
        for t in texts:
            if t in self._cache:
                self._cache.move_to_end(t)
        return np.stack([self._cache[t] for t in texts]) if texts else np.empty((0, self.dim))

    @property
    def dim(self) -> int:
        return self._model.get_sentence_embedding_dimension()
