"""Vector retrieval backends.

Three backends share one interface (``build(docs)`` / ``search(query, k)``), so
the pipelines are agnostic to which is active:

  * ``SparseTfidfIndex``   — exact TF-IDF cosine similarity over an inverted
    index. Deterministic, dependency-free and the default offline backend
    (no model download, no API key).
  * ``VectorIndex`` + ``HashingTfidfEmbedder`` — dense hashed TF-IDF vectors,
    usable when a fixed-width embedding matrix is required.
  * ``VectorIndex`` + ``SentenceTransformerEmbedder`` — real neural sentence
    embeddings, selected automatically when ``sentence-transformers`` is
    installed (see :func:`create_vector_backend`).

Every backend returns ``List[(chunk_index, score)]`` sorted by descending score.
"""

from __future__ import annotations

import hashlib
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .lexical import term_list

DEFAULT_DIM = 2048


def _features(text: str) -> List[str]:
    """Word unigrams plus adjacent bigrams (the retrieval feature space)."""
    terms = term_list(text)
    return list(terms) + [f"{a}_{b}" for a, b in zip(terms, terms[1:])]


# ── exact sparse TF-IDF ───────────────────────────────────────────────────
class SparseTfidfIndex:
    """Exact sparse TF-IDF cosine index (L2-normalised, lnc.ltc weighting)."""

    name = "tfidf-sparse"

    def __init__(self) -> None:
        self.inverted: Dict[str, List[Tuple[int, float]]] = {}
        self.norms: List[float] = []
        self.idf: Dict[str, float] = {}
        self.num_docs = 0
        self.dim = 0

    def build(self, documents: Sequence[str]) -> "SparseTfidfIndex":
        self.num_docs = len(documents)
        df: Dict[str, int] = {}
        doc_features: List[Dict[str, int]] = []
        for text in documents:
            counts: Dict[str, int] = {}
            for feat in _features(text):
                counts[feat] = counts.get(feat, 0) + 1
            doc_features.append(counts)
            for feat in counts:
                df[feat] = df.get(feat, 0) + 1

        self.idf = {f: math.log((1 + self.num_docs) / (1 + c)) + 1.0 for f, c in df.items()}
        self.dim = len(self.idf)
        self.inverted = {}
        self.norms = [0.0] * self.num_docs
        for idx, counts in enumerate(doc_features):
            norm_sq = 0.0
            weights: Dict[str, float] = {}
            for feat, tf in counts.items():
                weight = (1.0 + math.log(tf)) * self.idf[feat]
                weights[feat] = weight
                norm_sq += weight * weight
            norm = math.sqrt(norm_sq)
            self.norms[idx] = norm or 1.0
            for feat, weight in weights.items():
                self.inverted.setdefault(feat, []).append((idx, weight / self.norms[idx]))
        return self

    def _encode(self, text: str) -> Tuple[Dict[str, float], float]:
        counts: Dict[str, int] = {}
        for feat in _features(text):
            counts[feat] = counts.get(feat, 0) + 1
        weights: Dict[str, float] = {}
        norm_sq = 0.0
        for feat, tf in counts.items():
            idf = self.idf.get(feat, math.log(1 + self.num_docs) + 1.0)
            weight = (1.0 + math.log(tf)) * idf
            weights[feat] = weight
            norm_sq += weight * weight
        return weights, (math.sqrt(norm_sq) or 1.0)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        if not self.inverted:
            return []
        weights, qnorm = self._encode(query)
        scores: Dict[int, float] = {}
        for feat, weight in weights.items():
            for idx, dweight in self.inverted.get(feat, ()):  # type: ignore[arg-type]
                scores[idx] = scores.get(idx, 0.0) + weight * dweight
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(idx, score / qnorm) for idx, score in ranked]
# ── dense backends ────────────────────────────────────────────────────────
def _hash(token: str, dim: int) -> int:
    digest = hashlib.md5(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") % dim


class HashingTfidfEmbedder:
    """Deterministic hashed TF-IDF embedding (dense, fixed width)."""

    name = "hashing-tfidf"

    def __init__(self, dim: int = DEFAULT_DIM) -> None:
        self.dim = dim
        self.idf: Dict[str, float] = {}
        self.num_docs = 0

    def fit(self, documents: Sequence[str]) -> "HashingTfidfEmbedder":
        df: Dict[str, int] = {}
        self.num_docs = len(documents)
        for text in documents:
            for feat in set(_features(text)):
                df[feat] = df.get(feat, 0) + 1
        self.idf = {f: math.log((1 + self.num_docs) / (1 + c)) + 1.0 for f, c in df.items()}
        return self

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        counts: Dict[str, int] = {}
        for feat in _features(text):
            counts[feat] = counts.get(feat, 0) + 1
        for feat, tf in counts.items():
            vec[_hash(feat, self.dim)] += (1.0 + math.log(tf)) * self.idf.get(feat, 1.0)
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec

    def embed_batch(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), np.float32)
        return np.vstack([self.embed(t) for t in texts])


class SentenceTransformerEmbedder:
    """Real neural embeddings via sentence-transformers (optional dependency)."""

    name = "sentence-transformers"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2",
                 device: Optional[str] = None) -> None:
        from sentence_transformers import SentenceTransformer  # lazy import

        self.model_name = model_name
        self.dim = 0
        self._model = SentenceTransformer(model_name, device=device)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def fit(self, documents: Sequence[str]) -> "SentenceTransformerEmbedder":
        return self

    def embed(self, text: str) -> np.ndarray:
        vec = np.asarray(self._model.encode([text], normalize_embeddings=True)[0],
                         dtype=np.float32)
        return vec

    def embed_batch(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), np.float32)
        return np.asarray(self._model.encode(list(texts), normalize_embeddings=True,
                                            show_progress_bar=False), dtype=np.float32)


class VectorIndex:
    """Cosine-similarity index over densely embedded chunks."""

    def __init__(self, embedder=None) -> None:
        self.embedder = embedder or HashingTfidfEmbedder()
        self.matrix: np.ndarray = np.zeros((0, self.embedder.dim), dtype=np.float32)

    @property
    def name(self) -> str:
        return self.embedder.name

    def build(self, documents: Sequence[str]) -> "VectorIndex":
        if hasattr(self.embedder, "fit"):
            self.embedder.fit(documents)
        self.matrix = self.embedder.embed_batch(documents)
        return self

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        if self.matrix.size == 0:
            return []
        qvec = self.embedder.embed(query)
        scores = self.matrix @ qvec
        k = min(top_k, scores.shape[0])
        if k <= 0:
            return []
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [(int(i), float(scores[i])) for i in idx]


def create_vector_backend(prefer_neural: bool = True, dim: int = DEFAULT_DIM,
                          device: Optional[str] = None, verbose: bool = False):
    """Pick the best available vector backend.

    Neural embeddings are used when ``sentence-transformers`` is installed and
    ``prefer_neural`` is set; otherwise the exact sparse TF-IDF index is used,
    which needs no model and no API key.
    """
    if prefer_neural:
        try:
            embedder = SentenceTransformerEmbedder(device=device)
            if verbose:
                print(f"[retrieval] using neural embeddings: {embedder.model_name}")
            return VectorIndex(embedder), "sentence-transformers"
        except Exception as exc:  # ImportError, missing weights, no network, ...
            if verbose:
                print(f"[retrieval] neural embeddings unavailable ({exc.__class__.__name__}); "
                      "falling back to sparse TF-IDF")
    if verbose:
        print("[retrieval] using exact sparse TF-IDF vectors")
    return SparseTfidfIndex(), "tfidf-sparse"