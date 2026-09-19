"""Unified corpus index: chunking + lexical/vector/graph retrieval.

This is the shared substrate all three pipelines retrieve from, which keeps the
comparison fair: RAG uses only ``vector_search``, GraphRAG adds
``graph_search``, and the agentic pipeline may call any of them, repeatedly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from kg.builder import parse_infobox
from kg.model import KnowledgeGraph
from kg.textutil import normalize

from .lexical import BM25Index
from .vector import SparseTfidfIndex, VectorIndex, create_vector_backend

MAX_CHUNK_WORDS = 130
CHUNK_OVERLAP_WORDS = 30
MIN_CHUNK_WORDS = 12


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    index: int

    def to_dict(self) -> Dict[str, Any]:
        return {"chunk_id": self.chunk_id, "doc_id": self.doc_id,
                "title": self.title, "index": self.index, "text": self.text[:400]}


@dataclass
class Doc:
    doc_id: str
    title: str
    url: str = ""
    text: str = ""
    approx_tokens: int = 0
    infobox: Dict[str, str] = field(default_factory=dict)


def _split_passages(text: str) -> List[str]:
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def _window_passage(passage: str) -> List[str]:
    words = passage.split()
    if len(words) <= MAX_CHUNK_WORDS:
        return [passage]
    out, start = [], 0
    while start < len(words):
        out.append(" ".join(words[start:start + MAX_CHUNK_WORDS]))
        if start + MAX_CHUNK_WORDS >= len(words):
            break
        start += MAX_CHUNK_WORDS - CHUNK_OVERLAP_WORDS
    return out


def infobox_summary(doc: "Doc") -> str:
    """Render a document in the canonical infobox-chunk format.

    Graph-expanded evidence must use exactly the same rendering as the indexed
    infobox chunks so that the shared answer extractor can read it identically.
    """
    if not doc.infobox:
        return doc.text[:800]
    fields = " | ".join(f"{k}: {v}" for k, v in doc.infobox.items())
    return f"Infobox of {doc.title} -> {fields}"


class CorpusIndex:
    """Chunk store with BM25, hashed-TF-IDF vectors and graph lookup."""

    def __init__(self, kg: Optional[KnowledgeGraph] = None,
                 vector_backend: str = "auto", verbose: bool = False) -> None:
        self.kg = kg
        self.docs: Dict[str, Doc] = {}
        self.chunks: List[Chunk] = []
        self.bm25 = BM25Index()
        self.vectors = None
        self.vector_backend_name = "unbuilt"
        self._vector_backend_pref = vector_backend
        self._verbose = verbose
        self._built = False

    # ── construction ────────────────────────────────────────────────────
    def load(self, corpus_path: str, limit: Optional[int] = None) -> "CorpusIndex":
        with open(corpus_path, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if limit is not None and i >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                doc_id = raw.get("doc_id") or f"doc_{i}"
                text = raw.get("text") or ""
                self.docs[doc_id] = Doc(
                    doc_id=doc_id,
                    title=raw.get("title") or "",
                    url=raw.get("url") or "",
                    text=text,
                    approx_tokens=int(raw.get("approx_tokens") or 0),
                    infobox=parse_infobox(text),
                )
        return self

    def build(self) -> "CorpusIndex":
        """Chunk every document and build the lexical + vector indexes."""
        self.chunks = []
        for doc in self.docs.values():
            for i, chunk in enumerate(self._chunk_doc(doc)):
                self.chunks.append(Chunk(
                    chunk_id=f"{doc.doc_id}::{i}",
                    doc_id=doc.doc_id,
                    title=doc.title,
                    text=chunk,
                    index=i,
                ))
        texts = [f"{c.title}. {c.text}" for c in self.chunks]
        self.bm25.build(texts)
        if self.vectors is None:
            backend = None
            if self._vector_backend_pref in ("auto", "neural"):
                backend, self.vector_backend_name = create_vector_backend(
                    prefer_neural=True, verbose=self._verbose)
            elif self._vector_backend_pref == "sparse-tfidf":
                backend, self.vector_backend_name = SparseTfidfIndex(), "tfidf-sparse"
            elif self._vector_backend_pref == "hashing-tfidf":
                backend, self.vector_backend_name = VectorIndex(), "hashing-tfidf"
            else:
                backend, self.vector_backend_name = SparseTfidfIndex(), "tfidf-sparse"
            self.vectors = backend
        self.vectors.build(texts)
        if not self.vector_backend_name or self.vector_backend_name == "unbuilt":
            self.vector_backend_name = getattr(self.vectors, "name", "unknown")
        self._built = True
        return self

    @staticmethod
    def _chunk_doc(doc: Doc) -> List[str]:
        chunks: List[str] = []
        if doc.infobox:
            header = " | ".join(f"{k}: {v}" for k, v in doc.infobox.items())
            chunks.append(f"Infobox of {doc.title} -> {header}")
        for passage in _split_passages(doc.text):
            if passage.startswith("[Infobox"):
                continue
            for window in _window_passage(passage):
                if len(window.split()) >= MIN_CHUNK_WORDS:
                    chunks.append(window)
        if not chunks:
            chunks.append(f"{doc.title}. {doc.text[:600]}")
        return chunks
# ── retrieval ──────────────────────────────────────────────────────
    def vector_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        if self.vectors is None:
            return []
        return self._pack(self.vectors.search(query, top_k), "vector")

    def lexical_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        return self._pack(self.bm25.search(query, top_k), "lexical")

    def hybrid_search(self, query: str, top_k: int = 10,
                      alpha: float = 0.5) -> List[Dict[str, Any]]:
        """Reciprocal-rank fusion of lexical and vector candidates."""
        pool = max(top_k * 3, 20)
        lex = self.bm25.search(query, pool)
        vec = self.vectors.search(query, pool)
        fused: Dict[int, float] = {}
        for rank, (idx, _score) in enumerate(lex):
            fused[idx] = fused.get(idx, 0.0) + (1 - alpha) / (60 + rank + 1)
        for rank, (idx, _score) in enumerate(vec):
            fused[idx] = fused.get(idx, 0.0) + alpha / (60 + rank + 1)
        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return self._pack(list(ranked), "hybrid")

    def _pack(self, hits: Sequence[Tuple[int, float]], method: str) -> List[Dict[str, Any]]:
        out = []
        for idx, score in hits:
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]
            out.append({
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "title": chunk.title,
                "text": chunk.text,
                "score": round(float(score), 5),
                "method": method,
            })
        return out

    # ── graph-aware retrieval ───────────────────────────────────────────
    def graph_search(self, query: str, top_k: int = 10,
                     num_hops: int = 2) -> List[Dict[str, Any]]:
        """Seed with hybrid retrieval, then expand along knowledge-graph edges.

        A passage is worth more when the graph says it is structurally related
        to a seed passage: sibling editions (PREV/NEXT), same venue, same
        sport+games. This is the GraphRAG retrieval primitive.
        """
        if self.kg is None:
            return self.hybrid_search(query, top_k)
        seeds = self.hybrid_search(query, max(3, self.top_k_seeds(top_k)))
        # Normalise seed scores to [0, 1] by rank so that graph-structural
        # bonuses are comparable to (and cannot silently outrank) the seeds.
        n = max(1, len(seeds))
        expanded: Dict[str, Dict[str, Any]] = {
            s["doc_id"]: dict(s, hop=0, score=1.0 - (i / n))
            for i, s in enumerate(seeds)
        }

        frontier = list(expanded.keys())
        for hop in range(1, num_hops + 1):
            next_frontier: List[str] = []
            for doc_id in frontier:
                parent_score = float(expanded[doc_id].get("score", 0.0))
                for neighbour_id, relation in self._neighbours(doc_id):
                    inherited = parent_score * 0.6
                    existing = expanded.get(neighbour_id)
                    if existing is None:
                        expanded[neighbour_id] = {
                            "doc_id": neighbour_id, "title": neighbour_id, "text": "",
                            "score": inherited, "method": f"graph:{relation}", "hop": hop,
                        }
                        next_frontier.append(neighbour_id)
                    elif existing.get("hop", 0) > 0 and inherited > existing.get("score", 0.0):
                        existing["score"] = inherited
                        existing["method"] = f"graph:{relation}"
            frontier = next_frontier
            if not frontier:
                break

        results: List[Dict[str, Any]] = []
        for doc_id, item in expanded.items():
            doc = self.docs.get(doc_id)
            if doc is None:
                continue
            node = self.kg.events.get(doc_id)
            results.append({
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}::0",
                "title": doc.title,
                "text": infobox_summary(doc) if node else doc.text[:500],
                "score": round(float(item.get("score", 0.0)), 5),
                "method": item.get("method", "hybrid"),
                "hop": item.get("hop", 0),
            })
        results.sort(key=lambda r: (-r["score"], r.get("hop", 0)))
        return results[:top_k]

    @staticmethod
    def top_k_seeds(top_k: int) -> int:
        return max(3, top_k // 2)

    def _neighbours(self, doc_id: str) -> List[Tuple[str, str]]:
        """Structurally related documents, with the relation label."""
        if self.kg is None:
            return []
        out: List[Tuple[str, str]] = []
        node = self.kg.events.get(doc_id)
        if node is None:
            return out
        if node.prev_doc_id:
            out.append((node.prev_doc_id, "PREV_EDITION"))
        if node.next_doc_id:
            out.append((node.next_doc_id, "NEXT_EDITION"))
        for other in self.kg.events_for(node.sport, node.year, node.season):
            if other.doc_id != doc_id:
                out.append((other.doc_id, "SAME_SPORT_GAMES"))
        if node.venue:
            for other in self.kg.events_at_venue(node.venue)[:6]:
                if other.doc_id != doc_id:
                    out.append((other.doc_id, "SAME_VENUE"))
        return out[:24]

    # ── helpers ────────────────────────────────────────────────────────
    def get_doc(self, doc_id: str) -> Optional[Doc]:
        return self.docs.get(doc_id)

    def resolve_doc_id(self, title: str) -> Optional[str]:
        if self.kg is not None:
            hit = self.kg.title_index.get(normalize(title))
            if hit:
                return hit
        target = normalize(title)
        for doc in self.docs.values():
            if normalize(doc.title) == target:
                return doc.doc_id
        return None

    @property
    def num_chunks(self) -> int:
        return len(self.chunks)

    @property
    def num_docs(self) -> int:
        return len(self.docs)

    def stats(self) -> Dict[str, Any]:
        return {
            "num_docs": self.num_docs,
            "num_chunks": self.num_chunks,
            "embedding_backend": self.vector_backend_name,
            "embedding_dim": getattr(self.vectors, "dim", 0),
        }


def load_index(corpus_path: str, kg: Optional[KnowledgeGraph] = None,
               vector_backend: str = "auto", verbose: bool = False) -> CorpusIndex:
    """Load the corpus and build all indexes."""
    index = CorpusIndex(kg=kg, vector_backend=vector_backend, verbose=verbose)
    index.load(corpus_path)
    index.build()
    return index
