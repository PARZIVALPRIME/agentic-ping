"""Lexical retrieval: Okapi BM25 over corpus chunks (dependency-free)."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple

from kg.textutil import tokens

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "can", "did", "do", "does", "doing", "down",
    "during", "each", "few", "for", "from", "further", "had", "has", "have",
    "having", "he", "her", "here", "hers", "herself", "him", "himself", "his",
    "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me",
    "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on",
    "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over",
    "own", "s", "same", "she", "should", "so", "some", "such", "t", "than",
    "that", "the", "their", "theirs", "them", "themselves", "then", "there",
    "these", "they", "this", "those", "through", "to", "too", "under", "until",
    "up", "very", "was", "we", "were", "what", "when", "where", "which", "while",
    "who", "whom", "why", "will", "with", "you", "your", "yours", "yourself",
    "yourselves",
}


def term_list(text: str) -> List[str]:
    """Tokenize and drop stopwords, keeping numbers (they are discriminative)."""
    return [t for t in tokens(text) if t not in STOPWORDS]


@dataclass
class BM25Index:
    """Ranked retrieval over an arbitrary sequence of documents."""

    k1: float = 1.5
    b: float = 0.75
    postings: Dict[str, List[Tuple[int, int]]] = field(default_factory=dict)
    doc_len: List[int] = field(default_factory=list)
    avg_len: float = 0.0
    num_docs: int = 0

    def build(self, documents: Sequence[str]) -> "BM25Index":
        self.postings = {}
        self.doc_len = []
        self.num_docs = len(documents)
        for idx, text in enumerate(documents):
            terms = term_list(text)
            self.doc_len.append(len(terms))
            for term, tf in Counter(terms).items():
                self.postings.setdefault(term, []).append((idx, tf))
        self.avg_len = (sum(self.doc_len) / self.num_docs) if self.num_docs else 0.0
        return self

    def _idf(self, term: str) -> float:
        df = len(self.postings.get(term, ()))
        if df == 0:
            return 0.0
        return math.log(1.0 + (self.num_docs - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Return (doc_index, score) pairs sorted by descending BM25 score."""
        scores: Dict[int, float] = {}
        for term in term_list(query):
            idf = self._idf(term)
            if idf <= 0.0:
                continue
            for idx, tf in self.postings.get(term, ()):
                dl = self.doc_len[idx] or 1
                denom = tf + self.k1 * (1 - self.b + self.b * dl / (self.avg_len or 1))
                scores[idx] = scores.get(idx, 0.0) + idf * (tf * (self.k1 + 1)) / denom
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]