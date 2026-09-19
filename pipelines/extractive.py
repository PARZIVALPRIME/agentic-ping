"""Extractive answering from retrieved passages.

RAG and GraphRAG share this module so their comparison is purely about
*retrieval*: both read answers out of the passages they were given, using the
same field-parsing logic and the same fuzzy matching. The module deliberately
has **no access to the knowledge graph**, which is exactly why the two
retrieval-only pipelines cannot answer counterfactual/full-corpus questions
("how many ... had more than N competitors") when the required documents were
never retrieved.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from kg.textutil import best_match, normalize, similarity
from reasoning.query_parser import QuerySpec, parse_question

_FIELD_RE = re.compile(r"([a-z_]+):\s*(.*?)(?:\s\|\s|$)")


@dataclass
class ExtractResult:
    answer: str = ""
    confidence: float = 0.0
    citations: List[str] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    docs_seen: int = 0
    fields_parsed: int = 0
    unresolved: List[str] = field(default_factory=list)
    method: str = "extractive"

    @property
    def resolved(self) -> bool:
        return bool(self.answer)


def parse_chunk_fields(text: str) -> Dict[str, str]:
    """Recover infobox fields from a rendered infobox chunk."""
    if not text.startswith("Infobox of "):
        return {}
    _, _, body = text.partition("->")
    return {m.group(1): m.group(2).strip() for m in _FIELD_RE.finditer(body)}


def parse_chunk_title(text: str) -> str:
    if text.startswith("Infobox of "):
        return text[len("Infobox of "):].split("->")[0].strip()
    return ""


def to_int(value: str) -> Optional[int]:
    if not value:
        return None
    m = re.match(r"^\s*(\d+)", str(value))
    return int(m.group(1)) if m else None


def dedupe_by_doc(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One entry per document, preferring the chunk that carries its infobox.

    Retrieval ranks prose passages above the structured header for some queries,
    but the *fields* (competitors, nations, venue, gold) only exist in the
    infobox chunk. Keeping the infobox chunk for every retrieved document is
    what makes "read the answer out of the passages" work at all.
    """
    order: List[str] = []
    best: Dict[str, Dict[str, Any]] = {}
    for chunk in chunks:
        doc_id = chunk.get("doc_id", "")
        if doc_id not in best:
            best[doc_id] = chunk
            order.append(doc_id)
            continue
        if not parse_chunk_fields(best[doc_id].get("text", "")):
            if parse_chunk_fields(chunk.get("text", "")):
                best[doc_id] = chunk
    return [best[d] for d in order]


def _title_year(title: str) -> Optional[int]:
    m = re.search(r"\b(19\d{2}|20\d{2})\b", title or "")
    return int(m.group(1)) if m else None


def _title_matches_games(title: str, spec: QuerySpec) -> bool:
    """Does a retrieved title correspond to (sport, year, season)?"""
    if spec.year and str(spec.year) not in title:
        return False
    if spec.season and spec.season.lower() not in title.lower():
        return False
    if spec.sport:
        ns = normalize(spec.sport)
        nt = normalize(title)
        if ns not in nt and not any(tok in nt for tok in ns.split() if len(tok) > 3):
            return False
class ExtractiveAnswerer:
    """Answer questions by reading fields out of retrieved passages."""

    def answer(self, question: str, chunks: List[Dict[str, Any]],
               spec: Optional[QuerySpec] = None) -> ExtractResult:
        spec = spec or parse_question(question, None)
        docs = dedupe_by_doc(chunks)
        parsed = []
        for chunk in docs:
            fields = parse_chunk_fields(chunk.get("text", ""))
            title = parse_chunk_title(chunk.get("text", "")) or chunk.get("title", "")
            if fields:
                parsed.append((chunk, title, fields))
        result = ExtractResult(docs_seen=len(docs), fields_parsed=len(parsed))

        if spec.qtype == "lookup":
            return self._lookup(spec, parsed, result)
        if spec.qtype == "aggregation":
            return self._aggregation(spec, parsed, result)
        if spec.qtype == "superlative":
            return self._superlative(spec, parsed, result)
        if spec.qtype == "temporal":
            return self._temporal(spec, parsed, result)
        return self._multi_hop(spec, parsed, result)

    # ── per-type strategies ────────────────────────────────────────────
    def _lookup(self, spec: QuerySpec, parsed, res: ExtractResult) -> ExtractResult:
        if not parsed:
            res.unresolved.append("no_infobox_chunks_retrieved")
            return res
        titles = [t for _c, t, _f in parsed]
        match, score = best_match(spec.target_title, titles)
        if match is None or score < 0.8:
            res.unresolved.append("target_title_not_retrieved")
            return res
        for chunk, title, fields in parsed:
            if title == match:
                res.answer = fields.get("nations", "") or ""
                res.citations = [chunk["doc_id"]]
                res.confidence = round(0.55 + 0.4 * score, 3)
                res.evidence = [{"doc_id": chunk["doc_id"], "title": title, "fields": fields}]
                if not res.answer:
                    res.unresolved.append("nations_missing_in_retrieved_docs")
                return res
        res.unresolved.append("target_title_not_retrieved")
        return res

    def _aggregation(self, spec: QuerySpec, parsed, res: ExtractResult) -> ExtractResult:
        kept, total = 0, 0
        for chunk, title, fields in parsed:
            if not _title_matches_games(title, spec):
                continue
            value = to_int(fields.get("competitors", ""))
            if value is None:
                continue
            total += 1
            if value > (spec.threshold or 0):
                kept += 1
        if total == 0:
            res.unresolved.append("no_candidate_docs_retrieved")
            return res
        res.answer = str(kept)
        res.citations = [c["doc_id"] for c, _t, _f in parsed]
        # A document-level count is only trustworthy if the whole candidate set
        # was retrieved; without a graph there is no way to know. Confidence is
        # therefore capped, and the gap is recorded explicitly.
        res.confidence = round(min(0.7, 0.25 + 0.45 * min(1.0, total / 12)), 3)
        res.evidence = [{"doc_id": c["doc_id"], "title": t, "competitors": f.get("competitors")}
                        for c, t, f in parsed]
        res.unresolved.append("cannot_verify_candidate_set_completeness")
        return res
    def _superlative(self, spec: QuerySpec, parsed, res: ExtractResult) -> ExtractResult:
        best: Optional[Tuple[int, str, str]] = None
        for chunk, title, fields in parsed:
            if not _title_matches_games(title, spec):
                continue
            value = to_int(fields.get("competitors", ""))
            if value is None:
                continue
            if best is None or value > best[0]:
                best = (value, title, chunk["doc_id"])
        if best is None:
            res.unresolved.append("no_candidate_docs_retrieved")
            return res
        res.answer = best[1]
        res.citations = [best[2]]
        res.confidence = 0.45
        res.evidence = [{"doc_id": best[2], "title": best[1], "competitors": best[0]}]
        res.unresolved.append("candidate_set_may_be_incomplete")
        return res

    def _temporal(self, spec: QuerySpec, parsed, res: ExtractResult) -> ExtractResult:
        candidates = []
        for chunk, title, fields in parsed:
            year = _title_year(title)
            if year is None or year >= (spec.before_year or 10 ** 9):
                continue
            if spec.season and spec.season.lower() not in title.lower():
                continue
            score = similarity(spec.event_desc, fields.get("event", "") or title)
            candidates.append((score, year, title, chunk["doc_id"], fields))
        if not candidates:
            res.unresolved.append("no_prior_edition_retrieved")
            return res
        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        score, year, title, doc_id, fields = candidates[0]
        res.answer = fields.get("gold", "")
        res.citations = [doc_id]
        res.confidence = round(min(0.8, 0.3 + 0.5 * score), 3)
        res.evidence = [{"doc_id": doc_id, "title": title, "gold": fields.get("gold")}]
        if not res.answer:
            res.unresolved.append("gold_missing_in_retrieved_docs")
        return res

    def _multi_hop(self, spec: QuerySpec, parsed, res: ExtractResult) -> ExtractResult:
        candidates = []
        for chunk, title, fields in parsed:
            venue = fields.get("venue", "") or fields.get("venues", "")
            date = _date_of(fields)
            venue_score = similarity(spec.venue, venue) if spec.venue else 0.0
            date_score = similarity(spec.date_text, date) if spec.date_text else 0.0
            if spec.year:
                year = _title_year(title)
                if year is not None:
                    date_score = max(date_score, 0.8 if year == spec.year else 0.0)
            candidates.append((venue_score + date_score, title, chunk["doc_id"], fields,
                               venue, date))
        if not candidates:
            res.unresolved.append("no_infobox_chunks_retrieved")
            return res
        candidates.sort(key=lambda x: x[0], reverse=True)
        score, title, doc_id, fields, venue, date = candidates[0]
        if score < 0.5:
            res.unresolved.append("no_venue_date_match_in_retrieved_docs")
        res.answer = fields.get("gold", "")
        res.citations = [doc_id]
        res.confidence = round(min(0.85, 0.25 + 0.45 * score), 3)
        res.evidence = [{"doc_id": doc_id, "title": title, "venue": venue, "date": date,
                         "gold": fields.get("gold")}]
        if not res.answer:
            res.unresolved.append("gold_missing_in_retrieved_docs")
        return res


def _title_year(title: str) -> Optional[int]:
    m = re.search(r"\b(19\d{2}|20\d{2})\b", title or "")
    return int(m.group(1)) if m else None


def _title_matches_games(title: str, spec: QuerySpec) -> bool:
    """Does a retrieved title correspond to (sport, year, season)?"""
    if spec.year and str(spec.year) not in title:
        return False
    if spec.season and spec.season.lower() not in title.lower():
        return False
    if spec.sport:
        ns = normalize(spec.sport)
        nt = normalize(title)
        if ns not in nt and not any(tok in nt for tok in ns.split() if len(tok) > 3):
            return False
    return True


def _date_of(fields: Dict[str, str]) -> str:
    return f"{fields.get('date', '')} {fields.get('dates', '')}".strip()
