"""Question understanding: turn a natural-language question into a QuerySpec.

The parser is *deterministic and corpus-aware*: it uses the sport vocabulary of
the knowledge graph to resolve which sport a question is about, and regex
templates to recover the structured slot values (games year, season, threshold,
venue, date phrase, event descriptor). An LLM may later refine the result, but
the system is fully functional without one.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from kg.model import KnowledgeGraph
from kg.textutil import normalize

# ── question type detection ────────────────────────────────────────────────
AGG_RE = re.compile(
    r"how many\s+(?P<sport>.+?)\s+events?\s+at\s+the\s+(?P<year>\d{4})\s+"
    r"(?P<season>Summer|Winter)\s+Olympics\s+had\s+(?P<cmp>more|fewer|less|at least|at most)\s+than\s+"
    r"(?P<n>\d+)\s+competitors",
    re.I,
)
SUP_RE = re.compile(
    r"which\s+(?P<sport>.+?)\s+events?\s+at\s+the\s+(?P<year>\d{4})\s+"
    r"(?P<season>Summer|Winter)\s+Olympics\s+had\s+the\s+"
    r"(?P<dir>highest|lowest|most|fewest|largest|smallest)\s+number of competitors",
    re.I,
)
TEMPORAL_RE = re.compile(
    r"won the gold medal in the\s+(?P<desc>.+?)\s+at\s+the\s+(?P<season>Summer|Winter)\s+"
    r"Olympics\s+held\s+immediately\s+before\s+(?P<year>\d{4})",
    re.I,
)
MULTIHOP_RE = re.compile(
    r"event\s+held\s+at\s+(?P<venue>.+?)\s+on\s+(?P<date>.+?)"
    r"(?:\s+at\s+the\s+(?P<year>\d{4})\s+(?P<season>Summer|Winter)\s+Olympics)?\s*[?.]?\s*$",
    re.I,
)
LOOKUP_RE = re.compile(r"how many nations competed in\s+(?P<title>.+?)\s*\?*\s*$", re.I)

SUPERLATIVE_DIR = {
    "highest": "max", "most": "max", "largest": "max",
    "lowest": "min", "fewest": "min", "smallest": "min",
}
AGG_DIR = {
    "more": "gt", "at least": "gte", "at most": "lte", "fewer": "lt", "less": "lt",
}


@dataclass
class QuerySpec:
    question: str
    qtype: str = "unknown"
    sport: str = ""
    year: int = 0
    season: str = ""
    event_desc: str = ""
    venue: str = ""
    date_text: str = ""
    threshold: Optional[int] = None
    comparator: str = "gt"
    direction: str = "max"
    target_title: str = ""
    before_year: Optional[int] = None
    years_mentioned: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def detect_sport(text: str, vocab: List[str]) -> str:
    """Longest known sport name mentioned in ``text`` (corpus vocabulary)."""
    norm = f" {normalize(text)} "
    for sport in vocab:  # vocab is sorted longest-first
        ns = normalize(sport)
        if ns and f" {ns} " in norm:
            return sport
    return ""


def classify(question: str, kg: Optional[KnowledgeGraph] = None) -> str:
    """Rule-based question-type classifier aligned with the corpus taxonomy."""
    q = question.lower()
    if "how many nations competed in" in q:
        return "lookup"
    if "how many" in q and "competitors" in q:
        return "aggregation"
    if ("highest" in q or "lowest" in q or "most" in q or "fewest" in q) and "competitors" in q:
        return "superlative"
    if "immediately before" in q or "immediately after" in q:
        return "temporal"
    if "held at" in q:
        return "multi_hop"
    if "how many" in q or "number of" in q:
        return "aggregation"
    if any(w in q for w in ("most", "highest", "largest", "fastest", "record")):
        return "superlative"
    if any(w in q for w in ("before", "after", "when")):
        return "temporal"
    return "lookup"


def parse_question(question: str, kg: Optional[KnowledgeGraph] = None) -> QuerySpec:
    """Parse ``question`` into a :class:`QuerySpec`."""
    vocab = kg.sport_vocabulary if kg is not None else []
    qtype = classify(question, kg)
    spec = QuerySpec(question=question, qtype=qtype)
    spec.years_mentioned = [int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", question)]

    if qtype == "aggregation":
        m = AGG_RE.search(question)
        if m:
            spec.sport = detect_sport(m.group("sport"), vocab) or m.group("sport").strip()
            spec.year, spec.season = int(m.group("year")), m.group("season").title()
            spec.threshold = int(m.group("n"))
            spec.comparator = AGG_DIR.get(m.group("cmp").lower(), "gt")
        return spec

    if qtype == "superlative":
        m = SUP_RE.search(question)
        if m:
            spec.sport = detect_sport(m.group("sport"), vocab) or m.group("sport").strip()
            spec.year, spec.season = int(m.group("year")), m.group("season").title()
            spec.direction = SUPERLATIVE_DIR.get(m.group("dir").lower(), "max")
        return spec

    if qtype == "temporal":
        m = TEMPORAL_RE.search(question)
        if m:
            raw_desc = m.group("desc").strip()
            spec.season = m.group("season").title()
            spec.before_year = int(m.group("year"))
            sport = detect_sport(raw_desc, vocab)
            desc = raw_desc
            if sport:
                nd, ns = normalize(raw_desc), normalize(sport)
                if nd.endswith(ns):
                    desc = nd[: -len(ns)].strip()
                else:
                    desc = nd.replace(ns, " ").strip()
                spec.event_desc = desc
            else:
                spec.event_desc = normalize(raw_desc)
            spec.sport = sport
            # drop the trailing generic noun ("... event") so the descriptor
            # matches the corpus event name exactly
            spec.event_desc = re.sub(r"\s+(events?|competitions?|races?)$", "",
                                     spec.event_desc).strip()
        return spec

    if qtype == "multi_hop":
        m = MULTIHOP_RE.search(question)
        if m:
            spec.venue = m.group("venue").strip()
            spec.date_text = m.group("date").strip()
            if m.group("year"):
                spec.year, spec.season = int(m.group("year")), m.group("season").title()
        return spec

    # lookup
    m = LOOKUP_RE.search(question)
    if m:
        spec.target_title = m.group("title").strip().rstrip("?")
    return spec