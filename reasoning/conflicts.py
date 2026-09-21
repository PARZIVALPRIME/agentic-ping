"""Round 2 - reasoning over evolving, conflicting and uncertain facts.

Round 2 asks a system to detect conflicting versions of a fact, decide what
supersedes what, judge which source is more authoritative, and express the
residual uncertainty. This corpus already contains all four, because it spans
1987-2023 and the world changed underneath it:

* **Superseded records** - an Olympic record set in 1992 is not the record in
  2012; both documents state "Olympic record" in the present tense.
* **Dissolved and renamed nations** - the Soviet Union, Yugoslavia,
  Czechoslovakia and the two Germanys all appear as medallist nations, and
  their successor states appear later.
* **Renamed venues** - the same building carries different names across
  editions.
* **Reallocated medals** - doping disqualifications rewrite results years
  afterwards; the later document is authoritative.

The resolver below is deliberately *rule-based and explainable*: each decision
names the rule that produced it and the evidence it rested on, so a conflict
appears in the trace as a reasoned adjudication rather than a silent pick.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Predecessor -> successor. The successor supersedes when both appear and no
# date separates them. Keys are matched on a normalised (lowercased) form.
SUCCESSION: Dict[str, str] = {
    "soviet union": "Russia",
    "ussr": "Russia",
    "unified team": "Russia",
    "yugoslavia": "Serbia",
    "serbia and montenegro": "Serbia",
    "fr yugoslavia": "Serbia",
    "czechoslovakia": "Czech Republic",
    "east germany": "Germany",
    "west germany": "Germany",
    "gdr": "Germany",
    "frg": "Germany",
    "burma": "Myanmar",
    "zaire": "Democratic Republic of the Congo",
    "ceylon": "Sri Lanka",
    "rhodesia": "Zimbabwe",
    "netherlands antilles": "Netherlands",
    "chinese taipei": "Chinese Taipei",
}

# Phrases that mark a document as a correction of an earlier one. A source
# carrying one of these outranks a plain statement of the same fact.
AUTHORITY_MARKERS = (
    "disqualified", "stripped", "reallocated", "upgraded", "annulled",
    "doping", "corrected", "revised", "superseded",
)

# Ordered most-authoritative first; used only as a tie-break.
SOURCE_RANK = ("official_report", "result_page", "infobox", "prose", "unknown")


@dataclass
class Candidate:
    """One attested version of a fact."""

    value: Any
    doc_id: str = ""
    year: Optional[int] = None
    source_type: str = "unknown"
    text: str = ""

    def normalised(self) -> str:
        return str(self.value).strip().lower()

    def authority_markers(self) -> List[str]:
        blob = f"{self.text} {self.value}".lower()
        return [m for m in AUTHORITY_MARKERS if m in blob]

    def rank(self) -> int:
        try:
            return SOURCE_RANK.index(self.source_type)
        except ValueError:
            return len(SOURCE_RANK)


@dataclass
class Resolution:
    """The adjudicated value plus the reasoning that produced it."""

    field_name: str
    resolved: Any = None
    rule: str = "no_conflict"
    confidence: float = 1.0
    explanation: str = ""
    superseded: List[Dict[str, Any]] = field(default_factory=list)
    candidates: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def had_conflict(self) -> bool:
        return self.rule != "no_conflict"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field": self.field_name,
            "resolved": self.resolved,
            "rule": self.rule,
            "confidence": round(self.confidence, 3),
            "explanation": self.explanation,
            "had_conflict": self.had_conflict,
            "superseded": self.superseded,
            "candidates": self.candidates,
        }


def _as_candidates(raw: List[Any]) -> List[Candidate]:
    out: List[Candidate] = []
    for item in raw:
        if isinstance(item, Candidate):
            out.append(item)
        elif isinstance(item, dict):
            out.append(Candidate(
                value=item.get("value"),
                doc_id=str(item.get("doc_id", "")),
                year=item.get("year") if isinstance(item.get("year"), int) else None,
                source_type=str(item.get("source_type", "unknown")),
                text=str(item.get("text", "")),
            ))
        else:
            out.append(Candidate(value=item))
    return [c for c in out if c.value not in (None, "")]


def resolve(field_name: str, raw_candidates: List[Any]) -> Resolution:
    """Adjudicate competing versions of one fact.

    Precedence, highest first:

    1. **authority** - a source that explicitly corrects the record
       (disqualification, reallocation) beats a plain statement.
    2. **recency**   - a later document supersedes an earlier one.
    3. **succession**- a successor state supersedes its predecessor.
    4. **majority**  - otherwise the most-attested value wins, and the
       confidence reflects how contested it was.
    """
    cands = _as_candidates(raw_candidates)
    res = Resolution(field_name=field_name,
                     candidates=[{"value": c.value, "doc_id": c.doc_id,
                                  "year": c.year, "source_type": c.source_type}
                                 for c in cands])
    if not cands:
        res.rule = "no_evidence"
        res.confidence = 0.0
        res.explanation = "no candidate values were supplied"
        return res

    distinct = {c.normalised() for c in cands}
    if len(distinct) == 1:
        res.resolved = cands[0].value
        res.confidence = 1.0
        res.explanation = f"all {len(cands)} source(s) agree"
        return res

    # 1. explicit corrections outrank everything else
    corrected = [c for c in cands if c.authority_markers()]
    if corrected:
        winner = max(corrected, key=lambda c: (c.year or 0, -c.rank()))
        res.resolved = winner.value
        res.rule = "authority_correction"
        res.confidence = 0.95
        markers = ", ".join(winner.authority_markers())
        res.explanation = (f"{winner.doc_id or 'source'} corrects the record "
                           f"({markers}); a correction outranks a plain statement")
        res.superseded = [{"value": c.value, "doc_id": c.doc_id, "year": c.year}
                          for c in cands if c is not winner]
        return res

    # 2. recency
    dated = [c for c in cands if c.year is not None]
    if len({c.year for c in dated}) > 1:
        winner = max(dated, key=lambda c: c.year or 0)
        older = [c for c in cands if c is not winner]
        res.resolved = winner.value
        res.rule = "recency"
        res.confidence = 0.9
        res.explanation = (f"{winner.doc_id or 'source'} ({winner.year}) is the "
                           f"most recent statement and supersedes "
                           f"{len(older)} earlier version(s)")
        res.superseded = [{"value": c.value, "doc_id": c.doc_id, "year": c.year}
                          for c in older]
        return res

    # 3. entity succession
    for c in cands:
        successor = SUCCESSION.get(c.normalised())
        if successor and successor.strip().lower() in distinct:
            winner = next(x for x in cands
                          if x.normalised() == successor.strip().lower())
            res.resolved = winner.value
            res.rule = "entity_succession"
            res.confidence = 0.85
            res.explanation = (f"{c.value} was succeeded by {successor}; "
                               f"the successor entity is authoritative")
            res.superseded = [{"value": c.value, "doc_id": c.doc_id,
                               "year": c.year}]
            return res

    # 4. majority, with confidence reflecting how contested the fact is
    counts: Dict[str, int] = {}
    first_seen: Dict[str, Candidate] = {}
    for c in cands:
        key = c.normalised()
        counts[key] = counts.get(key, 0) + 1
        first_seen.setdefault(key, c)
    best_key = max(counts, key=lambda k: (counts[k], -first_seen[k].rank()))
    winner = first_seen[best_key]
    share = counts[best_key] / len(cands)
    res.resolved = winner.value
    res.rule = "majority"
    res.confidence = round(0.5 + 0.4 * share, 3)
    res.explanation = (f"{counts[best_key]}/{len(cands)} sources attest "
                       f"{winner.value!r}; no date or authority signal "
                       f"separated the versions")
    res.superseded = [{"value": c.value, "doc_id": c.doc_id, "year": c.year}
                      for c in cands if c.normalised() != best_key]
    return res


def detect_conflicts(facts: Dict[str, List[Any]]) -> List[Resolution]:
    """Resolve every field, returning only the ones that were contested."""
    resolutions = [resolve(name, values) for name, values in facts.items()]
    return [r for r in resolutions if r.had_conflict]


def summarise(resolutions: List[Resolution]) -> Dict[str, Any]:
    """Compact, trace-friendly summary for the dashboard and results file."""
    conflicted = [r for r in resolutions if r.had_conflict]
    by_rule: Dict[str, int] = {}
    for r in conflicted:
        by_rule[r.rule] = by_rule.get(r.rule, 0) + 1
    return {
        "fields_examined": len(resolutions),
        "conflicts_found": len(conflicted),
        "rules_applied": by_rule,
        "min_confidence": min([r.confidence for r in conflicted], default=1.0),
        "details": [r.to_dict() for r in conflicted],
    }
