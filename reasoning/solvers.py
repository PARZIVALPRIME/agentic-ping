"""Deterministic structured solvers over the knowledge graph.

Each solver answers one question family exactly:

  aggregation : count events for (sport, games) whose `competitors` pass a threshold
  superlative : argmax/argmin of `competitors` within (sport, games)
  temporal    : walk the PREV/NEXT edition chain to the games immediately
                before a target year, then resolve the event and its gold winner
  multi_hop   : link (venue, date) -> event page -> gold winner
  lookup      : resolve an article by title and read a single field

The solvers return complete provenance (candidate sets, filters, rejected
candidates) so the agentic pipeline can report *why* it is confident.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from kg.model import EventNode, KnowledgeGraph
from kg.textutil import best_match, normalize, similarity, tokens
from utils.thresholds import thresholds

from .query_parser import QuerySpec, parse_question


@dataclass
class SolveResult:
    answer: str = ""
    citations: List[str] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    method: str = ""
    candidates_considered: int = 0
    steps: List[Dict[str, Any]] = field(default_factory=list)
    unresolved: List[str] = field(default_factory=list)

    @property
    def resolved(self) -> bool:
        return bool(self.answer)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── shared helpers ─────────────────────────────────────────────────────────

def resolve_sport(name: str, kg: KnowledgeGraph) -> str:
    """Map a question's sport phrase onto the corpus sport vocabulary."""
    from utils.thresholds import thresholds

    if not name:
        return ""
    norm = normalize(name)
    for sport in kg.sport_vocabulary:
        if normalize(sport) == norm:
            return sport
    for sport in kg.sport_vocabulary:
        ns = normalize(sport)
        if ns and ns in norm:
            return sport
    match, score = best_match(name, kg.sport_vocabulary)
    return match if score >= thresholds().sport_match_min else ""


def resolve_venue(name: str, kg: KnowledgeGraph) -> Tuple[List[str], float]:
    """Return (venue keys, score) for a venue phrase in the question."""
    if not name:
        return [], 0.0
    norm = normalize(name)
    if norm in kg.venue_index:
        return [norm], 1.0

    # drop trailing/leading location qualifiers, e.g. "Centennial Parklands, Sydney"
    stripped = normalize(re.sub(r"[,()]", " ", name))
    if stripped in kg.venue_index:
        return [stripped], 1.0

    hits: List[Tuple[str, float]] = []
    qtok = set(tokens(name))
    for venue_key in kg.venue_index:
        score = similarity(name, venue_key)
        vtok = set(venue_key.split())
        if qtok and vtok:
            # containment helps with "X Beijing" style concatenations
            overlap = len(qtok & vtok) / min(len(qtok), len(vtok))
            score = max(score, overlap * 0.93)
        if score >= thresholds().venue_match_min:
            hits.append((venue_key, score))
    if not hits:
        return [], 0.0
    hits.sort(key=lambda x: x[1], reverse=True)
    top = hits[0][1]
    band = thresholds().venue_match_band
    return [k for k, s in hits
            if s >= max(thresholds().venue_strong_min, top - band)], top


def season_candidates(spec: QuerySpec, kg: KnowledgeGraph) -> List[str]:
    """The seasons a question may refer to, most-likely first.

    A question that says "Summer" or "Winter" pins the season. A question that
    does not state one is *undetermined*, not Summer: the candidates are then
    every season the corpus actually contains, and the caller resolves it from
    evidence (which season has the event the question describes). Returning an
    ordered list rather than a default is what makes the ambiguity visible - a
    silent fallback to Summer would answer a Winter question from the wrong
    Games with no trace of the substitution.
    """
    if spec.season:
        return [spec.season]
    seasons = _games_seasons(kg)
    present = [s for s in ("Summer", "Winter") if seasons.get(s)]
    if present:
        return present
    return ["Summer", "Winter"]


def seasons_for_sport(sport: str, kg: KnowledgeGraph) -> List[str]:
    """Seasons in which the corpus actually holds editions of ``sport``.

    Sports are effectively season-specific in the data (sailing has no Winter
    editions, biathlon no Summer ones), so this is the cheapest *evidence* for a
    season the question did not state - and unlike a default it can come back
    empty, which is the honest answer when the corpus does not decide.
    """
    if not sport:
        return []
    seen: List[str] = []
    for node in kg.events_for_sport(sport):
        if node.season and node.season not in seen:
            seen.append(node.season)
    return [s for s in ("Summer", "Winter") if s in seen]


def resolve_previous_edition(spec: QuerySpec, kg: KnowledgeGraph) -> Dict[str, Any]:
    """Resolve "the Games immediately before *target*" without assuming a season.

    Order of evidence, strongest first:
      1. the question states the season -> use it;
      2. only one season in the corpus holds this sport -> that season;
      3. otherwise compare the question's event descriptor against the events of
         each candidate edition and take a clear winner;
      4. otherwise report ``unresolved``/``ambiguous`` so the caller can mark the
         gap and recover, instead of silently answering from the Summer Games.
    """
    target = spec.before_year
    sport = spec.sport or resolve_sport(spec.event_desc, kg)
    candidates: List[Dict[str, Any]] = []
    for season in season_candidates(spec, kg):
        prev = _nearest_previous_games(season, target, kg) if target else None
        score = 0.0
        if prev is not None and sport and spec.event_desc:
            events = kg.events_for(sport, prev, season)
            score = max((_event_similarity(spec.event_desc, e) for e in events),
                        default=0.0)
        candidates.append({"season": season, "year": prev,
                           "event_score": round(score, 3)})

    out: Dict[str, Any] = {"season": "", "year": None, "method": "",
                           "target_year": target, "sport": sport,
                           "candidates": candidates}

    if spec.season:
        hit = next((c for c in candidates if c["season"] == spec.season), None)
        if hit and hit["year"] is not None:
            out.update(season=hit["season"], year=hit["year"], method="stated_season")
        else:
            out["method"] = "unresolved"
        return out

    # 2. sport evidence: a sport lives in exactly one season in this corpus
    sport_seasons = seasons_for_sport(sport, kg)
    if len(sport_seasons) == 1:
        hit = next((c for c in candidates if c["season"] == sport_seasons[0]), None)
        if hit and hit["year"] is not None:
            out.update(season=hit["season"], year=hit["year"], method="sport_season")
            return out

    # 3. event-descriptor evidence
    scored = [c for c in candidates if c["year"] is not None and c["event_score"] > 0]
    scored.sort(key=lambda c: c["event_score"], reverse=True)
    if len(scored) == 1:
        out.update(season=scored[0]["season"], year=scored[0]["year"],
                   method="event_match")
        return out
    if len(scored) > 1 and scored[0]["event_score"] > scored[1]["event_score"]:
        out.update(season=scored[0]["season"], year=scored[0]["year"],
                   method="event_match_margin")
        return out
    if len(scored) > 1:
        out["method"] = "ambiguous"
        out["ambiguous_between"] = [c["season"] for c in scored]
        return out
    out["method"] = "unresolved"
    return out







def _date_score(spec: QuerySpec, ev: EventNode) -> float:
    """Score how well an event's date fields match the question's date phrase."""
    if not spec.date_text:
        return 1.0
    best = 0.0
    for blob in (ev.date_raw, ev.dates_raw, f"{ev.date_raw} {ev.dates_raw}"):
        if not blob:
            continue
        best = max(best, similarity(spec.date_text, blob))
    qmonths = {m for m in tokens(spec.date_text) if m in _MONTH_SET}
    qdays = {d for d in tokens(spec.date_text) if d.isdigit() and 1 <= int(d) <= 31}
    emonths = set(ev.months_found)
    edays = set(ev.days_found)
    if qmonths:
        overlap = len(qmonths & emonths) / len(qmonths)
        best = max(best, 0.55 + 0.4 * overlap)
    if qdays:
        overlap = len(qdays & edays) / len(qdays)
        best = max(best, 0.5 + 0.45 * overlap)
    return best


_MONTH_SET = {
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
}

# a sign that is attached to a number, e.g. "+80 kg" or "-48 kg"
_NUM_SIGN_RE = re.compile(r"[+-]\s*\d")


def _games_seasons(kg: KnowledgeGraph) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {"Summer": [], "Winter": []}
    for key in kg.games:
        parts = key.split()
        if len(parts) == 2 and parts[0].isdigit():
            out.setdefault(parts[1], []).append(int(parts[0]))
    for season in out:
        out[season] = sorted(set(out[season]))
    return out


def _nearest_previous_games(season: str, before_year: int, kg: KnowledgeGraph) -> Optional[int]:
    years = _games_seasons(kg).get(season, [])
    prior = [y for y in years if y < before_year]
    return max(prior) if prior else None


def _event_similarity(query: str, ev: EventNode) -> float:
    """Similarity between a question's event descriptor and a graph event.

    Adds a symbol-aware correction so that '80 kg' is not confused with
    '+80 kg' / '-80 kg' (normalization drops the sign, which would otherwise
    make a weight-class question ambiguous).
    """
    if not query:
        return 0.0
    base = max(
        similarity(query, ev.event_name),
        similarity(query, ev.title),
        similarity(query, normalize(ev.title).split("olympics")[-1]),
    )
    return max(0.0, base - _symbol_penalty(query, ev.event_name))


def _symbol_penalty(query: str, event_name: str) -> float:
    """Penalise numeric sign mismatches that plain normalization erases.

    Only signs directly attached to a number count ('+80 kg'), so hyphenated
    names such as 'Greco-Roman' are unaffected.
    """
    q_sign = bool(_NUM_SIGN_RE.search(query or ""))
    n_sign = bool(_NUM_SIGN_RE.search(event_name or ""))
    return 0.35 if q_sign != n_sign else 0.0


def _venue_sport_affinity(venue_phrase: str, ev: EventNode) -> float:
    """How strongly the venue name announces the event's sport.

    Resolves corpus-level ambiguity where two different sports share a venue
    string (e.g. 'Laura Biathlon & Ski Complex' hosting both a biathlon relay
    and a cross-country race on the same day): the sport named inside the venue
    wins.
    """
    if not venue_phrase or not ev.sport:
        return 0.0
    venue_tokens = set(tokens(venue_phrase))
    sport_tokens = set(tokens(ev.sport))
    if not sport_tokens or not venue_tokens:
        return 0.0
    return len(sport_tokens & venue_tokens) / len(sport_tokens)

class StructuredSolver:
    """Runs the exact solver for a parsed :class:`QuerySpec`."""

    def __init__(self, kg: KnowledgeGraph) -> None:
        self.kg = kg

    # ── public entry point ─────────────────────────────────────────────
    def solve(self, question: str, spec: Optional[QuerySpec] = None) -> SolveResult:
        spec = spec or parse_question(question, self.kg)
        handler = {
            "aggregation": self.solve_aggregation,
            "superlative": self.solve_superlative,
            "temporal": self.solve_temporal,
            "multi_hop": self.solve_multi_hop,
            "lookup": self.solve_lookup,
        }.get(spec.qtype)
        if handler is None:
            return SolveResult(method="unsupported", unresolved=[f"qtype={spec.qtype}"])
        result = handler(spec)
        result.steps.insert(0, {"operation": "parse_question", "qtype": spec.qtype,
                                "slots": spec.to_dict()})
        return result

    # ── aggregation ────────────────────────────────────────────────────
    def solve_aggregation(self, spec: QuerySpec) -> SolveResult:
        res = SolveResult(method="aggregation_competitor_threshold")
        sport = spec.sport or resolve_sport(spec.sport, self.kg)
        if not sport:
            res.unresolved.append("sport")
            return res
        events = self.kg.events_for(sport, spec.year, spec.season)
        res.steps.append({
            "operation": "graph_lookup",
            "description": f"events(IN_SPORT={sport} AND PART_OF={spec.year} {spec.season})",
            "found": len(events),
        })
        if not events:
            res.unresolved.append("no_events_for_sport_games")
            return res
        res.candidates_considered = len(events)
        passed, failed, missing = [], [], []
        for ev in events:
            if ev.competitors is None:
                missing.append(ev)
                continue
            if _threshold_pass(ev.competitors, spec.comparator, spec.threshold):
                passed.append(ev)
            else:
                failed.append(ev)
        res.answer = str(len(passed))
        res.citations = [e.doc_id for e in events]
        res.evidence = [
            {"doc_id": e.doc_id, "title": e.title, "competitors": e.competitors,
             "kept": e in passed}
            for e in events
        ]
        completeness = len(passed) + len(failed)
        res.confidence = round(
            0.6 + 0.3 * (completeness / len(events)) + (0.1 if not missing else 0.0), 3
        )
        res.steps.append({
            "operation": "filter_competitors",
            "description": f"competitors {spec.comparator} {spec.threshold}",
            "kept": len(passed), "rejected": len(failed), "missing_field": len(missing),
            "kept_titles": [e.title for e in passed],
        })
        return res

    # ── superlative ─────────────────────────────────────────────────────
    def solve_superlative(self, spec: QuerySpec) -> SolveResult:
        res = SolveResult(method="superlative_competitor_argmax")
        sport = spec.sport or resolve_sport(spec.sport, self.kg)
        if not sport:
            res.unresolved.append("sport")
            return res
        events = self.kg.events_for(sport, spec.year, spec.season)
        candidates = [e for e in events if e.competitors is not None]
        res.steps.append({
            "operation": "graph_lookup",
            "description": f"events(IN_SPORT={sport} AND PART_OF={spec.year} {spec.season})",
            "found": len(events),
        })
        if not candidates:
            res.unresolved.append("no_events_with_competitors")
            return res
        res.candidates_considered = len(candidates)
        if spec.direction == "max":
            best = max(candidates, key=lambda e: e.competitors)
        else:
            best = min(candidates, key=lambda e: e.competitors)
        res.answer = best.title
        # every event page of that sport at those Games forms the evidence set
        res.citations = [e.doc_id for e in events]
        ranked = sorted(candidates, key=lambda e: e.competitors,
                        reverse=(spec.direction == "max"))
        res.evidence = [
            {"doc_id": e.doc_id, "title": e.title, "competitors": e.competitors,
             "rank": i + 1}
            for i, e in enumerate(ranked)
        ]
        runner_up = ranked[1].competitors if len(ranked) > 1 else None
        margin = (best.competitors - runner_up) if runner_up is not None else None
        res.confidence = round(0.8 + (0.15 if margin else 0.0), 3)
        res.steps.append({
            "operation": "arg_extreme",
            "description": f"{spec.direction}(competitors) over {len(candidates)} events",
            "winner": best.title, "value": best.competitors, "runner_up_value": runner_up,
            "margin": margin,
        })
        return res

    # ---- temporal ----
    def solve_temporal(self, spec: QuerySpec) -> SolveResult:
        res = SolveResult(method="temporal_prev_edition")
        if spec.before_year is None:
            res.unresolved.append("before_year")
            return res
        # Which season the previous edition belongs to is *evidence*, not a
        # default: "the Games immediately before 1994" contains no season, and
        # assuming Summer answers from the Summer Games when the question meant
        # Lillehammer. The shared resolver decides it from the question's sport
        # and event descriptor - the same decision the entity linker and the
        # graph traverser make - and reports which evidence produced it. When the
        # question does state a season this is exactly the old behaviour
        # ("stated_season" -> that season's nearest earlier edition).
        edition = resolve_previous_edition(spec, self.kg)
        season = str(edition.get("season") or "")
        year = edition.get("year")
        res.steps.append({
            "operation": "temporal_resolution",
            "description": (f"latest {season or '(unresolved)'} Games strictly "
                            f"before {spec.before_year} "
                            f"[{edition.get('method') or 'unresolved'}]"),
            "resolved_year": year,
            "season_method": edition.get("method"),
            "season_candidates": edition.get("candidates", []),
        })
        if not season:
            # Never guessed: an unstated season the evidence could not settle is
            # reported as a gap so the caller can widen or replan.
            res.unresolved.append("season_unresolved")
            return res
        if year is None:
            res.unresolved.append("no_prior_games")
            return res
        sport = spec.sport or resolve_sport(spec.event_desc, self.kg)
        if not sport:
            res.unresolved.append("sport")
            return res
        events = self.kg.events_for(sport, year, season)
        res.candidates_considered = len(events)
        scored = sorted(((_event_similarity(spec.event_desc, e), e) for e in events),
                        key=lambda x: x[0], reverse=True)
        res.steps.append({
            "operation": "event_resolution",
            "description": f"match event '{spec.event_desc}' in {sport} {year} {season}",
            "candidates": [{"title": e.title, "score": round(s, 3)} for s, e in scored[:3]],
        })
        if not scored or scored[0][0] < 0.55:
            res.unresolved.append("event_not_matched")
            return res
        best_score, best = scored[0]
        res.answer = best.gold
        if not res.answer:
            res.unresolved.append("gold_missing_in_corpus")
            return res
        # cite the resolved edition *and* the anchor edition named in the question
        res.citations = [best.doc_id]
        if best.next_doc_id:
            res.citations.append(best.next_doc_id)
        res.evidence = [
            {"doc_id": best.doc_id, "title": best.title, "gold": best.gold,
             "silver": best.silver, "bronze": best.bronze, "games": best.games_key}
        ]
        res.confidence = round(min(0.95, 0.55 + 0.4 * best_score), 3)
        res.steps.append({
            "operation": "medal_lookup",
            "description": f"gold of {best.title}",
            "answer": res.answer,
        })
        return res

    # ─ multi hop ───────────────────────────────────────────────────────
    def solve_multi_hop(self, spec: QuerySpec) -> SolveResult:
        res = SolveResult(method="venue_date_link")
        if not spec.venue:
            res.unresolved.append("venue")
            return res
        venue_keys, venue_score = resolve_venue(spec.venue, self.kg)
        res.steps.append({
            "operation": "entity_linking",
            "description": f"venue '{spec.venue}'",
            "resolved": venue_keys, "score": round(venue_score, 3),
        })
        if not venue_keys:
            res.unresolved.append("venue_not_found")
            return res

        candidate_ids: List[str] = []
        for key in venue_keys:
            candidate_ids.extend(self.kg.venue_index.get(key, []))
        candidates = [self.kg.events[d] for d in dict.fromkeys(candidate_ids)]
        if spec.year:
            filtered = [e for e in candidates if e.year == spec.year]
            if spec.season:
                filtered = [e for e in filtered if e.season == spec.season]
            if filtered:
                candidates = filtered
        res.candidates_considered = len(candidates)
        if not candidates:
            res.unresolved.append("no_events_at_venue")
            return res

        scored = sorted(((_date_score(spec, e), _venue_sport_affinity(spec.venue, e), e)
                         for e in candidates),
                        key=lambda x: (x[0], x[1]), reverse=True)
        res.steps.append({
            "operation": "date_filter",
            "description": f"date '{spec.date_text}'",
            "top_candidates": [
                {"title": e.title, "date_score": round(s, 3), "venue_affinity": round(a, 3)}
                for s, a, e in scored[:3]
            ],
        })
        best_score, best_affinity, best = scored[0]
        if best_score < 0.55:
            res.unresolved.append("date_not_matched")
            return res
        res.answer = best.gold
        if not res.answer:
            res.unresolved.append("gold_missing_in_corpus")
            return res
        res.citations = [best.doc_id]
        res.evidence = [
            {"doc_id": e.doc_id, "title": e.title, "date": e.date_raw or e.dates_raw,
             "venue": e.venue, "gold": e.gold, "score": round(s, 3),
             "venue_affinity": round(a, 3)}
            for s, a, e in scored[:5]
        ]
        res.confidence = round(min(0.94, 0.5 + 0.45 * best_score + 0.05 * best_affinity), 3)
        res.steps.append({
            "operation": "medal_lookup",
            "description": f"gold of {best.title}",
            "answer": res.answer,
        })
        return res

    # ── lookup ──────────────────────────────────────────────────────────
    def solve_lookup(self, spec: QuerySpec) -> SolveResult:
        res = SolveResult(method="title_field_lookup")
        target = spec.target_title
        if not target:
            res.unresolved.append("target_title")
            return res
        norm = normalize(target)
        doc_id = self.kg.title_index.get(norm)
        match_score = 1.0
        if doc_id is None:
            match, match_score = best_match(target, list(self.kg.title_index.keys()))
            if match is not None and match_score >= 0.85:
                doc_id = self.kg.title_index[match]
        res.steps.append({
            "operation": "title_resolution",
            "description": f"resolve article '{target}'",
            "doc_id": doc_id, "score": round(match_score, 3),
        })
        if doc_id is None:
            res.unresolved.append("title_not_found")
            return res
        ev = self.kg.events[doc_id]
        res.answer = "" if ev.nations is None else str(ev.nations)
        if not res.answer:
            res.unresolved.append("nations_missing_in_corpus")
            return res
        res.citations = [ev.doc_id]
        res.evidence = [{"doc_id": ev.doc_id, "title": ev.title, "nations": ev.nations,
                         "competitors": ev.competitors}]
        res.confidence = round(min(0.96, 0.6 + 0.35 * match_score), 3)
        res.candidates_considered = 1
        return res


def _threshold_pass(value: int, comparator: str, threshold: Optional[int]) -> bool:
    if threshold is None:
        return False
    return {
        "gt": value > threshold,
        "gte": value >= threshold,
        "lt": value < threshold,
        "lte": value <= threshold,
    }.get(comparator, value > threshold)

