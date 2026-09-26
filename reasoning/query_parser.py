"""Question understanding: turn a natural-language question into a QuerySpec.

THREE WAYS A QUESTION BECOMES A SPEC, AND WHICH ONE IS THE ENTRY POINT
---------------------------------------------------------------------
The regex templates in this module recognise the wording they were written
against (the public set) exactly, and nothing else. A system that only
understands the sentences it was written against is a lookup table for those
sentences, so the templates cannot be the entry point:

1. ``parse_question_semantic`` - THE ENTRY POINT the pipeline uses
   (``parse_question_with_llm`` wraps it). The LLM extracts the slots; the
   templates then VALIDATE every model slot against the corpus and FILL only
   what the model left empty. Every decision is recorded in a report that
   travels with the run (``AgentState.parse_report`` ->
   ``PipelineResult.metadata['parse_report']``), so a run can be audited for how
   much of its understanding came from the model rather than from wording.
2. ``parse_question`` - the template parser. Still the whole story on the
   deterministic (``--no-llm``) path and under ``PARSE_MODE=rules``, and the
   validation/fill reference described above. It is also what ``--no-llm``
   benchmarks exercise, which is what makes the "does the agent's LLM
   understanding pay for itself?" comparison meaningful.
3. ``classify`` - the keyword ladder behind (2), used by the classifier agent as
   one vote (see ``agents/classifier.py``).

An LLM is therefore required for (1) but never for correctness: if no model is
reachable, the semantic parser returns the template spec unchanged and says so
in its report. The ablation switch ``ABLATE_CLASSIFIER=1`` (or
``set_benchmark_parser(False)``) removes the templates from *both* roles, so a
run can be measured under a phrasing the system was never tuned on.
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

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
    #: True when the question does not state a season and none could be
    #: established from the corpus. Downstream code must then consider *both*
    #: seasons (or mark the gap) - it must never assume Summer.
    season_unresolved: bool = False
    #: How this spec was produced: "llm" (semantic parse), "rules" (the
    #: deterministic template parser acting as fallback), "llm+rules" (semantic
    #: parse with deterministic validation/fill), or "none".
    parse_method: str = "rules"

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


# ══════════════════════════════════════════════════════════════════════════
# Semantic parsing (primary path) with the template parser as fallback
# ══════════════════════════════════════════════════════════════════════════
#
# The five regexes above recognise the benchmark's own wording. They are exact
# on it and blind to anything phrased differently, so they cannot be the primary
# way a question becomes a QuerySpec: a system that only understands the phrases
# it was written against is a lookup table, not a question answering system.
#
# The primary parser is therefore the LLM, asked for the same structured slots.
# The template parser is kept for two jobs and no others:
#
#   1. VALIDATION - a slot is accepted only if it is well-formed for the corpus
#      (the sport must exist in the corpus vocabulary, the year must be
#      plausible, the season must be one the corpus has, ...).
#   2. FALLBACK - when the model is unavailable, returns unusable JSON, or
#      leaves a slot empty, the template value fills it.
#
# ``set_benchmark_parser(False)`` - or the environment switch below - removes the
# template parser entirely: the spec is then whatever the model produced,
# validated but never repaired from question wording.

#: Ablation switch. False means the template parser is not consulted at all.
BENCHMARK_PARSER_ENABLED = True

# How a question becomes a QuerySpec:
#   "semantic" - the LLM extracts the slots, the templates validate and fill
#                (``parse_question_semantic``, the default)
#   "rules"    - the templates alone (deterministic runs, ablations)
PARSE_MODES = ("semantic", "rules")
DEFAULT_PARSE_MODE = "semantic"


def parse_mode() -> str:
    """The configured slot-parse path, validated against :data:`PARSE_MODES`."""
    mode = os.getenv("PARSE_MODE", DEFAULT_PARSE_MODE).strip().lower()
    return mode if mode in PARSE_MODES else DEFAULT_PARSE_MODE


def template_parser_enabled() -> bool:
    """Whether the template parser may be consulted at all.

    False under ``ABLATE_CLASSIFIER=1``: the ablation removes the templates from
    *both* of their jobs (classification and slot parsing), which is what makes
    row D of the generalisation suite a measurement of the system on wording it
    was never tuned against. Also false after ``set_benchmark_parser(False)``.
    """
    if os.getenv("ABLATE_CLASSIFIER", "").strip().lower() in ("1", "true", "yes"):
        return False
    return BENCHMARK_PARSER_ENABLED

SLOT_DEFAULTS: Dict[str, Any] = {
    "qtype": "lookup", "sport": "", "year": 0, "season": "", "event_desc": "",
    "venue": "", "date_text": "", "threshold": 0, "comparator": "", "direction": "",
    "target_title": "", "before_year": 0,
}

SEMANTIC_PARSE_PROMPT = """You extract structured slots from a question about \
{corpus_label}. Reply with JSON only.

{{"qtype": "lookup|multi_hop|temporal|aggregation|superlative",
 "sport": "", "year": 0, "season": "", "event_desc": "", "venue": "",
 "date_text": "", "threshold": 0, "comparator": "", "direction": "",
 "target_title": "", "before_year": 0, "confidence": 0.0}}

Category meanings:
- lookup      : one fact about one named article
- multi_hop   : a venue and/or a date must be linked to an event (and its winner)
- temporal    : the edition immediately before/after a stated one - two editions
- aggregation : count events of a sport at a stated Games passing a threshold
- superlative : the event with the highest/lowest value of some field

Extraction rules - these matter more than filling every field:
- Use ONLY values the question states. Never infer a year, a sport, a season or
  a venue that the question does not give.
- season is "Summer" or "Winter" ONLY if the question says it. If it does not,
  return "" - an unstated season is genuinely unknown and guessing is worse.
- comparator is one of gt|gte|lt|lte; direction is max or min; "" if the
  question does not compare.
- number fields are 0 when the question gives no number.
- confidence is your own confidence that the qtype and slots are right.

Question: {question}
"""


def set_benchmark_parser(enabled: bool) -> None:
    """Enable/disable the template parser (ablation switch, never a default)."""
    global BENCHMARK_PARSER_ENABLED
    BENCHMARK_PARSER_ENABLED = bool(enabled)


def benchmark_parser_enabled() -> bool:
    """Whether the template parser may be used as fallback/validation."""
    return BENCHMARK_PARSER_ENABLED


def _as_int(value: Any) -> Optional[int]:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


def _validate_slot(field: str, value: Any, kg: Optional[KnowledgeGraph]) -> Any:
    """Coerce an LLM-provided slot to a corpus-legal value, or None to reject.

    Validation is a *corpus* check, not a wording check: a slot value is legal if
    it denotes something the corpus contains or a well-formed number. Nothing
    here inspects the question's phrasing.
    """
    from utils.thresholds import thresholds

    th = thresholds()
    if field == "qtype":
        text = str(value or "").strip().lower()
        return text if text in ("lookup", "multi_hop", "temporal",
                                "aggregation", "superlative") else None
    if field in ("year", "before_year"):
        number = _as_int(value)
        if number in (None, 0):
            return None
        return number if th.year_min <= number <= th.year_max else None
    if field == "threshold":
        number = _as_int(value)
        if number in (None, 0):
            return None
        return number if 0 < number < th.threshold_max else None
    if field == "season":
        text = str(value or "").strip().title()
        return text if text in ("Summer", "Winter") else None
    if field == "comparator":
        text = str(value or "").strip().lower()
        return text if text in ("gt", "gte", "lt", "lte") else None
    if field == "direction":
        text = str(value or "").strip().lower()
        return text if text in ("max", "min") else None
    if field == "sport":
        text = str(value or "").strip()
        if not text:
            return None
        if kg is None:
            return text
        from .solvers import resolve_sport

        return resolve_sport(text, kg) or None
    text = str(value or "").strip()
    return text[:200] if text else None



def parse_question_semantic(question: str, kg: Optional[KnowledgeGraph] = None,
                            llm: Any = None, counter: Any = None,
                            use_rules: Optional[bool] = None,
                            ) -> Tuple[QuerySpec, Dict[str, Any]]:
    """Primary parser: the LLM produces the slots, validated and filled by rules.

    Returns ``(spec, report)``. The report records which path produced the slots,
    so a run can be audited for how much of its understanding came from the model
    rather than from question-wording templates.
    """
    if use_rules is None:
        use_rules = template_parser_enabled()
    report: Dict[str, Any] = {"method": "rules" if use_rules else "none",
                              "llm_used": False, "filled_by_rules": [],
                              "rejected": [], "conflicts": [],
                              "confidence": 0.0, "corpus_validated": [],
                              "templates_enabled": use_rules,
                              "llm_error": None, "reason": ""}

    # The deterministic parser, when allowed, always runs: it is the validation
    # reference and the fill source. Skipped entirely under the ablation.
    rules_spec = parse_question(question, kg) if use_rules else None
    # Recorded before any return, so every report shape carries it: the caller
    # compares it with the final qtype to tell an adaptation from a rephrasing.
    report["rule_qtype"] = rules_spec.qtype if rules_spec is not None else ""

    llm_ready = llm is not None and bool(getattr(llm, "available", False))
    if not llm_ready:
        spec = rules_spec or QuerySpec(question=question, qtype="lookup",
                                       parse_method="none")
        if rules_spec is not None:
            spec.parse_method = "rules"
        report["reason"] = ("no model available: template parse only" if rules_spec
                            is not None else
                            "no model and no templates: nothing to parse with")
        return spec, report

    payload = llm.complete_json(
        SEMANTIC_PARSE_PROMPT.format(question=question,
                                     corpus_label=_corpus_label()),
        caller="parser.semantic", counter=counter,
        model=getattr(llm, "fast_model", None) or None)
    if not payload or payload.get("error"):
        # Unusable model output: fall back wholesale rather than half-parse.
        spec = rules_spec or QuerySpec(question=question, qtype="lookup",
                                       parse_method="none")
        if rules_spec is not None:
            spec.parse_method = "rules"
        report["llm_error"] = str(payload.get("error")) if payload else "empty"
        report["reason"] = ("unusable model output: template parse kept"
                            if rules_spec is not None else
                            "unusable model output and no templates")
        return spec, report

    report["llm_used"] = True
    try:
        report["confidence"] = round(float(payload.get("confidence", 0.0) or 0.0), 3)
    except (TypeError, ValueError):
        report["confidence"] = 0.0

    spec = QuerySpec(question=question)
    # 1. accept every model slot that validates against the corpus.
    #    Only fields the model *actually returned* are read: a field it omitted
    #    is missing evidence, and `SLOT_DEFAULTS` is a prompt convention (a
    #    compliant model writes qtype="lookup", year=0 for what it does not
    #    know), not a value the model can be credited with. Substituting the
    #    default here would let an unusable reply - e.g. `{"answer": ...}` with
    #    no slots at all - be scored as the model choosing "lookup", which then
    #    survives the rules fill below (the slot looks non-empty) and silently
    #    replaces the template's question type. That is exactly how a bad model
    #    turned an aggregation question into an unresolvable lookup.
    for field in SLOT_DEFAULTS:
        if field not in payload:
            continue
        raw = payload.get(field)
        if raw in (None, "", 0, []):
            continue
        coerced = _validate_slot(field, raw, kg)
        if coerced is None:
            report["rejected"].append({field: raw})
            continue
        setattr(spec, field, coerced)
        report["corpus_validated"].append({field: coerced})

    # 2. deterministic validation/fill for anything the model left out.
    #    "Left out" is tracked from what the model actually supplied and the
    #    corpus accepted - not from "the slot is empty" - because the dataclass
    #    placeholders (`qtype="unknown"`, `comparator="gt"`, `direction="max"`)
    #    and a rejected value are all non-empty and would otherwise block the
    #    template from filling the gap.
    supplied = {name for item in report["corpus_validated"] for name in item}
    if rules_spec is not None:
        for field in SLOT_DEFAULTS:
            if field in supplied:
                continue
            value = getattr(rules_spec, field, None)
            if value in (None, "", 0):
                continue
            # Only a fill that changes the spec is a template contribution. The
            # dataclass defaults already agree with the template on most
            # questions; recording those as template work would misreport where
            # the understanding actually came from.
            if getattr(spec, field, None) == value:
                continue
            setattr(spec, field, value)
            report["filled_by_rules"].append({field: value})
        # A model slot and a template slot that disagree is recorded rather than
        # hidden: on benchmark-shaped wording the disagreement is the template's
        # failure, on reworded questions it is usually the template matching noise.
        for field in ("qtype", "sport", "year", "season", "event_desc",
                      "venue", "date_text", "target_title", "before_year"):
            llm_value = payload.get(field)
            rule_value = getattr(rules_spec, field, None)
            if (llm_value not in (None, "", 0) and rule_value not in (None, "", 0)
                    and str(llm_value).strip().lower() != str(rule_value).strip().lower()):
                report["conflicts"].append({field: {"llm": llm_value,
                                                    "rules": rule_value}})

    spec.qtype = spec.qtype if spec.qtype in (
        "lookup", "multi_hop", "temporal", "aggregation", "superlative") else "lookup"
    spec.years_mentioned = [int(y) for y in
                            re.findall(r"\b(19\d{2}|20\d{2})\b", question)]
    # An unstated season stays unstated. Never silently "Summer".
    spec.season_unresolved = not bool(spec.season)
    report["method"] = "llm+rules" if report["filled_by_rules"] else "llm"
    report["reason"] = (f"model slots accepted: {len(report['corpus_validated'])}; "
                        f"filled by templates: {len(report['filled_by_rules'])}; "
                        f"model/template disagreements: {len(report['conflicts'])}")
    spec.parse_method = report["method"]
    return spec, report


def _corpus_label() -> str:
    """How prompts name the corpus: configuration, never hardcoded per domain."""
    from config import config as _app_config

    return getattr(getattr(_app_config, "domain", None), "corpus_label",
                   "a corpus of event articles")


def parse_question_with_llm(question: str, kg: Optional[KnowledgeGraph] = None,
                            llm: Any = None, counter: Any = None,
                            use_rules: Optional[bool] = None,
                            ) -> Tuple[QuerySpec, Dict[str, Any]]:
    """The pipeline's entry point: parse ``question``, preferring the model.

    ``PARSE_MODE=semantic`` (the default) routes to
    :func:`parse_question_semantic`, where the LLM produces the slots and the
    templates validate them. ``PARSE_MODE=rules`` routes to the template parser
    alone - the deterministic (``--no-llm``) path and the ablation setting.

    The second element of the return value is the audit report: which path ran,
    which slots the model contributed, which the templates had to fill, which
    model slots the corpus rejected, and where the two disagreed. It is carried
    on the run so the claim "the model does the understanding" is checkable
    rather than asserted.
    """
    mode = parse_mode()
    enabled = template_parser_enabled() if use_rules is None else bool(use_rules)
    if mode == "rules" or (not enabled and llm is None):
        spec = (parse_question(question, kg) if enabled
                else QuerySpec(question=question, qtype="lookup",
                               parse_method="none"))
        spec.season_unresolved = not bool(spec.season)
        report = {"method": "rules" if enabled else "none", "llm_used": False,
                  "parse_mode": mode, "templates_enabled": enabled,
                  "filled_by_rules": [], "rejected": [], "conflicts": [],
                  "corpus_validated": [], "confidence": 1.0 if enabled else 0.0,
                  "rule_qtype": spec.qtype,
                  "reason": "PARSE_MODE=rules" if mode == "rules"
                            else "template parser removed by the ablation"}
        return spec, report

    spec, report = parse_question_semantic(question, kg, llm=llm,
                                           counter=counter, use_rules=enabled)
    report["parse_mode"] = mode
    spec.season_unresolved = not bool(spec.season)
    return spec, report
