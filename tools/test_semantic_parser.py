"""Checks for the semantic (LLM) slot parser and its wiring into a run.

The claim these tests defend: *the model produces the query slots; the template
parser validates them against the corpus and fills only what the model left
empty*. Before this wiring existed, ``parse_question_semantic`` was written,
documented as the primary path and never called - the templates were the entry
point and the documentation said the opposite. A claim that cannot fail is not a
test, so every check below asserts a specific sink for a specific input:

  1. model slots win over template slots when both exist;
  2. an unstated season stays unstated (never "Summer");
  3. a slot the corpus cannot confirm is REJECTED and recorded, not used;
  4. a model that omits a slot gets the template value (fill, and only fill) -
     including the question type when the reply carries no slots at all, or
     carries a value the corpus rejects;
  5. unusable model output falls back wholesale to the templates, loudly;
  6. with no model reachable the templates run and the report says so;
  7. PARSE_MODE=rules never calls the model at all;
  8. ABLATE_CLASSIFIER=1 removes the templates from both roles, so nothing can
     be filled from question wording;
  9. the parse reaches the run: AgentState.parse_report -> to_dict().

Offline: no LLM, no network, no knowledge graph (kg=None), no benchmark run.
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import reasoning.query_parser as qp  # noqa: E402
from agents.state import AgentState  # noqa: E402
from reasoning.query_parser import (QuerySpec, parse_question,  # noqa: E402
                                   parse_question_with_llm,
                                   parse_question_semantic)

AGG_Q = ("how many biathlon events at the 2018 Winter Olympics had more than "
         "73 competitors?")
REWORDED_Q = ("Counting the biathlon programme of the 2018 Winter Games, how "
              "many of its events drew a field bigger than 73?")


class StubLLM:
    """A stand-in for a provider: returns a fixed payload, counts its calls."""

    available = True
    fast_model = "stub"

    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def complete_json(self, prompt, system_prompt="", caller="", counter=None,
                      model=None, **kwargs):
        self.calls += 1
        return dict(self.payload) if isinstance(self.payload, dict) else self.payload


class OfflineLLM:
    """Looks like a configured model but is not reachable."""

    available = False
    fast_model = "stub"

    def complete_json(self, *a, **k):  # pragma: no cover - must never run
        raise AssertionError("an unavailable model must not be called")


FAILURES = []


# ── 1. the model's slots are the ones that survive ─────────────────────────
def test_model_slots_win():
    stub = StubLLM({"qtype": "aggregation", "sport": "Biathlon", "year": 2018,
                    "season": "Winter", "threshold": 73, "comparator": "gt",
                    "confidence": 0.88})
    spec, report = parse_question_semantic(REWORDED_Q, None, llm=stub)

    check("model slots win: the templates cannot extract the slots",
          not parse_question(REWORDED_Q, None).threshold,
          f"template threshold={parse_question(REWORDED_Q, None).threshold!r}")
    check("model slots win: qtype from the model", spec.qtype == "aggregation",
          spec.qtype)
    check("model slots win: numbers from the model",
          (spec.year, spec.threshold, spec.comparator) == (2018, 73, "gt"),
          f"{spec.year}/{spec.threshold}/{spec.comparator}")
    check("model slots win: report names the model path",
          report["llm_used"] and report["method"] == "llm",
          f"method={report['method']} confidence={report['confidence']}")
    check("model slots win: parse_method recorded on the spec",
          spec.parse_method == "llm", spec.parse_method)
    check("model slots win: the model was asked once", stub.calls == 1,
          f"calls={stub.calls}")


# ── 2. a season the question does not state is never invented ──────────────
def test_unstated_season_stays_unstated():
    question = "Which athlete won the gold in the event held immediately before 2012?"
    stub = StubLLM({"qtype": "temporal", "before_year": 2012, "confidence": 0.9})
    spec, report = parse_question_semantic(question, None, llm=stub)
    check("unstated season stays unstated", spec.season == "",
          f"season={spec.season!r}")
    check("unstated season is flagged on the spec", spec.season_unresolved is True)
    check("unstated season keeps the year", spec.before_year == 2012,
          str(spec.before_year))


# ── 3. a slot the corpus cannot confirm is rejected and recorded ───────────
def test_illegal_slots_are_rejected():
    stub = StubLLM({"qtype": "aggregation", "sport": "Biathlon", "year": 1066,
                    "season": "Autumn", "threshold": 73, "comparator": "about",
                    "confidence": 0.9})
    spec, report = parse_question_semantic(AGG_Q, None, llm=stub)

    rejected = {k for item in report["rejected"] for k in item}
    check("illegal slots rejected: impossible year", "year" in rejected,
          str(report["rejected"]))
    check("illegal slots rejected: season outside the corpus",
          "season" in rejected, f"season={spec.season!r}")
    check("illegal slots rejected: undefined comparator",
          "comparator" in rejected, f"comparator={spec.comparator!r}")
    check("illegal slots rejected: templates filled the gaps instead",
          {"year", "season"} <= {k for item in report["filled_by_rules"]
                                 for k in item},
          str(report["filled_by_rules"]))
    check("illegal slots rejected: no rejected value was validated in",
          not {"year", "season", "comparator"} & {k for item in
                                                  report["corpus_validated"]
                                                  for k in item},
          str(report["corpus_validated"]))
    check("illegal slots rejected: a rejected value never reaches the spec",
          (spec.year, spec.comparator) == (2018, "gt"),
          f"{spec.year}/{spec.comparator}")



def check(name, condition, detail=""):
    mark = "PASS" if condition else "FAIL"
    print(f"{mark} {name}" + (f" - {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


# ── 4. templates may fill, and only fill ──────────────────────────────────
def test_templates_fill_empty_slots_only():
    stub = StubLLM({"qtype": "aggregation", "sport": "Biathlon", "year": 2018,
                    "confidence": 0.8})
    spec, report = parse_question_semantic(AGG_Q, None, llm=stub)
    filled = {k for item in report["filled_by_rules"] for k in item}
    check("fill: template supplied only the missing slots",
          {"threshold", "season"} <= filled and "year" not in filled,
          f"filled={sorted(filled)}")
    check("fill: the result is complete anyway",
          (spec.threshold, spec.comparator, spec.year) == (73, "gt", 2018),
          f"{spec.threshold}/{spec.comparator}/{spec.year}")
    check("fill: report marks the mixed path", report["method"] == "llm+rules",
          report["method"])

    stub2 = StubLLM({"qtype": "aggregation", "sport": "Biathlon", "year": 2014,
                     "season": "Winter", "threshold": 73, "comparator": "gt",
                     "confidence": 0.9})
    _, report2 = parse_question_semantic(AGG_Q, None, llm=stub2)
    conflicted = {k for item in report2["conflicts"] for k in item}
    check("fill: model/template disagreement is recorded, not hidden",
          "year" in conflicted, str(report2["conflicts"]))


# ── 5. unusable model output: fall back, loudly ────────────────────────────
def test_unusable_output_falls_back():
    spec, report = parse_question_semantic(
        AGG_Q, None, llm=StubLLM({"error": "json_parse_failed"}))
    check("unusable output: templates kept the question answerable",
          spec.qtype == "aggregation" and spec.threshold == 73,
          f"{spec.qtype}/{spec.threshold}")
    check("unusable output: the failure is recorded",
          report["llm_error"] == "json_parse_failed" and not report["llm_used"],
          str(report["llm_error"]))
    check("unusable output: parse_method says rules", spec.parse_method == "rules",
          spec.parse_method)

    spec2, report2 = parse_question_semantic(AGG_Q, None, llm=StubLLM(None))
    check("unusable output: an empty reply is a failure too",
          report2["llm_error"] == "empty" and spec2.threshold == 73,
          str(report2["llm_error"]))


# ── 6. no model reachable: templates, and the report admits it ─────────────
# 5b. a reply with no slots is not credited with the prompt's defaults.
# The failure this pins down: a small model answers the *parsing* request with
# an adjudication-shaped object (`{"answer": ..., "agree": ...}`) - no slots at
# all. Because the prompt shows `qtype: "lookup"` as a convention for "no
# category", the omitted field used to be read as the model choosing "lookup".
# That value then survived the template fill (the slot looked non-empty), so an
# aggregation question was planned as a single-article lookup, hit an
# unresolvable article and returned no answer at all. The invariant is that a
# slot the model did not return is *missing*, and missing means the templates
# supply it.
def test_slotless_reply_keeps_the_template_parse():
    stub = StubLLM({"answer": "Vladimir Smirnov", "agree": False,
                    "reason": "adversarial stub"})
    spec, report = parse_question_semantic(AGG_Q, None, llm=stub)

    check("slotless reply: the template qtype survives",
          spec.qtype == "aggregation", spec.qtype)
    check("slotless reply: the template slots are all filled",
          (spec.year, spec.threshold, spec.comparator) == (2018, 73, "gt"),
          f"{spec.year}/{spec.threshold}/{spec.comparator}")
    check("slotless reply: the model is credited with nothing",
          report["corpus_validated"] == [], str(report["corpus_validated"]))
    check("slotless reply: no phantom model/template disagreement",
          report["conflicts"] == [], str(report["conflicts"]))
    check("slotless reply: the path is reported as templates filling in",
          report["method"] == "llm+rules" and report["llm_used"],
          f"method={report['method']} llm_used={report['llm_used']}")

    # A slot the corpus rejects must behave the same way: rejected is not
    # "answered", so the template still gets to supply the question type.
    spec2, report2 = parse_question_semantic(
        AGG_Q, None, llm=StubLLM({"qtype": "counting", "confidence": 0.4}))
    check("rejected qtype: the template qtype is used instead",
          spec2.qtype == "aggregation", spec2.qtype)
    check("rejected qtype: the rejection is recorded",
          {"qtype"} == {k for item in report2["rejected"] for k in item},
          str(report2["rejected"]))


def test_no_model_degrades_loudly():
    spec, report = parse_question_with_llm(AGG_Q, None, llm=OfflineLLM())
    check("no model: templates still produce a full spec",
          (spec.qtype, spec.year, spec.threshold, spec.comparator) ==
          ("aggregation", 2018, 73, "gt"),
          f"{spec.qtype}/{spec.year}/{spec.threshold}/{spec.comparator}")
    check("no model: report says the model was not used",
          not report["llm_used"] and "no model" in report["reason"],
          report["reason"])
    check("no model: the default parse mode is semantic",
          report["parse_mode"] == "semantic", report["parse_mode"])


# ── 7. PARSE_MODE=rules never calls the model ─────────────────────────────
def test_rules_mode_never_calls_the_model():
    os.environ["PARSE_MODE"] = "rules"
    try:
        stub = StubLLM({"qtype": "superlative", "confidence": 1.0})
        spec, report = parse_question_with_llm(AGG_Q, None, llm=stub)
        check("rules mode: the model was never asked", stub.calls == 0,
              f"calls={stub.calls}")
        check("rules mode: templates decided", spec.qtype == "aggregation",
              spec.qtype)
        check("rules mode: report is explicit",
              report["parse_mode"] == "rules" and not report["llm_used"],
              report["reason"])
    finally:
        os.environ.pop("PARSE_MODE", None)


# ── 8. the ablation removes the templates from both roles ─────────────────
def test_ablation_removes_templates():
    os.environ["ABLATE_CLASSIFIER"] = "1"
    try:
        check("ablation: template_parser_enabled() is False",
              qp.template_parser_enabled() is False)
        # The model omits every slot: with the templates gone there is nothing to
        # fill from, so the slots must stay empty rather than be guessed.
        stub = StubLLM({"qtype": "aggregation", "confidence": 0.9})
        spec, report = parse_question_semantic(AGG_Q, None, llm=stub)
        check("ablation: no slot is filled from question wording",
              not report["filled_by_rules"] and not spec.threshold
              and not spec.year,
              f"filled={report['filled_by_rules']} "
              f"threshold={spec.threshold!r} year={spec.year!r}")
        check("ablation: the report records that templates were off",
              report["templates_enabled"] is False,
              str(report["templates_enabled"]))
        # The classifier must read the same switch, not its own copy of it.
        from agents.classifier import template_classifier_enabled
        check("ablation: the classifier agrees with the parser",
              template_classifier_enabled() is False)
    finally:
        os.environ.pop("ABLATE_CLASSIFIER", None)


# ── 9. the parse reaches the run's state and its serialisation ────────────
def test_parse_report_reaches_the_state():
    state = AgentState(AGG_Q, QuerySpec(question=AGG_Q, qtype="lookup"), "q1")
    stub = StubLLM({"qtype": "aggregation", "sport": "Biathlon", "year": 2018,
                    "season": "Winter", "threshold": 73, "comparator": "gt",
                    "confidence": 0.91})
    spec, report = parse_question_with_llm(AGG_Q, None, llm=stub)
    state.parse_report = report
    state.adopt_spec(spec, "aggregation")

    check("state: the parse reclassified the run",
          state.qtype == "aggregation" and state.kind == "aggregation",
          f"{state.qtype}/{state.kind}")
    check("state: spec and qtype agree after adoption",
          state.spec is spec and state.spec.qtype == state.qtype)
    dump = state.to_dict()
    check("state: parse_report is serialised into the result",
          dump["parse_report"].get("method") == "llm",
          str(dump["parse_report"].get("method")))


def _is_json(obj) -> bool:
    import json
    try:
        json.dumps(obj)
        return True
    except (TypeError, ValueError):
        return False


def main() -> int:
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    for _, fn in tests:
        fn()
    print()
    print("=" * 74)
    if FAILURES:
        print(f" SEMANTIC PARSER CHECKS - {len(FAILURES)} FAILED: "
              + ", ".join(FAILURES))
    else:
        print(" SEMANTIC PARSER CHECKS - all passed")
    print("=" * 74)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
