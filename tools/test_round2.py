"""Round 2 checks: conflicting versions, supersession and residual uncertainty.

Round 2 asks a system to notice that two documents state the same fact
differently, decide what supersedes what and which source is authoritative, and
carry the uncertainty that is left. These checks cover each link of that chain -
the version date on a fact, the resolver's precedence, the agent-side audit that
consumes it, the evidence gap it raises and the tool the model can call - so a
break shows up here rather than as a quietly wrong answer in a benchmark run.

    python tools/test_round2.py
"""
import datetime
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from config import config  # noqa: E402
from kg.builder import _iso_version_date, load_or_build  # noqa: E402
from retrieval import load_index  # noqa: E402
from reasoning.conflicts import resolve  # noqa: E402
from agents.gap_detector import GapDetector  # noqa: E402
from agents.gaps import Gap  # noqa: E402
from agents.orchestrator import OrchestratorAgent  # noqa: E402
from agents.tools import GraphTools  # noqa: E402
from utils.thresholds import thresholds  # noqa: E402

PASS, FAIL = [], []


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASS if ok else FAIL).append(name)
    print(f"{'PASS' if ok else 'FAIL'} {name}" + (f" - {detail}" if detail else ""))


class _FakeState:
    """The slice of AgentState the conflict audit actually reads."""

    def __init__(self, kind, documents, candidates, answer, confidence=0.72,
                 adjudication=None, citations=None):
        self.kind = kind
        self.documents = documents
        self.candidates = list(candidates)
        self.answer = answer
        self.confidence = confidence
        self.adjudication = adjudication or {"changed": False, "agree": True,
                                             "candidate": "", "reason": ""}
        self.citations = list(citations or [])
        self.missing_info = []
        self.proposed_gaps = []
        self.fact_conflicts = {}
        self.uncertainty = 0.0

    def note_gap(self, gap) -> None:
        if gap and gap not in self.missing_info:
            self.missing_info.append(gap)


def _event(doc_id, field, value, **kw):
    attrs = {field: value, "doc_id": doc_id, "source_type": "infobox",
             "fact_version_date": kw.pop("fact_version_date", ""),
             "year": kw.pop("year", None)}
    return SimpleNamespace(**attrs, **kw)


# The audit is a pure function of (state, thresholds) and touches no
# collaborator, so it is exercised on an instance built without __init__.
stub_self = object.__new__(OrchestratorAgent)


def audit(state):
    return OrchestratorAgent._audit_conflicts(stub_self, state)


# ── 1. every fact carries an "as of" date ───────────────────────────────
check("clear date -> ISO version date",
      _iso_version_date("28 July 2012", 2012) == "2012-07-28",
      _iso_version_date("28 July 2012", 2012))
check("month-first date -> ISO version date",
      _iso_version_date("July 28, 2012", 2012) == "2012-07-28",
      _iso_version_date("July 28, 2012", 2012))
check("undated fact falls back to the Games year",
      _iso_version_date("", 2012) == "2012",
      _iso_version_date("", 2012))
check("an unknown year stays empty rather than guessing",
      _iso_version_date("", 0) == "", repr(_iso_version_date("", 0)))
check("unparseable date keeps the year, invents nothing",
      _iso_version_date("2012 Summer", 2012) == "2012",
      _iso_version_date("2012 Summer", 2012))
check("an off-edition date is discarded for the edition year",
      _iso_version_date("10 August to 15 August 2012", 2008) == "2008",
      _iso_version_date("10 August to 15 August 2012", 2008))
check("a postponed edition keeps its real date",
      _iso_version_date("30 July 2021", 2020) == "2021-07-30",
      _iso_version_date("30 July 2021", 2020))

# ── 2. the resolver separates decisive rules from contested facts ───────
floor = thresholds().conflict_accept_confidence
decisive = resolve("gold", [
    {"value": "Athlete A", "doc_id": "d1", "year": 1992},
    {"value": "Athlete B", "doc_id": "d2", "year": 2012,
     "text": "upgraded after the winner was disqualified"},
])
contested = resolve("venue", [
    {"value": "London Velopark", "doc_id": "d1"},
    {"value": "Lee Valley VeloPark", "doc_id": "d2"},
])
check("an explicit correction is decisive", decisive.confidence >= floor,
      f"rule={decisive.rule} confidence={decisive.confidence}")
check("a bare 1-of-2 split is not decisive", contested.confidence < floor,
      f"rule={contested.rule} confidence={contested.confidence}")

# ── 3. the auditor compares versions of one fact ───────────────────────
def renamed_venue(new_year=2012, old_year=2008):
    """Two pages, two names for one building, one of them cited."""
    return {"d_new": _event("d_new", "venue", "Lee Valley VeloPark", year=new_year,
                            fact_version_date=f"{new_year}-08-01" if new_year else ""),
            "d_old": _event("d_old", "venue", "London Velopark", year=old_year,
                            fact_version_date=f"{old_year}-08-01" if old_year else "")}


docs = renamed_venue()
state = _FakeState("lookup", docs, ["d_new", "d_old"], "Lee Valley VeloPark",
                   confidence=0.92, citations=["d_new", "d_old"])
summary = audit(state)
check("sources that state the fact differently are adjudicated, not merged",
      summary.get("rule") == "recency" and
      [s["value"] for s in summary.get("superseded", [])] == ["London Velopark"],
      f"rule={summary.get('rule')} superseded={summary.get('superseded')}")
check("a contested fact caps the run's confidence",
      state.confidence == 0.9 and summary.get("confidence_after") == 0.9,
      f"confidence={state.confidence}")
check("a decisive rule leaves no evidence gap",
      not state.missing_info and summary.get("decisive") is True,
      f"decisive={summary.get('decisive')}")

docs = renamed_venue(new_year=None, old_year=None)
state = _FakeState("lookup", docs, ["d_new", "d_old"], "Lee Valley VeloPark",
                   confidence=0.92, citations=["d_new", "d_old"])
summary = audit(state)
check("an undated disagreement stays undecided",
      summary.get("rule") == "majority" and summary.get("decisive") is False,
      f"rule={summary.get('rule')} confidence={summary.get('confidence')}")
check("an unsettled conflict is recorded as an evidence gap",
      Gap.CONFLICTING_EVIDENCE.value in state.missing_info,
      f"missing_info={state.missing_info}")
check("an unsettled conflict cannot raise confidence",
      state.confidence < 0.92, f"confidence={state.confidence}")

# A different edition is a different building, not a renamed one: comparing
# across editions turned 415 of 535 event groups into conflicts, so a document
# the answer does not cite never contributes a version.
state = _FakeState("lookup", renamed_venue(), ["d_new", "d_old"],
                   "Lee Valley VeloPark", confidence=0.92, citations=["d_new"])
check("a document the answer does not cite supplies no version",
      state.documents["d_old"].venue == "London Velopark" and audit(state) == {}
      and not state.missing_info and state.confidence == 0.92)

# A derived answer is not a version of one fact: the candidate set of a count
# holds a different value in every document by construction, so auditing it
# would label every aggregation "majority, undecided" and tax a correct answer.
derived = {"a": _event("a", "competitors", 87), "b": _event("b", "competitors", 30)}
state = _FakeState("aggregation", derived, ["a", "b"], "5", confidence=0.6,
                   citations=["a", "b"])
check("a derived count is not audited as a conflict",
      audit(state) == {} and not state.missing_info and state.confidence == 0.6,
      f"confidence={state.confidence}")
state = _FakeState("superlative", derived, ["a", "b"], "a", confidence=0.65,
                   citations=["a"])
check("a derived extreme is not audited as a conflict",
      audit(state) == {} and not state.missing_info and state.confidence == 0.65)

agree_docs = {"d1": _event("d1", "nations", 204), "d2": _event("d2", "nations", 204)}
state = _FakeState("lookup", agree_docs, ["d1", "d2"], "204", confidence=0.9,
                   citations=["d1", "d2"])
summary = audit(state)
check("a fact every source agrees on is not a conflict",
      summary.get("rule", "no_conflict") == "no_conflict"
      and not state.missing_info,
      f"rule={summary.get('rule')}")
check("an answer that matches no field is left unaudited",
      audit(_FakeState("lookup", {"d1": _event("d1", "gold", "Athlete A")}, ["d1"],
                       "a claim no field states", citations=["d1"])) == {})
check("a run with no citations is left unaudited",
      audit(_FakeState("lookup", agree_docs, ["d1"], "204")) == {})

# ── 4. the gap has a real recovery, not a dead end ─────────────────────
detector = GapDetector()
action = detector._action_for(None, "lookup", Gap.CONFLICTING_EVIDENCE, None)
check("CONFLICTING_EVIDENCE has a recovery action",
      action is not None and action.recoverable and action.operation == "vector_search",
      f"{action.operation if action else 'none'} - {action.reason if action else ''}")

# ── 5. the corpus itself carries the version signal ────────────────────
kg = load_or_build(config.benchmark.corpus_path)
events = list(kg.events.values())
badged = [e for e in events if e.source_type == "infobox"]
check("every event fact records its source type",
      bool(events) and len(badged) == len(events),
      f"{len(badged)}/{len(events)} infobox-sourced")
dated = [e for e in events if e.year]
with_version = [e for e in dated if e.fact_version_date]
check("every dated event carries an 'as of' version date",
      bool(dated) and len(with_version) == len(dated),
      f"{len(with_version)}/{len(dated)} dated events")
iso = [e.fact_version_date for e in events if len(e.fact_version_date) == 10]
parsed = 0
for value in iso:
    try:
        datetime.date.fromisoformat(value)
        parsed += 1
    except ValueError:
        pass
check("full version dates are real ISO dates",
      bool(iso) and parsed == len(iso), f"{parsed}/{len(iso)} parse as ISO")
aligned = [e for e in events if len(e.fact_version_date) == 10 and e.year
           and e.fact_version_date[:4] == str(e.year)]
postponed = [e for e in events if len(e.fact_version_date) == 10 and e.year
             and e.fact_version_date[:4] == str(e.year + 1)]
check("version dates agree with the edition year, bar postponed editions",
      bool(iso) and len(aligned) + len(postponed) == len(iso),
      f"{len(aligned)}/{len(iso)} aligned, {len(postponed)} one year later "
      f"(a 2020 edition held in 2021)")

# ── 6. the model-facing tool returns the verdict, not just a value ─────
index = load_index(config.benchmark.corpus_path, kg,
                   vector_backend=config.benchmark.vector_backend)
tools = GraphTools(kg, index)
verdict = tools.detect_conflicts(
    field="olympic_record",
    values=["9.85", "9.69", "9.63"],
    years=[1988, 2008, 2012],
    sources=["infobox", "result_page", "official_report"])
check("detect_conflicts resolves with a named rule",
      verdict.get("resolved") == "9.63" and verdict.get("rule") == "recency",
      f"rule={verdict.get('rule')} resolved={verdict.get('resolved')}")
check("detect_conflicts reports the uncertainty it leaves",
      verdict.get("uncertainty") == round(1.0 - verdict.get("confidence", 0.0), 3),
      f"uncertainty={verdict.get('uncertainty')}")
check("detect_conflicts lists what it superseded",
      len(verdict.get("superseded", [])) == 2,
      f"superseded={[s.get('value') for s in verdict.get('superseded', [])]}")

result = tools.execute("detect_conflicts", {
    "field": "venue",
    "values": ["London Velopark", "Lee Valley VeloPark"],
    "doc_ids": ["a", "b"]})
check("execute() routes detect_conflicts and its summary names the rule",
      "majority" in tools.calls[-1].get("summary", "") and
      result.get("rule") == "majority" and result.get("had_conflict") is True,
      tools.calls[-1].get("summary", "")[:80])

print()
print("=" * 70)
print(f" ROUND 2 CHECKS - passed {len(PASS)}, failed {len(FAIL)}")
for name in FAIL:
    print(f"   FAILED: {name}")
print("=" * 70)

sys.exit(1 if FAIL else 0)

