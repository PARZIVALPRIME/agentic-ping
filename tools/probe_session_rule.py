"""Probe: validate corpus-derived checks for the multi-hop session rule.

Two questions whose hidden-set answer changed between the v2 snapshot and the
structured-aware run are examined against the corpus (eval-040 venue/date
multi-hop, eval-046 temporal light-flyweight boxing), and the exact-printed-date
rule used by ``agents.orchestrator._exact_date_event`` is scored against the
*public* multi-hop questions, where the gold answers are known.

Usage:
    python tools/probe_session_rule.py
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.tools import GraphTools  # noqa: E402
from agents.orchestrator import _exact_date_event  # noqa: E402
from agents.lookup_resolver import LookupResolver  # noqa: E402
from kg.backend import open_graph  # noqa: E402
from kg.textutil import normalize  # noqa: E402
from reasoning.query_parser import parse_question  # noqa: E402
from retrieval.index import CorpusIndex  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def main() -> int:
    from config import config

    kg = open_graph(config.benchmark.corpus_path)
    index = None
    try:
        index = CorpusIndex().build()
    except Exception as exc:  # venue/date resolution works from the KG alone
        print(f"# index unavailable ({exc.__class__.__name__}); KG-only probe")
    tools = GraphTools(kg, index)
    for venue, date, year in (("Stadium Australia", "30 September 2000", 2000),):
        out: Dict[str, Any] = tools.resolve_event(venue=venue, date=date, year=year)
        print(f"resolve_event({venue!r}, {date!r}) -> {json.dumps(out, ensure_ascii=False)[:900]}")
    # every boxing light flyweight event with its gold medalist, by year
    rows = []
    for ev in kg.events.values():
        text = f"{ev.title}".lower()
        sport = str(getattr(ev, "sport", "") or "").lower()
        if "boxing" in sport or "boxing" in text:
            if "light flyweight" in text or "light-flyweight" in text:
                year = getattr(ev, "year", 0) or 0
                gold = ""
                for field in ("gold", "gold_medalist", "winner"):
                    value = getattr(ev, field, None)
                    if value:
                        gold = value if isinstance(value, str) else ", ".join(map(str, value))
                        break
                rows.append((year, ev.title, gold))
    for year, title, gold in sorted(rows):
        print(f"  {year}  {title[:70]:<70s} gold={gold}")
    print()
    print("# 2000 Summer Olympics events at Stadium Australia with '30 September'")
    for ev in kg.events.values():
        title = str(getattr(ev, "title", "") or "")
        venue = str(getattr(ev, "venue", "") or "")
        date = str(getattr(ev, "date", "") or "")
        if "2000 Summer Olympics" in title and "stadium australia" in venue.lower() \
                and "30 september" in date.lower():
            gold = getattr(ev, "gold", "") or ""
            if not isinstance(gold, str):
                gold = ", ".join(map(str, gold))
            print(f"  {title[:75]:<75s} | {date[:45]:<45s} | gold={gold}")

    # ── what the resolver does with the two ambiguous questions ───────────
    from agents.lookup_resolver import LookupResolver
    from reasoning.query_parser import parse_question

    questions_path = ("questions-20260919T043312Z-1-001/questions/eval_hidden.jsonl")
    wanted = {"eval-040", "eval-046"}
    parser_kg = kg
    with open(questions_path, "r", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row.get("qid") not in wanted:
                continue
            spec = parse_question(row["question"], parser_kg)
            print(f"\n## {row['qid']} :: {row['question']}")
            print(f"   parsed: qtype={spec.qtype} venue={spec.venue!r} "
                  f"date_text={spec.date_text!r} year={spec.year}")
            result = LookupResolver(kg).resolve_venue_date(spec, top_n=5)
            print(f"   matched: {getattr(result.get('matched'), 'title', None)!r} "
                  f"score={result.get('score')}")
            for cand in result.get("candidates") or []:
                print(f"     {cand['date_score']:.3f} aff={cand['venue_affinity']:.2f} "
                      f"{cand['title'][:62]:<62s} date={str(cand['date'])[:38]:<38s} "
                      f"gold={cand['gold']}")
            for simulated in ("Derartu Tulu", "Nouria Mérah-Benida", "", "someone else"):
                detail = _exact_date_event(LookupResolver(kg), spec, simulated)
                state = "FIRES" if detail else "silent"
                print(f"   [{state}] submitted={simulated!r} -> {detail[:110]}")

    # ── precision of the rule on the public multi-hop questions (gold known) ──
    print("\n# public multi-hop: agreement of the exact-date event's gold with gold")
    resolver = LookupResolver(kg)
    fired = agreed = 0
    for row in _public_questions():
        spec = parse_question(row["question"], kg)
        if spec.qtype != "multi_hop":
            continue
        detail = _exact_date_event(resolver, spec, "definitely-not-the-answer")
        if not detail:
            continue
        fired += 1
        golds = {normalize(a) for a in (row.get("answer") or [])}
        top_gold = normalize(_exact_gold(resolver, spec))
        ok = top_gold in golds
        agreed += int(ok)
        mark = "OK " if ok else "MISS"
        print(f"  {mark} {row.get('qid')}: exact-date gold={_exact_gold(resolver, spec)!r} "
              f"gold={sorted(golds)}")
    print(f"  rule fired on {fired} public multi-hop question(s); "
          f"{agreed} agreed with gold")
    return 0


def _exact_gold(resolver, spec) -> str:
    result = resolver.resolve_venue_date(spec, top_n=8)
    exact = [c for c in (result.get("candidates") or [])
             if float(c.get("date_score") or 0.0) >= 0.999 and str(c.get("gold") or "")]
    return str(exact[0].get("gold") or "") if len(exact) == 1 else ""


def _public_questions():
    import json as _json

    path = "questions-20260919T043312Z-1-001/questions/eval_public.jsonl"
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield _json.loads(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
