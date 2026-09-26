"""Generate the generalisation evaluation sets (B: paraphrased, C: unseen).

Set B (``results/gen_paraphrased.jsonl``): every public question reworded into
frames that *cannot* match the deterministic keyword ladder in
``reasoning.query_parser.classify`` (no "how many", no "competitors", no
"immediately before", no "held at", no superlative keyword). Golds, qtype and
gold doc ids are carried over unchanged from the public set, so the only thing
under test is wording robustness.

Set C (``results/gen_unseen.jsonl``): 50 fresh questions (10 per type) authored
directly from the knowledge graph, with golds computed by *this script's own
independent code* (plain field reads and counting over EventNode rows - not the
system's solvers). Gold strings are verbatim corpus values, so the evaluator's
exact/containment layers grade them deterministically.

Both generators are seeded; rerunning reproduces the same files.
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kg.model import KnowledgeGraph  # noqa: E402
from reasoning.query_parser import (AGG_RE, LOOKUP_RE, MULTIHOP_RE,  # noqa: E402
                                    SUP_RE, TEMPORAL_RE)

PUBLIC_PATH = Path("questions-20260919T043312Z-1-001/questions/eval_public.jsonl")
KG_PATH = Path("results/knowledge_graph.json")
OUT_PARA = Path("results/gen_paraphrased.jsonl")
OUT_UNSEEN = Path("results/gen_unseen.jsonl")
SEED = 20260922

# â”€â”€ paraphrase frames â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Each frame deliberately avoids every keyword the rule ladder keys on, so a
# question built from it lands in the ladder's generic bucket and the semantic
# path has to carry the load. Entities stay verbatim (they are corpus strings).

PARA_AGG = [
    "Give the total count of {sport} events at the {year} {season} Olympics where the number of entrants exceeded {n}.",
    "From the supplied documents, count the {sport} events at the {year} {season} Games whose entry list topped {n} athletes.",
    "Tell me the count of {sport} events at the {year} {season} Olympics that drew over {n} entrants.",
]
PARA_SUP = [
    "Identify the {sport} event at the {year} {season} Olympics that attracted the biggest field of athletes.",
    "Among the {sport} events at the {year} {season} Olympics, name the one with the smallest entry list.",
    "From the corpus, name the {sport} event of the {year} {season} Games whose participation topped every other event of that sport.",
]
PARA_TEMP = [
    "In the {season} Games that preceded the {year} edition, which athlete claimed gold in {desc}?",
    "Name the gold medallist in {desc} at the {season} Olympics staged one edition prior to {year}.",
    "Which athlete took the top podium step in {desc} at the {season} Games immediately preceding the {year} edition?",
]
PARA_MH = [
    "Which athlete earned the top podium step in the competition staged at {venue} between {date}?",
    "Give the winner of the gold medal in the event at {venue} during {date} at the Games.",
    "Name the champion of the contest hosted at {venue} from {date}.",
]
PARA_LOOKUP = [
    "Give the count of participating countries in {title}.",
    "State the size of the national contingent in {title}.",
    "From the corpus, how large was the field of nations in {title}?",
]


def paraphrase(rows):
    out = []
    for r in rows:
        q = r["question"]
        qtype = r.get("qtype", "lookup")
        rng = random.Random(f"{SEED}-{r['qid']}")
        frame = None
        if qtype == "aggregation":
            m = AGG_RE.search(q)
            if m:
                frame = rng.choice(PARA_AGG).format(
                    sport=m.group("sport").strip(), year=m.group("year"),
                    season=m.group("season").title(), n=m.group("n"))
        elif qtype == "superlative":
            m = SUP_RE.search(q)
            if m:
                frame = rng.choice(PARA_SUP).format(
                    sport=m.group("sport").strip(), year=m.group("year"),
                    season=m.group("season").title())
        elif qtype == "temporal":
            m = TEMPORAL_RE.search(q)
            if m:
                frame = rng.choice(PARA_TEMP).format(
                    desc=m.group("desc").strip(),
                    season=m.group("season").title(), year=m.group("year"))
        elif qtype == "multi_hop":
            m = MULTIHOP_RE.search(q)
            if m:
                frame = rng.choice(PARA_MH).format(
                    venue=m.group("venue").strip().rstrip(" ?"),
                    date=m.group("date").strip().rstrip(" ?"))
        else:
            m = LOOKUP_RE.search(q)
            if m:
                frame = rng.choice(PARA_LOOKUP).format(
                    title=m.group("title").strip())
        out.append({
            "qid": f"para-{r['qid']}",
            "question": frame or q,   # unmatchable rows keep the original
            "qtype": qtype,
            "answer_named_in_question": r.get("answer_named_in_question", False),
            "guess_baseline": 0.0,
            "gold_doc_ids": r.get("gold_doc_ids", []),
            "answer_verified": r.get("answer_verified", False),
            "answer": r.get("answer", []),
            "source_qid": r["qid"],
        })
    return out


# PLACEHOLDER-COMMENT

# â”€â”€ unseen set: authored from the KG, golds from independent computation â”€â”€
def gen_unseen(kg, per_type: int = 10):
    rng = random.Random(SEED)
    events = list(kg.events.values())
    out = []

    groups = defaultdict(list)
    for e in events:
        if e.sport and e.year and e.season:
            groups[(e.sport, e.year, e.season)].append(e)
    usable = {k: v for k, v in groups.items() if len(v) >= 6}
    keys = sorted(usable)

    # â”€â”€ aggregation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    picked, seen = [], set()
    while len(picked) < per_type and len(seen) < len(keys):
        key = rng.choice(keys)
        if key in seen:
            continue
        seen.add(key)
        evs = usable[key]
        vals = sorted(e.competitors for e in evs if e.competitors is not None)
        if len(vals) < 6:
            continue
        t = vals[int(len(vals) * rng.uniform(0.5, 0.85))]
        want_gt = rng.random() < 0.5
        if want_gt:
            match = [e for e in evs
                     if e.competitors is not None and e.competitors > t]
            q = (f"Give the total count of {key[0]} events at the {key[1]} "
                 f"{key[2]} Olympics where the number of entrants exceeded {t}.")
        else:
            match = [e for e in evs
                     if e.competitors is not None and e.competitors < t]
            q = (f"Count the {key[0]} events at the {key[1]} {key[2]} Olympics "
                 f"whose entry lists stayed below {t} athletes.")
        if not match:
            continue
        picked.append({
            "qid": f"new-agg-{len(picked)+1:02d}", "question": q,
            "qtype": "aggregation", "answer_named_in_question": False,
            "guess_baseline": 0.0,
            "gold_doc_ids": sorted(e.doc_id for e in match),
            "answer_verified": True, "answer": [str(len(match))],
        })
    out.extend(picked)

    # â”€â”€ superlative: unique extreme only (ties make gold ambiguous) â”€â”€â”€â”€â”€â”€
    picked, seen = [], set()
    while len(picked) < per_type and len(seen) < len(keys):
        key = rng.choice(keys)
        if key in seen:
            continue
        seen.add(key)
        evs = [e for e in usable[key] if e.competitors is not None]
        if len(evs) < 6:
            continue
        want_max = rng.random() < 0.5
        extreme = (max if want_max else min)(e.competitors for e in evs)
        winners = [e for e in evs if e.competitors == extreme]
        if len(winners) != 1:
            continue
        w = winners[0]
        if want_max:
            q = (f"Identify the {key[0]} event at the {key[1]} {key[2]} "
                 f"Olympics that attracted the biggest field of athletes.")
        else:
            q = (f"Among the {key[0]} events at the {key[1]} {key[2]} Olympics, "
                 f"name the one with the smallest entry list.")
        picked.append({
            "qid": f"new-sup-{len(picked)+1:02d}", "question": q,
            "qtype": "superlative", "answer_named_in_question": False,
            "guess_baseline": 0.0, "gold_doc_ids": [w.doc_id],
            "answer_verified": True, "answer": [w.title],
        })
    out.extend(picked)

    # â”€â”€ temporal: prev-chain walk, gold = prev event's gold field â”€â”€â”€â”€â”€â”€â”€â”€
    picked = []
    pool = [e for e in events if e.prev_doc_id and e.year and e.season
            and e.sport and e.event_name]
    rng.shuffle(pool)
    for e in pool:
        if len(picked) >= per_type:
            break
        prev = kg.events.get(e.prev_doc_id)
        if prev is None or not prev.gold:
            continue
        q = (f"Name the gold medallist in {e.sport} {e.event_name} at the "
             f"{e.season} Olympics staged one edition prior to {e.year}.")
        picked.append({
            "qid": f"new-tmp-{len(picked)+1:02d}", "question": q,
            "qtype": "temporal", "answer_named_in_question": False,
            "guess_baseline": 0.0, "gold_doc_ids": [prev.doc_id],
            "answer_verified": True, "answer": [prev.gold],
        })
    out.extend(picked)

    # â”€â”€ multi_hop: (venue, date) resolving to exactly one event â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    picked = []
    vd = defaultdict(list)
    for e in events:
        if e.venue and e.gold and (e.date_raw or e.dates_raw):
            vd[(e.venue, e.date_raw or e.dates_raw)].append(e)
    unique = {k: v[0] for k, v in vd.items() if len(v) == 1}
    mh_keys = sorted(unique)
    rng.shuffle(mh_keys)
    for k in mh_keys:
        if len(picked) >= per_type:
            break
        e = unique[k]
        q = (f"Which athlete earned the top podium step in the competition "
             f"staged at {e.venue} between {e.date_raw or e.dates_raw}?")
        picked.append({
            "qid": f"new-mh-{len(picked)+1:02d}", "question": q,
            "qtype": "multi_hop", "answer_named_in_question": False,
            "guess_baseline": 0.0, "gold_doc_ids": [e.doc_id],
            "answer_verified": True, "answer": [e.gold],
        })
    out.extend(picked)

    # â”€â”€ lookup: nations / venue / bronze, verbatim field golds â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    picked = []
    pool = [e for e in events if e.title]
    rng.shuffle(pool)
    kinds = ["nations", "venue", "bronze"]
    for i, e in enumerate(pool):
        if len(picked) >= per_type:
            break
        kind = kinds[i % len(kinds)]
        if kind == "nations" and e.nations:
            q = f"Give the count of participating countries in {e.title}."
            gold = str(e.nations)
        elif kind == "venue" and e.venue:
            q = f"State the arena in which {e.title} was staged."
            gold = e.venue
        elif kind == "bronze" and e.bronze:
            q = f"Name the bronze medallist of {e.title}."
            gold = e.bronze
        else:
            continue
        picked.append({
            "qid": f"new-lk-{len(picked)+1:02d}", "question": q,
            "qtype": "lookup", "answer_named_in_question": False,
            "guess_baseline": 0.0, "gold_doc_ids": [e.doc_id],
            "answer_verified": True, "answer": [gold],
        })
    out.extend(picked)
    return out


def _write(path: Path, rows) -> None:
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8")


def main() -> None:
    rows = [json.loads(line) for line in
            PUBLIC_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    para = paraphrase(rows)
    _write(OUT_PARA, para)
    print(f"paraphrased: {len(para)} -> {OUT_PARA}")

    kg = KnowledgeGraph.from_dict(json.loads(KG_PATH.read_text(encoding="utf-8")))
    unseen = gen_unseen(kg)
    _write(OUT_UNSEEN, unseen)
    import collections
    print(f"unseen: {len(unseen)} -> {OUT_UNSEEN} "
          f"({dict(collections.Counter(r['qtype'] for r in unseen))})")
    for r in unseen[:2] + unseen[-2:]:
        print(f"  {r['qid']} [{r['qtype']}] {r['question'][:84]}")
        print(f"      gold={r['answer']} docs={len(r['gold_doc_ids'])}")


if __name__ == "__main__":
    main()

