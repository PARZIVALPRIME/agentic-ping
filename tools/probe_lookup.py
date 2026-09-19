"""Probe one lookup question: title resolution + infobox fields."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from config import config
from kg.builder import load_or_build
from kg.textutil import normalize
from reasoning.query_parser import parse_question
from reasoning.solvers import StructuredSolver

CORPUS = config.benchmark.corpus_path
QUESTIONS = config.benchmark.public_questions_path

kg = load_or_build(CORPUS)
solver = StructuredSolver(kg)

qs = []
with open(QUESTIONS, "r", encoding="utf-8") as fh:
    for line in fh:
        if line.strip():
            q = json.loads(line)
            if q["qtype"] == "lookup":
                qs.append(q)

for q in qs[:6]:
    spec = parse_question(q["question"], kg)
    print("=" * 90)
    print("Q:", q["question"])
    print("gold:", q["answer"], "| gold_doc_ids:", q.get("gold_doc_ids"))
    print("target_title:", repr(spec.target_title))
    key = normalize(spec.target_title)
    doc_id = kg.title_index.get(key)
    print("exact title_index hit:", doc_id)
    if doc_id is None:
        from kg.textutil import best_match
        match, score = best_match(spec.target_title, list(kg.title_index.keys()))
        print("best_match:", repr(match), score)
        if match:
            doc_id = kg.title_index[match]
    if doc_id:
        node = kg.events[doc_id]
        print("resolved:", node.title, "| nations:", node.nations,
              "| competitors:", node.competitors, "| venue:", node.venue)
    res = solver.solve_lookup(spec)
    print("solver ->", repr(res.answer), "| unresolved:", res.unresolved)