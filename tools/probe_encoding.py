"""Probe encoding repair and infobox/date parsing feasibility."""
import json
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

CORPUS = "corpus-20260919T043338Z-1-001/corpus/corpus.jsonl"
QP = "questions-20260919T043312Z-1-001/questions/eval_public.jsonl"


def repair(s: str) -> str:
    """Undo latin1-as-utf8 mojibake when detectable."""
    if not s:
        return s
    try:
        fixed = s.encode("latin-1", errors="strict").decode("utf-8", errors="strict")
        return fixed
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s


with open(CORPUS, "r", encoding="utf-8") as fh:
    first = json.loads(fh.readline())
print("RAW   title:", first["title"])
print("FIXED title:", repair(first["title"]))
i = first["text"].find("gold:")
print("RAW   gold :", first["text"][i:i + 40])
print("FIXED gold :", repair(first["text"][i:i + 40]))

with open(QP, "r", encoding="utf-8") as fh:
    q5 = json.loads(fh.readlines()[4])
print("RAW   q gold:", q5["answer"])
print("FIXED q gold:", [repair(a) for a in q5["answer"]])
