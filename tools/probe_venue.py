"""Probe venue strings for the pub-099 ambiguity and check temporal desc noise."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
from kg.builder import load_or_build
from kg.textutil import normalize, similarity

kg = load_or_build("corpus-20260919T043338Z-1-001/corpus/corpus.jsonl")
kg._rebuild_indexes()

print("venues containing 'laura':")
for v in sorted(kg.venues):
    if "laura" in normalize(v):
        print(f"  {v!r}  -> {len(kg.venue_index[normalize(v)])} events")

print("\nevents on 22 February 2014:")
for e in kg.events.values():
    if e.date_raw == "22 February 2014":
        print(f"  {e.title!r} venue={e.venue!r} gold={e.gold!r}")

print("\nnormalized venue comparison:")
q = "Laura Biathlon & Ski Complex"
print("  q norm:", normalize(q))
for k in kg.venue_index:
    if "laura" in k:
        print(f"  key={k!r} sim={similarity(q, k):.3f}")
