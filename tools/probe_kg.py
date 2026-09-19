"""Inspect specific gold-referenced docs to verify parsing assumptions."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
from kg.builder import build_kg

kg = build_kg("corpus-20260919T043338Z-1-001/corpus/corpus.jsonl")

print("\n--- temporal pub-002 gold docs Q1050909, Q26233122 ---")
for did in ["Q1050909", "Q26233122"]:
    e = kg.events.get(did)
    print(did, "|", e.title if e else "MISSING")
    if e:
        print("   sport=%r yr=%s season=%r event=%r prev=%r next=%r gold=%r"
              % (e.sport, e.year, e.season, e.event_name, e.prev, e.next, e.gold))

print("\n--- multi_hop gold docs ---")
for did in ["Q25239316", "Q2392979", "Q26233122"]:
    e = kg.events.get(did)
    if e:
        print(did, "|", e.title)
        print("   venue=%r date=%r dates=%r games=%r" % (e.venue, e.date_raw, e.dates_raw, e.games_label))

print("\n--- lookup pub-009 gold Q... titles containing 'RS:X' ---")
hits = [e for e in kg.events.values() if "RS:X" in e.title]
for e in hits[:4]:
    print("  ", e.title, "| nations=", e.nations)

print("\n--- nordic combined / speed skating event names sample ---")
for sp in ["Nordic combined", "Speed skating", "Fencing", "Wrestling"]:
    evs = kg.events_for_sport(sp)[:6]
    for e in evs:
        print(f"  [{sp}] {e.year} {e.season} event={e.event_name!r} gold={e.gold!r}")
    print()
