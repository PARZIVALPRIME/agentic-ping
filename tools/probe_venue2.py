"""Check the 2014 cross-country women's 30km venue/date fields."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
from kg.builder import load_or_build

kg = load_or_build("corpus-20260919T043338Z-1-001/corpus/corpus.jsonl")

for e in kg.events.values():
    if e.year == 2014 and "30 kilometre" in e.event_name and "Cross-country" in e.title:
        print(f"{e.title!r}\n  venue={e.venue!r}\n  date={e.date_raw!r}\n  dates={e.dates_raw!r}\n")