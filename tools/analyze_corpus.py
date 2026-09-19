"""One-off corpus analysis: understand infobox field coverage and title patterns.

Run:  python tools/analyze_corpus.py
"""
import json
import re
import sys
from collections import Counter

CORPUS = sys.argv[1] if len(sys.argv) > 1 else \
    "corpus-20260919T043338Z-1-001/corpus/corpus.jsonl"

FIELD_RE = re.compile(r"^\s{0,4}([a-z_]+)\s*:\s?(.*)$")


def infobox(text: str) -> dict:
    out = {}
    if not text.startswith("[Infobox"):
        return out
    for line in text.split("\n"):
        if line.startswith("["):
            if not line.startswith("[Infobox"):
                break
            continue
        if not line.strip():
            break
        m = FIELD_RE.match(line)
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


def main():
    keys = Counter()
    n = 0
    with_infobox = 0
    values_examples = {}
    titles = []
    with open(CORPUS, "r", encoding="utf-8") as fh:
        for line in fh:
            doc = json.loads(line)
            n += 1
            t = doc.get("text", "")
            ib = infobox(t)
            if ib:
                with_infobox += 1
            for k, v in ib.items():
                keys[k] += 1
                values_examples.setdefault(k, [])
                if len(values_examples[k]) < 3:
                    values_examples[k].append(v[:80])
            titles.append(doc["title"])

    print(f"docs={n} with_infobox={with_infobox}")
    print("\n=== FIELD COVERAGE ===")
    for k, c in keys.most_common(80):
        print(f"{k:24s} {c:6d}  e.g. {values_examples[k]}")

    print("\n=== TITLE PATTERNS ===")
    pat = Counter()
    for t in titles:
        if " at the " in t and "Olympics" in t:
            pat["<event> at the <games> Olympics"] += 1
        elif " at the " in t:
            pat["<x> at the <y>"] += 1
        elif "Olympics" in t:
            pat["Olympics (other)"] += 1
        else:
            pat["other"] += 1
    print(pat)
    print("\nsample Olympic titles:")
    shown = 0
    for t in titles:
        if "Olympics" in t:
            print("  ", t)
            shown += 1
            if shown >= 20:
                break
    print("\nsample non-Olympic titles:")
    shown = 0
    for t in titles:
        if "Olympics" not in t:
            print("  ", t)
            shown += 1
            if shown >= 15:
                break


if __name__ == "__main__":
    main()