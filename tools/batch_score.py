import argparse
import json
import os
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.audit_hidden import diff_answers  # noqa: E402


def accuracy(entries, pipeline):
    scored = [r for r in entries
              if (r.get("pipelines", {}).get(pipeline, {}).get("evaluation") or {}).get("is_correct") is not None]
    if not scored:
        return None
    return sum(1 for r in scored if r["pipelines"][pipeline]["evaluation"]["is_correct"]) / len(scored)


def record_totals(rec):
    calls = toks = 0
    for p in rec.get("pipelines", {}).values():
        calls += p.get("llm_calls", 0) or 0
        toks += p.get("total_tokens", 0) or 0
    return calls, toks


def pipeline_names(entries):
    """Every pipeline name present in the records, in first-seen order.

    Derived rather than hardcoded so the ablation arm (or any future arm) is
    reported without touching this file.
    """
    names = []
    for rec in entries:
        for name in (rec.get("pipelines") or {}):
            if name not in names:
                names.append(name)
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    ap.add_argument("--cumulative", required=True)
    ap.add_argument("--set", dest="qset", choices=["public", "hidden"], required=True)
    ap.add_argument("--baseline", default=None, help="previous full-run file for hidden answer diffs")
    ap.add_argument("--ledger", default="results/batch/ledger.jsonl")
    args = ap.parse_args()

    batch = json.load(open(args.batch, encoding="utf-8"))
    cum = json.load(open(args.cumulative, encoding="utf-8")) if os.path.exists(args.cumulative) else []

    # merge batch into cumulative (latest record per qid)
    merged = {r["qid"]: r for r in cum}
    for r in batch:
        merged[r["qid"]] = r
    merged = list(merged.values())
    with open(args.cumulative, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=1)

    lines = [f"=== batch {os.path.basename(args.batch)} ({len(batch)} questions, set={args.qset}) ==="]
    pipenames = pipeline_names(batch) or pipeline_names(merged)
    if args.qset == "public":
        for p in pipenames:
            b = accuracy(batch, p)
            c = accuracy(merged, p)
            fb = f"{b*100:.0f}%" if b is not None else "n/a"
            fc = f"{c*100:.0f}%" if c is not None else "n/a"
            lines.append(f"  {p:<18} batch {fb:>5}  cumulative {fc:>5}")
    else:
        if args.baseline and os.path.exists(args.baseline):
            base = json.load(open(args.baseline, encoding="utf-8"))
            from tools.audit_hidden import agreement
            agree = agreement(batch, base)
            for p in pipenames:
                counts = agree.get(p) or {}
                same = counts.get("same", 0)
                changed = counts.get("changed", 0)
                nob = counts.get("no_baseline", 0)
                extra = f" no_baseline={nob}" if nob else ""
                lines.append(f"  {p:<18} vs baseline: same={same} changed={changed}{extra}")
                if changed:
                    diff_answers(batch, base, p)  # prints question/old/new triples
        ag = [r.get("pipelines", {}).get("Agentic GraphRAG", {}) for r in batch]
        empty = sum(1 for a in ag if not (a.get("answer") or "").strip())
        lines.append(f"  empty agentic answers: {empty}/{len(batch)}")

    calls = toks = 0
    for r in batch:
        c, t = record_totals(r)
        calls += c
        toks += t
    lines.append(f"  llm calls: {calls}   tokens: {toks:,}")
    bad = [r["qid"] for r in batch
           if all((p.get("llm_calls") or 0) == 0 for p in r.get("pipelines", {}).values())]
    if bad:
        lines.append(f"  !! deterministic (no LLM) records: {bad}")

    # ledger line
    os.makedirs(os.path.dirname(args.ledger), exist_ok=True)
    entry = {"batch": os.path.basename(args.batch), "set": args.qset, "n": len(batch),
             "calls": calls, "tokens": toks}
    if args.qset == "public":
        entry["accuracy"] = {p: accuracy(batch, p) for p in pipenames}
    with open(args.ledger, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
