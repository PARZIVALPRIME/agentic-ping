"""Verify the numbers quoted in docs/ against the real result files.
Documentation that drifts from the data is worse than none.

Checks:
  * docs/ablation_study.md   vs results/ablation_summary.json
  * docs/baseline_ceiling.md vs results/baseline_sweep.json
  * docs/submission_writeup.md vs results/metrics_summary.json,
    results/public_results.json and results/hidden_submission.json
"""

from __future__ import annotations

import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARY = os.path.join(ROOT, "results", "ablation_summary.json")
SWEEP = os.path.join(ROOT, "results", "baseline_sweep.json")
WRITEUP = os.path.join(ROOT, "docs", "submission_writeup.md")
LIVE_SUMMARY = os.path.join(ROOT, "results", "metrics_summary.json")
PUBLIC_RESULTS = os.path.join(ROOT, "results", "public_results.json")
HIDDEN_SUBMISSION = os.path.join(ROOT, "results", "hidden_submission.json")

# What docs/ablation_study.md claims: name -> (overall, aggregation, superlative)
CLAIMED = {
    "RAG": (0.39, 0.00, 0.00),
    "GraphRAG": (0.61, 0.33, 0.50),
    "Agentic GraphRAG": (1.00, 1.00, 1.00),
    "Router": (1.00, 1.00, 1.00),
    "Agentic-NoPlanner": (0.00, 0.00, 0.00),
    "Agentic-NoEnumeration": (0.86, 0.38, 0.90),
    "Agentic-NoVerifier": (1.00, 1.00, 1.00),
    "Agentic-NoGapDetector": (1.00, 1.00, 1.00),
}

# What docs/baseline_ceiling.md claims: (pipeline, k) -> (accuracy, ctx_tokens)
CLAIMED_SWEEP = {
    ("RAG", 5): (0.39, 251),
    ("RAG", 160): (0.91, 9680),
    ("GraphRAG", 20): (0.64, 1768),
    ("GraphRAG", 40): (0.61, 3511),
    ("GraphRAG", 160): (0.74, 13785),
}

TOL = 0.015
TOK_TOL = 0.02  # 2% relative on token counts


def _acc(node) -> float:
    if not isinstance(node, dict):
        return float("nan")
    for key in ("accuracy", "exact_match", "acc"):
        if key in node:
            try:
                return float(node[key])
            except (TypeError, ValueError):
                pass
    correct, total = node.get("correct"), node.get("total")
    if isinstance(correct, (int, float)) and total:
        return correct / total
    return float("nan")


def _check_sweep() -> int:
    """docs/baseline_ceiling.md vs results/baseline_sweep.json."""
    if not os.path.exists(SWEEP):
        print(f"\nskip: no {SWEEP} yet "
              f"(run: python tools/baseline_sweep.py --ks 5,10,20,40,80,160)")
        return 0

    data = json.load(open(SWEEP, encoding="utf-8"))
    results = data.get("results", {})

    print(f"\n{'baseline sweep':<26} {'accuracy':>20} {'ctx tokens':>22}")
    print("-" * 84)

    failures = 0
    for (name, k), (c_acc, c_tok) in sorted(CLAIMED_SWEEP.items()):
        row = (results.get(name) or {}).get(str(k)) or (results.get(name) or {}).get(k)
        if not row:
            print(f"{name + f' k={k}':<26} {'MISSING from sweep':>20}")
            failures += 1
            continue
        a_acc = float(row.get("accuracy", float("nan")))
        a_tok = float(row.get("avg_context_tokens", float("nan")))
        acc_ok = abs(a_acc - c_acc) <= TOL
        tok_ok = abs(a_tok - c_tok) <= max(1.0, TOK_TOL * c_tok)
        failures += 0 if (acc_ok and tok_ok) else 1
        print(f"{name + f' k={k}':<26} "
              f"{f'{a_acc:.2f} vs {c_acc:.2f}' + ('' if acc_ok else ' X'):>20} "
              f"{f'{a_tok:,.0f} vs {c_tok:,}' + ('' if tok_ok else ' X'):>22}")
    return failures


def _check_readme() -> int:
    """README.md's headline tables vs the live run + the k-sweep.

    The README is the front page: it is the most likely file to keep a number
    from a superseded run, so it is checked with the same rigour as docs/.
    """
    need = (LIVE_SUMMARY, PUBLIC_RESULTS, WRITEUP)
    for path in need:
        if not os.path.exists(path):
            print(f"\nskip: no {path} yet")
            return 0

    readme = os.path.join(ROOT, "README.md")
    text = open(readme, encoding="utf-8").read()
    summary = json.load(open(LIVE_SUMMARY, encoding="utf-8"))
    rows = json.load(open(PUBLIC_RESULTS, encoding="utf-8"))

    qtypes = sorted({r["qtype"] for r in rows})
    live = {}
    for name, node in summary.get("pipelines", {}).items():
        by_type = {}
        for t in qtypes:
            sub = [r for r in rows if r["qtype"] == t]
            by_type[t] = sum(1 for r in sub
                             if r["pipelines"][name]["evaluation"]["is_correct"])
        live[name] = {
            "acc": node["accuracy"],
            "tok": node["avg_total_tokens"],
            "ctx": node["avg_context_tokens"],
            "by_type": by_type,
            "totals": {t: sum(1 for r in rows if r["qtype"] == t) for t in qtypes},
        }

    failures = 0
    flat = " ".join(text.split())

    # (a) every pipeline must still be named -- catches "three pipelines" drift
    missing = [n for n in live if n not in text]
    if missing:
        print(f"\nREADME   X   pipelines absent from README: {missing}")
        failures += 1
    else:
        print(f"\nREADME   OK  all {len(live)} pipelines named")

    # (b) the headline table
    section = text.split("## Headline result", 1)[-1].split("### ", 1)[0]
    head, table, mode = [], {}, None
    for line in section.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells[0] == "Pipeline":
            mode, head = "headline", cells
            continue
        if set("".join(cells)) <= set("- "):
            continue
        if mode and cells[0].replace("*", "") in live:
            table[cells[0].replace("*", "")] = dict(zip(head[1:], cells[1:]))

    print(f"{'README headline':<26} {'acc':>16} {'aggregation':>14} "
          f"{'superlative':>14} {'tok/q':>16}")
    print("-" * 92)
    for name, truth in live.items():
        got = table.get(name)
        if not got:
            print(f"{name:<26} {'MISSING from README':>16}")
            failures += 1
            continue

        def _pct(cell, correct, total):
            m = re.match(r"\**([\d.]+)%", cell or "")
            return (abs(float(m.group(1)) / 100 - correct / total) <= 0.006
                    if m and total else False)

        ok_acc = _pct(got.get("Accuracy"), round(truth["acc"] * len(rows)), len(rows))
        ok_agg = _pct(got.get("aggregation"),
                      truth["by_type"]["aggregation"], truth["totals"]["aggregation"])
        ok_sup = _pct(got.get("superlative"),
                      truth["by_type"]["superlative"], truth["totals"]["superlative"])
        m_tok = re.match(r"\**([\d,]+)\**", got.get("avg tokens/q") or "")
        ok_tok = bool(m_tok) and abs(
            int(m_tok.group(1).replace(",", "")) - truth["tok"]) <= 1
        bad = not all((ok_acc, ok_agg, ok_sup, ok_tok))
        failures += 1 if bad else 0
        cells = [f"{got.get(k, '?')}{'' if v else ' X'}"
                 for k, v in (("Accuracy", ok_acc), ("aggregation", ok_agg),
                              ("superlative", ok_sup), ("avg tokens/q", ok_tok))]
        print(f"{name:<26} {cells[0]:>16} {cells[1]:>14} {cells[2]:>14} "
              f"{cells[3]:>16}")

    # (c) the k-sweep cost table, against results/baseline_sweep.json
    if os.path.exists(SWEEP):
        sweep = json.load(open(SWEEP, encoding="utf-8")).get("results", {})
        for pipe, k in (("RAG", 160), ("GraphRAG", 160)):
            row = (sweep.get(pipe) or {}).get(str(k)) or {}
            if not row:
                continue
            acc_pct = round(float(row["accuracy"]) * 100)
            tok = round(float(row["avg_context_tokens"]))
            for pat in (f"{pipe} (best, k={k})",):
                ok = (f"| {pat} | {acc_pct}% | {tok:,} |" in text)
                print(f"README   {'OK ' if ok else 'X  '} {pat}: {acc_pct}% / "
                      f"{tok:,} ctx tokens")
                failures += 0 if ok else 1
        agent = live.get("Agentic GraphRAG")
        if agent:
            tok = round(agent["ctx"])
            ok = f"| **Agentic GraphRAG** | **{round(agent['acc'] * 100)}%** | **{tok:,}** |" in text
            print(f"README   {'OK ' if ok else 'X  '} Agentic row: "
                  f"{round(agent['acc'] * 100)}% / {tok:,} ctx tokens")
            failures += 0 if ok else 1

    # (d) the stale "cheapest pipeline"/0-token claim must stay gone
    for stale in ("at 0 prompt tokens", "**0**", "39× the context cost"):
        if stale in flat:
            print(f"README   X   stale claim still present: {stale!r}")
            failures += 1
    return failures


def _check_writeup() -> int:
    """docs/submission_writeup.md vs the live public/hidden result files.

    The writeup quotes a headline table and a per-question-type table; both are
    parsed here and recomputed from the records, so the document cannot quietly
    keep numbers that the run no longer supports.
    """
    for path in (LIVE_SUMMARY, PUBLIC_RESULTS, HIDDEN_SUBMISSION):
        if not os.path.exists(path):
            print(f"\nskip: no {path} yet")
            return 0

    summary = json.load(open(LIVE_SUMMARY, encoding="utf-8"))
    rows = json.load(open(PUBLIC_RESULTS, encoding="utf-8"))
    hidden = json.load(open(HIDDEN_SUBMISSION, encoding="utf-8"))["submission"]
    text = open(WRITEUP, encoding="utf-8").read()

    # ── recompute the truth ────────────────────────────────────────────────
    pipes = list(summary.get("pipelines", {}))
    qtypes = sorted({r["qtype"] for r in rows})
    live = {}
    for name in pipes:
        node = summary["pipelines"][name]
        by_type = {}
        for t in qtypes:
            sub = [r for r in rows if r["qtype"] == t]
            by_type[t] = sum(1 for r in sub
                             if r["pipelines"][name]["evaluation"]["is_correct"])
        live[name] = {
            "acc": node["accuracy"],
            "tok": node["avg_total_tokens"],
            "ctx": node["avg_context_tokens"],
            "lat": node["avg_latency_ms"] / 1000.0,
            "by_type": by_type,
            "totals": {t: sum(1 for r in rows if r["qtype"] == t) for t in qtypes},
        }

    # ── parse the markdown ─────────────────────────────────────────────────
    failure = 0

    def _cells(line: str):
        return [c.strip() for c in line.strip().strip("|").split("|")]

    table1, table2 = {}, {}
    section = text.split("## Results", 1)[-1].split("## Findings", 1)[0]
    mode, head = None, []
    for line in section.splitlines():
        if not line.startswith("|"):
            continue
        cells = _cells(line)
        if cells[0] == "Pipeline":
            mode = "headline" if "Accuracy" in cells else "qtype"
            head = cells
            continue
        if set("".join(cells)) <= set("- "):
            continue
        if mode and cells[0] in live:
            target = table1 if mode == "headline" else table2
            target[cells[0]] = dict(zip(head[1:], cells[1:]))

    print(f"\n{'writeup headline':<26} {'accuracy':>20} "
          f"{'tokens':>18} {'latency':>14}")
    print("-" * 84)
    for name, truth in live.items():
        got = table1.get(name)
        if not got:
            print(f"{name:<26} {'MISSING from writeup':>20}")
            failure += 1
            continue
        m = re.match(r"\**([\d.]+)%\**\s*\((\d+)/(\d+)\)", got["Accuracy"])
        if not m:
            print(f"{name:<26} {'UNPARSEABLE accuracy':>20}")
            failure += 1
            continue
        doc_acc, doc_cor, doc_tot = float(m.group(1)) / 100, int(m.group(2)), int(m.group(3))
        doc_tok = int(got["Avg total tokens"].replace(",", ""))
        doc_ctx = int(got["Avg context tokens"].replace(",", ""))
        doc_lat = float(got["Avg latency"].rstrip(" s"))

        real_cor = round(truth["acc"] * len(rows))
        checks = [
            abs(doc_acc - truth["acc"]) <= 0.0006,
            doc_cor == real_cor and doc_tot == len(rows),
            abs(doc_tok - truth["tok"]) <= 1,
            abs(doc_ctx - truth["ctx"]) <= 1,
            abs(doc_lat - truth["lat"]) <= 0.06,
        ]
        failure += 0 if all(checks) else 1
        doc_acc_s = f"{doc_acc:.3f} vs {truth['acc']:.3f}"
        doc_tok_s = f"{doc_tok:,} vs {truth['tok']:,.0f}"
        doc_lat_s = f"{doc_lat:.1f}s vs {truth['lat']:.1f}s"
        flag = "" if all(checks) else "  X"
        print(f"{name:<26} {doc_acc_s:>20} {doc_tok_s:>18} {doc_lat_s:>14}{flag}")

    print(f"\n{'writeup by-qtype':<26} " + " ".join(f"{t:>12s}" for t in qtypes))
    print("-" * 84)
    for name, truth in live.items():
        got = table2.get(name)
        if not got:
            print(f"{name:<26} {'MISSING from writeup':>20}")
            failure += 1
            continue
        cells, bad = [], False
        for t in qtypes:
            m = re.match(r"(\d+)\s*/\s*(\d+)", got.get(t, ""))
            ok = bool(m) and int(m.group(1)) == truth["by_type"][t] \
                and int(m.group(2)) == truth["totals"][t]
            bad = bad or not ok
            cells.append(f"{got.get(t, '?'):>12s}{'' if ok else ' X'}")
        failure += 1 if bad else 0
        print(f"{name:<26} " + " ".join(cells))

    # ── hidden-set claims ──────────────────────────────────────────────────
    n_done = hidden.get("num_completed")
    n_tok = hidden.get("total_tokens")
    # The writeup is wrapped prose, so compare against collapsed whitespace.
    flat = " ".join(text.split())
    claim_done = f"answered {n_done}/{n_done}"
    claim_tok = f"{n_tok:,} tokens"
    for claim in (claim_done, claim_tok):
        if claim in flat:
            print(f"hidden   OK  {claim}")
        else:
            print(f"hidden   X   writeup is missing {claim!r}")
            failure += 1
    return failure


def main() -> int:
    if not os.path.exists(SUMMARY):
        print(f"missing {SUMMARY}\nrun: python run_benchmark.py --no-llm "
              f"--no-tg --ablations --out results/ablation_study.json "
              f"--summary results/ablation_summary.json")
        return 1

    summary = json.load(open(SUMMARY, encoding="utf-8"))
    pipelines = summary.get("pipelines", {})

    print(f"{'pipeline':<26} {'overall':>18} {'aggregation':>18} {'superlative':>18}")
    print("-" * 84)

    failures = 0
    for name, (c_all, c_agg, c_sup) in CLAIMED.items():
        node = pipelines.get(name)
        if node is None:
            print(f"{name:<26} {'MISSING from summary':>18}")
            failures += 1
            continue
        by_type = node.get("by_type", {}) or {}
        actual = (_acc(node), _acc(by_type.get("aggregation", {})),
                  _acc(by_type.get("superlative", {})))
        claimed = (c_all, c_agg, c_sup)

        cells, bad = [], False
        for a, c in zip(actual, claimed):
            match = (a == a) and abs(a - c) <= TOL  # a == a filters NaN
            bad = bad or not match
            cells.append(f"{a:.2f} vs {c:.2f}{'' if match else ' X'}")
        failures += 1 if bad else 0
        print(f"{name:<26} " + " ".join(f"{c:>18}" for c in cells))

    failures += _check_sweep()
    failures += _check_readme()
    failures += _check_writeup()

    print("-" * 84)
    if failures:
        print(f"{failures} MISMATCH(ES): update the docs to match the data "
              f"(or re-run the study).")
        return 1
    print("docs match the result files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
