"""The benchmark harness: runs every pipeline on every question.

Design notes
------------
- **Incremental persistence**: the results file is rewritten after each
  question, so an interrupted run never loses completed work.
- **Per-question entry schema**: one JSON object per question with a nested
  ``pipelines`` map (pipeline name -> full PipelineResult + evaluation). The
  dashboard is generated directly from this file.
- **Streaming traces**: full step/evidence records are kept for the Agentic
  pipeline so the dashboard can render investigation waterfalls.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

from benchmark.evaluator import Evaluator
from benchmark.metrics import MetricsCollector

# Line-buffered so `python run_benchmark.py > log.txt` shows progress live.
# Without this the log file stays empty for minutes and the run *looks* hung.
try:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
except Exception:
    sys.stdout.reconfigure(encoding="utf-8")


def log(message: str) -> None:
    """Write a timestamped progress line and flush it immediately."""
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _short_name(name: str) -> str:
    """Compact pipeline label for the progress line (RAG/GraphRAG/Agentic)."""
    return {"RAG": "RAG", "GraphRAG": "Graph", "Agentic GraphRAG": "Agent"}.get(
        name, name.split()[0])


def load_questions(path: str) -> List[Dict[str, Any]]:
    questions = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                questions.append(json.loads(line))
    return questions


def load_entries(path: str) -> List[Dict[str, Any]]:
    """Read a previously written results file (for ``--resume``)."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _gold_of(q: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Normalise the gold fields; hidden questions carry no answers."""
    answers = q.get("answer") or q.get("answers") or []
    if isinstance(answers, str):
        answers = [answers]
    answers = [a for a in answers if a]
    doc_ids = q.get("gold_doc_ids") or []
    if answers or doc_ids:
        return {"answers": answers, "doc_ids": doc_ids}
    return None


def _record(result) -> Dict[str, Any]:
    """PipelineResult -> JSON record (keeps traces, drops nothing)."""
    rec = result.to_dict()
    # evaluation is attached after grading
    rec["evaluation"] = None
    return rec


class BenchmarkRunner:
    def __init__(self, pipelines: List[Any], evaluator: Evaluator,
                 results_dir: str = "results", llm: Any = None) -> None:
        self.pipelines = pipelines
        self.evaluator = evaluator
        self.results_dir = results_dir
        self.llm = llm
        os.makedirs(results_dir, exist_ok=True)

    # ── core loop ──────────────────────────────────────────────────────
    def _pacer_seconds(self) -> float:
        """Seconds the client token pacer has slept so far (0 if unavailable)."""
        try:
            return float((self.llm.stats() or {}).get("pace_slept_s", 0.0) or 0.0)
        except Exception:
            return 0.0

    def run(self, questions: List[Dict[str, Any]], out_path: str,
            summary_path: Optional[str] = None,
            initial_entries: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        collector = MetricsCollector()
        if initial_entries:                       # resume: keep completed work
            collector.entries.extend(initial_entries)
        done = {e.get("qid") for e in collector.entries}
        t_start = time.time()
        pacer_seen = self._pacer_seconds()
        for i, q in enumerate(questions, 1):
            qid = q.get("qid", f"q{i}")
            if qid in done:
                continue
            qtype = q.get("qtype", "unknown")
            question = q.get("question") or q.get("q") or ""
            gold = _gold_of(q)
            q_t0 = time.time()

            entry: Dict[str, Any] = {
                "qid": qid,
                "qtype": qtype,
                "question": question,
                "gold": gold,
                "pipelines": {},
            }
            marks = []
            for pipe in self.pipelines:
                try:
                    result = pipe.run(question, qid)
                except Exception as exc:  # a broken pipeline must not kill the run
                    result = None
                    entry["pipelines"][pipe.name] = {
                        "pipeline": pipe.name, "error": f"{exc.__class__.__name__}: {exc}",
                        "answer": "", "citations": [], "evaluation": None,
                    }
                if result is not None:
                    rec = _record(result)
                    if gold:
                        ev = self.evaluator.evaluate(
                            question, gold["answers"], result.answer,
                            result.citations, gold["doc_ids"])
                        rec["evaluation"] = ev.to_dict()
                    entry["pipelines"][pipe.name] = rec
                    marks.append((pipe.name, rec))

            collector.add(entry)
            self._write(out_path, collector.entries)

            # progress line - always flushed, with per-question timing so a
            # stalled question is visible instead of looking like a hang.
            tag = " ".join(
                f"{_short_name(name)}={'OK' if rec['evaluation'] and rec['evaluation']['is_correct'] else ('F' if rec['evaluation'] else '-')}"
                for name, rec in marks)
            toks = sum(rec.get("total_tokens", 0) for _, rec in marks)
            took = time.time() - q_t0
            rate = (time.time() - t_start) / max(1, len(collector.entries))
            eta = rate * max(0, len(questions) - len(collector.entries))
            print(f"[{time.strftime('%H:%M:%S')}] {i:>3}/{len(questions)} {qid} "
                  f"{qtype:<11s} {tag} tok={toks} took={took:.1f}s "
                  f"elapsed={time.time()-t_start:.0f}s eta={eta/60:.1f}m", flush=True)
            slow = [name for name, rec in marks
                    if rec.get("latency_ms", 0) > 60000]
            if slow:
                # Long latency is not automatically rate limiting: the client
                # token pacer deliberately sleeps to stay inside the provider's
                # tokens-per-minute budget. Report which of the two it was.
                now_pacer = self._pacer_seconds()
                paced_here = now_pacer - pacer_seen
                pacer_seen = now_pacer
                cause = (f"client-side token pacing ({paced_here:.0f}s slept"
                         f" this question, staying within TPM)"
                         if paced_here > 0.5 else
                         "provider latency or retries")
                print(f"           note: slow pipeline(s) {', '.join(slow)} - "
                      f"{cause}; see llm.circuit in the summary", flush=True)

        summary = collector.summarize(extra={
            "evaluator": self.evaluator.stats(),
            "llm": self.llm.stats() if hasattr(self.llm, "stats") else {},
            "wall_clock_s": round(time.time() - t_start, 1),
        })
        if summary_path:
            with open(summary_path, "w", encoding="utf-8") as fh:
                json.dump(summary, fh, indent=2, ensure_ascii=False)
        return {"summary": summary, "entries": collector.entries}

    # ── persistence ────────────────────────────────────────────────────
    def _write(self, out_path: str, entries: List[Dict[str, Any]]) -> None:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        tmp = out_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(entries, fh, indent=2, ensure_ascii=False)
        os.replace(tmp, out_path)


# ── CLI bootstrap ─────────────────────────────────────────────────────
def main(questions_path: str, out_path: str = "results/public_results.json",
         summary_path: str = "results/metrics_summary.json",
         pipelines: Optional[List[str]] = None,
         limit: int = 0, offset: int = 0, llm_judge: bool = True,
         resume: bool = False, no_llm: bool = False,
         types: Optional[List[str]] = None, per_type: int = 0) -> Dict[str, Any]:
    from config import config
    from kg.builder import load_or_build
    from pipelines import build_pipelines
    from retrieval import load_index
    from utils.llm import LLMHelper, build_llm

    log("building KG / index ...")
    kg = load_or_build(config.benchmark.corpus_path)
    index = load_index(config.benchmark.corpus_path, kg,
                       vector_backend=config.benchmark.vector_backend)
    if no_llm:
        # Deterministic-only mode: no provider calls at all, so the run always
        # finishes in minutes regardless of quota. Useful as the reproducible
        # baseline and as the fallback when the provider is rate limiting.
        llm = LLMHelper()
        log("LLM disabled (--no-llm): deterministic reasoning only")
    else:
        llm = build_llm(config)
    log(f"llm available: {llm.available} ({llm.chat_model}) | index: {index.stats()}")

    all_pipes = build_pipelines(index, llm, config)
    if pipelines:
        wanted = {p.lower() for p in pipelines}
        all_pipes = [p for p in all_pipes
                     if any(w in p.name.lower() for w in wanted)]
    log(f"pipelines: {[p.name for p in all_pipes]}")

    questions = load_questions(questions_path)[offset:]
    if types:
        wanted = {t.strip().lower() for t in types}
        questions = [q for q in questions if str(q.get("qtype", "")).lower() in wanted]
        log(f"qtype filter {sorted(wanted)} -> {len(questions)} questions")
    if per_type:
        # Quota-bounded sampling: LLM runs cost tokens per question, so a
        # stratified sample (N per question type) keeps every category covered
        # while fitting inside a daily budget.
        picked: List[Dict[str, Any]] = []
        counts: Dict[str, int] = {}
        for q in questions:
            qtype = str(q.get("qtype", "unknown"))
            if counts.get(qtype, 0) < per_type:
                counts[qtype] = counts.get(qtype, 0) + 1
                picked.append(q)
        questions = picked
        log(f"per-type sample ({per_type}/type) -> {len(questions)} questions: "
            f"{counts}")
    if limit:
        questions = questions[:limit]

    evaluator = Evaluator(llm, use_llm_judge=llm_judge and not no_llm)
    runner = BenchmarkRunner(all_pipes, evaluator,
                             results_dir=config.benchmark.results_dir, llm=llm)
    initial = None
    if resume and os.path.exists(out_path):
        initial = load_entries(out_path)
        log(f"resuming: {len(initial)} questions already completed")
    out = runner.run(questions, out_path, summary_path, initial_entries=initial)

    # console summary
    s = out["summary"]
    print("\n" + "=" * 76)
    print(f"RESULTS ({s['num_questions']} questions) -> {out_path}")
    for name, ps in s["pipelines"].items():
        acc = ps.get("accuracy")
        print(f"  {name:<18s} acc={acc:.1%} ({ps['correct']}/{ps['num_evaluated']})"
              if acc is not None else f"  {name:<18s} (no gold answers)")
        bt = ps.get("by_type", {})
        for qtype, t in bt.items():
            if t.get("n"):
                a = t.get("accuracy")
                if a is None:
                    # no gold answers for this question type (hidden set)
                    print(f"     {qtype:<12s} {t['n']} question(s) "
                          f"(unscored - no gold answers)")
                else:
                    print(f"     {qtype:<12s} {t['correct']:>3d}/{t['n']:<3d}"
                          f" = {a:.0%}")
        print(f"     avg tok={ps.get('avg_total_tokens')} "
              f"lat={ps.get('avg_latency_ms')}ms "
              f"steps={ps.get('avg_retrieval_steps')}")
    print(f"evaluator: {s.get('evaluator')}")
    if s.get("llm"):
        print(f"llm: {s['llm']}")
    return out


def cli(argv=None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Run the 3-pipeline benchmark")
    ap.add_argument("questions", nargs="?", default=None)
    ap.add_argument("--out", default="results/public_results.json")
    ap.add_argument("--summary", default="results/metrics_summary.json")
    ap.add_argument("--pipelines", default="",
                    help="comma filter, e.g. 'rag,graph,agentic'")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--resume", action="store_true",
                    help="skip questions already present in --out")
    ap.add_argument("--no-llm", action="store_true",
                    help="deterministic mode: make no provider calls (always fast)")
    ap.add_argument("--no-llm-judge", action="store_true")
    ap.add_argument("--agent-mode", default="",
                    choices=["", "react", "hybrid", "plan"],
                    help="agentic pipeline mode: react (LLM tool-calling only), "
                         "hybrid (LLM + deterministic rescue), plan (deterministic "
                         "planner/executor, no tool loop)")
    ap.add_argument("--types", default="",
                    help="comma filter on qtype, e.g. 'temporal,aggregation' "
                         "(useful for quota-bounded LLM samples)")
    ap.add_argument("--per-type", type=int, default=0,
                    help="with --limit 0: take at most N questions per qtype")
    args = ap.parse_args(argv)

    if args.agent_mode or args.types or args.per_type:
        from config import config
        if args.agent_mode:
            config.agent.mode = args.agent_mode

    qpath = args.questions
    if not qpath:
        from config import config
        qpath = config.benchmark.public_questions_path
    main(qpath, args.out, args.summary,
         pipelines=[p for p in args.pipelines.split(",") if p],
         limit=args.limit, offset=args.offset,
         llm_judge=not args.no_llm_judge, resume=args.resume,
         no_llm=args.no_llm,
         types=[t for t in args.types.split(",") if t],
         per_type=args.per_type)


if __name__ == "__main__":
    cli()



