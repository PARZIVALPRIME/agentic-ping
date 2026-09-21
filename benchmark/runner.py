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
from benchmark.metrics import (MetricsCollector, llm_activity,
                               llm_activity_warnings)
# Imported at module level (not inside main) because the runner records the
# active backend in the summary and the CLI prints it on every run.
from kg.backend import describe_backend, open_graph

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


def _provenance(entries: List[Dict[str, Any]], *, rebuilt_from: Optional[str],
                note: str) -> Dict[str, Any]:
    """The provider-contribution block every summary carries, live or rebuilt.

    One shape for both paths, so the dashboard and the audit tool never have to
    guess where ``llm_activity`` lives: a run whose provider calls all failed
    writes a complete, plausible-looking file in which every answer came from the
    deterministic path, and this block is what makes that visible.
    """
    activity = llm_activity(entries)
    return {
        "rebuilt_from": rebuilt_from,
        "note": note,
        "llm_activity": activity,
        "warnings": llm_activity_warnings(activity),
    }


def summarize_entries(entries: List[Dict[str, Any]], summary_path: Optional[str] = None,
                      source: str = "") -> Dict[str, Any]:
    """Rebuild a summary from an existing results file (``--summarize-only``).

    Provider/evaluator/backend telemetry is written by a live run and is not
    reconstructable from the answers, so those blocks are reported as ``null``
    instead of being guessed. ``provenance`` says so explicitly, and carries
    the LLM activity it *can* derive, so a rebuilt summary cannot be mistaken
    for a run in which the provider answered.
    """
    collector = MetricsCollector()
    collector.entries.extend(entries)
    activity = llm_activity(entries)
    summary = collector.summarize(extra={
        "evaluator": None,
        "llm": None,
        "backend": None,
        "wall_clock_s": None,
        # Derived from the records: a rebuilt summary still knows whether the
        # provider answered anything, it just cannot know the client telemetry.
        "run_mode": ("live" if any(slot.get("records_answering") for slot in activity.values())
                     else "provider-failed" if any(slot.get("calls") for slot in activity.values())
                     else "deterministic"),
        "provenance": _provenance(
            entries, rebuilt_from=source,
            note=("summary rebuilt from an existing results file; evaluator/llm/"
                  "backend telemetry is only known during a live run, so those "
                  "blocks are null rather than guessed")),
    })
    if summary_path:
        os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, ensure_ascii=False)
    return summary


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

        # "live" must mean the provider *answered*, not merely that one was
        # configured. A run whose calls all failed (429, timeout, bad JSON) still
        # writes a complete file with zeroed token counters and every answer from
        # the solvers; labelling that "live" is precisely how a deterministic run
        # gets mistaken for an LLM-backed one.
        activity = llm_activity(collector.entries)
        answered = any(slot.get("records_answering") for slot in activity.values())
        attempted = any(slot.get("calls") for slot in activity.values())
        if not getattr(self.llm, "available", False):
            mode = "deterministic"
        elif answered:
            mode = "live"
        elif attempted:
            mode = "provider-failed"
        else:
            mode = "live"          # configured, but nothing needed the LLM yet
        summary = collector.summarize(extra={
            "evaluator": self.evaluator.stats(),
            "llm": self.llm.stats() if hasattr(self.llm, "stats") else {},
            # "live" = a provider was configured for this run, "deterministic" =
            # --no-llm. Recorded because "0 LLM calls/q" is a property of the run,
            # never of the architecture, and the two are otherwise indistinguishable
            # once the numbers are on a slide.
            "run_mode": mode,
            "wall_clock_s": round(time.time() - t_start, 1),
            # Which graph store actually answered. Recorded in the summary so a
            # TigerGraph run is distinguishable from a local one after the fact -
            # a backend that silently fell back would otherwise look identical.
            "backend": describe_backend(),
            # Which pipelines the provider actually contributed to, in the same
            # block a rebuilt summary uses. A run whose calls all failed still
            # finishes and looks successful, so the summary states the
            # contribution explicitly instead of leaving it to be inferred from
            # the accuracy table ("0 LLM calls/q" must not read as a property of
            # the architecture when it is a property of the run).
            "provenance": _provenance(
                collector.entries, rebuilt_from=None,
                note="live run: llm/evaluator/backend telemetry recorded above"),
        })
        if summary_path:
            with open(summary_path, "w", encoding="utf-8") as fh:
                json.dump(summary, fh, indent=2, ensure_ascii=False)
        self._report_provider(mode, summary)
        return {"summary": summary, "entries": collector.entries}

    def _report_provider(self, mode: str, summary: Dict[str, Any]) -> None:
        """Say out loud whether the provider contributed to this run.

        A live run whose calls all failed still finishes and writes a complete,
        plausible-looking file with zeroed token counters, so the harness states
        the outcome at the moment it is known instead of leaving a reader to
        notice it later. ``tools/audit_results.py`` is the gate that blocks such
        a file from being published; this is the same information, printed.
        """
        activity = (summary.get("provenance") or {}).get("llm_activity") or {}
        calls = sum(int(slot.get("calls") or 0) for slot in activity.values())
        answering = sum(int(slot.get("records_answering") or 0)
                        for slot in activity.values())
        if mode == "deterministic":
            print("\nMODE: deterministic (--no-llm): no provider calls were made. "
                  "These accuracies are the solver baseline, not LLM results.",
                  flush=True)
            return
        questions = max(1, int(summary.get("num_questions") or 1))
        if calls and answering:
            print(f"\nMODE: live: the provider returned text for {answering} "
                  f"record(s) across {calls} call(s) "
                  f"({calls / questions:.2f} calls/question).", flush=True)
            return
        stats = summary.get("llm") or {}
        print("\n" + "!" * 72, flush=True)
        print("PROVIDER CALLS RECORDED NOTHING: every answer in this run came from\n"
              "the deterministic solvers. Check the failure counters below and the\n"
              "provider quota, then re-run with --resume (finished questions are\n"
              "kept, so only the failed ones are retried).", flush=True)
        print(f"  run_mode={mode} (recorded in the summary so the dashboard cannot "
              f"present this file as an LLM result)", flush=True)
        if stats:
            print(f"  available={stats.get('available')} "
                  f"calls={stats.get('num_calls')} "
                  f"failed={stats.get('failed_calls')} "
                  f"error={stats.get('last_error')!r}", flush=True)
        print("!" * 72, flush=True)

    # ── persistence ────────────────────────────────────────────────────
    def _write(self, out_path: str, entries: List[Dict[str, Any]],
               retries: int = 5) -> None:
        """Persist results, tolerating transient Windows file locks.

        ``os.replace`` is atomic but raises PermissionError when a sync client
        (OneDrive/Dropbox) or an AV scanner momentarily holds the destination
        open. That is transient and common on a synced folder, so we retry with
        a short backoff and then fall back to a direct write. Losing a
        100-question run to a locked file would be far worse than a
        non-atomic write.
        """
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        tmp = out_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(entries, fh, indent=2, ensure_ascii=False)

        for attempt in range(retries):
            try:
                os.replace(tmp, out_path)
                return
            except PermissionError:
                time.sleep(0.25 * (attempt + 1))

        try:
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(entries, fh, indent=2, ensure_ascii=False)
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError as exc:
            print(f"WARNING: could not update {out_path} ({exc}); "
                  f"progress preserved in {tmp}", flush=True)


# ── CLI bootstrap ─────────────────────────────────────────────────────
def main(questions_path: str, out_path: str = "results/public_results.json",
         summary_path: str = "results/metrics_summary.json",
         pipelines: Optional[List[str]] = None,
         limit: int = 0, offset: int = 0, llm_judge: bool = True,
         resume: bool = False, no_llm: bool = False,
         types: Optional[List[str]] = None, per_type: int = 0,
         no_tg: bool = False, ablations: bool = False,
         no_router: bool = False) -> Dict[str, Any]:
    from config import config
    from pipelines import build_pipelines
    from retrieval import load_index
    from utils.llm import LLMHelper, build_llm

    log("building KG / index ...")
    # open_graph decides local vs TigerGraph from config (TG_ENABLED), and always
    # keeps the corpus-built graph as the mirror/fallback. --no-tg forces local.
    kg = open_graph(config.benchmark.corpus_path, config,
                    cache_path=config.benchmark.kg_cache_path,
                    force_local=no_tg,
                    verbose=bool(getattr(config.tg, "verbose", False)))
    backend = describe_backend()
    log(f"graph backend: {backend['active']}"
        + (f" (graph={backend['graphname']})" if backend["active"] == "tigergraph"
           else f" [{backend['reason']}]"))
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
        # A run that asks for the LLM must not silently become a deterministic
        # one. Without this guard a missing/blank key produces a complete results
        # file whose every answer came from the solvers and whose dashboard reads
        # "0 LLM calls/q" - which is exactly how a deterministic run gets mistaken
        # for a failed LLM one. Better to stop before spending an hour on it.
        if not llm.available:
            local = getattr(config.llm, "provider", "") == "ollama"
            raise SystemExit(
                "no LLM provider available: "
                f"{llm.error or 'unknown reason'}\n"
                f"  provider={getattr(config.llm, 'provider', None)!r} "
                f"model={getattr(config.llm, 'chat_model', '')!r} "
                f"key={'set' if getattr(config.llm, 'api_key', None) else 'MISSING'}\n"
                + ("  fix: start the local server (`ollama serve`) and check "
                   f"OLLAMA_BASE_URL={getattr(config.llm, 'ollama_base_url', '')!r} "
                   "with `ollama list`\n"
                   if local else
                   "  fix: put LLM_PROVIDER + the matching *_API_KEY in .env "
                   "(e.g. LLM_PROVIDER=groq, GROQ_API_KEY=gsk_...)\n")
                + "  or pass --no-llm to run the deterministic baseline on purpose.")
    log(f"llm available: {llm.available} ({llm.provider}/{llm.chat_model}) "
        f"| index: {index.stats()}")

    all_pipes = build_pipelines(index, llm, config,
                                with_router=not no_router,
                                with_ablations=ablations)
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
    if s.get("backend"):
        b = s["backend"]
        extra = (f" graph={b['graphname']}" if b["active"] == "tigergraph"
                 else f" ({b['reason']})")
        print(f"backend: {b['active']}{extra}")
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
    ap.add_argument("--no-tg", action="store_true",
                    help="force the local corpus-built graph even if TG_ENABLED=true")
    ap.add_argument("--agent-mode", "--mode", default="",
                    choices=["", "react", "hybrid", "plan"],
                    help="agentic pipeline mode: react (LLM tool-calling only), "
                         "hybrid (LLM + deterministic rescue), plan (deterministic "
                         "planner/executor, no tool loop)")
    ap.add_argument("--summarize-only", action="store_true",
                    help="rebuild --summary from an existing --out results file "
                         "without running anything; provider telemetry is "
                         "reported as null because it is only known during a run")
    ap.add_argument("--types", default="",
                    help="comma filter on qtype, e.g. 'temporal,aggregation' "
                         "(useful for quota-bounded LLM samples)")
    ap.add_argument("--per-type", type=int, default=0,
                    help="with --limit 0: take at most N questions per qtype")
    ap.add_argument("--ablations", action="store_true",
                    help="also run the agentic ablation variants (no-planner, "
                         "no-verifier, no-gap-detector) to attribute the "
                         "agentic gain to a specific mechanism")
    ap.add_argument("--no-router", action="store_true",
                    help="skip the Router pipeline")
    args = ap.parse_args(argv)

    if args.summarize_only:
        # Rebuilding a summary must not need the KG, the provider or a key: it
        # only aggregates what an earlier run already wrote down.
        entries = load_entries(args.out)
        if not entries:
            raise SystemExit(f"--summarize-only: no results found at {args.out}")
        summary = summarize_entries(entries, args.summary, source=args.out)
        print(f"summary rebuilt from {args.out} "
              f"({summary['num_questions']} questions) -> {args.summary}")
        print(f"  pipelines : {', '.join(summary['pipelines'])}")
        for name, slot in (summary["provenance"]["llm_activity"] or {}).items():
            print(f"  {name:<18s} provider calls in "
                  f"{slot['records_with_calls']}/{slot['records']} records, "
                  f"{slot['total_tokens']} tokens")
        for warning in summary["provenance"]["warnings"]:
            print(f"  WARNING: {warning}")
        return

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
         per_type=args.per_type,
         no_tg=args.no_tg,
         ablations=args.ablations,
         no_router=args.no_router)


if __name__ == "__main__":
    cli()



