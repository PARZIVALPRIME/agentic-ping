"""End-to-end self-test. Run this on any new machine before trusting it.

    python tools/selftest.py

Exercises every deliverable in order and reports PASS/FAIL per stage. Exits
non-zero if anything is broken, so it can gate a submission. Makes no network
calls, so it is valid before an API key is configured.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PY = sys.executable


def _run(label: str, args: list, timeout: int = 900) -> bool:
    print(f"\n{'=' * 68}\n  {label}\n{'=' * 68}")
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(args, cwd=ROOT, timeout=timeout,
                              capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        print(f"  FAIL: timed out after {timeout}s")
        return False
    dt = time.perf_counter() - t0
    tail = (proc.stdout or "").strip().splitlines()[-6:]
    for line in tail:
        print(f"  | {line}")
    if proc.returncode != 0:
        err = (proc.stderr or "").strip().splitlines()[-8:]
        for line in err:
            print(f"  ! {line}")
        print(f"  FAIL ({dt:.1f}s, exit {proc.returncode})")
        return False
    print(f"  PASS ({dt:.1f}s)")
    return True


def _check_imports() -> bool:
    print(f"\n{'=' * 68}\n  imports\n{'=' * 68}")
    mods = [
        "config", "pipelines", "pipelines.router_pipeline", "pipelines.ablations",
        "reasoning.conflicts", "benchmark.runner", "kg.backend", "retrieval",
        "agents.orchestrator", "utils.llm",
    ]
    ok = True
    for name in mods:
        try:
            __import__(name)
            print(f"  ok   {name}")
        except Exception as exc:
            print(f"  FAIL {name}: {exc.__class__.__name__}: {exc}")
            ok = False
    return ok


def main() -> int:
    stages = []

    stages.append(("imports", _check_imports()))
    stages.append(("preflight", _run("preflight", [PY, "preflight.py"])))
    stages.append(("conflicts (Round 2)",
                   _run("conflict resolution", [PY, "tools/demo_conflicts.py"])))
    stages.append(("round 2 wiring",
                   _run("version dates, audit, gap + tool",
                        [PY, "tools/test_round2.py"])))
    stages.append(("semantic parser",
                   _run("LLM slot parse: validation, fill, fallback",
                        [PY, "tools/test_semantic_parser.py"])))
    stages.append(("benchmark smoke",
                   _run("benchmark: 5 questions, deterministic",
                        [PY, "run_benchmark.py", "--no-llm", "--no-tg",
                         "--limit", "5",
                         "--out", "results/_selftest.json",
                         "--summary", "results/_selftest_summary.json"])))
    stages.append(("hidden submission",
                   _run("hidden set: 5 questions",
                        [PY, "tools/submit_hidden.py", "--no-llm", "--no-tg",
                         "--limit", "5",
                         "--out", "results/_selftest_hidden.json"])))
    stages.append(("no data leakage",
                   _run("leakage audit", [PY, "tools/audit_leakage.py"])))
    stages.append(("dashboard page",
                   _run("page payload, provider counters, chart helpers",
                        [PY, "tools/validate_dashboard.py"])))
    stages.append(("llm robustness",
                   _run("adversarial LLM audit (a bad model must not hurt)",
                        [PY, "tools/audit_llm_robustness.py", "--limit", "20"])))
    stages.append(("baseline sweep",
                   _run("baseline ceiling: k=5,20 on 10 questions",
                        [PY, "tools/baseline_sweep.py", "--ks", "5,20",
                         "--limit", "10",
                         "--out", "results/_selftest_sweep.json"])))

    # Only meaningful once the ablation study has been run; skipped otherwise
    # so a fresh clone does not report a spurious failure.
    if os.path.exists(os.path.join(ROOT, "results", "ablation_summary.json")):
        stages.append(("docs match data",
                       _run("verify documented numbers",
                            [PY, "tools/verify_doc_numbers.py"])))
    else:
        print("\n  skip  docs match data (no results/ablation_summary.json yet)")

    print(f"\n{'=' * 68}\n  SUMMARY\n{'=' * 68}")
    for label, ok in stages:
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")

    failed = [label for label, ok in stages if not ok]
    print("=" * 68)
    if failed:
        print(f"  {len(failed)} STAGE(S) FAILED: {', '.join(failed)}")
        return 1
    print("  ALL STAGES PASS - this machine can produce a submission")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
