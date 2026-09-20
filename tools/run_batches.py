"""Run a contiguous range of benchmark batches and publish each one.

One batch = N questions through every pipeline, then merge into the cumulative
file, then generate that batch's dashboard page. Batches exist so a long live
run is observable: each one finishes, is scored and is published before the next
starts, and an interrupted batch is retried rather than restarted.

Usage:
    # public batches 2..10 (questions 11-100), publishing per-batch pages
    python tools/run_batches.py --set public --first 2 --last 10

    # the same, with the RAG ablation arm in every batch
    python tools/run_batches.py --set public --first 2 --last 10 --ablation

    # hidden set, batches 1..5, diffing answers against the previous full run
    python tools/run_batches.py --set hidden --first 1 --last 5 \
        --baseline results/hidden_results.json
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

BATCH_SIZE = 10


def run(cmd, log_path):
    """Run one step, teeing its output into a log file. Returns True on success."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as log:
        log.write("\n$ " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(cmd, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True,
                              encoding="utf-8", errors="replace")
        log.write(proc.stdout or "")
    print(proc.stdout or "", flush=True)
    return proc.returncode == 0


def complete(path, size):
    """Has this batch file already been produced in full?"""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return isinstance(data, list) and len(data) == size
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="qset", choices=["public", "hidden"],
                    required=True)
    ap.add_argument("--first", type=int, default=1)
    ap.add_argument("--last", type=int, default=10)
    ap.add_argument("--size", type=int, default=BATCH_SIZE)
    ap.add_argument("--questions", default="",
                    help="override the questions file (default: from config)")
    ap.add_argument("--ablation", action="store_true",
                    help="include the RAG (vector only) arm in every batch")
    ap.add_argument("--baseline", default="",
                    help="previous full-run results file for hidden answer diffs")
    ap.add_argument("--force", action="store_true",
                    help="re-run batches whose output already exists")
    args = ap.parse_args()

    from config import config
    questions = args.questions or (
        config.benchmark.public_questions_path if args.qset == "public"
        else config.benchmark.hidden_questions_path)
    out_dir = "results/batch"
    cumulative = os.path.join(out_dir, f"cumulative_{args.qset}.json")
    failures = []

    for n in range(args.first, args.last + 1):
        offset = (n - 1) * args.size
        tag = f"{args.qset[:3]}_{n:02d}"
        batch_json = os.path.join(out_dir, f"b_{tag}.json")
        summary = os.path.join(out_dir, f"b_{tag}_summary.json")
        log = os.path.join(out_dir, f"b_{tag}_v2.log")
        print(f"\n{'=' * 74}\nbatch {n} ({args.qset} {offset}-"
              f"{offset + args.size - 1})\n{'=' * 74}", flush=True)

        if not args.force and complete(batch_json, args.size):
            print(f"  {batch_json} already complete - post-processing only",
                  flush=True)
        else:
            cmd = [sys.executable, "run_benchmark.py", questions,
                   "--out", batch_json, "--summary", summary,
                   "--limit", str(args.size), "--offset", str(offset)]
            if args.ablation:
                cmd.append("--ablation")
            if not run(cmd, log):
                failures.append((n, "benchmark"))
                continue

        score = [sys.executable, "tools/batch_score.py", "--batch", batch_json,
                 "--cumulative", cumulative, "--set", args.qset]
        if args.baseline and args.qset == "hidden":
            score += ["--baseline", args.baseline]
        if not run(score, log):
            failures.append((n, "score"))
            continue

        page = [sys.executable, "-m", "benchmark.dashboard_generator",
                batch_json, "--summary", summary,
                "--name", f"batch_{tag}.html"]
        if not run(page, log):
            failures.append((n, "dashboard"))
            continue
        print(f"  batch {n} done -> dashboard/batch_{tag}.html", flush=True)

    print(f"\n{'=' * 74}")
    if failures:
        print("FAILED steps:", failures)
        return 1
    print(f"all batches {args.first}..{args.last} ({args.qset}) completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
