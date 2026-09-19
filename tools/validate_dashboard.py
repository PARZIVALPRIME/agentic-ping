"""Validates the generated dashboard: JS syntax + embedded DATA payload.

Uses node (if present) for a real syntax check, and pure Python json for the
payload. Also runs the dashboard's pure helper functions under a tiny DOM stub
to confirm the render path does not throw on real result data.
"""
import json
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8")

HTML = os.path.join(ROOT, "dashboard",
                    sys.argv[1] if len(sys.argv) > 1 else "index.html")
JS = os.path.join(ROOT, "dashboard", "dashboard.js")
fails = 0


def check(label, ok, extra=""):
    global fails
    if not ok:
        fails += 1
    print(f"{'PASS' if ok else 'FAIL'} {label}{(' — ' + extra) if extra else ''}")


# 1. files exist
check("dashboard/index.html exists", os.path.exists(HTML))
check("dashboard/dashboard.js exists", os.path.exists(JS))
check("dashboard/styles.css exists",
      os.path.exists(os.path.join(ROOT, "dashboard", "styles.css")))

html = open(HTML, encoding="utf-8").read()

# 2. embedded payload parses as JSON
m = re.search(r"const DATA = (\{.*?\});\n", html, re.S)
check("DATA payload present", bool(m))
entries, summary, data = [], {}, {}
if m:
    try:
        data = json.loads(m.group(1))
        entries = data.get("entries", [])
        summary = data.get("summary", {})
        check("DATA payload is valid JSON", True,
              f"{len(entries)} entries, {len(summary.get('pipelines', {}))} pipelines")
    except json.JSONDecodeError as exc:
        check("DATA payload is valid JSON", False, str(exc))

# 3. json escaping: '</script>' must not appear inside the payload
payload = m.group(1) if m else ""
check("no </script> breakout in payload", "</script>" not in payload)

# 3b. provenance: "0 LLM calls" must never be reported without a reason. Every
# pipeline present in the summary has to carry a provider-contribution slot whose
# counters are internally consistent, and a pipeline that called the provider yet
# got no text back must be called out in the page's warnings.
activity = (data.get("llm_activity") if m else None) or {}
warnings = (data.get("llm_warnings") if m else None) or []
if activity:
    bad = [n for n, a in activity.items()
           if not (0 <= int(a.get("records_answering") or 0)
                   <= int(a.get("records_with_calls") or 0)
                   <= int(a.get("records") or 0))]
    check("provider counters are consistent", not bad,
          f"inconsistent: {bad}" if bad else f"{len(activity)} pipeline slot(s)")
    wrong = [n for n, a in activity.items()
             if bool(a.get("llm_contributed"))
             != (int(a.get("records_answering") or 0) > 0)]
    check("llm_contributed matches the counters", not wrong, str(wrong))
    silent = [n for n, a in activity.items()
              if int(a.get("calls") or 0) > 0 and not int(a.get("records_answering") or 0)]
    unexplained = [n for n in silent
                   if not any(n in w for w in warnings)]
    check("a failed provider call is explained", not unexplained,
          f"no warning names {unexplained}" if unexplained else
          (f"declared: {silent}" if silent else "no silent provider failures"))
    missing = [n for n in (summary.get("pipelines") or {}) if n not in activity]
    check("every pipeline reports its provider share", not missing, str(missing))
else:
    print("SKIP no llm_activity in payload — page predates the provenance block")

# 4. js syntax via node (optional)
node = shutil.which("node")
if node:
    r = subprocess.run([node, "--check", JS], capture_output=True, text=True)
    check("node --check dashboard.js", r.returncode == 0, r.stderr.strip()[:200])

    # 5. exercise the pure helpers under a DOM stub, with DATA injected
    js_src = open(JS, encoding="utf-8").read()
    # drop the boot listener (needs a real DOM) and expose DATA globally
    js_src = re.sub(r"document\.addEventListener\(\"DOMContentLoaded\"[\s\S]*$",
                    "", js_src)
    harness = (
        "globalThis.DATA = " + payload + ";\n"
        + js_src + "\n"
        + """
const entries = DATA.entries || [];
let undef = 0;
for (const e of entries) {
  for (const [name, r] of Object.entries(e.pipelines || {})) {
    if (r.evaluation === undefined) undef++;
  }
  try { outcomeOf(e); } catch (err) {
    console.log('THREW in outcomeOf: ' + err.message); process.exit(1);
  }
}
const s = (DATA.summary || {}).pipelines || {};
for (const [name, ps] of Object.entries(s)) {
  if (ps.accuracy !== null && ps.accuracy !== undefined &&
      (ps.accuracy < 0 || ps.accuracy > 1)) {
    console.log('BAD accuracy ' + name + ' ' + ps.accuracy); process.exit(2);
  }
}
// colour map must cover every pipeline present in the data
const missing = Object.keys(s).filter(n => !(n in PIPE_COLORS));
if (missing.length) {
  console.log('no chart colour for: ' + missing.join(', ')); process.exit(3);
}
console.log('helpers ok; entries=' + entries.length +
  ' pipelines=' + Object.keys(s).length +
  ' delta_rows=' + ((DATA.summary || {}).when_agents_matter || []).length +
  ' undefined_eval=' + undef);
"""
    )
    # unique per process: two validators (or two pages) may run at once
    harness_path = os.path.join(ROOT, "tools",
                                f"_dash_harness_{os.getpid()}.js")
    with open(harness_path, "w", encoding="utf-8") as fh:
        fh.write(harness)
    r = subprocess.run([node, harness_path], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    check("dashboard helpers run on real data", r.returncode == 0,
          (r.stdout or r.stderr).strip()[-300:])
    try:
        os.remove(harness_path)
    except OSError:
        pass          # Windows may still hold the handle; leftovers are ignored
else:
    print("SKIP node not found — JS syntax check skipped")

print("\nFAILURES:", fails)
sys.exit(1 if fails else 0)