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

HTML = os.path.join(ROOT, "dashboard", "index.html")
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
entries, summary = [], {}
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
    harness_path = os.path.join(ROOT, "tools", "_dash_harness.js")
    with open(harness_path, "w", encoding="utf-8") as fh:
        fh.write(harness)
    r = subprocess.run([node, harness_path], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    check("dashboard helpers run on real data", r.returncode == 0,
          (r.stdout or r.stderr).strip()[-300:])
    os.remove(harness_path)
else:
    print("SKIP node not found — JS syntax check skipped")

print("\nFAILURES:", fails)
sys.exit(1 if fails else 0)