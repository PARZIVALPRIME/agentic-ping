"""Audits which third-party packages the codebase actually imports.

Walks every .py file (excluding vendored/third-party trees) and collects
top-level import names, then reports which ones are third-party vs stdlib vs
local modules. Used to keep requirements.txt honest.
"""
import ast
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP_DIRS = {".git", "__pycache__", ".cache", "venv", ".venv", "node_modules",
             "graphrag", "corpus-20260919T043338Z-1-001",
             "questions-20260919T043312Z-1-001", "tools"}
LOCAL = {d for d in os.listdir(ROOT)
         if os.path.isdir(os.path.join(ROOT, d)) and d != "tools"}

sys.stdout.reconfigure(encoding="utf-8")

# module -> set of files importing it
imports = defaultdict(set)
for dirpath, dirnames, filenames in os.walk(ROOT):
    dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
    for fn in filenames:
        if not fn.endswith(".py"):
            continue
        path = os.path.join(dirpath, fn)
        rel = os.path.relpath(path, ROOT)
        try:
            tree = ast.parse(open(path, encoding="utf-8").read())
        except SyntaxError as exc:
            print(f"  !! syntax error in {rel}: {exc}")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    imports[a.name.split(".")[0]].add(rel)
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0 and node.module:
                    imports[node.module.split(".")[0]].add(rel)

stdlib = set(getattr(sys, "stdlib_module_names", ()))
local = {m for m in imports if m in LOCAL}
third = {m for m in imports if m not in stdlib and m not in local}

print(f"local modules   ({len(local)}): {sorted(local)}")
print()
print(f"third-party     ({len(third)}):")
for m in sorted(third):
    files = sorted(imports[m])
    print(f"  {m:<22s} {len(files):>2d} file(s)  e.g. {files[0]}")
print()
print("stdlib used:", sorted(m for m in imports if m in stdlib))