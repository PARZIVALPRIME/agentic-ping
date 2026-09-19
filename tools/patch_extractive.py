"""Repair pipelines/extractive.py: move misplaced methods back into the class."""
import io

PATH = "pipelines/extractive.py"
with io.open(PATH, "r", encoding="utf-8") as fh:
    lines = fh.readlines()

head = lines[:157]          # 1..157  (through end of _aggregation)
methods = lines[163:234]    # 164..234 (_superlative, _temporal, _multi_hop)

helpers = '''

def _title_year(title: str) -> Optional[int]:
    m = re.search(r"\\b(19\\d{2}|20\\d{2})\\b", title or "")
    return int(m.group(1)) if m else None


def _title_matches_games(title: str, spec: QuerySpec) -> bool:
    """Does a retrieved title correspond to (sport, year, season)?"""
    if spec.year and str(spec.year) not in title:
        return False
    if spec.season and spec.season.lower() not in title.lower():
        return False
    if spec.sport:
        ns = normalize(spec.sport)
        nt = normalize(title)
        if ns not in nt and not any(tok in nt for tok in ns.split() if len(tok) > 3):
            return False
    return True


def _date_of(fields: Dict[str, str]) -> str:
    return f"{fields.get('date', '')} {fields.get('dates', '')}".strip()
'''

with io.open(PATH, "w", encoding="utf-8", newline="\n") as fh:
    fh.writelines(head)
    fh.writelines(methods)
    fh.write(helpers)

import ast
ast.parse(io.open(PATH, encoding="utf-8").read())
print("extractive.py repaired and parses OK")