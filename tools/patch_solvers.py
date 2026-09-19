"""One-off repair of a botched insert in reasoning/solvers.py."""
import io

PATH = "reasoning/solvers.py"
with io.open(PATH, "r", encoding="utf-8") as fh:
    lines = fh.readlines()

# Keep 1..252 (index 0..251), then re-emit the superlative tail, then resume at 304.
head = lines[:252]
tail = lines[303:]

mid = '''            "description": f"{spec.direction}(competitors) over {len(events)} events",
            "winner": best.title, "value": best.competitors, "runner_up_value": runner_up,
            "margin": margin,
        })
        return res

    # ---- temporal ----
    def solve_temporal(self, spec: QuerySpec) -> SolveResult:
        res = SolveResult(method="temporal_prev_edition")
        if spec.before_year is None:
            res.unresolved.append("before_year")
            return res
        season = spec.season or "Summer"
        year = _nearest_previous_games(season, spec.before_year, self.kg)
        res.steps.append({
            "operation": "temporal_resolution",
            "description": f"latest {season} Games strictly before {spec.before_year}",
            "resolved_year": year,
        })
        if year is None:
            res.unresolved.append("no_prior_games")
            return res
        sport = spec.sport or resolve_sport(spec.event_desc, self.kg)
        if not sport:
            res.unresolved.append("sport")
            return res
        events = self.kg.events_for(sport, year, season)
        res.candidates_considered = len(events)
        scored = sorted(((_event_similarity(spec.event_desc, e), e) for e in events),
                        key=lambda x: x[0], reverse=True)
        res.steps.append({
            "operation": "event_resolution",
            "description": f"match event '{spec.event_desc}' in {sport} {year} {season}",
            "candidates": [{"title": e.title, "score": round(s, 3)} for s, e in scored[:3]],
        })
        if not scored or scored[0][0] < 0.55:
            res.unresolved.append("event_not_matched")
            return res
        best_score, best = scored[0]
        res.answer = best.gold
        if not res.answer:
            res.unresolved.append("gold_missing_in_corpus")
            return res
        res.citations = [best.doc_id]
        res.evidence = [
            {"doc_id": best.doc_id, "title": best.title, "gold": best.gold,
             "silver": best.silver, "bronze": best.bronze, "games": best.games_key}
        ]
        res.confidence = round(min(0.95, 0.55 + 0.4 * best_score), 3)
        res.steps.append({
            "operation": "medal_lookup",
            "description": f"gold of {best.title}",
            "answer": res.answer,
        })
        return res

'''

with io.open(PATH, "w", encoding="utf-8", newline="\n") as fh:
    fh.writelines(head)
    fh.write(mid)
    fh.writelines(tail)
print("patched; line 250-256 now:")
with io.open(PATH, "r", encoding="utf-8") as fh:
    all_lines = fh.readlines()
for i in range(248, 258):
    print(f"{i+1:4d}| {all_lines[i].rstrip()}")