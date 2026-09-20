import json, sys
sys.stdout.reconfigure(encoding='utf-8')
ents = json.load(open('results/llm_results.json', encoding='utf-8'))
print('records:', len(ents))
pipes = ('RAG', 'GraphRAG', 'Agentic GraphRAG')
stats = {p: {'ok': 0, 'n': 0, 'tok': 0, 'lat': 0.0, 'with_calls': 0, 'contrib': 0} for p in pipes}
for e in ents:
    act = e.get('llm_activity', {})
    for p in pipes:
        r = e.get('pipelines', {}).get(p)
        if not r:
            continue
        s = stats[p]
        s['n'] += 1
        ok = (r.get('answer') in (e.get('gold') or {}).get('answers', []))
        s['ok'] += 1 if ok else 0
        s['tok'] += r.get('total_tokens', 0)
        s['lat'] += r.get('latency_ms', 0)
        if r.get('llm_calls', 0) or act:
            s['with_calls'] += 1
        if act.get('llm_contributed', {}).get(p) if isinstance(act.get('llm_contributed'), dict) else act.get('llm_contributed'):
            s['contrib'] += 1
for p, s in stats.items():
    if s['n']:
        print(f"{p:18s} acc={s['ok']}/{s['n']} = {100*s['ok']/s['n']:.0f}%  avg_tok={s['tok']/s['n']:.0f}  avg_lat={s['lat']/s['n']:.0f}ms  llm_contributed={s['contrib']}/{s['n']}")
# qtype breakdown for agentic
from collections import Counter, defaultdict
by = defaultdict(lambda: [0, 0])
for e in ents:
    r = e.get('pipelines', {}).get('Agentic GraphRAG')
    if r:
        ok = (r.get('answer') in (e.get('gold') or {}).get('answers', []))
        by[e.get('qtype', '?')][0] += 1 if ok else 0
        by[e.get('qtype', '?')][1] += 1
for q, (ok, n) in sorted(by.items()):
    print(f'  agentic {q:12s} {ok}/{n}')
llmkeys = [k for k in ents[0].keys() if 'llm' in k.lower()]
print('llm keys on record:', llmkeys)
