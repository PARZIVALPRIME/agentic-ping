import json
rows=json.load(open('results/batch/b_pub_all.json',encoding='utf-8'))
acc={}
for r in rows:
    for pn,pr in r.get('pipelines',{}).items():
        ev=pr.get('evaluation') or {}
        ic=ev.get('is_correct')
        if ic is None: continue
        acc.setdefault(pn,[0,0]); acc[pn][1]+=1; acc[pn][0]+= 1 if ic else 0
for k,(c,n) in sorted(acc.items()): print(f'{k:20s} {c:3d}/{n}  {100*c/n:.1f}%')
tok={}
for r in rows:
    for pn,pr in r.get('pipelines',{}).items():
        tok.setdefault(pn,[0,0]); tok[pn][0]+=pr.get('total_tokens',0); tok[pn][1]+=pr.get('llm_calls',0)
for k,(t,c) in sorted(tok.items()): print(f'{k:20s} tokens={t:,} llm_calls={c}')
