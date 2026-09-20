import json
rows=json.load(open('results/batch/b_pub_all.json',encoding='utf-8'))
miss=[r['qid'] for r in rows if 'RAG (vector only)' not in r['pipelines']]
print('missing vector-only on', len(miss), miss[:12])
