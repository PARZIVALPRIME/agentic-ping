import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8')
from config import config
from utils.llm import build_llm
llm = build_llm(config)
print('available:', llm.available, '| provider:', llm.provider, '| model:', llm.chat_model)
if not llm.available:
    sys.exit(1)
for i in (1, 2, 3):
    t = time.time()
    p = llm.complete_json('Return JSON {"answer":"pong%d"}' % i, 'You reply with JSON only.', caller='smoke%d' % i)
    print('call %d:' % i, p, '| %.1fs' % (time.time() - t))
print('calls:', llm.num_calls, 'failed:', llm.failed_calls)
