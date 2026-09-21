import sys, os
sys.path.insert(0, os.getcwd())
from agents.orchestrator import OrchestratorAgent, AgentUnavailableError

# Case A: react/hybrid mode but no working LLM -> must raise
class DeadLLM:  available = False
class FakeReact:  available = False
o = OrchestratorAgent(None, None, llm=DeadLLM(), cfg={"agent_mode":"hybrid"})
o.react = FakeReact()
try:
    o.preflight_or_raise()
    print("CASE A: FAIL - did not raise")
except AgentUnavailableError as e:
    print("CASE A: OK - raised:", str(e)[:70], "...")

# Case B: deterministic plan mode -> must NOT raise even with no react
o2 = OrchestratorAgent(None, None, llm=None, cfg={"agent_mode":"plan"})
o2.react = FakeReact()
try:
    o2.preflight_or_raise()
    print("CASE B: OK - plan mode did not raise")
except AgentUnavailableError as e:
    print("CASE B: FAIL - raised on deterministic:", e)
