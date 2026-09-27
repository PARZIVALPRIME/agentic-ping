# TigerGraph Model Context Protocol (MCP) Integration & Setup Guide

The **TigerGraph MCP Server** (`tools/mcp_server.py`) provides an official, zero-dependency **Model Context Protocol (MCP)** interface to **TigerGraph Cloud 4.2.5 Enterprise (`OlympicsKG`)**.

It implements the official JSON-RPC 2.0 stdio protocol specification, exposing graph primitives, GSQL queries, conflict resolution, and autonomous multi-agent investigations directly to **Claude Desktop**, **Cursor**, **LangChain**, **CrewAI**, or custom agent systems.

---

## 1. Quick Verification (Automated Self-Test)

Verify the MCP server locally with zero external configuration:

```powershell
python tools/mcp_server.py --test
```

**Expected output:**
```
============================================================
 TigerGraph Model Context Protocol (MCP) Server Self-Test
============================================================
[kg] backend=tigergraph graph=OlympicsKG host=https://...
  PASS  initialize handshake
  PASS  tools/list returned 6 tools: agentic_investigate, detect_conflicts, tg_aggregate_stats, tg_event, tg_filter_events, tg_neighbours
  PASS  tg_filter_events -> 5 events
  PASS  tg_neighbours from Q743905 -> 10 hops
  PASS  tg_event -> Q743905 (Athletics at the 2008 Summer Olympics – Men's 110 metres hurdles)
  PASS  tg_aggregate_stats -> 295 events, span 1988-2020
  PASS  detect_conflicts -> Athlete B (authority_correction, uncertainty 0.05)
  PASS  resources/read tigergraph://schema/OlympicsKG
  PASS  prompts/list returned 3 prompts: investigate_olympic_question, adjudicate_conflicting_facts, explore_athlete_career
  PASS  prompts/get investigate_olympic_question
  PASS  TigerGraphMCPBridge tool dispatch verified (2 events)
============================================================
 ALL MCP SERVER CHECKS PASSED (100% OPERATIONAL)
============================================================
```

---

## 2. One-Click Setup for Claude Desktop

Add the server to your `claude_desktop_config.json`:
* **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
* **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "tigergraph-olympics": {
      "command": "python",
      "args": [
        "D:\\gg.worktrees\\install-claude-code-from-repo\\tools\\mcp_server.py"
      ],
      "env": {
        "TG_ENABLED": "1",
        "TG_HOST": "https://tg-fed265f1-0603-4e99-b6f3-6efd6d7fd5c5.tg-2635877100.i.tgcloud.io",
        "TG_USERNAME": "tigergraph",
        "TG_PASSWORD": "your_password_here"
      }
    }
  }
}
```

Restart Claude Desktop. You will see the **tigergraph-olympics** hammer icon with 6 tools and 2 resources available in the prompt composer.

---

## 3. One-Click Setup for Cursor

Add to your project's `.cursor/mcp.json` or Global Cursor Settings:

```json
{
  "mcpServers": {
    "tigergraph-olympics": {
      "command": "python",
      "args": ["tools/mcp_server.py"],
      "env": {
        "TG_ENABLED": "1"
      }
    }
  }
}
```

---

## 4. MCP Server Capabilities

### A. Graph & Investigation Tools (`tools/list`)

| Tool Name | Parameters | Description |
|---|---|---|
| `tg_filter_events` | `sport`, `venue`, `season`, `year_from`, `year_to`, `limit` | Filters Olympic event vertices on TigerGraph with deterministic ordering. |
| `tg_neighbours` | `vertex_id` (required), `edge_type`, `limit` | Traverses multi-hop edges (`HELD_AT`, `WON_BY`, `PART_OF`, `PREV`, `NEXT`). |
| `tg_event` | `doc_id` (required) | Returns structured attributes and medal winners for an event node. |
| `tg_aggregate_stats`| `sport`, `venue`, `season`, `year_from`, `year_to` | Executes server-side GSQL accumulators (`SumAccum`, `MinAccum`, `MaxAccum`). |
| `detect_conflicts` | `field_name`, `candidates` | Adjudicates conflicting records via 4-tier precedence with uncertainty score. |
| `agentic_investigate`| `question` (required) | Conducts an autonomous multi-turn investigation via cached Agentic GraphRAG. |

### B. Graph Resources (`resources/list`)

* `tigergraph://schema/OlympicsKG`: Full GSQL graph schema definition (vertices, edge types, and attributes).
* `tigergraph://stats/OlympicsKG`: Live cluster statistics (vertex counts, edge counts, and host status).

### C. Expert Prompts (`prompts/list`)

* `investigate_olympic_question`: System prompt guiding models on multi-hop investigation techniques over TigerGraph.
* `adjudicate_conflicting_facts`: System prompt for resolving historical disputes with the 4-tier hierarchy.
* `explore_athlete_career`: Prompt for mapping an athlete's career trajectory across sports and Games editions.

---

## 5. Programmatic Python Client Bridge (`TigerGraphMCPBridge`)

Python code, agents, and test scripts can invoke MCP primitives in-process:

```python
from tools.mcp_server import TigerGraphMCPBridge

bridge = TigerGraphMCPBridge()

# 1. Filter events on TigerGraph Cloud
events = bridge.call_tool("tg_filter_events", {"sport": "Athletics", "limit": 5})
print(f"Found {events['count']} events")

# 2. Traverse hops
hops = bridge.call_tool("tg_neighbours", {"vertex_id": "Q743905", "limit": 10})
print(f"Traversed {hops['num_hops']} relationship hops")

# 3. Adjudicate conflicts
res = bridge.call_tool("detect_conflicts", {
    "field_name": "gold_medallist",
    "candidates": [
        {"value": "Marion Jones", "year": 2000, "text": "Won 100m gold"},
        {"value": "Stripped", "year": 2007, "text": "Disqualified for doping by IOC EB"},
    ],
})
print(f"Resolved to: {res['resolved']} via {res['rule']} (uncertainty: {res['uncertainty']})")
```
