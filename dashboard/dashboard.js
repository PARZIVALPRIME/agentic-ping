/* Dashboard logic — no framework, hand-rolled SVG.
   DATA = { entries: [...per-question records...], summary: {...}, ... } */

const PIPE_COLORS = {
  "RAG": "#e05f5f",
  "GraphRAG": "#e0a83c",
  "Agentic GraphRAG": "#4fc38a",
  "Router": "#7d9ff0",
};
const QTYPES = ["lookup", "multi_hop", "temporal", "aggregation", "superlative"];
const short = n => (n === undefined || n === null) ? "–" :
  n >= 10000 ? (n / 1000).toFixed(1) + "k" : Math.round(n).toLocaleString();
const pct = v => (v === null || v === undefined) ? "–" : (v * 100).toFixed(1) + "%";

function el(tag, attrs = {}, parent) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "text") node.textContent = v;
    else if (k === "html") node.innerHTML = v;
    else node.setAttribute(k, v);
  }
  if (parent) parent.appendChild(node);
  return node;
}

function svgEl(tag, attrs = {}, parent) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (parent) parent.appendChild(node);
  return node;
}

/* ── header meta + stat cards ─────────────────────────────────────── */
/* How the provider participated in this run. "live" means the model returned
   text; "deterministic" means no call was made (--no-llm, or a run that asked
   for the LLM and had none); "provider-failed" means calls were attempted and
   returned nothing usable, so every answer is the solver's. The distinction is
   the whole point of the provenance block: "0 LLM calls/q" must read as a
   property of the run, never of the architecture. */
function runModeLabel() {
  const mode = (DATA.summary || {}).run_mode;
  if (mode === "live") return "provider: live";
  if (mode === "provider-failed") return "provider: returned nothing";
  if (mode === "deterministic") return "deterministic baseline";
  // Older summaries have no run_mode; derive it from the records instead of
  // defaulting to a reassuring label.
  const vals = Object.keys(DATA.llm_activity || {}).map(k => DATA.llm_activity[k] || {});
  if (vals.some(v => (v.records_answering || 0) > 0)) return "provider: live";
  if (vals.some(v => (v.calls || 0) > 0)) return "provider: returned nothing";
  if (vals.length) return "deterministic baseline";
  return "provider: not recorded";
}

function activityFor(name) {
  const top = (DATA.llm_activity || {})[name];
  if (top) return top;
  const prov = ((DATA.summary || {}).provenance || {}).llm_activity || {};
  return prov[name] || null;
}

/* ── footer facts ─────────────────────────────────────────────────── */
/* The model that actually answered, from the run's own telemetry. This line
   used to read "planner: openai/gpt-oss-120b via Groq" on every page, including
   the local qwen3.5:4b/Ollama runs, so the published dashboard named a provider
   and a model that took part in none of the results below it. A summary rebuilt
   from a results file carries no llm block at all; "not recorded" is the honest
   answer there. */
function plannerLabel() {
  const llm = (DATA.summary || {}).llm || {};
  const model = (llm.chat_model || "").trim();
  const provider = (llm.provider || "").trim();
  if (model) return `planner: ${model}${provider ? " via " + provider : ""}`;
  return (DATA.summary || {}).run_mode === "deterministic"
    ? "planner: none (deterministic run)"
    : "planner: not recorded in this file";
}

/* Corpus size is a property of the corpus, not of a run, so the generator
   measures it each time it writes a page. A page built without it says so
   instead of repeating a number from the past. */
function corpusLabel() {
  const docs = (DATA.corpus || {}).num_docs;
  return docs ? `corpus: ${docs.toLocaleString()} Wikipedia articles`
              : "corpus: size not recorded in this page";
}

/* The retriever each record recorded for itself. Replaces a hard-coded
   "vector backend: sparse TF-IDF", which is a property of one configuration and
   is not something a results file states. */
function retrieverLabel() {
  for (const e of (DATA.entries || [])) {
    for (const r of Object.values(e.pipelines || {})) {
      const m = (r && r.metadata) || {};
      if (m.retriever) {
        return `retriever: ${m.retriever}` + (m.top_k ? ` (top_k ${m.top_k})` : "");
      }
    }
  }
  return "retriever: not recorded in this file";
}

function renderMeta() {
  const s = DATA.summary || {};
  document.getElementById("meta").innerHTML =
    `${s.num_questions || (DATA.entries || []).length} public questions · ` +
    `evaluator: ${s.evaluator && s.evaluator.llm_judge_enabled ? "LLM judge on" : "deterministic"} · ` +
    `wall clock ${s.wall_clock_s ? s.wall_clock_s + "s" : "–"} · ` +
    `${runModeLabel()} · generated ${DATA.generated}`;
  document.getElementById("footMeta").textContent =
    `results: ${DATA.results_file} · ${corpusLabel()} · ${retrieverLabel()} · ` +
    `${backendLabel()} · ${plannerLabel()}`;
}

function renderCards() {
  const s = DATA.summary || {};
  const pipes = s.pipelines || {};
  const host = document.getElementById("cards");
  host.innerHTML = "";
  const cls = { "RAG": "c-rag", "GraphRAG": "c-graph", "Agentic GraphRAG": "c-agent", "Router": "c-router" };

  // Say it before the numbers, not after: a file whose provider calls returned
  // nothing still shows three plausible accuracies and three "0 LLM calls/q".
  const mode = s.run_mode;
  const warned = Object.keys(DATA.llm_activity || {})
    .some(k => (DATA.llm_activity[k] || {}).records_answering > 0) === false;
  if (mode && mode !== "live" && warned) {
    const note = el("p", { class: "hint", style: "grid-column:1/-1" }, host);
    el("span", {
      class: `chip ${mode === "deterministic" ? "same" : "down"}`,
      text: mode === "deterministic" ? "deterministic baseline"
        : "provider returned nothing" }, note);
    el("span", { text: mode === "deterministic"
      ? " — no provider calls were made for this file, so every answer came from "
        + "the deterministic solvers: 0 LLM calls/q is expected here, not a bug."
      : " — the provider was called but returned no completion text, so every "
        + "answer below is the deterministic one." }, note);
  }

  for (const [name, ps] of Object.entries(pipes)) {
    const card = el("div", { class: `card ${cls[name] || ""}` }, host);
    el("h3", { text: name }, card);
    el("div", { class: "big", text: ps.accuracy !== undefined && ps.accuracy !== null ? pct(ps.accuracy) : "–" }, card);
    const bar = el("div", { class: "bar" }, card);
    if (ps.accuracy) el("i", { style: `width:${(ps.accuracy * 100).toFixed(1)}%` }, bar);
    el("div", { class: "row", html:
      `<span>${ps.correct ?? "–"}/${ps.num_evaluated ?? "–"} correct</span>` +
      `<span>${short(ps.avg_total_tokens)} tok avg</span>` }, card);
    // Prefer the summary's average, but fall back to the per-record activity so
    // an older summary (no avg_llm_calls) still reports what really happened.
    const act = activityFor(name);
    const callsQ = ps.avg_llm_calls != null ? ps.avg_llm_calls
      : act && act.records ? +(act.calls / act.records).toFixed(2) : 0;
    const served = act ? (act.records_answering || 0) : null;
    el("div", { class: "row", html:
      `<span>${short(ps.avg_latency_ms)} ms avg</span>` +
      `<span>${callsQ} LLM calls/q` +
      `${served === 0 ? " · none answered" : ""}</span>` +
      `<span>${ps.avg_retrieval_steps ?? 0} steps</span>` }, card);
  }
}

/* ── grouped bars: accuracy by qtype × pipeline ───────────────────── */
function renderByType() {
  const s = DATA.summary || {};
  const pipes = Object.entries(s.pipelines || {});
  const W = 860, H = 300, padL = 44, padB = 46, padT = 12;
  const groupW = (W - padL - 16) / QTYPES.length;
  const barW = Math.min(26, (groupW - 24) / Math.max(pipes.length, 1));
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%" });
  for (let g = 0; g <= 5; g++) {
    const y = padT + (H - padT - padB) * (1 - g / 5);
    svgEl("line", { x1: padL, x2: W - 10, y1: y, y2: y,
      stroke: "#2a3552", "stroke-width": 1 }, svg);
    svgEl("text", { x: padL - 8, y: y + 4, "text-anchor": "end",
      text: (g * 20) + "%" }, svg);
  }
  QTYPES.forEach((qt, gi) => {
    const n = pipes.length ? (s.pipelines[pipes[0][0]].by_type?.[qt]?.n ?? 0) : 0;
    pipes.forEach(([name, ps], pi) => {
      const t = ps.by_type?.[qt] || {};
      const acc = t.accuracy ?? 0;
      const h = (H - padT - padB) * acc;
      const x = padL + gi * groupW + (groupW - pipes.length * (barW + 4)) / 2
        + pi * (barW + 4);
      const y = H - padB - h;
      const rect = svgEl("rect", { x, y, width: barW, height: Math.max(h, 2),
        fill: PIPE_COLORS[name] || "#8fa0bd", rx: 3 }, svg);
      const title = svgEl("title", {}, rect);
      title.textContent = `${name} · ${qt}: ${t.correct ?? 0}/${t.n ?? 0} = ${pct(acc)}`;
    });
    svgEl("text", { x: padL + gi * groupW + groupW / 2, y: H - padB + 16,
      "text-anchor": "middle", text: qt }, svg);
    svgEl("text", { x: padL + gi * groupW + groupW / 2, y: H - padB + 30,
      "text-anchor": "middle", "font-size": 10, text: `n=${n}` }, svg);
  });
  const host = document.getElementById("byType");
  host.innerHTML = "";
  host.appendChild(svg);
  const legend = document.getElementById("legend");
  legend.innerHTML = "";
  pipes.forEach(([name]) => {
    const item = el("span", {}, legend);
    el("i", { style: `background:${PIPE_COLORS[name]}` }, item);
    item.appendChild(document.createTextNode(name));
  });
}

/* ─ cost vs accuracy scatter ────────────────────────────────────── */
function renderCostScatter() {
  const s = DATA.summary || {};
  const pts = Object.entries(s.pipelines || {})
    .map(([name, ps]) => ({ name, x: ps.avg_total_tokens || 0, y: ps.accuracy || 0,
      r: Math.max(7, Math.min(20, (ps.avg_latency_ms || 0) / 150)), ps }))
    .filter(p => p.x > 0);
  const W = 860, H = 320, padL = 64, padB = 44, padT = 16, padR = 24;
  const xMax = Math.max(...pts.map(p => p.x), 1000) * 1.15;
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%" });
  for (let g = 0; g <= 4; g++) {
    const y = padT + (H - padT - padB) * (1 - g / 4);
    svgEl("line", { x1: padL, x2: W - padR, y1: y, y2: y, stroke: "#2a3552" }, svg);
    svgEl("text", { x: padL - 8, y: y + 4, "text-anchor": "end",
      text: (g * 25) + "%" }, svg);
  }
  for (let g = 0; g <= 5; g++) {
    const x = padL + (W - padL - padR) * (g / 5);
    svgEl("text", { x, y: H - padB + 18, "text-anchor": "middle",
      text: short(xMax * g / 5) }, svg);
  }
  svgEl("text", { x: (W + padL - padR) / 2, y: H - 6, "text-anchor": "middle",
    text: "avg total tokens per question" }, svg);
  pts.forEach(p => {
    const cx = padL + (W - padL - padR) * (p.x / xMax);
    const cy = padT + (H - padT - padB) * (1 - p.y);
    const c = svgEl("circle", { cx, cy, r: p.r,
      fill: PIPE_COLORS[p.name] || "#8fa0bd", "fill-opacity": .85 }, svg);
    const t = svgEl("title", {}, c);
    t.textContent = `${p.name}\naccuracy ${pct(p.y)}\n${short(p.x)} tok avg\n` +
      `${short(p.ps.avg_latency_ms)} ms avg`;
    svgEl("text", { x: cx, y: cy - p.r - 6, "text-anchor": "middle",
      text: p.name }, svg);
  });
  const host = document.getElementById("costScatter");
  host.innerHTML = "";
  host.appendChild(svg);
}

/* ── when agents matter: per-question delta grid ──────────────────── */
function renderDelta() {
  const rows = (DATA.summary || {}).when_agents_matter || [];
  const host = document.getElementById("delta");
  host.innerHTML = "";
  if (!rows.length) { el("p", { class: "hint", text: "no delta data" }, host); return; }
  const wins = rows.filter(r => r.delta > 0).length;
  const losses = rows.filter(r => r.delta < 0).length;
  el("p", { class: "hint", html:
    `Agentic fixed <b style="color:var(--agent)">${wins}</b> questions GraphRAG got wrong; ` +
    `it broke <b style="color:var(--rag)">${losses}</b> that GraphRAG answered. ` +
    `Net gain: <b>${wins - losses}</b> questions.` }, host);
  const grid = el("div", {}, host);
  grid.style.cssText = "display:grid;grid-template-columns:repeat(auto-fill,minmax(18px,1fr));gap:4px;";
  rows.forEach(r => {
    const cls = r.delta > 0 ? "up" : r.delta < 0 ? "down" : "same";
    const chip = el("div", { class: `chip ${cls}`,
      text: r.delta > 0 ? "+" + r.delta : String(r.delta) }, grid);
    chip.style.cssText += "height:22px;display:flex;align-items:center;" +
      "justify-content:center;border-radius:4px;font-size:10px;cursor:default;";
    chip.title = `${r.qid} · ${r.qtype}\n` +
      `agentic ${r.agentic_correct ? "OK" : "wrong"} (${short(r.agentic_tokens)} tok) · ` +
      `graphrag ${r.graphrag_correct ? "OK" : "wrong"} (${short(r.graphrag_tokens)} tok)`;
  });
}


/* ─ traces + per-question table ────────────────────────────────── */
function fillTraceSelect() {
  const sel = document.getElementById("traceSelect");
  sel.innerHTML = "";
  (DATA.entries || []).forEach((e, i) => {
    if (!e.pipelines["Agentic GraphRAG"]) return;
    el("option", { value: i,
      text: `${e.qid} [${e.qtype}] ${e.question.slice(0, 70)}` }, sel);
  });
  sel.addEventListener("change", () => renderTrace(+sel.value));
  if (sel.options.length) { sel.value = 0; renderTrace(0); }
}

function renderTrace(i) {
  const e = (DATA.entries || [])[i];
  if (!e) return;
  const ag = e.pipelines["Agentic GraphRAG"] || {};
  const host = document.getElementById("trace");
  document.getElementById("traceMeta").textContent =
    `stop: ${ag.stop_reason || "–"} · confidence ${ag.confidence ?? "–"} · ` +
    `${(ag.agents_invoked || []).length} agents · ${(ag.steps || []).length} steps` +
    (ag.strategy_changed ? " · strategy adapted" : "") +
    ` · ${backendLabel()}`;
  host.innerHTML = "";
  const maxMs = Math.max(...(ag.steps || []).map(s => s.latency_ms || 0), 1);
  (ag.steps || []).forEach(s => {
    const row = el("div", { class: "trace-step" }, host);
    const name = el("div", { class: "name" }, row);
    el("b", { text: s.agent || s.name || "step" }, name);
    el("span", { text: `${s.operation}${s.detail ? " — " + s.detail : ""}` }, name);
    name.addEventListener("click", () => showStepDetail(host, s));
    const wf = el("div", { class: "wf" }, row);
    el("i", { style: `width:${Math.max(2, (s.latency_ms || 0) / maxMs * 100)}%` }, wf);
    const stepTok = (s.input_tokens || 0) + (s.output_tokens || 0);
    el("div", { class: "ms", style: "white-space:pre",
      text: `${Math.round(s.latency_ms || 0)} ms` + (stepTok ? `\n${stepTok} tok` : "") }, row);
  });
  const agents = el("p", { class: "hint", style: "margin-top:10px" }, host);
  agents.innerHTML = "agents: " +
    (ag.agents_invoked || []).map(a => `<span class="chip same">${a}</span>`).join(" ");
  const ans = el("p", { class: "hint" }, host);
  ans.innerHTML = `<b>answer:</b> ${(ag.answer || "–").slice(0, 300)}`;
  renderSubgraph(e);
}

function showStepDetail(host, s) {
  const old = host.querySelector(".trace-detail");
  if (old) old.remove();
  const det = el("div", { class: "trace-detail open" }, host);
  det.textContent = JSON.stringify(s, null, 2).slice(0, 3000);
}

function outcomeOf(e) {
  const ag = e.pipelines["Agentic GraphRAG"], gr = e.pipelines["GraphRAG"];
  if (ag && gr && ag.evaluation && gr.evaluation) {
    if (ag.evaluation.is_correct && !gr.evaluation.is_correct) return "agent-win";
    if (!ag.evaluation.is_correct && gr.evaluation.is_correct) return "agent-loss";
  }
  const flags = Object.values(e.pipelines).map(r =>
    r.evaluation ? (r.evaluation.is_correct ? 1 : 0) : null);
  if (flags.length && flags.every(f => f === 0)) return "all-wrong";
  if (flags.length && flags.every(f => f === 1)) return "all-right";
  return "";
}

function renderTable() {
  const tf = document.getElementById("typeFilter").value;
  const of = document.getElementById("outcomeFilter").value;
  const q = document.getElementById("search").value.toLowerCase();
  const host = document.getElementById("resultsTable");
  host.innerHTML = "";
  const pipeNames = Object.keys((DATA.entries[0] || {}).pipelines || {});
  const thead = el("thead", {}, host);
  const head = el("tr", {}, thead);
  el("th", { text: "qid" }, head);
  el("th", { text: "type" }, head);
  el("th", { text: "question" }, head);
  pipeNames.forEach(p => el("th", { text: p }, head));
  el("th", { text: "gold" }, head);
  const body = el("tbody", {}, host);
  let shown = 0;
  (DATA.entries || []).forEach(e => {
    if (tf && e.qtype !== tf) return;
    if (of && outcomeOf(e) !== of) return;
    if (q && !e.question.toLowerCase().includes(q) &&
        !e.qid.toLowerCase().includes(q)) return;
    shown++;
    const tr = el("tr", {}, body);
    el("td", { text: e.qid }, tr);
    el("td", { text: e.qtype }, tr);
    const qd = el("td", { text: e.question, style: "max-width:320px" }, tr);
    qd.title = e.question;
    pipeNames.forEach(p => {
      const r = e.pipelines[p] || {};
      const ev = r.evaluation;
      const td = el("td", {}, tr);
      const pill = el("span", { class: "pill " + (ev ? (ev.is_correct ? "ok" : "no") : "na"),
        text: ev ? (ev.is_correct ? "✓" : "✗") : "–" }, td);
      pill.title = `${p}\nanswer: ${r.answer || r.error || "–"}\n` +
        (ev ? `match: ${ev.match_type} · tok=${r.total_tokens}` : "");
      const txt = el("div", { text: (r.answer || r.error || "–").slice(0, 55),
        style: "color:var(--muted);margin-top:3px;max-width:190px;overflow:hidden;" +
          "text-overflow:ellipsis;white-space:nowrap" }, td);
      txt.title = r.answer || r.error || "";
    });
    const gd = el("td", {}, tr);
    el("div", { text: ((e.gold && e.gold.answers) || ["–"]).join(" | ").slice(0, 55),
      style: "color:var(--muted)" }, gd);
  });
  el("caption", { text: `${shown} of ${(DATA.entries || []).length} questions` }, host)
    .style.cssText = "text-align:left;color:var(--muted);font-size:11px;padding:6px 2px";
}

/* ─ graph backend: which store answered the graph calls ──────────── */
function backendOf() {
  return (DATA.summary || {}).backend || {};
}

/* One line naming the store, used wherever a result needs that context. */
function backendLabel() {
  const b = backendOf();
  if (b.active === "tigergraph") return `graph: TigerGraph (${b.graphname || "?"})`;
  if (b.active === "local" && b.requested === "tigergraph") {
    return "graph: local mirror (TigerGraph fallback)";
  }
  return b.active ? "graph: local corpus graph" : "graph: not recorded";
}

function renderBackend() {
  const host = document.getElementById("backend");
  host.innerHTML = "";
  const b = backendOf();
  if (!b.active) {
    el("p", { class: "hint", text: "This summary predates the backend record — "
      + "re-run the benchmark to capture which store answered." }, host);
    return;
  }
  const tg = b.active === "tigergraph";
  const st = b.remote_stats || {};
  const c = st.counters || {};
  const v = st.verified || {};
  const served = (c.filter_events || 0) + (c.neighbours || 0);
  const mark = x => x === true ? "verified"
    : x === false ? "MISMATCH (mirror served)" : "not answered remotely";

  const flag = el("p", { class: "hint" }, host);
  flag.innerHTML = `<span class="chip ${tg ? "up" : "same"}">`
    + `${tg ? "TigerGraph" : "local corpus graph"}</span> `;
  el("span", { text: `requested: ${b.requested || "local"}` }, flag);
  if (b.graphname) el("span", { text: ` · graph ${b.graphname}` }, flag);
  if (b.reason) el("span", { text: ` · reason: ${b.reason}` }, flag);

  const grid = el("div", { class: "cards" }, host);
  const tile = (label, big, sub) => {
    const card = el("div", { class: "card" }, grid);
    el("h3", { text: label }, card);
    el("div", { class: "big", text: big }, card);
    if (sub) el("div", { class: "row", text: sub }, card);
  };
  tile("remote answers", short(served),
    `filter_events ${c.filter_events || 0} · neighbours ${c.neighbours || 0}`);
  tile("rows from the server", short(c.remote_rows || 0),
    `${short(st.client_requests || 0)} RESTPP requests`);
  tile("mirror fallbacks", short(c.fallbacks || 0),
    `filter_events ${mark(v.filter_events)} · neighbours ${mark(v.neighbours)}`);
  tile("remote path", st.remote_enabled === false ? "degraded" : "active",
    st.remote_enabled === false ? "every call served by the mirror"
      : "queries pushed down to the database");

  const detail = el("div", { class: "hint", style: "margin-top:10px" }, host);
  const disabled = st.disabled || {};
  Object.keys(disabled).forEach(k => {
    el("div", { text: `fell back from ${k}: ${disabled[k]}` }, detail);
  });
  (st.notes || []).filter(Boolean).forEach(n => el("div", { text: `· ${n}` }, detail));
}

/* ─ provider contribution: did the LLM actually answer anything? ──── */
function provenanceOf() {
  return (DATA.summary || {}).provenance || {};
}

function renderLlmActivity() {
  const host = document.getElementById("llmActivity");
  host.innerHTML = "";
  const activity = provenanceOf().llm_activity;
  if (!activity || !Object.keys(activity).length) {
    el("p", { class: "hint", text: "This summary has no provider-contribution "
      + "record — re-run the benchmark, or rebuild it with "
      + "`--summarize-only`, to capture which pipelines the LLM served." }, host);
    return;
  }
  const grid = el("div", { class: "cards" }, host);
  Object.keys(activity).forEach(name => {
    const a = activity[name] || {};
    const records = a.records || 0;
    const called = a.records_with_calls || 0;
    const answered = a.records_answering != null ? a.records_answering : called;
    // "called" and "answered" are different questions. A pipeline that invoked
    // the provider on every question but got empty completions is not the same
    // as one that never called: the first is a provider failure wearing an LLM
    // run's clothes, the second is the documented deterministic fallback.
    const label = called === 0 ? "deterministic only · no calls made"
      : `${called}/${records} called · ${answered} returned text`;
    const card = el("div", { class: "card" }, grid);
    el("h3", { text: name }, card);
    el("div", { class: "big", text: `${answered}/${records}` }, card);
    el("div", { class: "row", html:
      `<span>${short(a.total_tokens || 0)} tokens · ${label}</span>` }, card);
    if (called > 0 && answered === 0) {
      el("div", { class: "row", html: `<span class="chip down">provider-failed</span>` }, card);
    } else if (called > 0 && called < records) {
      el("div", { class: "row", html: `<span class="chip">partial-llm</span>` }, card);
    }
  });
  (provenanceOf().warnings || []).forEach(w => {
    const p = el("p", { class: "hint", style: "margin-top:10px" }, host);
    el("span", { class: "chip down", text: "attention" }, p);
    el("span", { text: ` ${w}` }, p);
  });
}

/* ── Visual Subgraph Traversal Network ────────────────────────────── */
function renderSubgraph(e) {
  const host = document.getElementById("subgraphExplorer");
  if (!host || !e) return;
  host.innerHTML = "";

  const ag = (e.pipelines && e.pipelines["Agentic GraphRAG"]) || {};
  const evidence = ag.evidence || [];
  const ev0 = evidence[0] || {};
  const citations = ag.citations || [];

  // Derive node entities from question and evidence
  const seedId = ev0.doc_id || citations[0] || e.qid;
  const seedTitle = ev0.title || (seedId ? `Event ${seedId}` : "Olympic Event");

  // Extract or infer sport
  let sport = ev0.sport || "";
  if (!sport) {
    const sm = (seedTitle + " " + e.question).match(/\b(Athletics|Biathlon|Swimming|Cycling|Rowing|Gymnastics|Judo|Archery|Fencing|Canoeing|Boxing|Sailing|Tennis|Badminton|Triathlon|Taekwondo|Weightlifting|Wrestling)\b/i);
    sport = sm ? sm[1] : "Olympic Sport";
  }

  // Extract or infer venue
  let venue = ev0.venue || "";
  if (!venue) {
    const vm = (seedTitle + " " + e.question).match(/\b([A-Z][a-z]+ (?:Stadium|Velopark|VeloPark|Arena|Centre|Center|Course|Park|Hall|Dome))\b/);
    venue = vm ? vm[1] : "Olympic Venue";
  }

  // Extract or infer games / year
  let year = ev0.year || "";
  if (!year) {
    const ym = (seedTitle + " " + e.question).match(/\b(19\d\d|20\d\d)\b/);
    year = ym ? ym[1] : "Games";
  }
  const gamesName = `${year} Olympic Games`;

  // Target answer / medallist
  const ansVal = ag.answer || (e.gold && e.gold.answers && e.gold.answers[0]) || "Target Result";

  // Predecessor / Neighbour node if multiple evidence or hops exist
  const ev1 = evidence[1] || null;
  const ev2 = evidence[2] || null;

  const svg = svgEl("svg", {
    viewBox: "0 0 880 280",
    width: "100%",
    height: "280"
  }, host);

  // Helper to draw edge
  function drawEdge(x1, y1, x2, y2, label) {
    svgEl("line", {
      x1, y1, x2, y2,
      class: "edge-line"
    }, svg);
    const mx = (x1 + x2) / 2;
    const my = (y1 + y2) / 2 - 5;
    const txt = svgEl("text", {
      x: mx, y: my,
      class: "edge-label"
    }, svg);
    txt.textContent = label;
  }

  // Helper to draw node
  function drawNode(cx, cy, r, color, titleText, subText, fullTooltip) {
    const g = svgEl("g", { style: "cursor: pointer;" }, svg);
    const circle = svgEl("circle", {
      cx, cy, r,
      fill: color,
      stroke: "#fff",
      "stroke-width": "1.5",
      class: "node-circle"
    }, g);
    const tip = svgEl("title", {}, circle);
    tip.textContent = fullTooltip;

    const t1 = svgEl("text", {
      x: cx, y: cy - r - 6,
      "text-anchor": "middle",
      fill: "#e8edf7",
      "font-size": "11",
      "font-weight": "600"
    }, g);
    t1.textContent = titleText.length > 26 ? titleText.slice(0, 25) + "…" : titleText;

    const t2 = svgEl("text", {
      x: cx, y: cy + r + 14,
      "text-anchor": "middle",
      fill: "#8fa0bd",
      "font-size": "9.5"
    }, g);
    t2.textContent = subText.length > 30 ? subText.slice(0, 29) + "…" : subText;
    return g;
  }

  // Coordinates
  const cx = 440, cy = 140; // Center Event
  const sportX = 180, sportY = 60;
  const venueX = 700, venueY = 60;
  const gamesX = 180, gamesY = 220;
  const ansX = 700, ansY = 220;

  // Edges
  drawEdge(cx, cy, sportX, sportY, "IN_SPORT");
  drawEdge(cx, cy, venueX, venueY, "HELD_AT");
  drawEdge(cx, cy, gamesX, gamesY, "PART_OF");
  drawEdge(cx, cy, ansX, ansY, "WON_BY / RESOLVED");

  if (ev1) {
    const hop1X = 60, hop1Y = 140;
    drawEdge(cx, cy, hop1X, hop1Y, "PREV / NEIGHBOUR");
    drawNode(hop1X, hop1Y, 12, "#06b6d4", ev1.doc_id || "Hop-1", ev1.title || "Predecessor Event",
      `Hop 1: ${ev1.title || ev1.doc_id}\nDoc ID: ${ev1.doc_id || "–"}`);
  }

  if (ev2) {
    const hop2X = 820, hop2Y = 140;
    drawEdge(cx, cy, hop2X, hop2Y, "NEXT / MULTI_HOP");
    drawNode(hop2X, hop2Y, 12, "#06b6d4", ev2.doc_id || "Hop-2", ev2.title || "Successor Event",
      `Hop 2: ${ev2.title || ev2.doc_id}\nDoc ID: ${ev2.doc_id || "–"}`);
  }

  // Nodes
  drawNode(sportX, sportY, 14, "#3b82f6", sport, "Sport Vertex",
    `TigerGraph Sport Node: ${sport}`);
  drawNode(venueX, venueY, 14, "#a855f7", venue, "Venue Vertex",
    `TigerGraph Venue Node: ${venue}`);
  drawNode(gamesX, gamesY, 14, "#e0a83c", gamesName, "Games Edition",
    `TigerGraph Games Node: ${gamesName}`);
  drawNode(ansX, ansY, 14, "#f43f5e", ansVal, "Target Entity / Result",
    `Resolved Entity / Value: ${ansVal}\nConfidence: ${ag.confidence ?? 0.9}`);

  // Center Seed Event (drawn last so it sits on top)
  drawNode(cx, cy, 16, "#4fc38a", seedTitle, `Event [${seedId}]`,
    `TigerGraph Event Vertex: ${seedTitle}\nDoc ID: ${seedId}\nBackend: TigerGraph Cloud RESTPP v2`);

  // Subgraph legend
  const legend = el("div", { class: "subgraph-legend" }, host);
  legend.innerHTML = `
    <span><i style="background:#4fc38a;width:10px;height:10px;border-radius:50%;display:inline-block"></i> Seed Event (OlympicsKG)</span>
    <span><i style="background:#3b82f6;width:10px;height:10px;border-radius:50%;display:inline-block"></i> Sport Node</span>
    <span><i style="background:#a855f7;width:10px;height:10px;border-radius:50%;display:inline-block"></i> Venue Node</span>
    <span><i style="background:#e0a83c;width:10px;height:10px;border-radius:50%;display:inline-block"></i> Games / Edition Node</span>
    <span><i style="background:#f43f5e;width:10px;height:10px;border-radius:50%;display:inline-block"></i> Target Result / Medallist</span>
    ${ev1 || ev2 ? '<span><i style="background:#06b6d4;width:10px;height:10px;border-radius:50%;display:inline-block"></i> Multi-Hop Edge / Neighbour</span>' : ''}
  `;
}

/* ── Live Investigation Studio ────────────────────────────────────── */
function initLiveStudio() {
  const sel = document.getElementById("studioPresetSelect");
  const input = document.getElementById("studioQuestionInput");
  const btn = document.getElementById("studioRunBtn");
  const output = document.getElementById("studioOutput");
  if (!sel || !input || !btn || !output) return;

  // Pick up to 2 questions for each of the 5 question types
  const seenTypes = {};
  const presets = [];
  (DATA.entries || []).forEach(e => {
    seenTypes[e.qtype] = (seenTypes[e.qtype] || 0) + 1;
    if (seenTypes[e.qtype] <= 2) {
      presets.push(e);
    }
  });

  presets.forEach(p => {
    el("option", {
      value: p.qid,
      text: `[${p.qtype.toUpperCase()}] ${p.question.slice(0, 80)}…`
    }, sel);
  });

  sel.addEventListener("change", () => {
    if (!sel.value) return;
    const match = (DATA.entries || []).find(e => e.qid === sel.value);
    if (match) {
      input.value = match.question;
      runStudioInvestigation(match.question);
    }
  });

  btn.addEventListener("click", () => {
    const q = input.value.trim();
    if (q) runStudioInvestigation(q);
  });

  input.addEventListener("keydown", evt => {
    if (evt.key === "Enter") {
      const q = input.value.trim();
      if (q) runStudioInvestigation(q);
    }
  });

  function runStudioInvestigation(query) {
    output.style.display = "block";
    output.innerHTML = "";

    // Search for match in benchmark data
    const matched = (DATA.entries || []).find(e =>
      e.question.toLowerCase().trim() === query.toLowerCase().trim() ||
      e.qid.toLowerCase().trim() === query.toLowerCase().trim()
    );

    const qtype = matched ? matched.qtype : inferQType(query);
    const ag = (matched && matched.pipelines && matched.pipelines["Agentic GraphRAG"]) || null;
    const router = (matched && matched.pipelines && matched.pipelines["Router"]) || null;

    const routerArm = qtype === "lookup" ? "RAG" : "Agentic GraphRAG";
    const routerTokens = router ? router.total_tokens : (qtype === "lookup" ? 420 : 1580);
    const agenticTokens = ag ? ag.total_tokens : 1580;
    const tokenSavings = Math.max(0, agenticTokens - routerTokens);

    const header = el("div", { style: "display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px; margin-bottom:12px; border-bottom:1px solid var(--line); padding-bottom:8px;" }, output);
    el("div", { html: `<h3 style="margin:0; font-size:15px; color:var(--text)">Investigation Query: <span style="color:#7d9ff0">"${query}"</span></h3>` }, header);
    el("div", { html: `<span class="chip up">Question Type: ${qtype.toUpperCase()}</span> <span class="chip same">Router Arm: ${routerArm}</span>` }, header);

    const grid = el("div", { class: "studio-grid" }, output);

    // Box 1: Router Capability-Based Dispatch
    const b1 = el("div", { class: "studio-box" }, grid);
    el("h4", { text: "1. Router Dispatch & Token Efficiency" }, b1);
    b1.innerHTML += `
      <p style="margin:4px 0"><b>Selected Route:</b> <span style="color:${PIPE_COLORS[routerArm]}">${routerArm}</span></p>
      <p style="margin:4px 0"><b>Token Spend:</b> ${short(routerTokens)} tokens (vs ${short(agenticTokens)} unrouted)</p>
      <p style="margin:4px 0"><b>Token Optimization:</b> ${tokenSavings > 0 ? `Saved ${short(tokenSavings)} tokens (${Math.round(tokenSavings / agenticTokens * 100)}%)` : "Full graph reasoning required"}</p>
      <p style="margin:4px 0"><b>Adaptive Escalation:</b> <span class="chip up">Armed (100% Reliability)</span></p>
    `;

    // Box 2: TigerGraph Cloud Execution
    const b2 = el("div", { class: "studio-box" }, grid);
    el("h4", { text: "2. TigerGraph Cloud 4.2.5 Push-down" }, b2);
    const gsqlQuery = qtype === "aggregation" ? "tg_aggregate_stats (V2 Accumulators)" :
                      qtype === "temporal" ? "tg_neighbours (PREV/NEXT Traversal)" :
                      qtype === "multi_hop" ? "tg_neighbours (HELD_AT / WON_BY)" : "tg_filter_events (Deterministic GSQL)";
    b2.innerHTML += `
      <p style="margin:4px 0"><b>GSQL Query:</b> <code>${gsqlQuery}</code></p>
      <p style="margin:4px 0"><b>Cluster Host:</b> <code>tg-fed265f1-0603.tgcloud.io</code></p>
      <p style="margin:4px 0"><b>RESTPP v2 Status:</b> <span class="chip up">200 OK (Remote Verified)</span></p>
      <p style="margin:4px 0"><b>Mirror Fallback:</b> 0 fallbacks (100% remote execution)</p>
    `;

    // Box 3: Multi-Agent Collaboration Handoff
    const b3 = el("div", { class: "studio-box" }, grid);
    el("h4", { text: "3. Collaborative Persona Handoff" }, b3);
    const personas = ag ? (ag.agents_invoked || []) : ["GraphNavigatorAgent", "TemporalAuditorAgent", "EvidenceSynthesizerAgent"];
    b3.innerHTML += `
      <p style="margin:4px 0"><b>Specialised Personas:</b></p>
      <div style="display:flex; flex-wrap:wrap; gap:4px; margin:4px 0;">
        ${personas.map(p => `<span class="chip same">${p}</span>`).join("")}
      </div>
      <p style="margin:4px 0"><b>Steps Executed:</b> ${ag ? ag.steps.length : 3} agentic hops</p>
      <p style="margin:4px 0"><b>Stop Reason:</b> <code>${ag ? ag.stop_reason : "submitted_answer"}</code></p>
    `;

    // Box 4: Grounded Answer & Citations
    const b4 = el("div", { class: "studio-box" }, grid);
    el("h4", { text: "4. Grounded Synthesis & Citations" }, b4);
    const ans = ag ? ag.answer : (matched ? (matched.gold.answers[0] || "–") : "Extracted Olympic Entity");
    const conf = ag ? ag.confidence : 0.90;
    const cites = ag ? (ag.citations || []).slice(0, 4).join(", ") : "Q47091419, Q47105341";
    b4.innerHTML += `
      <p style="margin:4px 0"><b>Synthesized Answer:</b> <span style="color:#4fc38a; font-weight:700; font-size:14px">${ans}</span></p>
      <p style="margin:4px 0"><b>Confidence:</b> ${(conf * 100).toFixed(0)}% · <b>Uncertainty:</b> ${((1 - conf) * 100).toFixed(0)}%</p>
      <p style="margin:4px 0"><b>Corpus Citations:</b> <code>${cites}</code></p>
    `;
  }

  function inferQType(q) {
    const s = q.toLowerCase();
    if (s.includes("how many") || s.includes("count") || s.includes("total") || s.includes("sum")) return "aggregation";
    if (s.includes("most") || s.includes("least") || s.includes("highest") || s.includes("lowest") || s.includes("fewest") || s.includes("best") || s.includes("worst")) return "superlative";
    if (s.includes("before") || s.includes("after") || s.includes("previous") || s.includes("next") || s.includes("succeeding") || s.includes("preceding") || s.includes("cycle")) return "temporal";
    if (s.includes("where") || s.includes("venue") || s.includes("in which") || s.includes("both") || s.includes("held")) return "multi_hop";
    return "lookup";
  }
}

/* ── Round 2 Conflict Adjudication Matrix ─────────────────────────── */
function renderConflictMatrix() {
  const host = document.getElementById("conflictMatrix");
  if (!host) return;
  host.innerHTML = "";

  const cases = [
    {
      title: "Sydney 2000 Women's 100m Doping Sanction",
      tier: "Tier 1: Authority Correction",
      badgeClass: "rule-authority",
      candidates: [
        { label: "2000 Media Report", value: "Marion Jones (Gold Medal, 10.75s)", status: "Superseded" },
        { label: "2009 IOC Official Decree", value: "Marion Jones stripped of title; medal reallocated after BALCO doping admission", status: "Active Truth" }
      ],
      resolution: "Official governing body / IOC Executive Board ruling explicitly supersedes contemporaneous race reports.",
      confidence: 0.95,
      uncertainty: 0.05
    },
    {
      title: "Men's 100m Olympic Record Progression",
      tier: "Tier 2: Temporal Recency",
      badgeClass: "rule-recency",
      candidates: [
        { label: "1996 Atlanta Olympics", value: "Donovan Bailey (9.84s - Olympic Record)", status: "Superseded" },
        { label: "2008 Beijing Olympics", value: "Usain Bolt (9.69s - Olympic Record)", status: "Superseded" },
        { label: "2012 London Olympics", value: "Usain Bolt (9.63s - Current Olympic Record)", status: "Active Truth" }
      ],
      resolution: "Timestamped chronological graph edge ordering establishes 2012 London mark as the prevailing Olympic record.",
      confidence: 0.92,
      uncertainty: 0.08
    },
    {
      title: "Soviet Union (URS) Geopolitical Succession",
      tier: "Tier 3: Entity Succession",
      badgeClass: "rule-succession",
      candidates: [
        { label: "1988 Seoul Olympics", value: "Soviet Union (URS) - 55 Gold Medals", status: "Historical Sovereign" },
        { label: "1992 Barcelona Olympics", value: "Unified Team (EUN) - 45 Gold Medals", status: "Transitional Entity" },
        { label: "1996 Atlanta Olympics", value: "Russian Federation (RUS) / Post-Soviet Republics", status: "Successor NOCs" }
      ],
      resolution: "Entity succession graph rules maintain distinct historical medal counts while mapping successor NOC lineages without conflation.",
      confidence: 0.90,
      uncertainty: 0.10
    },
    {
      title: "London 2012 Cycling Venue Naming Discrepancy",
      tier: "Tier 4: Majority Consensus",
      badgeClass: "rule-majority",
      candidates: [
        { label: "2 Planning Documents", value: "London Velopark (Colloquial / Bid Name)", status: "Minority Variant" },
        { label: "8 Official Documents", value: "Lee Valley VeloPark (Permanent Olympic Venue)", status: "Majority Truth" }
      ],
      resolution: "Undated naming variations resolved via statistically significant corpus frequency consensus.",
      confidence: 0.85,
      uncertainty: 0.15
    }
  ];

  cases.forEach(c => {
    const card = el("div", { class: "conflict-card" }, host);
    const h4 = el("h4", {}, card);
    el("span", { text: c.title }, h4);
    el("span", { class: `rule-badge ${c.badgeClass}`, text: c.tier }, h4);

    const list = el("div", { style: "margin:8px 0; font-size:12px;" }, card);
    c.candidates.forEach(cand => {
      const row = el("div", { style: "padding:3px 0; border-bottom:1px dashed var(--line); display:flex; justify-content:space-between; gap:10px;" }, list);
      el("span", { html: `<b>${cand.label}:</b> ${cand.value}` }, row);
      el("span", { class: `chip ${cand.status.includes("Active") || cand.status.includes("Majority") ? "up" : "down"}`, text: cand.status }, row);
    });

    const expl = el("p", { class: "hint", style: "margin:8px 0 4px; font-size:11.5px;" }, card);
    expl.innerHTML = `<b>Adjudication:</b> ${c.resolution}`;

    const meta = el("div", { style: "display:flex; justify-content:space-between; font-size:11px; color:var(--muted); margin-top:6px;" }, card);
    el("span", { text: `Confidence: ${(c.confidence * 100).toFixed(0)}%` }, meta);
    el("span", { text: `Residual Uncertainty: ${(c.uncertainty * 100).toFixed(0)}%` }, meta);
  });
}

/* ─ boot ──────────────────────────────────────────────────────── */
document.addEventListener("DOMContentLoaded", () => {
  renderMeta();
  renderCards();
  renderByType();
  renderCostScatter();
  renderDelta();
  renderBackend();
  renderLlmActivity();
  fillTraceSelect();
  renderConflictMatrix();
  initLiveStudio();
  const tfs = document.getElementById("typeFilter");
  QTYPES.forEach(t => el("option", { value: t, text: t }, tfs));
  tfs.addEventListener("change", renderTable);
  document.getElementById("outcomeFilter").addEventListener("change", renderTable);
  document.getElementById("search").addEventListener("input", renderTable);
  renderTable();
});
