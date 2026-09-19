/* Dashboard logic — no framework, hand-rolled SVG.
   DATA = { entries: [...per-question records...], summary: {...}, ... } */

const PIPE_COLORS = {
  "RAG": "#e05f5f",
  "GraphRAG": "#e0a83c",
  "Agentic GraphRAG": "#4fc38a",
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

function renderMeta() {
  const s = DATA.summary || {};
  document.getElementById("meta").innerHTML =
    `${s.num_questions || (DATA.entries || []).length} public questions · ` +
    `evaluator: ${s.evaluator && s.evaluator.llm_judge_enabled ? "LLM judge on" : "deterministic"} · ` +
    `wall clock ${s.wall_clock_s ? s.wall_clock_s + "s" : "–"} · ` +
    `${runModeLabel()} · generated ${DATA.generated}`;
  document.getElementById("footMeta").textContent =
    `results: ${DATA.results_file} · corpus: 2,951 Wikipedia articles (1987–2023) · ` +
    `vector backend: sparse TF-IDF · planner: openai/gpt-oss-120b via Groq`;
}

function renderCards() {
  const s = DATA.summary || {};
  const pipes = s.pipelines || {};
  const host = document.getElementById("cards");
  host.innerHTML = "";
  const cls = { "RAG": "c-rag", "GraphRAG": "c-graph", "Agentic GraphRAG": "c-agent" };

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
  const tfs = document.getElementById("typeFilter");
  QTYPES.forEach(t => el("option", { value: t, text: t }, tfs));
  tfs.addEventListener("change", renderTable);
  document.getElementById("outcomeFilter").addEventListener("change", renderTable);
  document.getElementById("search").addEventListener("input", renderTable);
  renderTable();
});
