const fs = require("fs"), vm = require("vm"), path = require("path");

function makeNode(tag) {
  return { tagName: String(tag).toUpperCase(), children: [], style: {},
    setAttribute() {}, appendChild(c) { this.children.push(c); return c; },
    addEventListener() {}, innerHTML: "", textContent: "", text: "", value: "",
    options: [] };
}

const DATA = JSON.parse(/const DATA = (\{[\s\S]*?\});\s*<\/script>/
  .exec(fs.readFileSync("dashboard/dashboard_llm.html", "utf8"))[1]);

const ids = ["meta", "cards", "byType", "legend", "costScatter", "delta",
  "backend", "llmActivity", "traceSelect", "traceMeta", "trace", "typeFilter",
  "outcomeFilter", "search", "resultsTable", "footMeta"];
const registry = {};
ids.forEach(id => { registry[id] = makeNode(id === "typeFilter" || id === "outcomeFilter" || id === "traceSelect" ? "select" : "div"); });

const document = {
  createElement: makeNode,
  createElementNS: (ns, tag) => makeNode(tag),
  createTextNode: text => ({ nodeType: 3, textContent: text }),
  getElementById: id => registry[id] || null,
  _listeners: {},
  addEventListener(evt, fn) { (this._listeners[evt] = this._listeners[evt] || []).push(fn); },
};

const sandbox = { document, console, JSON, Math, Object, Number, String, Array, DATA };
sandbox.window = sandbox;
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync("dashboard/dashboard.js", "utf8"), sandbox, { filename: "dashboard.js" });
for (const fn of (document._listeners.DOMContentLoaded || [])) fn();

console.log("cards children:", registry.cards.children.length);
for (const c of registry.cards.children) {
  const h3 = c.children.find(k => k.tagName === "H3");
  console.log("  card:", h3 ? h3.text : c.tagName);
}
