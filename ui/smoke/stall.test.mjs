// Smoke test of the "no sign of progress" banner (any job channel), without a window. Run with `node --test ui/smoke`.
import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const source = fs.readFileSync(path.join(here, "..", "app.js"), "utf8");

async function browser() {
  const els = {};
  const el = (id) => (els[id] ??= { id, innerHTML: "", textContent: "", hidden: true, checked: false, value: "", classList: { toggle() {} }, querySelectorAll: () => [], onclick: null, onchange: null, oninput: null, onkeydown: null, style: {}, dataset: {} });
  const listeners = {};
  const invoke = async (cmd) => (cmd === "get_state" ? { project: null, steps: [], bundle: null, settings: { theme: "light", expert_mode: false, show_help_panel: true }, worlds: [], recent: [] } : cmd === "help" ? "<p>help</p>" : null);
  const win = { __TAURI__: { core: { invoke }, dialog: {}, event: { listen: async (name, fn) => { (listeners[name] ??= []).push(fn); } }, webview: { getCurrentWebview: () => ({ onDragDropEvent: async () => {} }) } }, matchMedia: () => ({ matches: false }) };
  const doc = { getElementById: el, querySelectorAll: () => [], querySelector: () => null, createElement: () => el("tmp"), body: { appendChild() {} }, documentElement: { setAttribute() {}, removeAttribute() {} } };
  const ctx = vm.createContext({ window: win, document: doc, console, confirm: () => true, setTimeout, clearTimeout });
  vm.runInContext(source, ctx);
  await new Promise((r) => setTimeout(r, 20));
  return { els, el, send: (channel, payload) => (listeners[channel] || []).forEach((f) => f({ payload })), listeners };
}

test("a stall event on any job channel shows the banner and the next event hides it", async () => {
  const b = await browser();
  for (const ch of ["train-event", "reshape-event", "export-event", "imatrix-event"]) assert.ok((b.listeners[ch] || []).length >= 1, ch);
  assert.equal(b.el("stall").hidden, true);
  b.send("reshape-event", { kind: "stall", minutes: 20 });
  assert.equal(b.el("stall").hidden, false);
  assert.match(b.el("stall").textContent, /^No sign of progress for 20 minutes\. The job may be stuck; Stop is safe\.$/);
  b.send("train-event", { kind: "stall", minutes: 40 });
  assert.match(b.el("stall").textContent, /for 40 minutes/);
  b.send("export-event", { type: "stall", minutes: 1 });                  // a raw worker event has "type" only
  assert.match(b.el("stall").textContent, /for 1 minute\./);
  b.send("train-event", { kind: "progress", step: "x", value: 0.1, text: "" });
  assert.equal(b.el("stall").hidden, true);
  assert.equal(b.el("stall").textContent, "");
});
