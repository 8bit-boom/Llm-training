// The stand-in browser the smoke tests run app.js in (no window): `makeBrowser({ answers, state })` evaluates the whole of app.js in a vm context with a fake `invoke`.
// `answers` maps a command name to the answer (or a function of the arguments; a thrown value is a refusal); every call is recorded in `calls`.
// Run all with `node --test ui/smoke`. Files starting with an underscore are helpers, not tests.
import fs from "node:fs";
import vm from "node:vm";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
export const source = fs.readFileSync(path.join(here, "..", "app.js"), "utf8");

export const baseState = () => ({ project: { name: "P", folder: "/p" }, steps: [], bundle: { world: "W" }, settings: { theme: "light", expert_mode: false, show_help_panel: true }, worlds: [], recent: [] });
export const plain = (x) => JSON.parse(JSON.stringify(x));
export const tick = (ms = 20) => new Promise((r) => setTimeout(r, ms));

// Options: `answers`; `state` (a function answering get_state; null: get_state is not answered); `code` (the program text, default app.js); `dialogs` (what the file dialog returns, one per call);
// `cloneArgs` (record a copy of the arguments, {} for none); `groups` (what querySelectorAll(selector) finds); the result has `confirm(v)` to answer the next confirm() calls.
export function makeBrowser({ answers = {}, state = baseState, code = source, dialogs = [], cloneArgs = false, groups = {} } = {}) {
  const els = {};
  const el = (id) => (els[id] ??= { id, innerHTML: "", textContent: "", hidden: true, checked: false, value: "", classList: { toggle() {} }, querySelectorAll: () => [], onclick: null, onchange: null, oninput: null, onkeydown: null, scrollTop: 0, style: {}, dataset: {} });
  const calls = [];
  const listeners = {};
  const invoke = async (cmd, args) => {
    calls.push([cmd, cloneArgs ? JSON.parse(JSON.stringify(args ?? {})) : args]);
    if (cmd in answers) {
      const a = answers[cmd];
      if (typeof a === "function") return a(args);
      if (a instanceof Error) throw a;
      return a;
    }
    if (cmd === "get_state" && state) return state();
    if (cmd === "help") return "<p>help</p>";
    return null;
  };
  const win = { __TAURI__: { core: { invoke }, dialog: { open: async (o) => { calls.push(["dialog.open", o]); return dialogs.shift() ?? null; } }, event: { listen: async (name, fn) => { (listeners[name] ??= []).push(fn); } }, webview: { getCurrentWebview: () => ({ onDragDropEvent: async () => {} }) } }, matchMedia: () => ({ matches: false }) };
  const doc = { getElementById: el, querySelectorAll: (sel) => groups[sel] || [], querySelector: () => null, createElement: () => el("tmp"), body: { appendChild() {} }, documentElement: { setAttribute() {}, removeAttribute() {} } };
  let confirmed = true;
  const ctx = vm.createContext({ window: win, document: doc, console, confirm: () => confirmed, alert: () => {}, setTimeout, clearTimeout });
  vm.runInContext(code, ctx);
  return { ctx, els, el, calls, groups, listeners, confirm: (v) => { confirmed = v; }, send: (channel, payload) => (listeners[channel] || []).forEach((f) => f({ payload })), run: (c) => vm.runInContext(c, ctx) };
}
