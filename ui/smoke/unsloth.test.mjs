// Smoke test of the Unsloth Desktop card in "Training setup" without a window: app.js runs in a stand-in browser that records HTML.
import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const source = fs.readFileSync(path.join(here, "..", "app.js"), "utf8");

function browser(answers = {}) {
  const els = {};
  const el = (id) => (els[id] ??= { id, innerHTML: "", textContent: "", checked: false, value: "", classList: { toggle() {} }, querySelectorAll: () => [], onclick: null, onchange: null, oninput: null, onkeydown: null, scrollTop: 0 });
  const calls = [];
  const invoke = async (cmd, args) => {
    calls.push([cmd, args]);
    if (cmd in answers) { const a = answers[cmd]; return typeof a === "function" ? a(args) : a; }
    if (cmd === "get_state") return baseState();
    if (cmd === "help") return "<p>help</p>";
    if (cmd === "env_status") return { checks: [], env_installed: true, can_install: true };
    return null;
  };
  const win = { __TAURI__: { core: { invoke }, dialog: {}, event: { listen: async () => {} }, webview: { getCurrentWebview: () => ({ onDragDropEvent: async () => {} }) } }, matchMedia: () => ({ matches: false }) };
  const doc = { getElementById: el, querySelectorAll: () => [], querySelector: () => null, createElement: () => el("tmp"), body: { appendChild() {} }, documentElement: { setAttribute() {}, removeAttribute() {} } };
  const ctx = vm.createContext({ window: win, document: doc, console, confirm: () => true, setTimeout, clearTimeout });
  vm.runInContext(source, ctx);
  return { ctx, els, calls, run: (code) => vm.runInContext(code, ctx) };
}

function baseState(unsloth = null) {
  return { project: { name: "P", folder: "/p" }, steps: [], bundle: { world: "W" }, settings: { theme: "light", expert_mode: false, show_help_panel: true, unsloth }, worlds: [], recent: [] };
}

const tick = () => new Promise((r) => setTimeout(r, 0));

test("the setup page shows the Unsloth card with Find, Test and Save", () => {
  const b = browser();
  b.run(`state = ${JSON.stringify(baseState())}; env.status = {checks: [], env_installed: true, can_install: true}; page = "setup";`);
  const html = b.run("renderSetup()");
  assert.match(html, /Unsloth Desktop \(optional\)/);
  assert.match(html, /id="un-find"/); assert.match(html, /id="un-test"/); assert.match(html, /id="un-save" disabled/);
  assert.match(html, /type="password" id="un-key"/);
  assert.doesNotMatch(html, /sk-unsloth-[a-z0-9]/);
});

test("a result from Unsloth is shown with its version, and the saved connection is listed without any key", () => {
  const b = browser();
  b.run(`state = ${JSON.stringify(baseState({ name: "Unsloth Desktop", url: "http://127.0.0.1:8888" }))}; env.status = {checks: []}; page = "setup";`);
  b.run(`onUnslothEvent({kind: "done", data: {summary: "Unsloth Desktop 2026.11.1 on this computer, NVIDIA GeForce RTX 5090 31.8 GB, training API: yes", url: "http://127.0.0.1:8888"}})`);
  const html = b.run("renderSetup()");
  assert.match(html, /Unsloth Desktop 2026\.11\.1 on this computer/);
  assert.match(html, /Saved: Unsloth Desktop at http:\/\/127\.0\.0\.1:8888/);
  assert.match(html, /id="un-remove"/);
  b.run(`onUnslothEvent({kind: "error", text: "Unsloth refused the API key."})`);
  assert.match(b.run("renderSetup()"), /class="err">Unsloth refused the API key/);
});

test("Test and Find call the right command with what was typed; an address on the internet needs a tick before Save", async () => {
  const b = browser({ unsloth_set: baseState({ name: "Unsloth Desktop", url: "http://127.0.0.1:8888" }) });
  b.run(`state = ${JSON.stringify(baseState())}; env.status = {checks: []}; page = "setup"; un.url = "127.0.0.1:8888"; un.key = "sk-unsloth-abc";`);
  b.run("wireUnsloth()");
  b.els["un-test"].onclick();
  b.els["un-find"].onclick();
  const asked = b.calls.filter((c) => c[0] === "unsloth_check").map((c) => c[1]);
  assert.equal(JSON.stringify(asked), JSON.stringify([{ url: "127.0.0.1:8888", key: "sk-unsloth-abc", find: false }, { url: "127.0.0.1:8888", key: "sk-unsloth-abc", find: true }]));
  // the internet, plain http: refused until ticked
  assert.equal(b.run("unRisk('http://203.0.113.5:8888')"), true);
  assert.equal(b.run("unRisk('https://unsloth.example.com')"), false);
  assert.equal(b.run("[unRisk('127.0.0.1:8888'), unRisk('http://192.168.1.20:8888'), unRisk('truenas:8888'), unRisk('http://[::1]:8888')].join()"), "false,false,false,false");
  b.run(`un.busy = false; un.url = "http://203.0.113.5:8888"; un.ack = false; wireUnsloth()`);
  await b.els["un-save"].onclick();
  assert.equal(b.calls.filter((c) => c[0] === "unsloth_set").length, 0);
  assert.match(b.run("un.error"), /Tick the box/);
  b.run(`un.url = "127.0.0.1:8888"; un.error = null; wireUnsloth()`);
  await b.els["un-save"].onclick();
  await tick();
  const set = b.calls.find((c) => c[0] === "unsloth_set");
  assert.equal(set[1].key, "sk-unsloth-abc");
  assert.equal(b.run("un.key"), "", "the key is dropped from the page once saved");
});
