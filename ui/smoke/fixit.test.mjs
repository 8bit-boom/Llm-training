// Smoke test of the fix-it buttons under worker errors ([fix:<code>] marks), without a window. Run with `node --test ui/smoke`.
import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const source = fs.readFileSync(path.join(here, "..", "app.js"), "utf8");

function browser() {
  const els = {};
  const el = (id) => (els[id] ??= { id, innerHTML: "", textContent: "", checked: false, value: "", classList: { toggle() {} }, querySelectorAll: () => [], onclick: null, onchange: null, oninput: null, onkeydown: null, style: {}, dataset: {} });
  const calls = [];
  const invoke = async (cmd, args) => { calls.push([cmd, args]); if (cmd === "get_state") return { project: { name: "P", folder: "/p" }, steps: [], bundle: { world: "W" }, settings: { theme: "light", expert_mode: false, show_help_panel: true }, worlds: [], recent: [] }; return cmd === "help" ? "<p>help</p>" : null; };
  const win = { __TAURI__: { core: { invoke }, dialog: {}, event: { listen: async () => {} }, webview: { getCurrentWebview: () => ({ onDragDropEvent: async () => {} }) } }, matchMedia: () => ({ matches: false }) };
  const doc = { getElementById: el, querySelectorAll: () => [], querySelector: () => null, createElement: () => el("tmp"), body: { appendChild() {} }, documentElement: { setAttribute() {}, removeAttribute() {} } };
  const ctx = vm.createContext({ window: win, document: doc, console, confirm: () => true, alert: () => {}, setTimeout, clearTimeout });
  vm.runInContext(source, ctx);
  return { ctx, calls, run: (code) => vm.runInContext(code, ctx) };
}
const plain = (x) => JSON.parse(JSON.stringify(x));

test("the marks are taken out of the text and each known code gets its button", () => {
  const b = browser();
  const f = (t) => plain(b.run(`fixActions(${JSON.stringify(t)})`));
  let r = f("There is not enough free space. (details: logs/jobs/x.log) [fix:disk_full]");
  assert.equal(r.text, "There is not enough free space. (details: logs/jobs/x.log)");
  assert.match(r.html, /data-fix="open">Open the project folder<\/button>/);
  r = f("The graphics card ran out of memory. [fix:distill_oom]");
  assert.equal(r.text, "The graphics card ran out of memory.");
  assert.match(r.html, /data-fix="safer">Use safer settings<\/button>/);
  r = f("A training library is missing. [fix:missing_module]");
  assert.match(r.html, /data-fix="repair">Repair the training software<\/button>/);
  for (const code of ["distill_unstable", "reshape_refused", "bad_job"]) {
    r = f(`Words. [fix:${code}]`);
    assert.deepEqual(r, { text: "Words.", html: "" }, code);
  }
  assert.deepEqual(f("Plain."), { text: "Plain.", html: "" });
  assert.deepEqual(plain(b.run(`fixActions(null)`)), { text: null, html: "" });
  assert.match(f("a [fix:disk_full] b [fix:disk_full]").html, /^<div class="row fix-row"><button[^>]*>Open the project folder<\/button><\/div>$/, "the same button is not repeated");
});

test("error boxes show the clean text with the buttons, and the llama.cpp mark still brings its installer", () => {
  const b = browser();
  const html = b.run(`errBox("Disk. [fix:disk_full]", "all")`);
  assert.match(html, /<div class="err">Disk\.<\/div><div class="row fix-row">/);
  assert.doesNotMatch(html, /\[fix:/);
  assert.match(b.run(`errDiv("Long. [fix:missing_module]", "p")`), /^<p class="err">Long\.<\/p><div class="row fix-row"><button class="secondary" data-fix="repair">/);
  const llama = b.run(`errBox("llama.cpp was not found. [llama_cpp_missing]", "all")`);
  assert.match(llama, /llama-notice/);
  assert.doesNotMatch(llama, /\[llama_cpp_missing\]/);
});

test("the buttons do what they say", async () => {
  const b = browser();
  b.run(`state = { project: { folder: "/my/project" } }`);
  await b.run(`fixDo("open")`);
  assert.deepEqual(plain(b.calls.find((c) => c[0] === "imatrix_open_folder")[1]), { path: "/my/project" });
  // safer settings: taken from what the program offers for the model
  b.run(`rs.hl = { name: "m", opts: { safer: { checkpointing: "on", batch: 1, per_update: 32, kl_chunk: 64, kbit_upcast: false } } }; rs.hlSpec = {}`);
  assert.equal(b.run(`healUseSafer()`), true);
  assert.deepEqual(plain(b.run(`rs.hlSpec`)), { checkpointing: "on", batch: 1, per_update: 32, kl_chunk: 64, kbit_upcast: null });
  b.run(`rs.hl = { name: "m", opts: { safer: null } }`);
  assert.equal(b.run(`healUseSafer()`), false, "nothing safer is left");
});
