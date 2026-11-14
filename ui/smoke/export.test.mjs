// Smoke test of the Step 5 export page without a window (the importance-matrix box): app.js runs in a stand-in browser that records HTML.
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
  const el = (id) => (els[id] ??= { id, innerHTML: "", textContent: "", checked: false, value: "", classList: { toggle() {} }, querySelectorAll: () => [], onclick: null, onchange: null, oninput: null, onkeydown: null, scrollTop: 0 });
  const calls = [];
  const invoke = async (cmd, args) => { calls.push([cmd, args]); if (cmd === "help") return "<p>help</p>"; return null; };
  const win = { __TAURI__: { core: { invoke }, dialog: {}, event: { listen: async () => {} }, webview: { getCurrentWebview: () => ({ onDragDropEvent: async () => {} }) } }, matchMedia: () => ({ matches: false }) };
  const doc = { getElementById: el, querySelectorAll: () => [], querySelector: () => null, createElement: () => el("tmp"), body: { appendChild() {} }, documentElement: { setAttribute() {}, removeAttribute() {} } };
  const ctx = vm.createContext({ window: win, document: doc, console, confirm: () => true, setTimeout, clearTimeout });
  vm.runInContext(source, ctx);
  return { els, calls, run: (code) => vm.runInContext(code, ctx) };
}

const est = (c) => ({ chunks: c, ctx: 512, tokens: c * 512, gpu_minutes: c / 32, cpu_minutes: c });
const exportState = (over = {}, imatrix = {}) => ({
  runs: [{ name: "run1", model: "Qwen3.5 9B", status: "done", loss: 0.8 }], exports: [], servers: [], suggested_name: "w", env_ready: true,
  quants: [["q4_k_m", "Q4_K_M: smallest"], ["q6_k", "Q6_K: larger"], ["q8_0", "Q8_0: large"], ["f16", "F16: original"]].map(([id, name]) => ({ id, name })),
  imatrix: { chunks: [64, 128, 256], default_chunks: 128, ctx: 512, general_share: 0.2, no_matrix_for: ["q8_0", "f16"], saved: {}, estimates: { run1: { 64: est(64), 128: est(128), 256: est(256) } }, ...imatrix },
  ...over,
});
const page = (b, st) => { b.run(`state = {project: {name: "P", folder: "/p"}, steps: [], bundle: {world: "W"}, settings: {theme: "light"}, worlds: [], recent: []}; page = "use"; us.state = ${JSON.stringify(st)};`); const html = b.run("renderUse()"); b.run("wireUse()"); return html; };

test("the importance matrix is off by default, with its checkbox and a short explanation, and nothing else of it is shown", () => {
  const b = browser();
  const html = page(b, exportState());
  assert.match(html, /id="us-im" >/);                                   // not ticked
  assert.match(html, /Better small sizes \(importance matrix\)/);
  assert.match(html, /Q8_0 and F16 stay as they are/);
  assert.doesNotMatch(html, /us-imchunks|us-imreuse/);
});

test("ticking it shows the amount of text (64 / 128 / 256, 128 chosen) and a time guess", () => {
  const b = browser();
  page(b, exportState());
  b.els["us-im"].checked = true; b.els["us-im"].onchange();
  const html = b.run("renderUse()");
  assert.match(html, /id="us-im" checked/);
  assert.match(html, /<option value="64" >64 chunks \(quick\)/); assert.match(html, /<option value="128" selected>128 chunks \(recommended\)/); assert.match(html, /<option value="256" >256 chunks \(thorough\)/);
  assert.match(html, /65,536 tokens, about 4 min on a graphics card \(up to 128 min on the processor\)/);
  assert.match(html, /No saved matrix of this run yet/);
  b.run("wireUse()"); b.els["us-imchunks"].value = "256"; b.els["us-imchunks"].onchange();
  assert.match(b.run("renderUse()"), /<option value="256" selected>/);
  assert.match(b.run("renderUse()"), /131,072 tokens/);
});

test("only a GGUF export shows the box", () => {
  const b = browser();
  page(b, exportState());
  b.run(`us.what = "merged"`);
  assert.doesNotMatch(b.run("renderUse()"), /importance matrix/);
  b.run(`us.what = "gguf_merged"`);
  assert.match(b.run("renderUse()"), /us-im/);
});

test("a saved matrix gives a reuse box that is on by default and can be unticked", () => {
  const b = browser();
  page(b, exportState({}, { saved: { run1: { dir: "/p/exports/run1-q4_k_m", chunks: 128, ctx: 512, created: "2026-11-01T10:00:00Z", general_used: true } } }));
  b.run("us.im = true");
  let html = b.run("renderUse()");
  assert.match(html, /id="us-imreuse" checked> Reuse the saved matrix \(128 chunks, made 2026-11-01/);
  b.run("wireUse()"); b.els["us-imreuse"].checked = false; b.els["us-imreuse"].onchange();
  assert.match(b.run("renderUse()"), /id="us-imreuse" > Reuse/);
});

test("Make the files sends the matrix options only when ticked", async () => {
  const b = browser();
  page(b, exportState({}, { saved: { run1: { dir: "/d", chunks: 64, created: "2026-11-01" } } }));
  b.els["us-make"].onclick();
  assert.deepEqual(b.calls.at(-1)[1].imatrix, null);
  assert.equal(b.calls.at(-1)[0], "export_start");
  b.run("us.im = true; us.imChunks = 64; us.quants = ['q4_k_m', 'q8_0']"); b.run("renderUse(); wireUse()");
  b.els["us-make"].onclick();
  assert.deepEqual(JSON.parse(JSON.stringify(b.calls.at(-1)[1].imatrix)), { enabled: true, chunks: 64, reuse: true, nextn: false, general: null });
  assert.deepEqual(JSON.parse(JSON.stringify(b.calls.at(-1)[1].quants)), ["q4_k_m", "q8_0"]);
  b.run("us.imReuse = false"); b.run("wireUse()");
  b.els["us-make"].onclick();
  assert.equal(b.calls.at(-1)[1].imatrix.reuse, false);
  b.run("us.what = 'merged'"); b.run("renderUse(); wireUse()");
  b.els["us-make"].onclick();
  assert.equal(b.calls.at(-1)[1].imatrix, null);
});

test("with only Q8_0 / F16 chosen the page says there is nothing to improve", () => {
  const b = browser();
  page(b, exportState());
  b.run("us.im = true; us.quants = ['q8_0']");
  assert.match(b.run("renderUse()"), /None of the chosen sizes is below Q8_0/);
  b.run("us.quants = ['q4_k_m']");
  assert.doesNotMatch(b.run("renderUse()"), /None of the chosen sizes/);
});

test("the files list says which size used the matrix and shows the plain-language note; Cancel is there while it runs", () => {
  const b = browser();
  const matrix = (over) => ({ requested: true, applied: true, chunks: 128, reused: false, note: "", ...over });
  const html = page(b, exportState({ exports: [
    { name: "run1-q4_k_m", gguf_file: "m.gguf", size_mb: 5000, dir: "/p/exports/run1-q4_k_m", imatrix: matrix({ note: "The general text could not be read. The matrix is made from your project's text only." }) },
    { name: "run1-q8_0", gguf_file: "m.gguf", size_mb: 9000, dir: "/p/exports/run1-q8_0", imatrix: matrix({ applied: false, reason: "high precision" }) },
    { name: "run1-q6_k", gguf_file: "m.gguf", size_mb: 7000, dir: "/p/exports/run1-q6_k", imatrix: matrix({ reused: true }) },
    { name: "old", gguf_file: "m.gguf", size_mb: 1, dir: "/p/exports/old" },
  ] }));
  assert.match(html, /importance matrix used \(128 chunks\)/); assert.match(html, /importance matrix used \(128 chunks, saved one\)/);
  assert.match(html, /no importance matrix \(high precision\)/);
  assert.match(html, /Importance matrix: The general text could not be read/);
  assert.equal((html.match(/importance matrix/gi) || []).length >= 5, true);
  assert.doesNotMatch(html, /id="us-cancel"/);
  b.run(`us.running = true; us.pending = "make"; us.step = "Chunk 3 of 128"; us.progress = 0.02`);
  assert.match(b.run("renderUse()"), /id="us-cancel"/);
  b.run("wireUse()"); b.els["us-cancel"].onclick();
  assert.equal(b.calls.at(-1)[0], "env_cancel");
});

test("the result of an export names the matrix note", () => {
  const b = browser();
  page(b, exportState());
  b.run(`onExportEvent({kind: "done", data: {name: "run1-q4_k_m", run: "run1", imatrix: {note: "This llama.cpp has no llama-imatrix."}}})`);
  assert.match(b.run("us.ok"), /Importance matrix: This llama\.cpp has no llama-imatrix\./);
  b.run(`onExportEvent({kind: "error", text: "Cancelled."})`);
  assert.match(b.run("us.error"), /^Cancelled\. The export was stopped/);
});

test("the speculative-decoding head option is shown only for a model that has one, on by default, and is sent as chosen", () => {
  const b = browser();
  page(b, exportState({}, { mtp: { run1: false } }));
  b.run("us.im = true");
  assert.doesNotMatch(b.run("renderUse()"), /us-imnextn|speculative-decoding/);
  page(b, exportState({}, { mtp: { run1: null } }));
  b.run("us.im = true");
  assert.doesNotMatch(b.run("renderUse()"), /us-imnextn/);                         // unknown: not offered
  page(b, exportState({}, { mtp: { run1: true } }));
  b.run("us.im = true; us.quants = ['q4_k_m']");
  let html = b.run("renderUse()");
  assert.match(html, /id="us-imnextn" checked\s*> Include the speculative-decoding head \(MTP\) in the matrix/);
  b.run("renderUse(); wireUse()"); b.els["us-make"].onclick();
  assert.equal(b.calls.at(-1)[1].imatrix.nextn, true);
  b.run("wireUse()"); b.els["us-imnextn"].checked = false; b.els["us-imnextn"].onchange();
  assert.match(b.run("renderUse()"), /id="us-imnextn"\s+>/);
  b.run("wireUse()"); b.els["us-make"].onclick();
  assert.equal(b.calls.at(-1)[1].imatrix.nextn, false);
});

test("the head option is greyed with the reason when llama-imatrix is known not to support it, and the matrix is then asked without it", () => {
  const b = browser();
  page(b, exportState({}, { mtp: { run1: true }, nextn_supported: false }));
  b.run("us.im = true");
  const html = b.run("renderUse()");
  assert.match(html, /id="us-imnextn"\s+disabled>/);
  assert.match(html, /Not available: this llama\.cpp has no --nextn option \(an older build\)\. The matrix is made without the head\./);
  b.run("renderUse(); wireUse()"); b.els["us-make"].onclick();
  assert.equal(b.calls.at(-1)[1].imatrix.nextn, false);
  page(b, exportState({}, { mtp: { run1: true }, nextn_supported: false, nextn_reason: "the installed llama-imatrix is too old" }));
  b.run("us.im = true");
  assert.match(b.run("renderUse()"), /Not available: the installed llama-imatrix is too old/);
});

test("the head option is not shown while the matrix is off or for a merged-only export", () => {
  const b = browser();
  page(b, exportState({}, { mtp: { run1: true } }));
  assert.doesNotMatch(b.run("renderUse()"), /us-imnextn/);
  b.run("us.im = true; us.what = 'merged'");
  assert.doesNotMatch(b.run("renderUse()"), /us-imnextn/);
});
