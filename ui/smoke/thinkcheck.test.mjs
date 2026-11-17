// Smoke test of the "Thinking check" card on the result of a made model (Slim or grow page), without a window (the same stand-in browser as reshape.test.mjs). Run with `node --test ui/smoke`.
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
    return null;
  };
  const win = { __TAURI__: { core: { invoke }, dialog: {}, event: { listen: async () => {} }, webview: { getCurrentWebview: () => ({ onDragDropEvent: async () => {} }) } }, matchMedia: () => ({ matches: false }) };
  const doc = { getElementById: el, querySelectorAll: () => [], querySelector: () => null, createElement: () => el("tmp"), body: { appendChild() {} }, documentElement: { setAttribute() {}, removeAttribute() {} } };
  const ctx = vm.createContext({ window: win, document: doc, console, confirm: () => true, setTimeout, clearTimeout });
  vm.runInContext(source, ctx);
  return { ctx, els, calls, run: (code) => vm.runInContext(code, ctx) };
}

const baseState = () => ({ project: { name: "P", folder: "/p" }, steps: [], bundle: { world: "W" }, settings: { theme: "light", expert_mode: false, show_help_panel: true }, worlds: [], recent: [] });
const NAME = "20261113-150000-qwen3.5-27b-slim-w10752";
const model = (over = {}) => ({ name: NAME, status: "done", model_id: "custom-w", healed: false, healed_model_id: null, heal: null, test: null, test_healed: null, size_gb: 42, cache_gb: 0, healed_size_gb: 0, scores: null,
  summary: { mode: "prune", shape: "width", layers_before: 64, layers_after: 64, dropped: [], duplicated: [], params_before: 27.8e9, params_after: 21.1e9, loss_before: 2.0, loss_after: 3.1, notes: [], chosen_blocks: [], boundary_distance: null }, ...over });
const pageState = (made) => ({ models: [], made, dataset_version: 3, env_ready: true, vram_gb: 32, gpu: "RTX 5090", free_disk_gb: 500 });
const main = (closed, acc, loops, median) => ({ n: 16, think_rate: 1, closed_rate: closed, accuracy: acc, loop_rate: loops, garbage_rate: 0, truncated_rate: 0.0625, median_think_tokens: median, p90_think_tokens: median * 2 });
const row = (m, over = {}) => ({ suite_version: 1, backend: "hf", main: m, control: { n: 8, accuracy: 0.875 }, created: "2026-11-17T10:00:00Z", ...over });
const gate = (verdict, failed = [], warned = []) => ({ verdict, failed, warned, checks: [] });
const results = (over = {}) => ({ suite_version: 1, targets: { original: true, cut: true, healed: false }, ggufs: [], modes: { full: { rows: {}, gates: {} }, quick: { rows: {}, gates: {} } }, ...over });

function open(b, made, data, extra = "") {
  b.run(`state = ${JSON.stringify(baseState())}; page = "reshape"; rs.state = ${JSON.stringify(pageState(made))}; rs.model = "qwen3.5-27b"; rs.view = ${JSON.stringify(made[0].name)};
    rs.options = null; rs.tk.name = ${JSON.stringify(made[0].name)}; rs.tk.data = ${JSON.stringify(data)}; ${extra}`);
}
const plain = (x) => JSON.parse(JSON.stringify(x));
const card = (b) => b.run("renderReshape()");

test("a made model has the card; with nothing checked yet it offers the original and the cut, and the healed button only for a healed model", () => {
  const b = browser();
  open(b, [model()], results());
  const html = card(b);
  assert.match(html, /<h3>Thinking check<\/h3>/);
  assert.match(html, /data-tkgo="original"/); assert.match(html, /data-tkgo="cut"/); assert.match(html, /data-tkgo="gguf"/); assert.doesNotMatch(html, /data-tkgo="healed"/);
  assert.doesNotMatch(html, /<th>Original<\/th>/);                                    // no table before a check
  assert.match(html, /Quick \(16 questions\)/);
  assert.match(html, /data-tkgo="gguf"[^>]*disabled/);                                   // no GGUF chosen yet
  open(b, [model({ healed: true, healed_model_id: "custom-w-healed" })], results({ targets: { original: true, cut: true, healed: true } }));
  assert.match(card(b), /data-tkgo="healed"/);
  const failed = model({ status: "failed", summary: null });
  open(b, [failed], results());
  assert.doesNotMatch(card(b), /Thinking check/);
});

test("quick is the default for a large model and the choice is kept; the table shows the numbers of the mode", () => {
  const b = browser();
  const data = results({ targets: { original: true, cut: true, healed: true }, modes: {
    quick: { rows: { original: row(main(1, 0.75, 0, 800)), cut: row(main(0.5, 0.25, 0.3, 4000)) }, gates: { cut: gate("fail", ["closed_rate", "accuracy"]) } },
    full: { rows: { original: row(main(0.97, 0.7, 0, 900)) }, gates: {} } } });
  open(b, [model({ healed: true })], data);
  assert.equal(b.run("tkQuick(rs.state.made[0])"), true);
  let html = card(b);
  assert.match(html, /<th>Original<\/th><th>Cut model<\/th>/);
  assert.match(html, /<td>Right answers<\/td><td>75 %<\/td><td>25 %<\/td>/);
  assert.match(html, /<td>Typical length of the reasoning \(tokens\)<\/td><td>800<\/td><td>4,000<\/td>/);
  assert.match(html, /data-tkverdict="cut">Fail<\/strong><br><span class="dim small">too weak: finishing its reasoning, right answers<\/span>/);
  assert.match(html, /Thinking off \(control\): right answers<\/td><td>88 %<\/td>/);
  b.run("rs.tk.quick = false");
  html = card(b);
  assert.match(html, /<td>Right answers<\/td><td>70 %<\/td><\/tr>/); assert.doesNotMatch(html, /Cut model<\/th>/);
  assert.match(html, /Check the original first|data-tkgo="original"/);
  b.run(`rs.state.made[0].summary.params_after = 9e9; rs.tk.quick = null`);
  assert.equal(b.run("tkQuick(rs.state.made[0])"), false);                                // a small model: the full suite by default
});

test("verdicts: pass, close to the limit and the note when the original is missing", () => {
  const b = browser();
  const data = results({ targets: { original: true, cut: true, healed: true }, modes: { quick: { rows: { original: row(main(1, 0.75, 0, 800)), cut: row(main(0.95, 0.7, 0.03, 900)), healed: row(main(0.9375, 0.7, 0, 900), { backend: "llama" }) },
    gates: { cut: gate("pass"), healed: gate("warn", [], ["closed_rate"]) } }, full: { rows: {}, gates: {} } } });
  open(b, [model({ healed: true })], data);
  const html = card(b);
  assert.match(html, /data-tkverdict="cut">Pass<\/strong>/);
  assert.match(html, /data-tkverdict="healed">Close to the limit<\/strong><br><span class="dim small">near the limit: finishing its reasoning<\/span>/);
  assert.match(html, /\(llama\.cpp\)/);
  open(b, [model()], results({ modes: { quick: { rows: { cut: row(main(0.5, 0.2, 0.3, 3000)) }, gates: {} }, full: { rows: {}, gates: {} } } }));
  const h2 = card(b);
  assert.match(h2, /check the original first/); assert.match(h2, /Check the original first: the other numbers/);
});

test("pressing a button starts the check with the quick choice, and the events fill the progress and the end reloads the numbers", async () => {
  const answers = { think_check_start: null, think_check_results: results({ modes: { quick: { rows: { original: row(main(1, 0.75, 0, 800)) }, gates: {} }, full: { rows: {}, gates: {} } } }) };
  const b = browser(answers);
  open(b, [model()], results());
  b.run(`tkStart(rs.state.made[0], "original")`);
  assert.deepEqual(plain(b.calls.find((c) => c[0] === "think_check_start")[1]), { spec: { name: NAME, target: "original", quick: true, gguf: null } });
  assert.match(card(b), /Starting…/); assert.match(card(b), /id="tk-stop"/);
  assert.match(card(b), /data-tkgo="cut"[^>]*disabled/);                                  // one check at a time
  b.run(`onReshapeEvent({ phase: "think", kind: "progress", step: "think", value: 0.25, text: "Question 4 of 24" })`);
  assert.match(card(b), /Question 4 of 24/); assert.match(card(b), /value="0.25"/);
  b.run(`onReshapeEvent({ phase: "think", kind: "think_item", id: "arith1", ok: true, closed: true, think_tokens: 90 })`);
  assert.equal(b.run("rs.tk.run.done"), 1);
  b.run(`onReshapeEvent({ phase: "think", kind: "done", data: { name: ${JSON.stringify(NAME)}, target: "original", stopped: false, cached: false } })`);
  await new Promise((r) => setTimeout(r, 10));
  assert.equal(b.run("rs.tk.run"), null);
  assert.ok(b.calls.some((c) => c[0] === "think_check_results" && c[1].name === NAME));
  assert.match(card(b), /Done\. The numbers are in the table\./); assert.match(card(b), /<th>Original<\/th>/);
});

test("a GGUF is chosen from the exports or typed, sent with the llama way, and a stop or an error ends the run in plain words", async () => {
  const b = browser();
  open(b, [model()], results({ ggufs: [{ name: "run1-q4_k_m", path: "/p/exports/run1-q4_k_m/model-q4_k_m.gguf" }] }));
  assert.match(card(b), /<option value="\/p\/exports\/run1-q4_k_m\/model-q4_k_m.gguf"/);
  b.run(`rs.tk.gguf = "/p/exports/run1-q4_k_m/model-q4_k_m.gguf"`);
  assert.doesNotMatch(card(b), /data-tkgo="gguf"[^>]*disabled/);
  b.run(`tkStart(rs.state.made[0], "gguf")`);
  assert.deepEqual(plain(b.calls.find((c) => c[0] === "think_check_start")[1].spec), { name: NAME, target: "gguf", quick: true, gguf: "/p/exports/run1-q4_k_m/model-q4_k_m.gguf" });
  b.run(`onReshapeEvent({ phase: "think", kind: "error", text: "llama.cpp was not found, so the thinking check can run the GGUF file." })`);
  assert.equal(b.run("rs.tk.run"), null);
  assert.match(card(b), /class="err">llama\.cpp was not found/);
  b.run(`tkStart(rs.state.made[0], "cut"); onReshapeEvent({ phase: "think", kind: "done", data: { stopped: true } })`);
  await new Promise((r) => setTimeout(r, 10));
  assert.match(card(b), /Stopped\. Nothing was saved from this check\./);
});

test("a start that the program refuses shows its message and frees the buttons", async () => {
  const b = browser({ think_check_start: () => { throw "Another job is still running. Wait for it or press Cancel."; } });
  open(b, [model()], results());
  b.run(`tkStart(rs.state.made[0], "cut")`);
  await new Promise((r) => setTimeout(r, 10));
  assert.equal(b.run("rs.tk.run"), null);
  assert.match(card(b), /Another job is still running/);
});
