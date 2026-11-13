// Smoke test of the "Slim or grow a model" page (and the two small changes in Steps 3 and 4) without a window: app.js runs in a stand-in browser
// that only records HTML. Run with `node --test ui/smoke`.
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

function baseState(over = {}) {
  return { project: { name: "P", folder: "/p" }, steps: [], bundle: { world: "W" }, settings: { theme: "light", expert_mode: false, show_help_panel: true }, worlds: [], recent: [], ...over };
}

const option = (blocks, layers, params, extra = {}) => ({ blocks, layers, params_b: params, disk_gb: +(params * 2).toFixed(1), vram_lora_gb: params * 2.2 + 1, vram_qlora_gb: params * 0.7 + 2, fits_lora: params * 2.2 + 1 <= 32, fits_qlora: true, disk_needed_gb: +(params * 2.2).toFixed(1), warnings: [], ...extra });
const inspectOf = (over = {}) => ({ model_type: "qwen3_5", supported: true, why: null, layers: 64, layer_types: ["linear_attention", "linear_attention", "linear_attention", "full_attention"].concat(Array(60).fill("x")), period: 4, blocks: 16, total_params: 27.8e9, has_vision: true, local: true, source_bytes: 0, ...over });
const optionsFor = (mode) => ({ model: { id: "qwen3.5-27b" }, mode, inspect: inspectOf(), default_blocks: 4, load: "4bit", vram_gb: 32,
  options: [1, 2, 3, 4, 5].map((b) => option(b, mode === "prune" ? 64 - 4 * b : 64 + 4 * b, mode === "prune" ? +(27.8 - 1.4 * b).toFixed(2) : +(27.8 + 1.4 * b).toFixed(2), { warnings: mode === "prune" ? ["A slimmed model is worse than the original until it is trained again (Step 3). Test it first, then train it."].concat(b * 4 > 16 ? ["More than a quarter of the layers are removed."] : []) : ["Expect little or no gain from your world's data alone: it uses about 5 % more memory."] })) });
const pageState = (over = {}) => ({ models: [{ id: "qwen3.5-27b", name: "Qwen3.5 27B", params_b: 27.8, custom: false }, { id: "custom-a", name: "My slim", params_b: 21.6, custom: true, reshape: "slimmed: 64 → 48 layers" }],
  made: [], dataset_version: 3, env_ready: true, vram_gb: 32, gpu: "RTX 5090", free_disk_gb: 500, unsloth: true, ...over });

const doneModel = (over = {}) => ({ name: "20261113-141500-qwen3.5-27b-slim-48L", status: "done", model_id: "custom-a", size_gb: 43.1, tested: false, test: null,
  summary: { mode: "prune", layers_before: 64, layers_after: 48, dropped: [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51], duplicated: [], params_before: 27.8e9, params_after: 21.6e9, loss_before: 1.92, loss_after: 2.35, stable: true,
    chosen_blocks: [9, 10, 11, 12], boundary_distance: Array.from({ length: 16 }, (_, i) => 0.1 + (i % 5) * 0.05), notes: [], load: "4bit" }, scores: null, ...over });

function open(b, rsState, mode = "prune") {
  b.run(`state = ${JSON.stringify(baseState())}; page = "reshape"; rs.state = ${JSON.stringify(rsState)}; rs.model = "qwen3.5-27b"; rs.mode = ${JSON.stringify(mode)}; rs.options = ${JSON.stringify(optionsFor(mode))}; rs.blocks = 4;`);
}

test("helpers: layer ranges and the layer list typed in Expert mode", () => {
  const b = browser();
  assert.equal(b.run("ranges([36,37,38,39,51,3])"), "3, 36–39, 51");
  assert.equal(b.run("ranges([])"), "");
  const layers = (text) => JSON.parse(b.run(`JSON.stringify(parseLayers(${JSON.stringify(text)}))`));
  assert.deepEqual(layers("36-39, 51"), [36, 37, 38, 39, 51]);
  assert.deepEqual(layers(""), []);
  assert.equal(b.run("parseLayers('5-3')"), null);
  assert.equal(b.run("parseLayers('abc')"), null);
});

test("the page waits for a project and for its data", () => {
  const b = browser();
  b.run(`state = ${JSON.stringify(baseState({ bundle: null }))}; page = "reshape"; rs.state = null;`);
  assert.match(b.run("renderReshape()"), /Import your world data first/);
  b.run(`state = ${JSON.stringify(baseState())};`);
  assert.match(b.run("renderReshape()"), /Loading/);
});

test("slimming: the size slider, the numbers, the warnings and the button", () => {
  const b = browser();
  open(b, pageState());
  const html = b.run("renderReshape()");
  assert.match(html, /Slim it/);
  assert.match(html, /3 fast linear-attention layers \+ 1 full-attention layer, repeated 16×/);
  assert.match(html, /<strong>48 layers · 22.2 B parameters<\/strong>/);           // 4 blocks: 64 - 16 layers, 27.8 - 5.6 B
  assert.match(html, /LoRA ≈ [\d.]+ GB [✓✗], QLoRA ≈ [\d.]+ GB ✓ on this card \(32 GB\)/);
  assert.match(html, /<input type="range" id="rs-slider" min="0" max="4" step="1" value="3"/);
  assert.match(html, /worse than the original until it is trained again/);
  assert.match(html, /data version v3/);
  assert.match(html, /in 4-bit \(NF4\)/);
  assert.doesNotMatch(html, /id="rs-go" disabled/);
  assert.match(html, /about 4\d\.\d GB of free disk space/);
  b.run("rs.blocks = 5");
  assert.match(b.run("renderReshape()"), /More than a quarter of the layers/);
});

test("slimming needs a saved data version, enough disk and the training software", () => {
  const b = browser();
  open(b, pageState({ dataset_version: null }));
  let html = b.run("renderReshape()");
  assert.match(html, /Save a training-data version in Step 2 first/);
  assert.match(html, /id="rs-go" disabled/);
  open(b, pageState({ free_disk_gb: 10 }));
  html = b.run("renderReshape()");
  assert.match(html, /Not enough free disk space/);
  assert.match(html, /id="rs-go" disabled/);
  open(b, pageState({ env_ready: false }));
  assert.match(b.run("renderReshape()"), /Training setup/);
  // explicit layers skip the measuring, so no data version is needed
  open(b, pageState({ dataset_version: null }));
  b.run(`state.settings.expert_mode = true; rs.dropText = "36-51"`);
  html = b.run("renderReshape()");
  assert.match(html, /id="rs-drop" value="36-51"/);
  assert.doesNotMatch(html, /id="rs-go" disabled/);
});

test("growing: the exact-copy start, and the honest note stays next to it", () => {
  const b = browser();
  open(b, pageState(), "grow");
  const html = b.run("renderReshape()");
  assert.match(html, /Grow it<\/strong> — copy a stretch of layers in <span class="dim">\(little or no gain from one world's text; about 40 % more memory\)/);
  assert.match(html, /Honest expectation:<\/strong> from one world's text the gain is usually small or none/);
  assert.match(html, /name="rs-init" value="identity" checked/);
  assert.match(html, /As an exact copy of the original/);
  assert.match(html, /Expect little or no gain from your world&#39;s data alone/);
  assert.doesNotMatch(html, /Measure on:/);
  assert.match(html, /Add <strong id="rs-blocks-n">4<\/strong> blocks of 4 layers/);
  // the grow button needs no data version
  open(b, pageState({ dataset_version: null }), "grow");
  assert.doesNotMatch(b.run("renderReshape()"), /id="rs-go" disabled/);
});

test("a model that cannot be reshaped says why", () => {
  const b = browser();
  open(b, pageState());
  b.run(`rs.options = ${JSON.stringify({ ...optionsFor("prune"), inspect: inspectOf({ supported: false, why: "Gemma 4 stores per-layer inputs." }), options: [] })}`);
  const html = b.run("renderReshape()");
  assert.match(html, /Gemma 4 stores per-layer inputs\./);
  assert.match(html, /id="rs-go" disabled/);
  b.run(`rs.options = null; rs.optErr = "Hugging Face could not be reached."`);
  assert.match(b.run("renderReshape()"), /Hugging Face could not be reached/);
});

test("the result card draws one bar per block and marks the removed ones", () => {
  const b = browser();
  open(b, pageState({ made: [doneModel()] }));
  const html = b.run("renderReshape()");
  assert.equal((html.match(/<rect /g) || []).length, 16);
  assert.equal((html.match(/fill="var\(--err\)"/g) || []).length, 4);                // the four removed blocks are coloured
  assert.match(html, /Removed layers 36–51 \(4 blocks\)/);
  assert.match(html, /64 → 48 layers · 27.80 → 21.60 billion parameters · 43.1 GB on disk/);
  assert.match(html, /Text-prediction error on your data: 1.92 → 2.35 before training it again/);
  assert.match(html, /data-rstest=/);
  assert.match(html, /data-rstrain="custom-a"/);
  assert.match(html, /data-rsdelete=/);
  assert.match(html, /Models made so far/);
  b.run(`rs.state.made[0].summary.stable = false`);
  assert.match(b.run("renderReshape()"), /measurement was not stable/);
});

test("a grown model that started as an exact copy says so, and a test is shown as a table", () => {
  const b = browser();
  const test = { n: 40, method: "overlap", created: "2026-11-13T15:00:00Z", against: "origin", loaded: { original: "4-bit", trained: "16-bit" },
    verdict: { headline: "Better than the original on 20 % of lore questions (worse on 10 %).", better: true, lines: ["General chat is as good as before."] },
    scores: { original: { lore: { mean: 0.4 }, general: { mean: 1 } }, trained: { lore: { mean: 0.5 }, general: { mean: 1 } } } };
  const grown = doneModel({ name: "g1", test, summary: { mode: "grow", layers_before: 32, layers_after: 48, dropped: [], duplicated: [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], params_before: 9.6e9, params_after: 13.1e9, loss_before: 1.5, loss_after: 1.5, notes: ["The new layers start switched off."] } });
  open(b, pageState({ made: [grown] }), "grow");
  const html = b.run("renderReshape()");
  assert.match(html, /Added 16 layers \(copies of layers 8–23\)/);
  assert.match(html, /Unchanged: it starts as an exact copy of the original/);
  assert.match(html, /The new layers start switched off\./);
  assert.match(html, /Better than the original on 20 % of lore questions/);
  assert.match(html, /<td>Lore questions[^<]*<\/td><td>40 %<\/td><td><strong>50 %<\/strong><\/td>/);
  assert.match(html, /original loaded in 4-bit, this model in 16-bit/);
});

test("events: progress, log, scores, done and errors", () => {
  const b = browser({ reshape_state: pageState({ made: [doneModel()] }), get_state: baseState() });
  open(b, pageState());
  b.run(`rs.running = true;`);
  b.run(`onReshapeEvent({ kind: "progress", step: "write", value: 0.4, text: "Writing the new model: 17 of 43 GB" })`);
  b.run(`onReshapeEvent({ kind: "log", text: "Removing layers 36 to 51" })`);
  let html = b.run("renderReshape()");
  assert.match(html, /Writing the new model: 17 of 43 GB/);
  assert.match(html, /<progress max="1" value="0.4">/);
  assert.match(html, /Removing layers 36 to 51/);
  assert.match(html, /id="rs-stop" >Stop/);
  b.run(`onReshapeEvent({ kind: "done", data: { name: "x", model_id: "custom-a", stopped: false } })`);
  assert.equal(b.run("rs.running"), false);
  assert.match(b.run("rs.ok"), /Done: x is ready/);
  b.run(`rs.running = true; onReshapeEvent({ kind: "error", text: "There is not enough free space: 5 GB needed." })`);
  assert.equal(b.run("rs.running"), false);
  assert.match(b.run("renderReshape()"), /There is not enough free space/);
  b.run(`rs.running = true; onReshapeEvent({ kind: "done", data: { stopped: true, name: "y" } })`);
  assert.match(b.run("rs.ok"), /Stopped\. The half-written model was removed\./);
});

test("the job the button sends", () => {
  const b = browser();
  open(b, pageState());
  b.run(`rs.samples = 256; rs.strategy = "separate"; rs.pf = 2; rs.dropText = ""`);
  assert.deepEqual(JSON.parse(JSON.stringify(b.run("rsSpec()"))), { source_model_id: "qwen3.5-27b", mode: "prune", blocks: 4, strategy: "separate", protect_first: 2, protect_last: 1, init: "identity",
    calib_samples: 256, calib_len: 512, load: "auto", drop_layers: null });
  b.run(`rs.dropText = "36-39"`);
  assert.deepEqual(JSON.parse(JSON.stringify(b.run("rsSpec().drop_layers"))), [36, 37, 38, 39]);
  b.run(`rs.mode = "grow"; rs.init = "copy"`);
  const g = JSON.parse(JSON.stringify(b.run("rsSpec()")));
  assert.equal(g.mode, "grow"); assert.equal(g.init, "copy"); assert.equal(g.drop_layers, null);
});

test("the sidebar has the page, unlocked once a world is imported", () => {
  const b = browser();
  b.run(`state = ${JSON.stringify(baseState())}; renderSteps();`);
  assert.match(b.els.steps.innerHTML, /data-key="reshape"[^>]*>🧬 Slim or grow a model/);
  assert.doesNotMatch(b.els.steps.innerHTML.match(/<button[^>]*data-key="reshape"[^>]*>/)[0], /disabled/);
  b.run(`state = ${JSON.stringify(baseState({ bundle: null }))}; renderSteps();`);
  assert.match(b.els.steps.innerHTML.match(/<button[^>]*data-key="reshape"[^>]*>/)[0], /disabled/);
});

test("Step 3 shows where a made model came from and offers 'train only the new layers' for a grown one", () => {
  const b = browser();
  const m = { id: "custom-g", name: "Qwen3.5 9B · grow 48 layers", params_b: 13.1, license: "apache-2.0", good_for: "Made on the Slim or grow page.", custom: true, reshape: "grown: 32 → 48 layers", derived_from: "unsloth/Qwen3.5-9B" };
  const fit = { fit: "yes", est_vram_gb: 12, measured: null };
  b.run(`state = ${JSON.stringify(baseState({ settings: { theme: "light", expert_mode: true, show_help_panel: true } }))}; page = "train";
    tr.state = ${JSON.stringify({ version: { number: 1, counts: { world_book_texts: 5, train_chat: 20 } }, gpu: null, gpu_problem: "none", env_ready: true, servers: [], runs: [],
      models: [{ model: m, lora: fit, qlora: fit, full: { ...fit, available: false, reasons: [], disk_gb: 1 }, recommended: "lora" }] })};
    tr.model = "custom-g"; tr.method = "lora"; tr.plan = { hyper: { rank: 16, lr: 0.0002, epochs: 2, max_steps: null, batch: 2, grad_accum: 4, max_seq: 2048, stages: "mixed", book_epochs: 2 }, steps: 100, est_vram_gb: 12, fit: "yes", notes: [] };`);
  const html = b.run("renderTrain()");
  assert.match(html, /grown: 32 → 48 layers · from Qwen3.5-9B/);
  assert.match(html, /id="tr-newonly"/);
  b.run(`tr.state.models[0].model.reshape = "slimmed: 64 → 48 layers"`);
  assert.doesNotMatch(b.run("renderTrain()"), /id="tr-newonly"/);
});

test("Step 4 can compare a run on a slimmed base with the original", () => {
  const b = browser();
  const run = (name, from) => ({ name, method: "lora", tested: false, reshaped_from: from });
  b.run(`state = ${JSON.stringify(baseState())}; page = "test";
    te.state = ${JSON.stringify({ runs: [run("r-slim", "unsloth/Qwen3.5-27B"), run("r-plain", null)], compare: [{ name: "model:m1", model: "qwen3.5-27b", method: "slim", stages: null, lore: 0.5, lore_original: 0.4, rules: null, invented: 0, general: 1 }], env_ready: true, teacher_ready: false, default_system: "s" })};
    te.run = "r-slim"; te.against = "origin";`);
  let html = b.run("renderTest()");
  assert.match(html, /id="te-against"/);
  assert.match(html, /<option value="origin" selected>the original: Qwen3.5-27B<\/option>/);
  assert.match(html, /model:m1<\/td><td>qwen3.5-27b<\/td><td>slim<\/td>/);              // a made model's row has no stages
  b.run(`te.run = "r-plain"`);
  html = b.run("renderTest()");
  assert.doesNotMatch(html, /id="te-against"/);
  assert.equal(b.run("te.against"), "base");
});

test("the start button sends the choice to the program", async () => {
  const sent = [];
  const b = browser({ reshape_start: (a) => { sent.push(a); return null; } });
  open(b, pageState());
  b.run(`rs.options = ${JSON.stringify(optionsFor("prune"))};`);
  b.run(`renderReshape(); wireReshape();`);
  await b.els["rs-go"].onclick();
  assert.equal(sent.length, 1);
  assert.equal(sent[0].spec.mode, "prune"); assert.equal(sent[0].spec.blocks, 4);
  assert.equal(b.run("rs.running"), true);
});
