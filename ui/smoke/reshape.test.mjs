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

function browser(answers = {}, code = source) {
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
  vm.runInContext(code, ctx);
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

// ------------------------------------------------------------------------------------------------------------- thinner layers and healing

const FULL = 17408;
const widthOption = (i, blocks = 0) => { const params = +(27.78 - (FULL - i) * 3 * 5120 * 64 / 1e9 - blocks * 1.4).toFixed(2); return option(blocks, 64 - 4 * blocks, params, { intermediate: i, exact: true, warnings: i < 13000 ? ["Removing 38 % of the MLP: answers get clearly worse until the model is healed (Heal step).", "Healing is strongly recommended for width slimming: the original model teaches the thinner one to answer like it did."] : ["Healing is strongly recommended for width slimming: the original model teaches the thinner one to answer like it did."] }); };
const widths = Array.from({ length: 35 }, (_, k) => (34 + k) * 256);
const widthMeta = (over = {}) => ({ supported: true, why: null, dims: { hidden: 5120, intermediate: FULL, heads: 24, kv_heads: 4, lin_v_heads: 48 }, choices: { intermediate: widths, lin_v_heads: [48, 32], heads: [24, 20, 16] },
  suggested: { width: 13312, both: { blocks: 1, intermediate: 12288 } }, heal_tokens_m: [10, 20, 40, 80], default_heal_tokens_m: 20, ...over });
const optionsWidth = (shape) => ({ ...optionsFor("prune"), shape, width: widthMeta(),
  width_options: shape === "width" ? [{ blocks: 0, options: widths.map((i) => widthOption(i)) }] : [1, 2, 3, 4, 5].map((b) => ({ blocks: b, options: widths.map((i) => widthOption(i, b)) })) });

function openWidth(b, rsState, shape = "width", over = "") {
  b.run(`state = ${JSON.stringify(baseState())}; page = "reshape"; rs.state = ${JSON.stringify(rsState)}; rs.model = "qwen3.5-27b"; rs.mode = "prune"; rs.shape = ${JSON.stringify(shape)}; rs.options = ${JSON.stringify(optionsWidth(shape))}; rs.blocks = 4; rs.wi = null; rsPickWidth(); ${over}`);
}

test("a model that allows it offers three ways, and the others only say why not", () => {
  const b = browser();
  open(b, pageState());
  assert.doesNotMatch(b.run("renderReshape()"), /name="rs-shape"/);                         // an answer without width info: as before
  b.run(`rs.options = ${JSON.stringify({ ...optionsFor("prune"), width: widthMeta() })}`);
  const html = b.run("renderReshape()");
  assert.match(html, /name="rs-shape" value="depth" checked/);
  assert.match(html, /Remove whole layers<\/strong> — the result runs faster/);
  assert.match(html, /Make every layer thinner<\/strong> — keeps more quality for the same size \(needs healing\)/);
  assert.match(html, /Both<\/strong> — remove a block and make the rest thinner/);
  assert.match(html, /Thinner layers<\/strong> \(Qwen3.5 and 3.8 only\)/);
  b.run(`rs.options = ${JSON.stringify({ ...optionsFor("prune"), width: widthMeta({ supported: false, why: "Thinning the layers is only available for dense Qwen3.5 / Qwen3.8 models so far. You can still remove whole layers." }) })}`);
  const no = b.run("renderReshape()");
  assert.doesNotMatch(no, /name="rs-shape"/);
  assert.match(no, /only available for dense Qwen3.5 \/ Qwen3.8 models so far/);
  b.run(`rs.mode = "grow"; rs.options = ${JSON.stringify({ ...optionsFor("grow"), width: widthMeta() })}`);
  assert.doesNotMatch(b.run("renderReshape()"), /name="rs-shape"/);                         // growing has no width
});

test("thinner layers start on the mild cut and show the exact size", () => {
  const b = browser();
  openWidth(b, pageState());
  assert.equal(b.run("rs.wi"), 13312);                                                       // the owner's decision: 23.7 B first
  const html = b.run("renderReshape()");
  assert.match(html, /name="rs-shape" value="width" checked/);
  assert.match(html, /MLP <strong id="rs-w-n">13,312<\/strong> of 17,408/);
  assert.match(html, /<input type="range" id="rs-wslider" min="0" max="34" step="1" value="18"/);
  assert.match(html, /MLP 13,312 of 17,408 \(−24 %\) · <strong>64 layers · [\d.]+ B parameters \(exact\)<\/strong>/);
  assert.match(html, /Healing is strongly recommended for width slimming/);
  assert.doesNotMatch(html, /id="rs-slider"/);                                               // width only: no block slider
  assert.match(html, /Make the model/);
  assert.doesNotMatch(html, /id="rs-go" disabled/);
  b.run("rs.wi = 10752");
  const thin = b.run("renderReshape()");
  assert.match(thin, /MLP 10,752 of 17,408 \(−38 %\)/);
  assert.match(thin, /Removing 38 % of the MLP: answers get clearly worse until the model is healed/);
  assert.match(thin, /Measure on: data version v3 \(world book \+ practice answers\), 256 samples/);
});

test("Both: one block and 12,288 first, and a second slider for the blocks", () => {
  const b = browser();
  openWidth(b, pageState(), "both", `rs.blocks = null; rs.wi = null; rsPickWidth();`);
  assert.equal(b.run("rs.blocks"), 1);
  assert.equal(b.run("rs.wi"), 12288);
  const html = b.run("renderReshape()");
  assert.match(html, /name="rs-shape" value="both" checked/);
  assert.match(html, /and remove <strong id="rs-blocks-n">1<\/strong> block of 4 layers/);
  assert.match(html, /<input type="range" id="rs-slider"/);
  assert.match(html, /MLP 12,288 of 17,408 \(−29 %\) · <strong>60 layers/);
  b.run("rs.blocks = 2; rsPickWidth()");
  assert.match(b.run("renderReshape()"), /<strong>56 layers/);
});

test("the Expert axes are labelled experimental and stay off until the graphics-card checks are done", () => {
  const b = browser();
  openWidth(b, pageState());
  assert.doesNotMatch(b.run("renderReshape()"), /thin the attention layers as well/);       // normal mode: not shown at all
  b.run(`state.settings.expert_mode = true`);
  const html = b.run("renderReshape()");
  assert.match(html, /Experimental: thin the attention layers as well/);
  assert.match(html, /Switched off for now: nobody has published how well this works/);
  assert.match(html, /id="rs-lv" +disabled/);
  assert.match(html, /48 → 32 value heads\) <span class="tag">experimental<\/span>/);
  assert.match(html, /<select id="rs-hh" disabled>/);
  assert.match(html, /id="rs-wsamples"/);
  assert.equal(b.run("EXPERIMENTAL_AXES_PASSED"), false);
  assert.equal(b.run("rsSpec().width.lin_v_heads"), null);
});

test("with the checks done, the axes are enabled and go into the job", () => {
  // the flag is a const of the page: a copy of the page with it switched on stands in for "the checklist is passed"
  const flipped = source.replace("const EXPERIMENTAL_AXES_PASSED = false;", "const EXPERIMENTAL_AXES_PASSED = true;");
  assert.notEqual(flipped, source);
  const b = browser({}, flipped);
  openWidth(b, pageState());
  b.run(`state.settings.expert_mode = true`);
  const html = b.run("renderReshape()");
  assert.doesNotMatch(html, /Switched off for now/);
  assert.doesNotMatch(html, /id="rs-lv" +disabled/);
  assert.match(html, /<select id="rs-hh" ><option value="">no<\/option><option value="20" >24 → 20 heads<\/option><option value="16" >24 → 16 heads<\/option>/);
  b.run(`rs.lv = true; rs.hh = 16`);
  assert.deepEqual(JSON.parse(JSON.stringify(b.run("rsSpec().width"))), { intermediate: 13312, lin_v_heads: 32, heads: 16 });
});

test("the job for thinner layers: width only has no blocks, both keeps them", () => {
  const b = browser();
  openWidth(b, pageState());
  let spec = JSON.parse(JSON.stringify(b.run("rsSpec()")));
  assert.equal(spec.blocks, 0);
  assert.deepEqual(spec.width, { intermediate: 13312, lin_v_heads: null, heads: null });
  assert.equal(spec.width_samples, 256);
  assert.equal(spec.drop_layers, null);
  b.run(`rs.shape = "both"; rs.blocks = 1; rs.wi = 12288; rs.wsamples = 512; rs.dropText = "36-39"`);
  spec = JSON.parse(JSON.stringify(b.run("rsSpec()")));
  assert.equal(spec.blocks, 1);
  assert.deepEqual(spec.drop_layers, [36, 37, 38, 39]);
  assert.equal(spec.width.intermediate, 12288);
  assert.equal(spec.width_samples, 512);
  // plain layer removal sends exactly what it always did
  b.run(`rs.shape = "depth"`);
  assert.equal(JSON.parse(JSON.stringify(b.run("'width' in rsSpec()"))), false);
});

test("measuring: reused when it exists, and the button needs a data version", () => {
  const b = browser();
  openWidth(b, pageState());
  b.run(`rs.meas = { exists: false }`);
  let html = b.run("renderReshape()");
  assert.match(html, /Not measured yet: it is measured first/);
  assert.match(html, /id="rs-measure-now"/);
  b.run(`rs.meas = { exists: true, tokens: 131072, load: "4bit" }`);
  html = b.run("renderReshape()");
  assert.match(html, /Measured earlier on this data \(131,072 tokens, loaded as 4bit\): <strong>reused<\/strong>/);
  assert.match(html, /id="rs-measure-again"/);
  openWidth(b, pageState({ dataset_version: null }));
  html = b.run("renderReshape()");
  assert.match(html, /Save a training-data version in Step 2 first/);
  assert.match(html, /id="rs-go" disabled/);
  // explicit layers do not make width skip the measuring
  b.run(`state.settings.expert_mode = true; rs.dropText = "36-51"`);
  assert.match(b.run("renderReshape()"), /id="rs-go" disabled/);
});

test("the start button sends the thinner-layers choice", async () => {
  const sent = [];
  const b = browser({ reshape_start: (a) => { sent.push(a); return null; } });
  openWidth(b, pageState());
  b.run(`renderReshape(); wireReshape();`);
  await b.els["rs-go"].onclick();
  assert.equal(sent[0].spec.width.intermediate, 13312);
  assert.equal(sent[0].spec.blocks, 0);
});

const thinModel = (over = {}) => doneModel({ name: "20261113-150000-qwen3.5-27b-slim-w10752", model_id: "custom-w", healed: false, healed_model_id: null, heal: null, test_healed: null, cache_gb: 0, healed_size_gb: 0,
  summary: { mode: "prune", shape: "width", layers_before: 64, layers_after: 64, dropped: [], duplicated: [], params_before: 27.8e9, params_after: 21.1e9, loss_before: 2.0, loss_after: 3.1, stable: null, chosen_blocks: [], boundary_distance: null, notes: ["The multi-token head was cut to fit but was not measured or healed; do not use it for speculative decoding."], load: "4bit",
    width: { before: { intermediate_size: 17408 }, after: { intermediate_size: 10752 }, stable_overlap: 0.93, scores_key: "k", mtp: "sliced by weight size, not measured" } },
  width_scores: { layers: 64, mlp_mean: Array.from({ length: 64 }, (_, i) => 0.2 + (i % 7) * 0.05), kept_share: 0.81, stable_overlap: 0.93 }, ...over });

const healOpts = (over = {}) => ({ name: "20261113-150000-qwen3.5-27b-slim-w10752", plan: { mode: "cache", student_method: "qlora", est_gb: 18.7, fits: true, why: null }, defaults: { seq: 1024, batch: 2, grad_accum: 16, checkpointing: true, rank: 32, kl_chunk: 128, qlora: true }, safer: null, default_tokens_m: 20, general_dataset: "mlabonne/open-perfectblend", has_data: true, vram_gb: 32, healed: false, heal: null, metrics: [], loss: { original: 2.0, thinned: 3.1 },
  tokens: [{ tokens_m: 10, cache_gb: 2, hours: [3.1, 5.9] }, { tokens_m: 20, cache_gb: 3.9, hours: [6.2, 11.8] }, { tokens_m: 40, cache_gb: 7.8, hours: [12.4, 23.6] }, { tokens_m: 80, cache_gb: 15.7, hours: [24.8, 47.2] }], ...over });

test("the result card of a thinner model: the neurons chart, the stability and the honest numbers", () => {
  const b = browser();
  open(b, pageState({ made: [thinModel()] }));
  const html = b.run("renderReshape()");
  assert.equal((html.match(/<rect /g) || []).length, 64);                                   // one bar per layer: the mean MLP score
  assert.match(html, /Made every layer thinner \(MLP 17,408 → 10,752\)/);
  assert.match(html, /64 → 64 layers · 27.80 → 21.10 billion parameters/);
  assert.match(html, /Text-prediction error on your data: 1.92? → 3.10|Text-prediction error on your data: 2.00 → 3.10/);
  assert.match(html, /The choice of neurons was stable \(93 % the same/);
  assert.match(html, /hold 81 % of the total score/);
  assert.match(html, /do not use it for speculative decoding/);
  assert.match(html, /Layers<\/th>/);
  assert.match(html, /MLP 17,408 → 10,752/);
  b.run(`rs.state.made[0].summary.width.stable_overlap = 0.6`);
  assert.match(b.run("renderReshape()"), /only 60 % the same neurons/);
});

test("the Heal card: explicit about the download, the budgets, the memory and the Expert settings", () => {
  const b = browser();
  const m = thinModel();
  open(b, pageState({ made: [m] }));
  b.run(`rs.hl = { name: ${JSON.stringify(m.name)}, opts: ${JSON.stringify(healOpts())} }`);
  let html = b.run("renderReshape()");
  assert.match(html, /Heal it \(recommended\)/);
  assert.match(html, /The original model teaches the thinner one to answer like it did/);
  assert.match(html, /id="hl-general" checked/);                                             // owner decision: general text on by default
  assert.match(html, /It is downloaded from Hugging Face<\/strong> \(or read from a file you choose\)/);
  assert.match(html, /Without it, or with this box unticked, healing uses your world's text only/);
  assert.match(html, /<option value="20" selected>20 million tokens<\/option>/);              // 20 M for a student of 15 B or more
  assert.match(html, /<option value="80" >80 million tokens<\/option>/);
  assert.match(html, /about 6.2–11.8 hours \(an estimate until this card has been measured\) · 3.9 GB of disk for the original's saved answers/);
  assert.match(html, /they take turns\): about 18.7 GB of memory \(this card has 32 GB\)/);
  assert.match(html, /trained as QLoRA/);
  assert.match(html, /Before healing: error 3.10 on your data; the original has 2.00/);
  assert.match(html, /id="hl-go" >Start healing/);
  assert.doesNotMatch(html, /id="hl-dataset"/);                                              // Expert only
  b.run(`state.settings.expert_mode = true`);
  html = b.run("renderReshape()");
  assert.doesNotMatch(html, /id="hl-dataset"/);                                              // the dataset is chosen in the chooser for everybody, not in Expert
  assert.match(html, /id="hl-rank" value="32"/);                                             // the automatic value for a 21 B student
  assert.match(html, /id="hl-lr" value="0.0001"/);
  // a 9B-sized student: 40 M by default, and LoRA in 16-bit
  b.run(`rs.hl.opts.default_tokens_m = 40; rs.hl.opts.plan = { mode: "online", student_method: "lora", est_gb: 28.4, fits: true }`);
  html = b.run("renderReshape()");
  assert.match(html, /<option value="40" selected>/);
  assert.match(html, /run at the same time/);
  assert.match(html, /trained as LoRA/);
});

test("healing that does not fit says so and cannot be started; offline general text is shown plainly", () => {
  const b = browser();
  const m = thinModel();
  open(b, pageState({ made: [m] }));
  b.run(`rs.hl = { name: ${JSON.stringify(m.name)}, opts: ${JSON.stringify(healOpts({ plan: { mode: "cache", student_method: "qlora", est_gb: 18.7, fits: false, why: "Healing a model of 21 B parameters needs about 19 GB with the teacher's answers saved first; this card has 16 GB. Choose a smaller model or heal on a bigger card." } }))} }`);
  let html = b.run("renderReshape()");
  assert.match(html, /this card has 16 GB\. Choose a smaller model or heal on a bigger card/);
  assert.match(html, /id="hl-go" disabled/);
  b.run(`rs.hl.opts.plan.fits = true; rs.hl.opts.heal = { status: "stopped", step: "train", note: "The general text (org/x) could not be downloaded: offline. Healing goes on with your world's text only; try again later for better results." }`);
  html = b.run("renderReshape()");
  assert.match(html, /could not be downloaded: offline/);
  assert.match(html, /Continue healing/);
  assert.match(html, /Healing was stopped during step train\. Continue goes on where it stopped\./);
  b.run(`rs.hl.opts.heal.status = "failed"; rs.hl.opts.heal.error = "Healing stopped: the training became unstable."`);
  assert.match(b.run("renderReshape()"), /Healing stopped with an error: Healing stopped: the training became unstable/);
});

test("healing events: the three steps, the charts and the end", async () => {
  const b = browser({ reshape_state: pageState({ made: [thinModel({ healed: true, healed_model_id: "custom-w-healed", healed_size_gb: 42.3 })] }), get_state: baseState(), heal_options: healOpts() });
  const m = thinModel();
  open(b, pageState({ made: [m] }));
  b.run(`rs.hl = { name: ${JSON.stringify(m.name)}, opts: ${JSON.stringify(healOpts())}, stale: false }; rs.hlRun = { running: true, step: "cache", text: "", progress: null, live: [] }`);
  b.run(`onReshapeEvent({ kind: "progress", phase: "heal", heal_step: "cache", step: "cache", value: 0.25, text: "The original model answers your texts: 5 of 20" })`);
  let html = b.run("renderReshape()");
  assert.match(html, /1 of 3 · The original model answers your texts · The original model answers your texts: 5 of 20/);
  assert.match(html, /<progress max="1" value="0.25">/);
  assert.match(html, /id="hl-stop" >Stop/);
  for (let i = 1; i <= 6; i++) b.run(`onReshapeEvent({ kind: "distill_metric", phase: "heal", heal_step: "train", step: ${i * 10}, kd: ${1.2 - i * 0.1}, ce: 0, tokens: ${i * 1000}, total_steps: 60 })`);
  b.run(`onReshapeEvent({ kind: "distill_eval", phase: "heal", heal_step: "train", step: 0, calib_loss: 3.1 })`);
  b.run(`onReshapeEvent({ kind: "distill_eval", phase: "heal", heal_step: "train", step: 50, calib_loss: 2.4 })`);
  html = b.run("renderReshape()");
  assert.match(html, /2 of 3 · The thinner model learns from the original/);
  assert.match(html, /How far the thinner model&#39;s answers are from the original&#39;s \(KL\) \(lower is better\): 1.100 → 0.600/);
  assert.match(html, /Text-prediction error on your data \(lower is better\): 3.100 → 2.400/);
  b.run(`onReshapeEvent({ kind: "progress", phase: "heal", heal_step: "merge", step: "merge", value: 0.5, text: "Saving the healed model: 20 of 42 GB" })`);
  assert.match(b.run("renderReshape()"), /3 of 3 · Saving the healed model/);
  b.run(`onReshapeEvent({ kind: "log", phase: "heal", text: "Healing: cache mode" })`);
  assert.match(b.run("rs.log.join('|')"), /Healing: cache mode/);
  b.run(`onReshapeEvent({ kind: "done", phase: "heal", data: { name: ${JSON.stringify(m.name)}, model_id: "custom-w-healed" } })`);
  assert.equal(b.run("rs.hlRun.running"), false);
  assert.match(b.run("rs.ok"), /Healed: .* now has a healed version/);
  b.run(`rs.hlRun.running = true; onReshapeEvent({ kind: "error", phase: "heal", text: "The teacher's answers need 15.7 GB in x; 1.5 GB free." })`);
  assert.equal(b.run("rs.hlRun.running"), false);
  assert.match(b.run("rs.error"), /1.5 GB free/);
  b.run(`rs.hlRun.running = true; onReshapeEvent({ kind: "done", phase: "heal", data: { stopped: true, name: "x" } })`);
  assert.match(b.run("rs.ok"), /Stopped\. Healing can be continued/);
});

test("the Heal buttons send the choice, and the measure events end quietly", async () => {
  const sent = [];
  const m = thinModel();
  const b = browser({ heal_start: (a) => { sent.push(a); return null; }, heal_stop: () => { sent.push("stop"); return null; } });
  open(b, pageState({ made: [m] }));
  b.run(`rs.hl = { name: ${JSON.stringify(m.name)}, opts: ${JSON.stringify(healOpts())} }; renderReshape(); wireReshape();`);
  await b.els["hl-go"].onclick();
  assert.equal(sent[0].name, m.name);
  assert.deepEqual(JSON.parse(JSON.stringify(sent[0].spec)), { tokens_m: 20, general_share: 0.5, general: null, alpha: 1, lr: 0.0001, mode: null, rank: null, seq: null, batch: null, per_update: null, checkpointing: "auto", method: null,
    optimizer: null, kl_chunk: null, k: null, teacher_load: null, warmup: null, kbit_upcast: null });
  assert.equal(b.run("rs.hlRun.running"), true);
  b.run(`rs.hlSpec.general = false; rs.hlSpec.tokens_m = 40`);
  assert.equal(b.run("rsHealSpec().general_share"), 0);
  assert.equal(b.run("rsHealSpec().tokens_m"), 40);
  b.run(`rs.running = true; onReshapeEvent({ kind: "done", phase: "measure", data: { key: "k" } })`);
  assert.equal(b.run("rs.running"), false);
  assert.match(b.run("rs.ok"), /Measured/);
});

test("the table of the three models: original, thinner, healed", () => {
  const t = (lore, general) => ({ n: 40, method: "overlap", created: "2026-11-13", against: "origin", loaded: {}, verdict: { headline: lore > 0.6 ? "Close to the original." : "Worse than the original.", better: false, lines: [] },
    scores: { original: { lore: { mean: 0.8 }, rules: { mean: 0.7 }, general: { mean: 1 } }, trained: { lore: { mean: lore }, rules: { mean: lore - 0.1 }, general: { mean: general } } } });
  const b = browser();
  const m = thinModel({ healed: true, healed_model_id: "custom-w-healed", healed_size_gb: 42.3, cache_gb: 3.9, test: t(0.3, 0.6), test_healed: t(0.7, 0.95),
    heal: { status: "done", result: { calib_loss_after: 2.4 } } });
  open(b, pageState({ made: [m], models: [{ id: "custom-w-healed", name: "Qwen3.5 27B · slim MLP 10,752 · healed", params_b: 21.1, custom: true }] }));
  const html = b.run("renderReshape()");
  assert.match(html, /Original · thinner · healed/);
  assert.match(html, /<td>Lore questions[^<]*<\/td><td>80 %<\/td><td>30 %<\/td><td><strong>70 %<\/strong><\/td>/);
  assert.match(html, /<td>General chat sensible<\/td><td>100 %<\/td><td>60 %<\/td><td><strong>95 %<\/strong><\/td>/);
  assert.match(html, /error on your data \(lower is better\)[^<]*<\/td><td>2.00<\/td><td>3.10<\/td><td><strong>2.40<\/strong><\/td>/i);
  assert.match(html, /Close to the original\./);
  assert.match(html, /Healed: <strong>Qwen3.5 27B · slim MLP 10,752 · healed<\/strong> is in the model list of Step 3 \(42.3 GB\)\. The original's saved answers \(3.9 GB\)/);
  assert.match(html, /data-rstest-healed=/);
  assert.match(html, /data-rstrain="custom-w-healed"[^>]*>Train the healed model on your world/);
  assert.match(html, /<span class="tag">healed<\/span>/);
  assert.doesNotMatch(html, /id="hl-go"/);
});

test("a depth-only model gets an optional Heal card; a grown one gets none", () => {
  const b = browser();
  const m = doneModel();
  open(b, pageState({ made: [m] }));
  b.run(`rs.hl = { name: ${JSON.stringify(m.name)}, opts: ${JSON.stringify(healOpts({ name: m.name }))} }`);
  assert.match(b.run("renderReshape()"), /Heal it \(optional\)/);
  open(b, pageState({ made: [doneModel({ summary: { ...m.summary, mode: "grow", duplicated: [8, 9] } })] }));
  b.run(`rs.hl = { name: ${JSON.stringify(m.name)}, opts: ${JSON.stringify(healOpts({ name: m.name }))} }`);
  assert.doesNotMatch(b.run("renderReshape()"), /Heal it/);
});

const estOf = (over = {}, plan = {}) => ({ plan: { mode: "cache", student_method: "qlora", est_gb: 23.1, fits: true, why: null, verdict: "fits", verdict_text: "Fits: about 23 GB of the 32 GB of this card (an estimate).",
  breakdown: { weights_gb: 16.4, lora_gb: 2.1, activations_gb: 1.2, kl_gb: 0.3, overhead_gb: 1, total_gb: 23.1 }, ...plan }, knobs: { seq: 1024, batch: 2, grad_accum: 16, checkpointing: true, rank: 32, kl_chunk: 128, qlora: true }, ...over });

test("Healing settings: only in Expert mode, with the automatic values shown, the live estimate in plain words and Reset", async () => {
  const asked = [];
  const m = thinModel();
  const b = browser({ heal_estimate: (a) => { asked.push(JSON.parse(JSON.stringify(a.spec))); return a.spec.seq === 4096 ? estOf({}, { fits: false, why: "Healing a model of 16 B parameters needs about 41 GB (an estimate); this card has 32 GB.", verdict: "over", verdict_text: "Will probably run out of memory: about 41 GB needed, this card has 32 GB (an estimate)." }) : estOf(); } });
  open(b, pageState({ made: [m] }));
  b.run(`rs.hl = { name: ${JSON.stringify(m.name)}, opts: ${JSON.stringify(healOpts())} }`);
  assert.doesNotMatch(b.run("renderReshape()"), /Healing settings/);                           // the safe defaults apply without Expert mode
  b.run(`state.settings.expert_mode = true`);
  let html = b.run("renderReshape()");
  assert.match(html, /<summary>Healing settings<\/summary>/);
  assert.match(html, /<option value="1024" selected>1024<\/option>/);
  assert.match(html, /<option value="2" selected>2<\/option>/);
  assert.match(html, /id="hl-perupdate" value="32"/);
  assert.match(html, /= 2 × 16 steps added up/);
  assert.match(html, /<option value="auto" selected>Auto \(on\)<\/option>/);
  assert.match(html, /Automatic \(QLoRA 4-bit\)/);
  assert.match(html, /8-bit AdamW/);
  assert.match(html, /id="hl-klchunk" value="128"/);
  assert.match(html, /id="hl-reset"[^>]*>Reset to safe defaults/);
  b.run(`renderReshape(); wireReshape();`);
  b.els["hl-seq"].value = "4096";
  await b.els["hl-seq"].onchange();
  await new Promise((r) => setTimeout(r, 5));
  assert.equal(asked.at(-1).seq, 4096);
  b.run(`state.settings.expert_mode = true`);                                                 // (the page re-render after a change may have reloaded the settings)
  html = b.run("renderReshape()");
  assert.match(html, /id="hl-verdict">Will probably run out of memory: about 41 GB needed/);
  assert.match(html, /Weights 16.4 GB, LoRA and optimizer 2.1 GB, activations 1.2 GB/);
  assert.match(html, /<option value="4096" selected>4096<\/option>/);
  b.run(`state.settings.expert_mode = true; wireReshape()`);
  await b.els["hl-reset"].onclick();
  await new Promise((r) => setTimeout(r, 5));
  assert.equal(b.run("rs.hlSpec.seq"), null);
  assert.equal(asked.at(-1).seq, null);
  b.run(`state.settings.expert_mode = true`);
  assert.match(b.run("renderReshape()"), /id="hl-verdict">Fits: about 23 GB/);
});

test("Healing settings: the 32-bit embedding copy is off by default, its memory comes from the estimate, ticking it is sent, Reset and the safer step clear it", async () => {
  const asked = [];
  const m = thinModel();
  const b = browser({ heal_estimate: (a) => { asked.push(JSON.parse(JSON.stringify(a.spec))); const on = a.spec.kbit_upcast === true;
    return estOf({ upcast_gb: 5.6, knobs: { seq: 1024, batch: 2, grad_accum: 16, checkpointing: true, rank: 32, kl_chunk: 128, qlora: true, upcast: on } },
      on ? { est_gb: 28.7, verdict: "tight", verdict_text: "Tight: about 29 GB of the 32 GB of this card (an estimate)." } : {}); } });
  open(b, pageState({ made: [m] }));
  b.run(`rs.hl = { name: ${JSON.stringify(m.name)}, opts: ${JSON.stringify(healOpts({ upcast_gb: 5.6 }))} }`);
  b.run(`state.settings.expert_mode = true`);
  let html = b.run("renderReshape()");
  assert.match(html, /<input type="checkbox" id="hl-upcast" +>/);                                                   // unchecked: the new default
  assert.match(html, /Keep the embeddings and output layer in 32-bit \(the old behaviour: about <span id="hl-upcast-gb">5\.6<\/span> GB more memory\)/);
  assert.equal(b.run("rsHealSpec().kbit_upcast"), null);                                                             // nothing sent: the program's default (off)
  b.run(`renderReshape(); wireReshape();`);
  b.els["hl-upcast"].checked = true;
  await b.els["hl-upcast"].onchange();
  await new Promise((r) => setTimeout(r, 5));
  assert.equal(asked.at(-1).kbit_upcast, true);
  b.run(`state.settings.expert_mode = true`);
  html = b.run("renderReshape()");
  assert.match(html, /<input type="checkbox" id="hl-upcast" checked +>/);
  assert.match(html, /id="hl-verdict">Tight: about 29 GB/);                                                          // the live estimate followed
  assert.equal(b.run("rsHealSpec().kbit_upcast"), true);
  b.run(`wireReshape()`);
  await b.els["hl-reset"].onclick();
  await new Promise((r) => setTimeout(r, 5));
  assert.equal(b.run("rs.hlSpec.kbit_upcast"), null);
  assert.equal(asked.at(-1).kbit_upcast, null);
  // LoRA 16-bit has no such copy: the box is greyed out and says so
  b.run(`state.settings.expert_mode = true; rs.hlEst = { name: ${JSON.stringify(m.name)}, key: "", est: ${JSON.stringify(estOf({ knobs: { seq: 1024, batch: 2, grad_accum: 16, checkpointing: true, rank: 32, kl_chunk: 128, qlora: false } }))} }`);
  html = b.run("renderReshape()");
  assert.match(html, /id="hl-upcast" +disabled>.*only for QLoRA/);
});

test("After an out-of-memory failure the first safer step can be the 32-bit embedding copy turned off", async () => {
  const m = thinModel();
  const b = browser({ heal_estimate: () => estOf() });
  open(b, pageState({ made: [m] }));
  const heal = { status: "failed", step: "train", oom: true, error: "The graphics card ran out of memory during step train.", retries: [] };
  const safer = { label: "the 32-bit embedding copy turned off", kbit_upcast: false, checkpointing: "on", batch: 2, per_update: 32, kl_chunk: 128 };
  b.run(`rs.hl = { name: ${JSON.stringify(m.name)}, opts: ${JSON.stringify(healOpts({ heal, safer }))} }`);
  b.run(`rs.hlSpec.kbit_upcast = true`);
  assert.match(b.run("renderReshape()"), /<strong>the 32-bit embedding copy turned off<\/strong>/);
  b.run(`wireReshape()`);
  await b.els["hl-safer"].onclick();
  assert.equal(b.run("rs.hlSpec.kbit_upcast"), null);
});

test("After an out-of-memory failure the card offers safer settings instead of a blind Continue", async () => {
  const m = thinModel();
  const b = browser({ heal_estimate: () => estOf() });
  open(b, pageState({ made: [m] }));
  const heal = { status: "failed", step: "train", oom: true, error: "The graphics card ran out of memory during step train. Settings used: sequence length 1024, batch 2 x 16 steps added up. What to change next, in this order: 1. a shorter sequence length.", retries: [{ attempt: 1, change: "gradient checkpointing" }] };
  const safer = { label: "a smaller batch (1 per step, 32 steps added up)", checkpointing: "on", batch: 1, per_update: 32, kl_chunk: 128 };
  b.run(`rs.hl = { name: ${JSON.stringify(m.name)}, opts: ${JSON.stringify(healOpts({ heal, safer }))} }`);
  let html = b.run("renderReshape()");
  assert.match(html, /Healing stopped: The graphics card ran out of memory during step train/);
  assert.match(html, /Tried automatically: gradient checkpointing\./);
  assert.match(html, /The next safer setting that keeps what the model learns: <strong>a smaller batch \(1 per step, 32 steps added up\)<\/strong>/);
  assert.match(html, /id="hl-safer">Try again with safer settings/);
  assert.match(html, /id="hl-go" >Continue with the settings above/);
  assert.doesNotMatch(html, /Continue goes on where it stopped/);
  assert.match(html, /makes the original answer again or starts the learning again/);
  b.run(`wireReshape()`);
  await b.els["hl-safer"].onclick();
  assert.deepEqual(JSON.parse(b.run("JSON.stringify([rs.hlSpec.checkpointing, rs.hlSpec.batch, rs.hlSpec.per_update, rs.hlSpec.kl_chunk])")), ["on", 1, 32, 128]);
  // used up: the plain list of what is left
  b.run(`rs.hl.opts.safer = null`);
  assert.match(b.run("renderReshape()"), /The automatic steps are used up\. Next, in this order: a shorter sequence length, QLoRA, a lower LoRA size/);
  // another failure keeps the old words and the old Continue
  b.run(`rs.hl.opts.heal = { status: "failed", step: "train", error: "Healing stopped: the training became unstable." }`);
  html = b.run("renderReshape()");
  assert.match(html, /Continue healing/);
  assert.doesNotMatch(html, /hl-safer/);
});
