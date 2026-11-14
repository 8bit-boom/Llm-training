// Smoke test of the "Try first" card of the Slim-or-grow page, without a window (the same stand-in browser as reshape.test.mjs). Run with `node --test ui/smoke`.
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

function open(b, rsState, mode = "prune") {
  b.run(`state = ${JSON.stringify(baseState())}; page = "reshape"; rs.state = ${JSON.stringify(rsState)}; rs.model = "qwen3.5-27b"; rs.mode = ${JSON.stringify(mode)}; rs.options = ${JSON.stringify(optionsFor(mode))}; rs.blocks = 4;`);
}

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
const BANDS = { edges: [0.03, 0.1], names: ["small", "noticeable", "large"], agree_pairs: 0.8 };
const row = (loss, base, extra = {}) => { const rel = loss / base - 1; return { tried: true, loss, rel, delta: loss - base, ppl: +Math.exp(loss).toFixed(3), band: rel < 0.03 ? "small" : rel < 0.1 ? "noticeable" : "large", ...extra }; };
// 16 blocks of the 27B: blocks 1 to 14 may go; block b alone costs 1 to 5 %, influence rises with the block number
function analysis(over = {}) {
  const base = 2.0;
  const blocks = Array.from({ length: 14 }, (_, i) => row(base * (1 + 0.01 * (1 + ((i + 1) % 5))), base, { block: i + 1, layers: [4 * (i + 1)], influence: 0.1 + 0.01 * i }));
  const windows = Array.from({ length: 11 }, (_, i) => row(base * (1 + 0.02 * (1 + Math.abs(i - 4))), base, { start: i + 1, blocks: [i + 1, i + 2, i + 3, i + 4], layers: Array.from({ length: 16 }, (_, j) => 4 * (i + 1) + j), distance: 0.3 + 0.01 * i }));
  const run4 = { k: 4, windows, best: windows[4], default: windows[7], complete: true, stability: { agree: true, best_agree: true, pairs: 0.95, n: 11 } };
  return { base: { loss: base, ppl: +Math.exp(base).toFixed(3), tokens: 5000 }, bands: BANDS, blocks, blocks_stability: { agree: true, best_agree: true, pairs: 0.9, n: 14 }, runs: { "4": run4 },
    widths: [], widths_stability: null, choices: [], passes: 26, ...over };
}
const wrow = (i, loss, base = 2.0) => row(loss, base, { key: `d=|w=${i}`, intermediate: i });
const sel = (b) => JSON.parse(b.run("JSON.stringify(rs.tr.sel)"));

test("helpers: the time, the percentages, the default number of samples", () => {
  const b = browser();
  assert.equal(b.run("fmtDur(30)"), "about 30 seconds"); assert.equal(b.run("fmtDur(600)"), "about 10 minutes"); assert.equal(b.run("fmtDur(7200)"), "about 2.0 hours"); assert.equal(b.run("fmtDur(null)"), "?");
  assert.equal(b.run("pct(0.0412)"), "+4.1 %"); assert.equal(b.run("pct(-0.02)"), "−2.0 %"); assert.equal(b.run("pct(null)"), "–");
  open(b, pageState());
  assert.equal(b.run("trySamples()"), 64);                       // 27.8 B: Quick
  b.run(`rs.options.inspect.total_params = 9e9`); assert.equal(b.run("trySamples()"), 128);
  b.run(`rs.tr.quick = true`); assert.equal(b.run("trySamples()"), 64);
  b.run(`rs.tr.quick = false; rs.options.inspect.total_params = 27.8e9`); assert.equal(b.run("trySamples()"), 128);
});

test("the request follows what is chosen and refuses what makes no sense", () => {
  const b = browser();
  open(b, pageState());
  const spec = () => JSON.parse(b.run("JSON.stringify(trySpec())"));
  let s = spec();
  assert.deepEqual([s.model_id, s.samples, s.len, s.blocks, s.runs, s.widths, s.choices, s.measure_samples, s.protect_first, s.params_b], ["qwen3.5-27b", 64, 512, true, [], [], [], 256, 1, 27.8]);
  b.run(`rs.tr.what = "runs"`); assert.deepEqual(spec().runs, [4]);
  b.run(`rs.tr.what = "choice"`); assert.equal(spec(), null);                                  // nothing chosen yet
  b.run(`rs.tr.sel = [9, 3]`); assert.deepEqual(spec().choices, [{ blocks: [3, 9] }]);
  b.run(`rs.tr.what = "widths"`); assert.equal(spec(), null);                                  // depth only: no widths to offer
  openWidth(b, pageState(), "width");
  b.run(`rs.tr.what = "widths"`); s = spec();
  assert.equal(s.widths.length, 34); assert.ok(!s.widths.includes(17408)); assert.equal(b.run("tryNeedsMeasure(trySpec())"), true);
  b.run(`rs.meas = { exists: true }`); assert.equal(b.run("tryNeedsMeasure(trySpec())"), false);
  b.run(`rs.tr.what = "choice"; rs.tr.sel = [5]; rs.wi = 13312`); assert.deepEqual(spec().choices, [{ blocks: [], width: 13312 }]);          // width only: clicked blocks do not count
  openWidth(b, pageState(), "both");
  b.run(`rs.tr.what = "choice"; rs.tr.sel = [5]; rs.wi = 12288`); assert.deepEqual(spec().choices, [{ blocks: [5], width: 12288 }]);
  open(b, pageState({ dataset_version: null })); assert.equal(spec(), null);
  open(b, pageState(), "grow"); assert.equal(b.run("tryCard()"), "");
});

test("the card says what it will cost before anything runs, and a guess is called a guess", async () => {
  const b = browser({ reshape_try_plan: () => ({ passes_total: 15, passes_cached: 1, passes_todo: 14, seconds_per_pass: 600, seconds: 8400, guess: true, needs_measurement: false }), reshape_try_results: () => ({ exists: false }) });
  open(b, pageState());
  b.run("renderReshape(); wireReshape();");
  await new Promise((r) => setTimeout(r, 20));
  const html = b.run("tryCard()");
  assert.match(html, /<strong>14<\/strong> passes over your text \(1 already kept\) × 10 minutes each = <strong>about 2\.3 hours<\/strong>/);
  assert.match(html, /a guess from the model's size/);
  assert.match(html, /Quick \(64 samples\)/); assert.match(html, /Quick is the default for models of 15 B or more/);
  assert.match(html, /Nothing is saved except a small results file/);
  assert.match(html, /id="tr-go"(?! disabled)/); assert.match(html, /id="tr-stop" disabled/);
  b.run(`rs.tr.plan = { passes_total: 3, passes_cached: 3, passes_todo: 0, seconds_per_pass: 5, seconds: 0, guess: false }`);
  assert.match(b.run("tryCard()"), /All 3 answers are already kept/);
  b.run(`rs.tr.plan = { passes_total: 3, passes_cached: 0, passes_todo: 3, seconds_per_pass: 5, seconds: 15, guess: false }`);
  assert.match(b.run("tryCard()"), /from a pass that was timed/);
});

test("a width asks for the measurement first", () => {
  const b = browser();
  openWidth(b, pageState(), "width", `rs.tr.what = "widths"; rs.meas = null;`);
  const html = b.run("tryCard()");
  assert.match(html, /needs the neuron measurement/); assert.match(html, /id="tr-measure"/); assert.match(html, /id="tr-go" disabled/);
});

test("the chart: a bar per block with the band colour, the influence marker, the chosen blocks outlined, protected ends greyed", () => {
  const b = browser();
  open(b, pageState());
  b.run(`rs.tr.analysis = ${JSON.stringify(analysis())}; rs.tr.sel = [3, 4]`);
  const html = b.run("tryCard()");
  assert.equal((html.match(/data-trblock=/g) || []).length, 14);                      // the first and the last block are not clickable
  assert.ok(!/data-trblock="0"/.test(html) && !/data-trblock="15"/.test(html));
  assert.match(html, /block 0: protected, never removed/);
  assert.equal((html.match(/stroke-width="2\.5"/g) || []).length, 2);                 // blocks 3 and 4 outlined
  assert.match(html, /fill="var\(--band-small\)"/); assert.match(html, /fill="var\(--band-noticeable\)"/);
  assert.match(html, /◆ = the program's influence score/); assert.match(html, /block 3 \(layers 12–15\): error \+[\d.]+ %, perplexity [\d.]+, influence 0\.12/);
  assert.match(html, /under \+3 % .* \+3 to \+10 % .* over \+10 % — rules of thumb, not measurements/);
  b.run(`rs.pf = 2`); assert.match(b.run("tryCard()"), /block 1: protected/);
});

test("the window table names the cheapest run and the program's own choice", () => {
  const b = browser();
  open(b, pageState());
  b.run(`rs.tr.analysis = ${JSON.stringify(analysis())}`);
  const html = b.run("tryCard()");
  assert.match(html, /Runs of 4 blocks/);
  assert.match(html, /The program's default choice \(blocks 8–11\) costs <strong>\+8\.0 %<\/strong>; the cheapest run \(blocks 5–8\) costs <strong>\+2\.0 %<\/strong>/);
  assert.equal((html.match(/data-trrun=/g) || []).length, 11); assert.match(html, /★ cheapest/); assert.match(html, /◆ the program's choice/);
  b.run(`rs.blocks = 3`); assert.doesNotMatch(b.run("tryCard()"), /Runs of 3/);                          // only the length on the slider is shown
  b.run(`rs.blocks = 4; rs.tr.analysis.runs["4"].default = rs.tr.analysis.runs["4"].windows[4]`);
  assert.match(b.run("tryCard()"), /is also the cheapest run/);
  b.run(`rs.tr.analysis.runs["4"].stability = { agree: false, best_agree: false, pairs: 0.5, n: 11 }; rs.tr.analysis.blocks_stability = { agree: false, best_agree: true, pairs: 0.6, n: 14 }`);
  const w = b.run("tryCard()");
  assert.match(w, /two halves of your text disagree about which run is cheapest/); assert.match(w, /disagree about which blocks are cheapest to skip/);
});

test("toggling blocks changes the choice, and the verdict and the button follow what was tried", () => {
  const b = browser();
  open(b, pageState());
  b.run(`rs.tr.analysis = ${JSON.stringify(analysis())}`);
  assert.match(b.run("tryCard()"), /<h4>Verdict<\/h4>/);
  assert.match(b.run("tryCard()"), /id="tr-use" disabled/);
  b.run("tryToggle(5); tryToggle(6); tryToggle(5); tryToggle(7)");
  assert.deepEqual(sel(b), [6, 7]);
  assert.ok(b.run("tryCard()").includes("has not been tried yet"));                       // the pair 6+7 is not a tried window
  b.run(`rs.tr.sel = [5, 6, 7, 8]`);
  const html = b.run("tryCard()");
  assert.match(html, /Your choice \(blocks 5–8\)<\/strong>: the text-prediction error on your own text goes from 2\.000 to 2\.040 \(<strong>\+2\.0 %<\/strong>; perplexity 7\.389 → 7\.69\d\)/);
  assert.match(html, /a small rise/); assert.match(html, /This is the error <strong>before<\/strong> healing/); assert.match(html, /not a guarantee that it heals well/);
  assert.match(html, /id="tr-use"(?! disabled)/);
});

test("a click on a table row selects that run", () => {
  const b = browser();
  open(b, pageState());
  b.run(`rs.tr.analysis = ${JSON.stringify(analysis())}`);
  const rowEl = { dataset: { trrun: "8" }, onclick: null };
  b.ctx.document.querySelectorAll = (q) => q === "[data-trrun]" ? [rowEl] : [];
  b.run("wireTry()"); rowEl.onclick();
  assert.deepEqual(sel(b), [8, 9, 10, 11]);
});

test("Use this choice fills the Make form with exactly the tried layers and width", () => {
  const b = browser();
  openWidth(b, pageState(), "both");
  b.run(`rs.tr.analysis = ${JSON.stringify(analysis({ choices: [row(2.3, 2.0, { blocks: [9, 10, 11, 12], layers: [], intermediate: 12288 })], widths: [wrow(12288, 2.5)] }))}; rs.tr.sel = [9, 10, 11, 12]; rs.wi = 12288; rs.blocks = 2`);
  assert.match(b.run("tryCard()"), /id="tr-use"(?! disabled)/);
  b.run("tryUse()");
  assert.equal(b.run("rs.dropText"), "36–51"); assert.equal(b.run("rs.blocks"), 4); assert.equal(b.run("rs.wi"), 12288);
  assert.deepEqual(JSON.parse(b.run("JSON.stringify(parseLayers(rs.dropText))")), Array.from({ length: 16 }, (_, i) => 36 + i));
  assert.match(b.run("rs.ok"), /cuts exactly what was tried: layers 36–51, MLP 12,288/);
  const spec = JSON.parse(b.run("JSON.stringify(rsSpec())"));
  assert.deepEqual([spec.drop_layers.length, spec.drop_layers[0], spec.width.intermediate], [16, 36, 12288]);
  assert.match(b.run("renderReshape()"), /Your choice from “Try first” is used: layers 36–51/);
  const d = browser(); open(d, pageState());                       // depth only: no width is added to a depth cut
  d.run(`rs.tr.analysis = ${JSON.stringify(analysis())}; rs.tr.sel = [5, 6, 7, 8]; tryUse()`);
  assert.equal(d.run("rs.dropText"), "20–35"); assert.equal(d.run("rs.shape"), "depth");
});

test("the width curve: a coloured point per width and the numbers", () => {
  const b = browser();
  openWidth(b, pageState(), "width", `rs.tr.what = "widths"; rs.meas = { exists: true }; rs.wi = 12288;`);
  const ws = [13312, 12288, 10752, 8704].map((i, k) => wrow(i, [2.05, 2.18, 2.5, 3.2][k]));
  b.run(`rs.tr.analysis = ${JSON.stringify(analysis({ widths: ws, blocks: [], runs: {} }))}`);
  const html = b.run("tryCard()");
  assert.equal((html.match(/<circle data-trwidth=/g) || []).length, 4); assert.match(html, /quality against size/);
  assert.match(html, /MLP 13,312 \(−24 %\): error \+2\.5 %, perplexity 7\.768/); assert.match(html, /fill="var\(--band-large\)"/);
  assert.match(html, /<tr data-trwidth="12288" style="cursor:pointer;background:var\(--sel\)">/);
  assert.match(b.run("tryCard()"), /Your choice \(MLP 12,288\)<\/strong>: .* \(<strong>\+9\.0 %<\/strong>/);
  b.run(`rs.tr.analysis.widths_stability = { agree: false, best_agree: false, pairs: 0.3, n: 4 }`);
  assert.match(b.run("tryCard()"), /disagree about the order of the widths/);
});

test("starting, the live events, stopping and the end", async () => {
  const sent = [], stopped = [];
  const b = browser({ reshape_try_start: (a) => { sent.push(a); return null; }, reshape_stop: () => { stopped.push(1); return null; }, reshape_try_plan: () => ({ passes_total: 2, passes_cached: 0, passes_todo: 2, seconds_per_pass: 5, seconds: 10, guess: true }) });
  open(b, pageState());
  b.run("renderReshape(); wireReshape();");
  await b.els["tr-go"].onclick();
  assert.equal(sent.length, 1); assert.equal(sent[0].spec.blocks, true); assert.equal(b.run("rs.tr.running"), true);
  let html = b.run("renderReshape()");
  assert.match(html, /id="tr-go" disabled/); assert.match(html, /id="rs-go" disabled/); assert.doesNotMatch(html, /id="tr-stop" disabled/);
  b.run(`onReshapeEvent({ phase: "try", kind: "progress", step: "try", value: 0.25, text: "Cut 2 of 8: block 5 skipped" })`);
  b.run(`onReshapeEvent({ phase: "try", kind: "try_pass", key: "d=5|w=", done: 2, total: 8, seconds_per_pass: 7, analysis: ${JSON.stringify(analysis())} })`);
  html = b.run("renderReshape()");
  assert.match(html, /Cut 2 of 8: block 5 skipped · 2 of 8 done/); assert.match(html, /<progress max="1" value="0.25">/); assert.match(html, /Skipping one block at a time/);
  b.run("wireReshape()"); b.els["tr-stop"].onclick(); assert.equal(stopped.length, 1);
  b.run(`onReshapeEvent({ phase: "try", kind: "done", data: { stopped: true, passes_done: 3, passes_total: 8, analysis: ${JSON.stringify(analysis())} } })`);
  assert.equal(b.run("rs.tr.running"), false); assert.match(b.run("rs.ok"), /Stopped\. 3 of 8 cuts are kept; Try goes on from there/);
  b.run(`onReshapeEvent({ phase: "try", kind: "error", text: "Measure the model first." })`);
  assert.equal(b.run("rs.error"), "Measure the model first."); assert.equal(b.run("rs.tr.running"), false);
});

test("kept answers are shown when the page opens", async () => {
  const b = browser({ reshape_try_results: () => ({ exists: true, analysis: analysis(), seconds_per_pass: 12 }), reshape_try_plan: () => null });
  open(b, pageState());
  b.run("renderReshape(); wireReshape();");
  await new Promise((r) => setTimeout(r, 20));
  assert.equal(b.run("rs.tr.analysis.passes"), 26); assert.equal(b.run("rs.tr.sps"), 12);
  assert.match(b.run("tryCard()"), /Skipping one block at a time/);
});
