// Smoke test of the "Try first" card of the Slim-or-grow page, without a window (the same stand-in browser as reshape.test.mjs). Run with `node --test ui/smoke`.
import test from "node:test";
import assert from "node:assert/strict";
import { makeBrowser, source } from "./_browser.mjs";


function browser(answers = {}, code = source) {
  return makeBrowser({ answers, code, state: baseState });
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

test("a model that thinks is told that a lower error is not a thinking test, and a lower-than-original choice is flagged", () => {
  const b = browser();
  open(b, pageState());
  b.run(`rs.tr.analysis = ${JSON.stringify(analysis())}; rs.tr.sel = [5, 6, 7, 8]`);
  let html = b.run("tryCard()");
  assert.match(html, /id="tr-think-note"/);
  assert.match(html, /A lower text-prediction error does not mean the model can still think\. Make the cut, then run “Check the cut model” in the Thinking check before healing\./);
  assert.doesNotMatch(html, /lower than the original's/, "this choice is worse than the original");
  // a choice that predicts the text better than the original: the measurement is doubted
  b.run(`rs.tr.analysis = ${JSON.stringify(analysis({ windows: undefined }))}; rs.tr.analysis.runs["4"].windows[4].loss = 1.8; rs.tr.analysis.runs["4"].windows[4].rel = -0.1; rs.tr.sel = [5, 6, 7, 8]`);
  html = b.run("tryCard()");
  assert.match(html, /This choice's error is lower than the original's: check that both were measured the same way/);
  // a model that does not think has no such note
  b.run(`rs.options.inspect.model_type = "llama"`);
  assert.doesNotMatch(b.run("tryCard()"), /id="tr-think-note"/);
});

// ------------------------------------------------------------------------------------------------ the automatic search
const SE_ANALYSIS = (n = 3) => ({ base: { loss: 2.0, ppl: 7.39 }, bands: BANDS, search: { stop_reason: null,
  ranking: { 0: ["d=3,7,11|w=10752", "d=3,5,7|w=10752", "d=1,2,3|w=10752"].slice(0, n) },
  rows: { "d=3,7,11|w=10752": { loss: 2.04, loss_h1: 2.04, loss_h2: 2.041 }, "d=3,5,7|w=10752": { loss: 2.2, loss_h1: 2.2, loss_h2: 2.21 }, "d=1,2,3|w=10752": { loss: 2.5, loss_h1: 2.6, loss_h2: 2.4 } } } });
const SE_ROWS = [{ key: "d=3,7,11|w=10752", group: 0, blocks: [3, 7, 11], k: 3, intermediate: 10752, params_b: 21.1, loss: 2.04, rel: 0.02, loss_h1: 2.04, loss_h2: 2.041, noise: 0.001, guardrails: [], think: null }];
const SE_RESULT = (over = {}) => ({ id: "20261119-s", status: "done", rows: SE_ROWS, recommendation: { key: "d=3,7,11|w=10752", group: 0, reason: "Blocks 3, 7, 11 out, MLP 10752, 21.1 B: the text-prediction error rises by 2.0 %. Thinking not checked: a low error does not prove that the model can still think.", confidence: "clear", thinking_checked: false, alternatives: [] }, ...over });

test("the card offers the automatic search first and the request follows the choices", () => {
  const b = browser();
  open(b, pageState());
  const html = b.run("tryCard()");
  assert.ok(html.indexOf("Find the best cut for me (automatic)") < html.indexOf("Every block on its own"));
  b.run(`rs.tr.what = "search"`);
  const s = JSON.parse(b.run("JSON.stringify(seSpec())"));
  assert.deepEqual([s.model_id, s.target, s.budget_min, s.keep_thinking, s.samples, s.axes, s.think_check, s.confirm_full], ["qwen3.5-27b", { b: 21 }, 20, true, 64, "both", "top2", true]);
  b.run(`rs.tr.se.target = "gb"; rs.tr.se.gb = "12.5"; rs.tr.se.budget = 60; rs.tr.se.think = false`);
  const g = JSON.parse(b.run("JSON.stringify(seSpec())"));
  assert.deepEqual([g.target, g.budget_min, g.think_check], [{ gb: 12.5 }, 60, "off"]);
  b.run(`rs.tr.se.target = "pct"`); assert.deepEqual(JSON.parse(b.run("JSON.stringify(seSpec().target)")), { kept_pct: 75 });
  b.run(`rs.tr.se.target = "tradeoff"`); assert.equal(b.run("seSpec().target"), "tradeoff");
  b.run(`rs.tr.se.target = "gb"; rs.tr.se.gb = ""`); assert.equal(b.run("seSpec()"), null);
  const card = b.run("tryCard()");
  assert.match(card, /id="se-card"/); assert.match(card, /name="se-target" value="tradeoff"/); assert.doesNotMatch(card, /id="tr-go"/, "the manual Try button is replaced");
  assert.match(card, /does not measure the quality after healing|a low error does not prove|Keep it able to think/);
});

test("the plan line says passes, time and what fits; the request goes through reshape_search_plan", async () => {
  const plan = { passes: 72, rounds: [{ name: "singles", count: 15 }, { name: "sets_k", count: 27 }, { name: "refine1", count: 6 }], pass_min: [0.2, 0.4], load_min: [3, 5], total_min: [20, 34], budget_min: 20, fit_passes: 40, fit_rounds: 2, complete: false, notes: ["A low error on text does not prove that a model can still think."] };
  const b = browser({ reshape_search_plan: () => plan, reshape_try_results: () => ({ exists: false }), reshape_try_plan: () => null });
  open(b, pageState());
  b.run(`rs.tr.what = "search"; renderReshape(); wireReshape();`);
  await new Promise((r) => setTimeout(r, 20));
  const call = b.calls.find((c) => c[0] === "reshape_search_plan");
  assert.ok(call && call[1].spec.target && call[1].inspect.supported);
  const html = b.run("tryCard()");
  assert.match(html, /About <strong>72<\/strong> measurements/);
  assert.match(html, /40 fit the 20 minutes: rounds 1 to 2/);
  assert.match(html, /does not prove that a model can still think/);
});

test("while it runs the card shows round, pass, time left and a live table of the best cuts", () => {
  const b = browser();
  open(b, pageState());
  b.run(`rs.tr.what = "search"; rs.tr.se.running = true; rs.tr.se.plan = { rounds: [{ count: 15 }, { count: 9 }, { count: 6 }, { count: 6 }, { count: 6 }], passes: 42, pass_min: [0.2, 0.3], load_min: [2, 3], total_min: [10, 16], budget_min: 20, fit_passes: 42, fit_rounds: 5, complete: true, notes: [] }`);
  b.run(`onSearchEvent(${JSON.stringify({ kind: "try_pass", phase: "search", done: 34, total: 72, planned_total: 72, round: "sets_k", round_label: "sets of blocks", round_index: 3, budget_left_s: 540, best: { key: "d=3,7,11|w=10752", loss: 2.04 }, analysis: SE_ANALYSIS() })})`);
  const html = b.run("tryCard()");
  assert.match(html, /Round 3 of 5 \(sets of blocks\) · pass 34 of about 72 · up to 10 min left/);
  assert.match(html, /<progress max="1" value="0.47\d*" id="se-bar">/);
  assert.match(html, /id="se-live-table"/);
  assert.match(html, /data-serow="d=3,7,11\|w=10752"/);
  assert.match(html, /<td>3, 7, 11<\/td><td>10,752<\/td>/);
  assert.match(html, /\+2\.0 %|2\.0 %/);
  assert.match(html, /id="se-stop"(?! disabled)/); assert.match(html, /id="se-go" disabled/);
});

test("the recommendation says how sure it is, whether thinking was checked, and offers the three buttons", () => {
  const b = browser();
  open(b, pageState());
  b.run(`rs.tr.what = "search"; rs.tr.se.result = ${JSON.stringify(SE_RESULT())}; rs.tr.se.id = "20261119-s"`);
  let html = b.run("tryCard()");
  assert.match(html, /id="se-rec"/); assert.match(html, /Blocks 3, 7, 11 out, MLP 10,752/);
  assert.match(html, /A clear winner\./);
  assert.match(html, /Thinking not checked/); assert.match(html, /id="se-think-go"/);
  assert.match(html, /does not measure the quality after healing; for a model that thinks only the thinking check decides/);
  assert.match(html, /id="se-use-ac"[^>]*>Use this cut in Auto-cut/); assert.match(html, /id="se-cut-now"/); assert.match(html, /id="se-cut-heal"[^>]*>Cut and heal it \(Auto-cut, one stage\)/);
  // a close call and a checked thinking result
  const rows = [{ ...SE_ROWS[0], think: { verdict: "kept", score: 0.9, main: { closed_rate: 0.95, loop_rate: 0.0 }, original: { closed_rate: 0.97 } } }];
  b.run(`rs.tr.se.result = ${JSON.stringify(SE_RESULT({ rows, recommendation: { ...SE_RESULT().recommendation, confidence: "close_call", thinking_checked: true } }))}`);
  html = b.run("tryCard()");
  assert.match(html, /A close call: the second best is within the noise/);
  assert.match(html, /it kept its reasoning/); assert.match(html, /the original: closed 97 %/); assert.doesNotMatch(html, /id="se-think-go"/);
});

test("Stop keeps what was measured; the three buttons set the Make form, ac.pin and the stages", async () => {
  const partial = { rows: SE_ROWS, recommendation: SE_RESULT().recommendation };
  const b = browser({ reshape_search_status: () => ({ search: null, partial }) });
  open(b, pageState());
  b.run(`rs.tr.what = "search"; rs.tr.se.running = true`);
  b.run(`onSearchEvent(${JSON.stringify({ kind: "done", phase: "search", data: { id: "x", status: "stopped", rows: [] } })})`);
  await new Promise((r) => setTimeout(r, 20));
  assert.ok(b.calls.some((c) => c[0] === "reshape_search_status"));
  assert.equal(b.run("rs.tr.se.running"), false);
  assert.match(b.run("tryCard()"), /id="se-rec"/, "the partial result is shown");
  assert.match(b.run("rs.ok"), /Stopped\. What was measured is kept/);
  // buttons
  b.run(`rs.tr.se.id = "x"; sePin(null)`);
  assert.deepEqual(JSON.parse(b.run("JSON.stringify(ac.pin)")), { blocks: [3, 7, 11], intermediate: 10752, loss: 2.04, search_id: "x", think: null });
  b.run(`ac.pin = null; sePin(1)`); assert.equal(b.run("ac.stages"), 1); assert.ok(b.run("ac.pin"));
  b.run(`seCutNow()`);
  assert.deepEqual(JSON.parse(b.run("JSON.stringify(rs.tr.sel)")), [3, 7, 11]);
  assert.match(b.run("rs.dropText"), /12-15|12–15/);
  assert.equal(b.run("rs.wi"), 10752);
  assert.match(b.run("rs.ok"), /cuts exactly what the search found: blocks 3, 7, 11/);
});

test("starting a search sends the spec, and the thinking check of the top 2 resumes the same search", async () => {
  const b = browser({ reshape_search_start: () => null, reshape_search_plan: () => ({ passes: 10, rounds: [], pass_min: [1, 1], load_min: [1, 1], total_min: [1, 1], budget_min: 20, fit_passes: 10, fit_rounds: 1, complete: true, notes: [] }), reshape_try_plan: () => null, reshape_try_results: () => ({ exists: false }) });
  open(b, pageState());
  b.run(`rs.tr.what = "search"; renderReshape(); wireReshape();`);
  await new Promise((r) => setTimeout(r, 20));
  b.run(`document.getElementById("se-go").onclick()`);
  await new Promise((r) => setTimeout(r, 10));
  const start = b.calls.find((c) => c[0] === "reshape_search_start");
  assert.ok(start && start[1].spec.target && start[1].resume === null);
  b.run(`rs.tr.se.running = false; rs.tr.se.result = ${JSON.stringify(SE_RESULT())}; rs.tr.se.id = "20261119-s"; renderReshape(); wireReshape();`);
  b.run(`document.getElementById("se-think-go").onclick()`);
  await new Promise((r) => setTimeout(r, 10));
  const again = b.calls.filter((c) => c[0] === "reshape_search_start")[1];
  assert.equal(again[1].resume, "20261119-s");
  assert.equal(again[1].spec.think_check, "top2");
});

test("ranking the neurons by their output: Expert only, sent only when ticked, and the comparison table is honest", () => {
  const b = browser();
  openWidth(b, pageState(), "width", `rs.tr.what = "widths"; rs.meas = { exists: true }; rs.wi = 12288;`);
  const spec = () => JSON.parse(b.run("JSON.stringify(trySpec())"));
  assert.doesNotMatch(b.run("tryCard()"), /id="tr-scoring"/);                                  // Expert only
  b.run(`state.settings.expert_mode = true`);
  assert.match(b.run("tryCard()"), /<input type="checkbox" id="tr-scoring" +>/);
  assert.deepEqual(spec().choices, []);                                                         // the default sends nothing new
  b.run(`rs.tr.scoring = true`);
  let s = spec();
  assert.equal(s.widths.length, 34); assert.equal(s.choices.length, 34);
  assert.deepEqual(s.choices[0], { blocks: [], width: s.widths[0], scoring: "output" });
  b.run(`rs.tr.what = "choice"; rs.wi = 12288`);
  assert.deepEqual(spec().choices, [{ blocks: [], width: 12288 }, { blocks: [], width: 12288, scoring: "output" }]);
  // the Make form: the setting is in the spec only when chosen
  assert.equal(JSON.parse(b.run("JSON.stringify(rsSpec())")).width_scoring, undefined);
  assert.match(b.run("renderReshape()"), /id="rs-scoring"/);
  b.run(`rs.scoring = "output"`);
  assert.equal(JSON.parse(b.run("JSON.stringify(rsSpec())")).width_scoring, "output");
  assert.match(b.run("renderReshape()"), /<option value="output" selected>their expected output \(experimental\)/);
  // the table of variants
  const v = (key, loss, plain, both) => ({ key, tried: true, loss, rel: loss / 2 - 1, label: "MLP 12,288, neurons ranked by their output", rel_vs_plain: loss / plain - 1, better_both_halves: both, options: { s: "output" } });
  b.run(`rs.tr.analysis = ${JSON.stringify(analysis({ widths: [wrow(12288, 2.4)], blocks: [], runs: {}, variants: [v("d=|w=12288|s=output", 2.3, 2.4, true)] }))}`);
  const html = b.run("tryCard()");
  assert.match(html, /id="tr-variants"/);
  assert.match(html, /<td>MLP 12,288, neurons ranked by their output<\/td><td>2\.300<\/td><td>\+15\.0 %<\/td><td>−4\.2 %<\/td><td>yes<\/td>/);
  assert.match(html, /a gain on both halves is a reason to try the same cut healed, not a promise/);
  b.run(`rs.tr.analysis.variants[0].better_both_halves = false`);
  assert.match(b.run("tryCard()"), /−4\.2 %<\/td><td>no<\/td>/);
  b.run(`rs.tr.analysis.variants = []`);
  assert.doesNotMatch(b.run("tryCard()"), /tr-variants/);
});

test("refitting the kept neurons: Expert only, both ways are asked for, and the table shows what the refit gives back on rows it was not fitted on", () => {
  const b = browser();
  openWidth(b, pageState(), "width", `rs.tr.what = "choice"; rs.meas = { exists: true }; rs.wi = 12288; state.settings.expert_mode = true;`);
  const spec = () => JSON.parse(b.run("JSON.stringify(trySpec())"));
  assert.match(b.run("tryCard()"), /<input type="checkbox" id="tr-recon" +>/);
  assert.deepEqual(spec().choices, [{ blocks: [], width: 12288 }]);                                       // off: nothing new is sent
  b.run(`rs.tr.recon = true`);
  assert.deepEqual(spec().choices, [{ blocks: [], width: 12288 }, { blocks: [], width: 12288, reconstruct: "pruned" }, { blocks: [], width: 12288, reconstruct: "original" }]);
  b.run(`rs.tr.what = "widths"`);
  const s = spec(); assert.equal(s.choices.length, 68);
  const v = (key, label, loss, both, recon) => ({ key, tried: true, loss, rel: loss / 2 - 1, label, rel_vs_plain: loss / 2.4 - 1, better_both_halves: both, recon });
  b.run(`rs.tr.analysis = ${JSON.stringify(analysis({ widths: [wrow(12288, 2.4)], blocks: [], runs: {}, variants: [
    v("d=|w=12288|r=pruned", "MLP 12,288, kept neurons refitted (from the cut model's activations)", 2.3, true, { median_removed_share: 0.21, worst_removed_share: -0.04, overlap: 0 }),
    v("d=|w=12288|r=original", "MLP 12,288, kept neurons refitted (from the original's activations)", 2.35, false, { median_removed_share: 0.15, worst_removed_share: -0.2, overlap: 3 })] }))}`);
  const html = b.run("tryCard()");
  assert.match(html, /kept neurons refitted \(from the cut model(?:'|&#39;)s activations\)<\/td><td>2\.300<\/td>[\s\S]*?<td>yes<\/td><td>21\.0 % \/ −4\.0 %<\/td>/);
  assert.match(html, /<td>15\.0 % \/ −20\.0 % · 3 rows overlap Try first<\/td>/);
  assert.match(html, /the refit is algebra, not a result/);
});
