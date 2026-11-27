// Smoke test of the mixture-of-experts card on the Slim page (Inkling-2-Mini): facts, the software check, the config-only cuts and what the button sends.
import test from "node:test";
import assert from "node:assert/strict";
import { makeBrowser, source, baseState } from "./_browser.mjs";

const sizes = [{ experts: 128, params: 26.4e9, packed_bytes: 19.5e9 }, { experts: 96, params: 20.3e9, packed_bytes: 15.6e9 }, { experts: 64, params: 14.2e9, packed_bytes: 11.7e9 }];
const moeInspect = (over = {}) => ({ model_type: "inkling2_loop", supported: true, why: null, layers: 24, layer_types: Array(24).fill("sliding_attention"), period: 6, blocks: 4, total_params: 26388641280, has_vision: true, local: false,
  moe: { arch: "inkling2_loop", experts: 128, top_k: 8, passes: 3, varloop: false, core: [6, 18], visits: 48, active_params: 5.4e9, packed_bytes: 19525870752,
         sizes }, ...over });
const finding = (tool, status, extra = {}) => ({ tool, need: "x", have: "1", status, message: `${tool} ${status}`, fix: status === "ok" ? "" : "do this", command: null, rollback: null, blocks: "all", ...extra });
const pageState = (over = {}) => ({ models: [{ id: "inkling-2-mini", name: "Inkling-2-Mini", params_b: 26.39 }], made: [], dataset_version: 3, env_ready: true, free_disk_gb: 500, ...over });

function open(answers = {}) {
  const b = makeBrowser({ answers: { moe_compat: { family: "inkling2", findings: [finding("transformers", "too_old", { command: "uv pip install \"transformers>=5.24.0,<6\"", rollback: "uv pip install \"transformers==5.17.0\"" }), finding("llama.cpp", "ok")], summary: "Upgrade needed for Inkling-2-Mini: transformers.", repo: null }, ...answers }, state: () => baseState() });
  b.run(`state = ${JSON.stringify(baseState())}; page = "reshape"; rs.state = ${JSON.stringify(pageState())}; rs.model = "inkling-2-mini"; rs.mode = "prune"; rs.options = ${JSON.stringify({ model: { id: "inkling-2-mini" }, mode: "prune", inspect: moeInspect(), options: [], default_blocks: 0, load: "auto", vram_gb: 32 })};`);
  return b;
}

test("the card shows the facts, replaces the block slider and offers only cuts that exist", () => {
  const b = open();
  const html = b.run("renderReshape()");
  assert.match(html, /128 experts in each of 24 layers, 8 used per token/);
  assert.match(html, /run 3 times, so a token passes through 48 layer slots/);
  assert.match(html, /19\.5 GB in the authors' packed format/);
  assert.doesNotMatch(html, /rs-slider/);
  assert.match(html, /id="moe-passes"/);
  assert.match(html, /2 passes \(never trained so/);
  assert.doesNotMatch(html, /<option value="1">1 pass/);             // a fixed loop cannot go to 1 pass
  assert.match(html, /<option value="96" >96 \(15\.6 GB packed/);
  assert.match(html, /Not measured yet/);
  assert.match(html, /id="moe-go" disabled/);                          // nothing chosen yet
});

test("an old transformers is explained with the command and the way back", async () => {
  const b = open();
  await b.run("moeLoad(false)");
  const html = b.run("renderReshape()");
  assert.match(html, /Upgrade needed for Inkling-2-Mini: transformers\./);
  assert.match(html, /uv pip install &quot;transformers&gt;=5\.24\.0,&lt;6&quot;/);
  assert.match(html, /to go back:/);
  assert.deepEqual(JSON.parse(JSON.stringify(b.calls.filter((c) => c[0] === "moe_compat").map((c) => c[1]))), [{ model: "inkling-2-mini", checkRepo: false }]);
});

test("the repository check is asked for, and a change is shown", async () => {
  const b = open({ moe_compat: (a) => ({ family: "inkling2", findings: [], summary: null, repo: a.checkRepo ? { changed: true, message: "thinkingmachines/Inkling-2-Mini has new changes (abcdef0 instead of c9d0283)." } : null }) });
  await b.run("moeLoad(true)");
  assert.match(b.run("renderReshape()"), /has new changes \(abcdef0 instead of c9d0283\)/);
});

test("the button sends the chosen cuts and nothing else", async () => {
  const b = open();
  b.run(`moe.passes = "2"; moe.topK = "6";`);
  assert.doesNotMatch(b.run("renderReshape()"), /id="moe-go" disabled/);
  assert.deepEqual(JSON.parse(b.run("JSON.stringify(moeSpec())")), { source_model_id: "inkling-2-mini", mode: "prune", blocks: 0, calib_samples: 64, calib_len: 2048, load: "auto", passes: 2, top_k: 6 });
  b.run(`moe.passes = "";`);
  assert.deepEqual(JSON.parse(b.run("JSON.stringify(moeSpec())")), { source_model_id: "inkling-2-mini", mode: "prune", blocks: 0, calib_samples: 64, calib_len: 2048, load: "auto", top_k: 6 });
});

test("a dense model's page is the one it always was", () => {
  const b = open();
  b.run(`rs.options.inspect = { model_type: "qwen3_5", supported: true, layers: 8, layer_types: [], period: 4, blocks: 2, total_params: 1e9 };`);
  assert.equal(b.run("moeOn()"), false);
});

test("removing experts needs the measurement, then sends the count with its defaults", async () => {
  const b = open({ reshape_measurement: { exists: false } });
  b.run(`moe.experts = "96";`);
  let html = b.run("renderReshape()");
  assert.match(html, /id="moe-go" disabled/);
  assert.match(html, /Measure the experts first/);
  b.run(`moe.meas = { exists: true, tokens: 131072, load: "packed", data_version: 3 };`);
  html = b.run("renderReshape()");
  assert.doesNotMatch(html, /id="moe-go" disabled/);
  assert.match(html, /Measured on data version v3 \(131,072 tokens/);
  assert.deepEqual(JSON.parse(b.run("JSON.stringify(moeSpec())")), { source_model_id: "inkling-2-mini", mode: "prune", blocks: 0, calib_samples: 64, calib_len: 2048, load: "auto", experts: { keep: 96, scoring: "reap", rebalance: true } });
  await b.run("moeMeasure(true)");
  assert.deepEqual(JSON.parse(JSON.stringify(b.calls.filter((c) => c[0] === "reshape_measure").map((c) => c[1]))), [{ model: "inkling-2-mini", samples: 64, len: 2048, load: "auto", force: true }]);
});

test("the recommendations panel shows the order of work, what to watch and what is not known", async () => {
  const reco = { goal: "size", blocked: null, order: [{ id: "goal", title: "Decide the goal", text: "A smaller file or faster answers." }, { id: "cut", title: "Remove experts: 128 to 96", text: "A byte copy." }],
    first_try: ["base", "e=96"], expected: [{ key: "e=96", experts: 96, top_k: 8, passes: 3, q4_gb: 12.4, packed_gb: 15.6, active_b: 4.6, heal_tokens_m: 20, heal_hours: [4.3, 8.9], heal_gb: 24.8, disk_peak_gb: 61 }],
    warnings: ["16-bit LoRA needs about 59 GB."], unknowns: ["Every time and memory figure is an estimate."] };
  const b = open({ moe_recommend: reco });
  await b.run("moeReco()");
  const html = b.run("renderReshape()");
  assert.match(html, /id="moe-reco"/); assert.match(html, /<li><strong>Remove experts: 128 to 96<\/strong>/);
  assert.match(html, /<td>e=96<\/td><td>96<\/td><td>8<\/td><td>3<\/td><td>12\.4 GB<\/td>/);
  assert.match(html, /16-bit LoRA needs about 59 GB/); assert.match(html, /Not known yet/);
  assert.match(html, /name="moe-reco-goal" value="size" checked/);
  assert.deepEqual(JSON.parse(JSON.stringify(b.calls.filter((c) => c[0] === "moe_recommend").map((c) => c[1]))), [{ model: "inkling-2-mini", goal: "size", targetGb: null }]);
});
