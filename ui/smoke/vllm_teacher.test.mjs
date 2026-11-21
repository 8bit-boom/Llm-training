// The original's answers through vLLM (experimental): the Settings block, the Heal card option with its numbers, the check button and its verdict. Run with `node --test ui/smoke`.
import test from "node:test";
import assert from "node:assert/strict";
import { makeBrowser } from "./_browser.mjs";

const baseState = (vllm) => ({ project: { name: "P", folder: "/p" }, steps: [], bundle: { world: "W" }, settings: { theme: "light", expert_mode: false, show_help_panel: true, ...(vllm ? { vllm } : {}) }, worlds: [], recent: [] });
const model = () => ({ name: "slim-w", status: "done", model_id: "custom-w", size_gb: 43, tested: false, test: null, healed: false, healed_model_id: null, heal: null, test_healed: null, cache_gb: 0, healed_size_gb: 0,
  summary: { mode: "prune", shape: "width", layers_before: 64, layers_after: 64, dropped: [], duplicated: [], params_before: 27.8e9, params_after: 21.1e9, loss_before: 2.0, loss_after: 3.1, stable: null, chosen_blocks: [], boundary_distance: null, notes: [] } });
const opts = () => ({ name: "slim-w", plan: { mode: "cache", student_method: "qlora", est_gb: 18.7, fits: true, why: null }, defaults: { seq: 8192, batch: 1, grad_accum: 32, checkpointing: true, rank: 32, kl_chunk: 128, qlora: true }, safer: null,
  tokens: [10, 20, 40, 80].map((t) => ({ tokens_m: t, cache_gb: t / 5, hours: [t / 3, t / 2] })), recipe: null });
const vt = (over = {}) => ({ asked: true, settings_on: true, route: { route: "bnb", quant: "bnb-nf4", quantized_repo: "", reason: "No confirmed quantized copy of Someone/Finetune: the original's own folder is loaded and quantized to 4 bits while loading." },
  identity: { engine: "vllm-0.34.0", teacher_quant: "bnb-nf4", route: "bnb" }, tokens_m: 40, standard_min: [480, 1200], vllm_min: [260, 420], startup_min: [7.2, 10.6], startup_measured: true, scoring_measured: false, engine_starts: 1, faster: true,
  text: "The original's answers: about 8.0 to 20.0 hours with the standard engine, about 4.3 to 7.0 hours with vLLM (start-up 7.2 to 10.6 minutes measured by your self-test, scoring the research report's estimate, not measured here).",
  warnings: ["The vLLM teacher has not been checked against the standard teacher yet: use \"Check the vLLM teacher\" first (about 100 rows)."], basis: ["The engine's start-up is 8.5 minutes: measured by your own vLLM self-test."], check: null, required: false, ...over });
const est = (v) => ({ plan: { mode: "cache", student_method: "qlora", est_gb: 18.7, fits: true }, knobs: { seq: 8192, batch: 1, grad_accum: 32, checkpointing: true, rank: 32, kl_chunk: 128, qlora: true }, vllm_teacher: v });
const on = { enabled: true, kind: "auto", teacher: { enabled: true, route: "auto", pairs: { "Qwen/Qwen3.8-27B": "cyankiwi/Qwen3.8-27B-AWQ-INT4" }, confirmed: {}, required: false } };

function open(vllm, v) {
  const b = makeBrowser({ answers: {}, state: () => baseState(vllm) });
  const m = model();
  b.run(`state = ${JSON.stringify(baseState(vllm))}; page = "reshape"; rs.state = ${JSON.stringify({ models: [{ id: "qwen3.5-27b", name: "Q", params_b: 27.8, custom: false }], made: [m], dataset_version: 3, env_ready: true, vram_gb: 32, free_disk_gb: 500 })}; rs.model = "qwen3.5-27b"; rs.view = ${JSON.stringify(m.name)}; rs.options = null;`);
  b.run(`rs.hl = { name: ${JSON.stringify(m.name)}, opts: ${JSON.stringify(opts())} }; rs.hlEst = { name: ${JSON.stringify(m.name)}, key: "", est: ${JSON.stringify(est(v || vt()))} };`);
  return b;
}

test("the Settings block is off by default and explains the identity rule", () => {
  const b = makeBrowser({ answers: {}, state: () => baseState() });
  b.run(`state = ${JSON.stringify(baseState())}`);
  const html = b.run("renderVllm()");
  assert.match(html, /The original&#39;s answers through vLLM \(experimental\)|The original's answers through vLLM \(experimental\)/);
  assert.doesNotMatch(html, /id="vl-t-on"[^>]*checked/);
  assert.match(html, /must be your ORIGINAL model/);
  assert.match(html, /Qwen\/Qwen3\.8-27B = cyankiwi\/Qwen3\.8-27B-AWQ-INT4/);
  assert.match(html, /Only a confirmed pre-quantized copy \(refuses otherwise\)/);
});

test("the Heal card has no vLLM option while Settings leave it off, and the request is as it always was", () => {
  const b = open(null);
  const html = b.run("renderReshape()");
  assert.doesNotMatch(html, /id="hl-vllm"/);
  b.run("rs.hlSpec.vllm = true");
  assert.equal("vllm" in JSON.parse(JSON.stringify(b.run("rsHealSpec()"))), false, "switched off in Settings: never sent");
});

test("with Settings on the option is offered, unticked, and ticking it sends the key", () => {
  const b = open(on);
  let html = b.run("renderReshape()");
  assert.match(html, /id="hl-vllm"/);
  assert.doesNotMatch(html, /id="hl-vllm"[^>]*checked/);
  assert.doesNotMatch(html, /id="hl-vllm-box"/);
  assert.equal("vllm" in JSON.parse(JSON.stringify(b.run("rsHealSpec()"))), false);
  b.run("rs.hlSpec.vllm = true");
  html = b.run("renderReshape()");
  assert.match(html, /id="hl-vllm-box"/);
  assert.deepEqual(JSON.parse(JSON.stringify(b.run("rsHealSpec()"))).vllm, { teacher_cache: true });
});

test("the box shows the route, the time from your numbers with its source, the warning and the check button", () => {
  const b = open(on);
  b.run("rs.hlSpec.vllm = true");
  const html = b.run("renderReshape()");
  assert.match(html, /id="hl-vllm-route"[^>]*>No confirmed quantized copy of Someone\/Finetune/);
  assert.match(html, /about 8\.0 to 20\.0 hours with the standard engine, about 4\.3 to 7\.0 hours with vLLM/);
  assert.match(html, /measured by your self-test, scoring the research report&#39;s estimate, not measured here|not measured here/);
  assert.match(html, /has not been checked against the standard teacher yet/);
  assert.match(html, /id="hl-vllm-check"/);
  assert.match(html, /Where these numbers come from/);
  assert.match(html, /8\.5 minutes: measured by your own vLLM self-test/);
});

test("a heal in rounds shows the engine-restart warning and a saved verdict is shown", () => {
  const b = open(on, vt({ engine_starts: 8, warnings: ["A heal in rounds asks the original for one chunk of answers per round, and every chunk starts the engine again: about 7 to 11 minutes of start-up per round, 8 rounds at most."],
    check: { verdict: "different", current: true, text: "The vLLM teacher differs too much from the standard teacher: it is NOT used for healing.", vllm: { startup_seconds: 510.6, tokens_per_s: 2900 } } }));
  b.run("rs.hlSpec.vllm = true");
  const html = b.run("renderReshape()");
  assert.match(html, /starts the engine again/);
  assert.match(html, /id="hl-vllm-verdict" class="err"><strong>Check: different<\/strong>/);
  assert.match(html, /Start-up 511 s, scoring 2900 tokens\/s/);
});

test("the check button starts the worker job and a running check shows its progress", async () => {
  const calls = [];
  const b = makeBrowser({ answers: { vllm_teacher_check_start: (a) => { calls.push(a); return null; } }, state: () => baseState(on) });
  b.run(`state = ${JSON.stringify(baseState(on))}; page = "reshape"; rs.state = ${JSON.stringify({ models: [], made: [model()], dataset_version: 3, env_ready: true, vram_gb: 32, free_disk_gb: 500 })}; rs.view = "slim-w"; rs.options = null;
    rs.hl = { name: "slim-w", opts: ${JSON.stringify(opts())} }; rs.hlEst = { name: "slim-w", key: "", est: ${JSON.stringify(est(vt()))} }; rs.hlSpec.vllm = true;`);
  b.run("vch.run = { text: 'Scoring', progress: 0.4 }");
  const html = b.run("renderReshape()");
  assert.match(html, /id="hl-vllm-checkstop"/);
  assert.match(html, /<progress max="1" value="0.4">/);
  assert.match(html, /id="hl-vllm-check" disabled/);
});

test("the self-test card shows the report inline (also after a failure), the folder as text and the copy / open buttons", () => {
  const b = makeBrowser({ answers: {}, state: () => baseState(on) });
  b.run(`state = ${JSON.stringify(baseState(on))}`);
  b.run(`vl.report = { verdict: "fail", verdict_text: "The engine did not become ready.", criteria: [], answer_tokens: { median: 80 }, long_tokens_per_s: 210.5 };
         vl.files = { exists: true, dir: "C:\\\\Users\\\\<user>\\\\AppData\\\\Roaming\\\\NDLoreForge\\\\vllm_selftest", markdown: "# vLLM self-test\\n**FAIL**: not ready\\nCUDA out of memory" }`);
  const html = b.run("renderVllm()");
  assert.match(html, /The report is saved in:/);
  assert.match(html, /<code class="vl-path" data-path>C:\\Users\\&lt;user&gt;\\AppData\\Roaming\\NDLoreForge\\vllm_selftest<\/code>/);
  assert.match(html, /data-vlcopy="C:/); assert.match(html, /Open the folder/); assert.match(html, /Copy the whole report/);
  assert.match(html, /<details open><summary class="small">The report \(text\)<\/summary><pre class="vl-report small" id="vl-files-text"[^>]*>[^<]*CUDA out of memory/);
  assert.match(html, /Answers: median 80 tokens\. Long answers: 210\.5 tokens\/s\./);
  assert.match(html, /Empty = the default, the configured pre-quantized original \(about 21 GB, 4-bit\)/);
});

test("the teacher check card shows its file path and text", () => {
  const files = { exists: true, dir: "/p/models/slim-w", file: "/p/models/slim-w/vllm_teacher_check.json", text: "{\n \"verdict\": \"close\"\n}" };
  const b = open(on, vt({ files, check: { verdict: "close", current: true, text: "Close." } }));
  b.run("rs.hlSpec.vllm = true");
  const html = b.run("renderReshape()");
  assert.match(html, /The check is saved in:/);
  assert.match(html, /\/p\/models\/slim-w/);
  assert.match(html, /id="hl-vllm-files-text"[^>]*>[^<]*&quot;verdict&quot;: &quot;close&quot;/);
  assert.match(html, /Open the folder/);
});

// ---- Auto-cut

const acBase = () => ({ project: { name: "P", folder: "/p" }, steps: [], bundle: { world: "W" }, settings: { theme: "light", expert_mode: false, show_help_panel: true, vllm: { enabled: true, kind: "auto" } }, worlds: [], recent: [] });
const acStage = (i, b) => ({ index: i, of: 2, params_b: b, blocks_cut: 0, width: 14080, cut_pct: 12, tokens_m: 40, hours: [8, 14], heal_gb: 26, verdict: "fits" });
const acPlan = (extra = {}) => ({ ok: true, blockers: [], warnings: [], candidates: [], stages: [acStage(1, 24.46), acStage(2, 21.14)], disk: { peak_gb: 196.8, peak_step: "x", final_gb: 86, free_gb: 500 }, time: { steps: [], total_h: [30, 40], notes: [], basis: "", extra: [], ...extra },
  speed: { presets: [], knobs: [{ knob: "vllm", label: "vLLM", risk: "r", options: [{ label: "Off (recipe)", current: true, delta_min: [0, 0], set: { vllm: null } }, { label: "The original's answers with vLLM", current: false, delta_min: [-1500, -400], set: {} },
    { label: "The thinking checks with vLLM", current: false, delta_min: [20, 60], set: {} }, { label: "Both", current: false, delta_min: [-1400, -300], set: {} }, { label: "Automatic: vLLM only where it is faster", current: false, delta_min: [-1500, -400], set: {} }] }], current_preset: "balanced", default_preset: "balanced", total_h: [30, 40] } });

function acOpen(settings, planExtra) {
  const b = makeBrowser({ answers: {}, state: () => acBase() });
  const st = acBase();
  st.settings.vllm = settings;
  b.run(`state = ${JSON.stringify(st)}; page = "reshape"; rs.state = ${JSON.stringify({ models: [], made: [], dataset_version: 3, env_ready: true, vram_gb: 32, gpu: "RTX 5090", free_disk_gb: 500 })}; rs.model = "qwen3.8-27b";
    rs.options = { inspect: { supported: true, model_type: "qwen3_5", total_params: 27.78e9 }, options: [], width: null }; ac.plan = ${JSON.stringify(acPlan(planExtra))};`);
  return b;
}

test("the Auto-cut card has no vLLM section while Settings keep the engine off, and the spec never carries the key", () => {
  const b = acOpen({ enabled: false });
  assert.doesNotMatch(b.run("autoCard()"), /id="ac-vllm"/);
  b.run("ac.sp.vl = { teacher_cache: true, checks: false, automatic: false }");
  assert.equal("vllm" in JSON.parse(JSON.stringify(b.run("acSpec()"))), false);
});

test("the vLLM section shows the two switches and automatic with the time change, the prerequisites, and the plan's reasons", () => {
  const b = acOpen({ enabled: true, kind: "auto" }, { vllm: { numbers: {}, decisions: [{ step: "s1.heal:cache", kind: "teacher_cache", mode: "automatic", use_vllm: true, reason: "Automatic: vLLM for the original's answers, about 3.0 to 3.8 hours instead of 8.2 to 20.6 hours (from the research report's estimate, not measured on this PC)." },
    { step: "checks", kind: "checks", mode: "automatic", use_vllm: false, reason: "Automatic: the standard engine for this thinking check: vLLM would take about 9 to 14 minutes against 10 to 14 minutes; the engine's start-up eats the gain." }], notes: ["Scoring speed 3000 to 3800 tokens per second: the research report's estimate, NOT measured on this PC"] } });
  b.run(`vl.report = { verdict: "partial", model_info: { model: "org/Model-9B", params_b: 9, quant: "quantized (awq)" }, long_tokens_per_s: 210 }`);
  const html = b.run("autoCard()");
  assert.match(html, /<legend>vLLM \(experimental\)<\/legend>/);
  for (const id of ["ac-vl-t", "ac-vl-c", "ac-vl-a"]) assert.match(html, new RegExp(`id="${id}"`));
  assert.match(html, /The original&#39;s answers with vLLM <span class="dim small ac-delta">|The original's answers with vLLM <span class="dim small ac-delta">/);
  assert.match(html, /measured on org\/Model-9B \(9 B\), quantized \(awq\)/);
  assert.match(html, /Speed on long answers measured: 210 tokens per second/);
  assert.match(html, /Automatic: the standard engine for this thinking check/);
  assert.match(html, /it does not pay for a short thinking check/);
});

test("ticking a switch puts the key in the spec (and makes the setting Custom); a preset switches it off again", () => {
  const b = acOpen({ enabled: true, kind: "auto" });
  b.run("ac.sp.vl = { teacher_cache: true, checks: false, automatic: false }");
  assert.deepEqual(JSON.parse(JSON.stringify(b.run("acSpec().vllm"))), { teacher_cache: true, checks: false, automatic: false });
  assert.equal(b.run("acPresetNow()"), "custom");
  b.run(`acApplyPreset("balanced")`);
  assert.equal(b.run("ac.sp.vl"), null);
  assert.equal("vllm" in JSON.parse(JSON.stringify(b.run("acSpec()"))), false);
});

// ---- the optional 8-bit KV cache

test("Settings: the 8-bit KV cache is off by default, explained, and the teacher has its own confirmation", () => {
  const b = makeBrowser({ answers: {}, state: () => baseState() });
  b.run(`state = ${JSON.stringify(baseState())}`);
  const html = b.run("renderVllm()");
  assert.match(html, /8-bit KV cache \(experimental, off\)/);
  assert.match(html, /71,141 cached tokens with auto, 85,196 with fp8 \(only about 1\.2 times, not 2 times\)/); assert.match(html, /only its full-attention layers have a KV cache/);
  assert.match(html, /<option value="auto" selected>auto \(the default\)/); assert.doesNotMatch(html, /id="vl-kv-teacher"[^>]*checked/);
  assert.match(html, /<option value="f16" selected>f16 \(the default\)/);
  assert.match(html, /id="vl-tkv"/);
});

test("Auto-cut: the 8-bit KV cache switch is for the checks and goes into the spec only when ticked", () => {
  const b = acOpen({ enabled: true, kind: "auto" });
  assert.match(b.run("autoCard()"), /id="ac-vl-kv"/);
  b.run("ac.sp.vl = { teacher_cache: false, checks: true, automatic: false, kv_fp8: true }");
  assert.deepEqual(JSON.parse(JSON.stringify(b.run("acSpec().vllm"))), { teacher_cache: false, checks: true, automatic: false, kv_fp8: true });
  b.run("ac.sp.vl = { teacher_cache: false, checks: true, automatic: false }");
  assert.equal("kv_fp8" in JSON.parse(JSON.stringify(b.run("acSpec().vllm"))), false);
});

test("the Heal card can measure the cost of the 8-bit KV cache on the teacher's answers", () => {
  const b = open(on);
  b.run("rs.hlSpec.vllm = true");
  assert.match(b.run("renderReshape()"), /id="hl-vllm-checkkv"[^>]*>Check with the 8-bit KV cache/);
});

// ---- Measure heal speed

test("the Heal card has a Measure heal speed box: off until pressed, with the report path, buttons and the Windows hint", () => {
  const b = open(null);
  b.run(`hsp.name = "slim-w"; hsp.files = { exists: true, dir: "C:\\\\p\\\\benchmarks\\\\heal_speed-slim-w", markdown: "# Heal speed report\\nGatedDeltaNet kernels take about 31 % of a step", windows_hints: ["Hardware-accelerated GPU scheduling: off. Windows Settings ..."] }`);
  const html = b.run("renderReshape()");
  assert.match(html, /id="hl-speed"[^>]*open/); assert.match(html, /<strong>Measure heal speed<\/strong> \(where does the time go/);
  assert.match(html, /hypotheses until this measurement confirms them/);
  assert.match(html, /id="hl-speed-go"/); assert.match(html, /id="hl-speed-hint">Hardware-accelerated GPU scheduling: off/);
  assert.match(html, /heal_speed-slim-w/); assert.match(html, /Open the folder/); assert.match(html, /Copy path/);
  assert.doesNotMatch(html, /id="hl-speed-stop"/);
  b.run(`hsp.run = { text: "Profiling" }`);
  assert.match(b.run("renderReshape()"), /id="hl-speed-stop"/);
});

test("Test the container copy: the button starts the probe with the real keys, shows which way worked and which failed", async () => {
  const b = open(on);
  b.run("rs.hlSpec.vllm = true");
  let html = b.run("renderReshape()");
  assert.match(html, /id="hl-vllm-probe"/);
  b.run(`vch.probing = true; vch.probeText = "Starting…"`);
  b.run(`onVprobeEvent(${JSON.stringify({ phase: "vllm_probe", kind: "progress", text: "Testing the vLLM container" })})`);
  assert.match(b.run("renderReshape()"), /Testing the vLLM container/);
  b.run(`onVprobeEvent(${JSON.stringify({ phase: "vllm_probe", kind: "done", data: { ok: true, text: "Files go in by cp:dir and out by cp:file.", tried: [{ route: "bind /p/w", ok: false, detail: "refused" }, { route: "cp:dir (in)", ok: true, detail: "1 KB went in" }] } })})`);
  html = b.run("renderReshape()");
  assert.match(html, /id="hl-vllm-probe-result" class="ok"/);
  assert.match(html, /failed: bind \/p\/w — refused/);
  assert.match(html, /worked: cp:dir \(in\)/);
  b.run(`onVprobeEvent(${JSON.stringify({ phase: "vllm_probe", kind: "error", text: "No way of moving files into the container works. Tried: ..." })})`);
  assert.match(b.run("renderReshape()"), /class="err"><strong>The container copy does not work/);
});

test("the KV warning for the teacher is part of the numbers the Heal card shows", () => {
  const b = open(on, vt({ warnings: ["8-bit KV cache for the teacher: measured 1.2x more cached tokens and no speed gain on this PC"] }));
  b.run("rs.hlSpec.vllm = true");
  assert.match(b.run("renderReshape()"), /8-bit KV cache for the teacher: measured 1\.2x more cached tokens and no speed gain/);
});

test("the Settings card has a button that tests the in-flight 4-bit route alone, with the real keys", async () => {
  const b = makeBrowser({ answers: {}, state: () => baseState({ enabled: true, kind: "auto", teacher: { enabled: true, route: "auto", pairs: {}, confirmed: {}, required: false } }) });
  b.run(`state = ${JSON.stringify(baseState({ enabled: true, kind: "auto", teacher: { enabled: true, route: "auto", pairs: {}, confirmed: {}, required: false } }))}`);
  const html = b.run("renderVllm()");
  assert.match(html, /id="vl-inflight"/);
  assert.match(html, /Test the in-flight 4-bit route on my original/);
});

test("the Heal card has a button for the auto against 8-bit KV cache comparison", () => {
  const b = open(on);
  b.run("rs.hlSpec.vllm = true");
  assert.match(b.run("renderReshape()"), /id="hl-vllm-checkcmp"[^>]*>Check auto and 8-bit cache/);
});
