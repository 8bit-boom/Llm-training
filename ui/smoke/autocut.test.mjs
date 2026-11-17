// Smoke test of the "Auto-cut (one click)" card on the Slim page, without a window (the same stand-in browser as thinkcheck.test.mjs). Run with `node --test ui/smoke`.
import test from "node:test";
import assert from "node:assert/strict";
import { makeBrowser, source } from "./_browser.mjs";


function browser(answers = {}) {
  return makeBrowser({ answers, state: baseState });
}

const baseState = () => ({ project: { name: "P", folder: "/p" }, steps: [], bundle: { world: "W" }, settings: { theme: "light", expert_mode: false, show_help_panel: true }, worlds: [], recent: [] });
const pageState = (over = {}) => ({ models: [], made: [], dataset_version: 3, env_ready: true, vram_gb: 32, gpu: "RTX 5090", free_disk_gb: 500, ...over });
const insp = (over = {}) => ({ supported: true, model_type: "qwen3_5", total_params: 27.78e9, ...over });
const stage = (i, of, b, extra = {}) => ({ index: i, of, params_b: b, blocks_cut: 0, width: 14080, cut_pct: 12, tokens_m: 40, hours: [8, 14], heal_gb: 26, verdict: "fits", ...extra });
const plan = (over = {}) => ({ ok: true, blockers: [], warnings: [], candidates: [], stages: [stage(1, 2, 24.46), stage(2, 2, 21.14, { width: 10752 })], disk: { peak_gb: 196.8, peak_step: "stage 2: the healed model", final_gb: 110.7, free_gb: 500 }, ...over });
const step = (id, label, status, note = null) => ({ id, label, status, note });
const run = (status, over = {}) => ({ id: "20261117-100000-qwen3-8-27b-21b", status, error: null, steps: [step("plan", "Check the model", "done"), step("measure", "Measure the model", "running"), step("s1.cut", "Stage 1 of 2: cut", "pending")], ...over });

function open(b, over = {}, ac = "") {
  b.run(`state = ${JSON.stringify(baseState())}; page = "reshape"; rs.state = ${JSON.stringify(pageState(over))}; rs.model = "qwen3.8-27b"; rs.options = { inspect: ${JSON.stringify(insp())}, options: [], width: null, width_options: null }; ${ac}`);
}
const card = (b) => b.run("autoCard()");
const plain = (x) => JSON.parse(JSON.stringify(x));
const tick = () => new Promise((r) => setTimeout(r, 10));

test("the card has a size slider for the model, keep thinking on for Qwen3.5/3.8, two stages, and the plan once it is there", () => {
  const b = browser();
  open(b, {}, `ac.plan = ${JSON.stringify(plan())}`);
  const html = card(b);
  assert.match(html, /<h2>Auto-cut \(one click\)<\/h2>/);
  assert.match(html, /id="ac-target" min="9" max="26" step="0.5" value="21"/);
  assert.match(html, /id="ac-think" checked/); assert.match(html, /name="ac-stages" value="2" checked/); assert.match(html, /name="ac-quick" value="full" checked/);
  assert.match(html, /<td>1 of 2<\/td><td>24\.5 B, MLP 14,080<\/td><td>12 % of the original<\/td><td>40 M tokens, about 8–14 h<\/td>/);
  assert.match(html, /Disk: up to 197 GB at "stage 2: the healed model", 111 GB left/);
  assert.doesNotMatch(html, /data-acstep/); assert.doesNotMatch(html, /id="ac-stop"/); assert.doesNotMatch(html, /id="ac-continue"/);
  assert.doesNotMatch(html, /id="ac-go"[^>]*disabled/);
  // another model family: thinking is not offered
  b.run(`rs.options = { inspect: ${JSON.stringify(insp({ model_type: "llama" }))}, options: [], width: null }; ac.thinking = null; ac.target = null`);
  assert.match(card(b), /id="ac-think"  disabled/);
  assert.equal(b.run("acSpec().keep_thinking"), false);
});

test("blockers disable Start and are shown; warnings are shown; the confirmation box appears only when blocks need it", () => {
  const b = browser();
  open(b, {}, `ac.plan = ${JSON.stringify(plan({ ok: false, blockers: ["The run needs up to 197 GB of disk; 120 GB are free."], warnings: ["Expect clearly weaker answers below 20 B."], candidates: [{ needs_confirmation: true }] }))}`);
  const html = card(b);
  assert.match(html, /id="ac-go"[^>]*disabled/); assert.match(html, /class="err">The run needs up to 197 GB/); assert.match(html, /class="warn">Expect clearly weaker/);
  assert.match(html, /id="ac-many"/);
  open(b, { env_ready: false }, `ac.plan = ${JSON.stringify(plan())}`);
  assert.match(card(b), /id="ac-go"[^>]*disabled/); assert.match(card(b), /training software is not installed/);
});

test("the spec follows the choices, the plan is asked with the model's description, and Start sends the spec", async () => {
  const b = browser({ auto_plan: { plan: plan() }, auto_start: null });
  open(b);
  b.run(`ac.target = 18; ac.stages = 1; ac.quick = true; ac.keep = true; ac.many = true; ac.publish = true; acRefresh()`);
  await tick();
  const c = b.calls.find((x) => x[0] === "auto_plan");
  assert.deepEqual(plain(c[1].spec), { model_id: "qwen3.8-27b", target_b: 18, keep_thinking: true, think_gate: true, think_gate_n: 2, stages: 1, quick: true, keep_intermediate: true, allow_many_blocks: true, quants: ["q4_k_m"], studio_publish: true });
  assert.equal(c[1].inspect.total_params, 27.78e9);
  assert.equal(b.run("ac.plan.ok"), true);
  const n = b.calls.filter((x) => x[0] === "auto_plan").length;
  b.run("acRefresh()"); await tick();
  assert.equal(b.calls.filter((x) => x[0] === "auto_plan").length, n, "the same request is not asked twice");
  b.run("wireAuto()"); b.els["ac-go"].onclick(); await tick();
  const s = b.calls.find((x) => x[0] === "auto_start");
  assert.equal(s[1].resume, null); assert.equal(s[1].spec.target_b, 18);
  assert.match(card(b), /id="ac-stop"/); assert.match(card(b), /Starting…/);
  assert.match(card(b), /id="ac-target"[^>]*disabled/);
});

test("the stepper shows every step with its state, the events move the text and the bar, and the end reloads the run", async () => {
  const r = run("running");
  let now = { run: r, running: true };
  const b = browser({ auto_status: () => now });
  open(b, {}, `ac.plan = ${JSON.stringify(plan())}; ac.run = ${JSON.stringify(r)}; ac.running = true`);
  let html = card(b);
  assert.match(html, /<li class="ac-done" data-acstep="plan"><span>✓<\/span> Check the model/);
  assert.match(html, /<li class="ac-running" data-acstep="measure"><span>▶<\/span>/); assert.match(html, /<li class="ac-pending" data-acstep="s1\.cut"><span>○<\/span>/);
  assert.match(html, /id="ac-status">Running/);
  b.run(`onReshapeEvent({ phase: "auto", kind: "progress", step: "measure", value: 0.5, text: "Measuring 64 of 128" })`);
  html = card(b); assert.match(html, /Measuring 64 of 128/); assert.match(html, /value="0.5"/);
  b.run(`onReshapeEvent({ phase: "auto", kind: "auto", auto_step: "measure", status: "done" })`);
  await tick();
  assert.ok(b.calls.some((x) => x[0] === "auto_status"));
  const done = run("done", { steps: [step("plan", "Check the model", "done"), step("export", "Make the GGUF files", "done")] });
  now = { run: done, running: false };
  b.run(`onReshapeEvent({ phase: "auto", kind: "done", data: ${JSON.stringify(done)} })`);
  await tick();
  assert.equal(b.run("ac.running"), false);
  assert.match(card(b), /Finished/); assert.match(card(b), /id="ac-report-btn"/); assert.doesNotMatch(card(b), /id="ac-stop"/);
});

test("stop asks the program, a stopped or failed run can be continued with its id, and an error is shown in plain words", async () => {
  const r = run("running");
  const b = browser();
  open(b, {}, `ac.plan = ${JSON.stringify(plan())}; ac.run = ${JSON.stringify(r)}; ac.running = true`);
  b.run("wireAuto()"); b.els["ac-stop"].onclick();
  assert.ok(b.calls.some((x) => x[0] === "auto_stop")); assert.match(card(b), /Stopping…/);
  const stopped = run("stopped");
  b.run(`ac.running = false; ac.run = ${JSON.stringify(stopped)}`);
  assert.match(card(b), /id="ac-continue"/); assert.match(card(b), /Stopped<\/strong>/);
  b.run("wireAuto()"); b.els["ac-continue"].onclick(); await tick();
  assert.deepEqual(plain(b.calls.find((x) => x[0] === "auto_start")[1]), { spec: null, resume: stopped.id });
  b.run(`onReshapeEvent({ phase: "auto", kind: "error", text: "The graphics card ran out of memory." })`);
  await tick();
  assert.match(card(b), /class="err">The graphics card ran out of memory\./);
});

test("a run that gave up says so, with the gates, and the report can be shown", async () => {
  const r = run("gates_failed", { error: "The thinking check of the healed model of stage 1 failed (closed_rate) after 3 tries; the run stops here.", steps: [step("s1.check_heal", "Stage 1 of 2: thinking check of the healed model", "failed", "closed_rate")] });
  const b = browser({ auto_report: { run: r.id, report: {}, markdown: "# Auto-cut report: Qwen3.8 27B to about 21.0 B\n\n**Stopped: the thinking checks failed.**" } });
  open(b, {}, `ac.plan = ${JSON.stringify(plan())}; ac.run = ${JSON.stringify(r)}`);
  let html = card(b);
  assert.match(html, /Stopped: the thinking checks failed<\/strong>: The thinking check of the healed model of stage 1 failed \(closed_rate\)/);
  assert.match(html, /<li class="ac-failed"/); assert.doesNotMatch(html, /id="ac-continue"/);
  b.run("wireAuto()"); b.els["ac-report-btn"].onclick(); await tick();
  assert.deepEqual(plain(b.calls.find((x) => x[0] === "auto_report")[1]), { run: r.id });
  assert.match(card(b), /<pre class="log"># Auto-cut report: Qwen3\.8 27B/);
});

test("the Studio share fields appear with the publish box, are saved and checked, and the answers are shown", async () => {
  const b = browser({ studio_share_set: baseState(), studio_share_check: { folder: true, note: "This is Studio's folder." } });
  open(b, {}, `ac.plan = ${JSON.stringify(plan())}`);
  assert.doesNotMatch(card(b), /id="ac-share"/);
  b.run("ac.publish = true");
  const html = card(b);
  assert.match(html, /id="ac-share"/); assert.match(html, /id="ac-sh-reg" checked/); assert.doesNotMatch(html, /id="ac-sh-hf" checked/, "the fine-tune handoff is off by default");
  b.run("wireAuto(); ['ac-sh-host', 'ac-sh-cont', 'ac-sh-reg', 'ac-sh-hf'].forEach($)");
  b.els["ac-sh-host"].value = "Y:\\studio"; b.els["ac-sh-cont"].value = "/workspace"; b.els["ac-sh-reg"].checked = true; b.els["ac-sh-hf"].checked = false;
  b.els["ac-sh-check"].onclick(); await tick();
  assert.deepEqual(plain(b.calls.find((x) => x[0] === "studio_share_check")[1].share), { host_path: "Y:\\studio", container_path: "/workspace", register_scan_folder: true, finetune_handoff: false });
  assert.match(card(b), /This is Studio&#39;s folder\./);
  b.run("wireAuto()"); b.els["ac-sh-save"].onclick(); await tick();
  assert.equal(b.calls.find((x) => x[0] === "studio_share_set")[1].share.host_path, "Y:\\studio");
  assert.match(card(b), /id="ac-sh-note">Saved\./);
});

test("the thinking gate has a checkbox that is on with keep thinking, shows its minutes and goes into the spec", async () => {
  const b = browser({ auto_plan: { plan: { ...plan(), think_gate_minutes: 61.4 } }, auto_start: null });
  open(b);
  b.run("acRefresh()"); await tick();
  const html = card(b);
  assert.match(html, /id="ac-gate" checked/);
  assert.match(html, /Check the thinking of the best 2 cuts before healing \(about 61 min more\)/);
  assert.equal(b.run("acSpec().think_gate"), true);
  b.run("ac.gate = false");
  assert.equal(b.run("acSpec().think_gate"), false);
  assert.doesNotMatch(card(b), /id="ac-gate" checked/);
  b.run("ac.gate = true; ac.thinking = false");
  assert.equal(b.run("acSpec().think_gate"), false, "without keep thinking there is no gate");
  assert.doesNotMatch(card(b), /id="ac-gate"/);
});
