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
  assert.match(html, /id="ac-think" checked/); assert.match(html, /name="ac-stages" value="2" checked/);
  // a model of 27.8 B has Balanced chosen for it (a heal of a big model takes days); Try first is Quick in it
  assert.match(html, /name="ac-preset" value="balanced" checked/); assert.match(html, /name="ac-quick" value="quick" checked/);
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
  b.run(`acDefaults(); ac.target = 18; ac.stages = 1; ac.keep = true; ac.many = true; ac.publish = true; acRefresh()`);
  await tick();
  const c = b.calls.find((x) => x[0] === "auto_plan");
  // Balanced was chosen for the big model; one stage instead of two makes it Custom. The speed fields are all sent.
  assert.deepEqual(plain(c[1].spec), { model_id: "qwen3.8-27b", notify_stages: true, target_b: 18, keep_thinking: true, think_gate: true, think_gate_n: 2, stages: 1, quick: true, keep_intermediate: true, allow_many_blocks: true, quants: ["q4_k_m"], studio_publish: true,
    preset: "custom", stage_a_tokens_m: 20, last_tokens_m: null, think_quick: true, check_cut: true, retry_heal: true, back_off: true, share_teacher_cache: true, imatrix: true });
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

test("an optional target size makes a measured file and compares it: the field, the words, the spec and what the plan adds", async () => {
  const b = browser({ auto_plan: { plan: plan({ measured_minutes: [8, 13] }) }, auto_start: null });
  open(b, {}, `ac.plan = ${JSON.stringify(plan({ measured_minutes: [8, 13] }))}`);
  let html = card(b);
  assert.match(html, /Also make a file of about <input type="text" id="ac-gb" value="" placeholder="10.6"[^>]*> GB \(measured mixed precision\) and compare it with the plain sizes/);
  assert.doesNotMatch(html, /id="ac-gbnote"/); assert.doesNotMatch(html, /id="ac-gbon" checked/);
  assert.equal(b.run("acSpec().target_gb"), undefined, "no target without the tick");
  b.run(`ac.gbOn = true`);
  html = card(b);
  assert.match(html, /id="ac-gbon" checked/); assert.match(html, /id="ac-gbwhy">Give the size in GB\./);
  assert.match(html, /Compare quants<\/strong> \(KL divergence\) checks the file against the plain sizes[^]*published only when it is clearly better/);
  assert.match(html, /About 8 min to measure and 13 min to compare \(guesses\)\. The 16-bit file is kept until the comparison is done/);
  assert.doesNotMatch(html, /Dynamic|UD-/i);
  b.run("wireAuto()"); b.els["ac-gb"].value = " 10.6 "; b.els["ac-gb"].onchange();
  assert.equal(b.run("acSpec().target_gb"), 10.6); assert.doesNotMatch(card(b), /id="ac-gbwhy"/);
  b.run("acRefresh()"); await tick();
  assert.equal(plain(b.calls.filter((x) => x[0] === "auto_plan").at(-1)[1].spec).target_gb, 10.6);
  b.run("wireAuto()"); b.els["ac-go"].onclick(); await tick();
  assert.equal(b.calls.find((x) => x[0] === "auto_start")[1].spec.target_gb, 10.6);
  b.els["ac-gbon"].checked = false; b.run("ac.running = false"); b.run("wireAuto()"); b.els["ac-gbon"].onchange();
  assert.equal(b.run("acSpec().target_gb"), undefined);
});

// ---- how long it will take (plan.time: docs/AUTO_CUT.md, "How long it takes")
const tstep = (id, label, lo, hi, basis = "formula", extra = {}) => ({ id, label, minutes: [lo, hi], basis, note: null, parts: [], ...extra });
const timed = (over = {}) => ({
  steps: [tstep("plan", "Check the model, the card and the disk", 0.2, 0.5), tstep("measure", "Measure the model", 11, 32), tstep("s1.heal", "Stage 1 of 2: heal", 1400, 2700, "your_history", { parts: [{ id: "cache", label: "the original writes its answers (40 M tokens)", minutes: [600, 1100], basis: "your_history" }, { id: "train", label: "the student learns (40 M tokens)", minutes: [500, 900], basis: "formula" }, { id: "merge", label: "add the learning into the model", minutes: [10, 40], basis: "formula" }] }),
    tstep("export", "Make the GGUF files", 7.5, 18.5)],
  total_h: [9.5, 14.25], history_steps: 1, step_count: 4, basis: "1 of 4 steps use your own earlier runs (from the saved job logs); the other 3 are formulas built on one real run on a 32 GB card.",
  lever_h: [18.3, 33.3], notes: ["An estimate, not a promise: the teacher's answers and the training of the heal are most of it.", "Not in the total: waiting in the queue."],
  extra: [{ id: "retry_heal", label: "If the last stage fails its thinking check: heal it again with 40 M more tokens", minutes: [60, 120], note: "" }], ...over });

test("before Start the card says how long everything takes: the total as a range, the clock time, a table of the steps with where each number comes from, and the basis", () => {
  const b = browser();
  open(b, {}, `ac.plan = ${JSON.stringify(plan({ time: timed({ total_h: [9.5, 14.25], lever_h: [10, 12] }) }))}`);
  const now = new Date(2026, 0, 1, 12, 0, 0).getTime();
  const html = b.run(`acTimeBlock(ac.plan, ${now})`);
  assert.match(html, /<h3>How long this will take<\/h3>/);
  assert.match(html, /id="ac-time-total">All of it together takes about <strong>9 h 30 min to 14 h 30 min<\/strong>/, "the existing rounding: 5 minutes below 10 hours, 30 from 10 hours up");
  assert.match(html, /id="ac-time-done">If you press Start now it should be done around 21:30–tomorrow 02:15\./);
  assert.match(html, /<details id="ac-time-steps"><summary>Every step<\/summary>/);
  assert.match(html, /<tr data-actime="measure"><td>Measure the model<\/td><td>11–32 min<\/td><td><span class="tag ac-formula">formula<\/span><\/td><\/tr>/);
  assert.match(html, /<tr data-actime="plan"><td>[^<]*<\/td><td>under 1 min<\/td>/);
  assert.match(html, /<tr data-actime="s1\.heal">[^]*the original writes its answers \(40 M tokens\): 10 h to 18 h 30 min[^]*<td>23 h 30 min to 45 h<\/td><td><span class="tag ac-hist">from your history<\/span>/);
  assert.match(html, /id="ac-time-basis">1 of 4 steps use your own earlier runs/);
  assert.match(html, /id="ac-time-caveat">An estimate, not a promise/);
  assert.match(html, /id="ac-time-lever">Most of the time is the teacher's answers and the heal: 10 h to 12 h\. Fewer tokens or one stage shortens it\./);
  assert.match(html, /data-actime-extra="retry_heal">Possible extra, not counted above: If the last stage fails[^<]*: 1 h to 2 h\./);
  // it is in the card before the Start button; without a plan or without the estimate there is no block
  assert.ok(card(b).indexOf('id="ac-time"') > 0 && card(b).indexOf('id="ac-time"') < card(b).indexOf('id="ac-go"'));
  assert.equal(b.run("acTimeBlock(null)"), ""); assert.equal(b.run("acTimeBlock({ ok: true })"), "");
  // a single value, no lever hint when the heal is not most of it
  const one = b.run(`acTimeBlock(${JSON.stringify(plan({ time: timed({ total_h: [2, 2], lever_h: [0.5, 0.5], extra: [] }) }))}, ${now})`);
  assert.match(one, /about <strong>2 h<\/strong>/); assert.match(one, /done around 14:00\./); assert.doesNotMatch(one, /ac-time-lever/); assert.doesNotMatch(one, /Possible extra/);
});

test("the estimate follows the choices: a changed size, stage count or Quick asks again and shows the new total; during a run the block is not shown", async () => {
  const totals = [[9.5, 14.25], [40, 60]];
  let n = 0;
  const b = browser({ auto_plan: () => ({ plan: plan({ time: timed({ total_h: totals[n++ % 2] }) }) }), auto_start: null });
  open(b);
  b.run('page = "reshape"; acRefresh()'); await tick();
  assert.match(card(b), /id="ac-time-total">[^]*9 h 30 min to 14 h 30 min/);
  b.run('page = "reshape"; ac.stages = 1; acRefresh()'); await tick();
  assert.equal(b.calls.filter((x) => x[0] === "auto_plan").length, 2, "the plan, and so the estimate, is asked again when the spec changes");
  assert.match(card(b), /id="ac-time-total">[^]*40 h to 60 h/);
  b.run(`ac.run = ${JSON.stringify(run("running"))}; ac.running = true`);
  assert.doesNotMatch(card(b), /id="ac-time"/);
});

test("during a run the time left is the sum of the steps not done yet (a running step has its elapsed time taken off); an old run without the estimate keeps the old rule", () => {
  const b = browser();
  const now = new Date(2026, 0, 1, 12, 0, 0).getTime();
  const st = (id, status, started = null) => ({ id, status, started });
  const t = timed();
  const left = (steps, eta) => JSON.parse(JSON.stringify(b.run(`acEtaRange(${JSON.stringify({ status: "running", steps, plan: { stages: [], time: t } })}, ${JSON.stringify(eta || null)}, ${now})`)));
  assert.deepEqual(left([st("plan", "done"), st("measure", "done"), st("s1.heal", "pending"), st("export", "pending")]), { lo: (1400 + 7.5) / 60, hi: (2700 + 18.5) / 60 });
  // the running measure started 10 minutes ago (1 minute of its 11 are left; never less than a minute)
  const started = new Date(now - 10 * 60e3).toISOString();
  const r = left([st("measure", "running", started), st("export", "pending")]);
  assert.ok(Math.abs(r.lo - (0.02 + 7.5 / 60)) < 0.001 && Math.abs(r.hi - (22 / 60 + 18.5 / 60)) < 0.001, JSON.stringify(r));
  // the running heal reports its own time left (1 h of training): that and its merge, then the export
  const h = left([st("s1.heal", "running", started), st("export", "pending")], { s: 3600, step: "s1.heal", at: now });
  assert.ok(Math.abs(h.lo - (1 + 10 / 60 + 7.5 / 60)) < 0.001 && Math.abs(h.hi - (1 + 40 / 60 + 18.5 / 60)) < 0.001, JSON.stringify(h));
  // the text
  assert.equal(b.run(`acEtaText(${JSON.stringify({ status: "running", steps: [st("export", "pending")], plan: { time: t } })}, null, ${now})`), "about 10 min to 20 min left (done around 12:07–12:18)");
  // a step the estimate does not know falls back to the fixed allowance
  assert.deepEqual(JSON.parse(JSON.stringify(b.run(`acEtaRange({ status: "running", steps: [{ id: "check_gguf", status: "pending" }], plan: { stages: [], time: ${JSON.stringify(t)} } }, null, ${now})`))), { lo: 0.25, hi: 0.25 });
});

test("the speed presets of the window are the table of auto_speed.json", async () => {
  const { readFileSync } = await import("node:fs");
  const file = JSON.parse(readFileSync(new URL("../../crates/ndforge-core/src/reshape/auto_speed.json", import.meta.url), "utf8"));
  const b = browser();
  assert.deepEqual(plain(b.run("AC_SPEED_PRESETS")), file.presets);
  assert.deepEqual(file.presets.map((p) => p.id), ["thorough", "balanced", "fast", "custom"]);
});


const speedPlan = (over = {}) => plan({
  speed: { current_preset: "balanced", default_preset: "balanced", total_h: [33, 61],
    presets: [["thorough", [57, 108], [23, 37]], ["balanced", [33, 61], [23, 37]], ["fast", [11, 22], null], ["custom", [33, 61], null]].map(([id, t, r]) => ({ id, label: id, total_h: t, retry_h: r })),
    knobs: [
      { knob: "last_tokens_m", label: "Heal tokens of the last stage", risk: "Fewer tokens is faster.", options: [10, 20, 40, 80].map((t) => ({ label: `${t} M${t === 80 ? " (recipe)" : ""}`, set: { last_tokens_m: t === 80 ? null : t }, current: t === 80, total_h: [10 + t / 2, 20 + t], delta_min: t === 80 ? [0, 0] : [-(80 - t) * 8, -(80 - t) * 12] })) },
      { knob: "share_teacher_cache", label: "Sequence length of the stages", risk: "Shorter sequence.", options: [{ label: "Same in both stages", set: { share_teacher_cache: true }, current: true, total_h: [33, 61], delta_min: [0, 0] }, { label: "Each on its own", set: { share_teacher_cache: false }, current: false, total_h: [40, 70], delta_min: [420, 540] }] },
      { knob: "stages", label: "Stages", risk: "x", options: [{ label: "Two stages (recipe)", set: { stages: 2 }, current: true, total_h: [33, 61], delta_min: [0, 0] }, { label: "One stage", set: { stages: 1 }, current: false, total_h: [20, 40], delta_min: [-780, -1260] }] },
      { knob: "imatrix", label: "Importance matrix for the export", risk: "Less exact small sizes.", options: [{ label: "On (recipe)", set: { imatrix: true }, current: true, total_h: [33, 61], delta_min: [0, 0] }, { label: "Off", set: { imatrix: false }, current: false, total_h: [32.8, 60.7], delta_min: [-10, -20] }] }] },
  ...over });

test("the presets show their totals, a preset sets the spec, and any change of a knob turns the choice to Custom", async () => {
  const fake = (value, dataset = {}) => ({ value, dataset, checked: false, onchange: null });
  const radios = ["thorough", "balanced", "fast", "custom"].map((v) => fake(v));
  const b = makeBrowser({ answers: { auto_plan: { plan: speedPlan() } }, state: baseState, groups: { "input[name=ac-preset]": radios } });
  open(b, {}, `ac.plan = ${JSON.stringify(speedPlan())}; acDefaults()`);
  let html = card(b);
  for (const id of ["thorough", "balanced", "fast", "custom"]) assert.match(html, new RegExp(`name="ac-preset" value="${id}"`));
  assert.match(html, /data-acpreset="thorough"[\s\S]*?about 57 h to 108 h/); assert.match(html, /data-acpreset="fast"[\s\S]*?about 11 h to 22 h/);
  assert.match(html, /name="ac-preset" value="balanced" checked/);
  assert.match(html, /If a thinking check fails, a retry adds 23 h to 37 h/);
  // Thorough: the recipe
  b.run(`acApplyPreset("thorough")`);
  let s = plain(b.run("acSpec()"));
  assert.deepEqual([s.preset, s.stages, s.stage_a_tokens_m, s.last_tokens_m, s.quick, s.think_quick, s.share_teacher_cache, s.check_cut, s.retry_heal, s.back_off, s.imatrix], ["thorough", 2, null, null, false, null, false, true, true, true, true]);
  // Fast
  b.run(`acApplyPreset("fast")`);
  s = plain(b.run("acSpec()"));
  assert.deepEqual([s.preset, s.stages, s.last_tokens_m, s.quick, s.think_quick, s.think_gate_n, s.check_cut, s.retry_heal, s.back_off, s.imatrix], ["fast", 1, 20, true, true, 1, false, false, false, false]);
  // one knob changed: Custom, and the radio says so
  b.run(`acApplyPreset("balanced"); acApplySet({ imatrix: false })`);
  assert.equal(b.run("acSpec().preset"), "custom"); assert.match(card(b), /name="ac-preset" value="custom" checked/);
  b.run(`acApplySet({ imatrix: true })`);
  assert.equal(b.run("acSpec().preset"), "balanced", "set back, it is the preset again");
  // the radio buttons do it
  b.run("wireAuto()");
  const fast = radios.find((r) => r.value === "fast"); fast.checked = true; fast.onchange();
  assert.equal(b.run("acSpec().stages"), 1); assert.equal(b.run("acSpec().last_tokens_m"), 20);
});

test("the biggest levers are shown with the change each choice makes, and a choice sets its fields", () => {
  const knobRadios = [0, 1, 2, 3].map((i) => ({ value: "", dataset: { acknob: "last_tokens_m", acopt: String(i) }, checked: false, onchange: null }));
  const b = makeBrowser({ state: baseState, groups: { "input[data-acknob]": knobRadios } });
  open(b, {}, `ac.plan = ${JSON.stringify(speedPlan())}; acDefaults()`);
  const html = card(b);
  assert.match(html, /id="ac-levers"[\s\S]*Heal tokens of the last stage/);
  assert.match(html, /data-acopt="0"[^>]*>\s*10 M\s*<span[^>]*>−\d+ h/);
  assert.match(html, /80 M \(recipe\)\s*<span class="dim small ac-delta">now/);
  assert.match(html, /Each on its own\s*<span[^>]*>\+/);
  // the rest is behind "More speed settings"
  assert.match(html, /<details id="ac-adv"[^>]*><summary>More speed settings<\/summary>[\s\S]*Importance matrix/);
  assert.doesNotMatch(html.slice(0, html.indexOf("More speed settings")), /Importance matrix/);
  // stages and Try first keep their own radios, with the change next to them
  assert.match(html, /One stage <span class="dim small ac-delta">−13 h to −21 h/);
  // picking 20 M sets the field; picking the recipe puts the recipe back (null), so the preset stays a preset
  b.run("wireAuto()");
  const pick = (i) => { const r = knobRadios[i]; r.checked = true; r.onchange(); };
  b.run(`ac.plan = ${JSON.stringify(speedPlan())}`);
  pick(1); assert.equal(b.run("acSpec().last_tokens_m"), 20); assert.equal(b.run("acSpec().preset"), "custom");
  b.run(`ac.plan = ${JSON.stringify(speedPlan())}`);
  pick(3); assert.equal(b.run("acSpec().last_tokens_m"), null); assert.equal(b.run("acSpec().preset"), "balanced");
  assert.equal(b.run("acDelta([-360, -660])"), "−6 h to −11 h"); assert.equal(b.run("acDelta([0, 0])"), "no change"); assert.equal(b.run("acDelta([420, 540])"), "+7 h to +9 h");
});

test("Fast shows a red warning for a model that thinks and not for one that does not; the plan's warnings name every value below the recipe", () => {
  const b = browser();
  open(b, {}, `ac.plan = ${JSON.stringify(speedPlan({ warnings: ["The last stage heals on 20 M tokens; the recipe for a cut of this size is 80 M. In an earlier real run, healing below the recipe failed the thinking checks."] }))}; acDefaults(); acApplyPreset("fast")`);
  let html = card(b);
  assert.match(html, /class="err" data-acspeed-warn="fast">[^<]*very likely lose its reasoning/);
  assert.match(html, /class="warn">The last stage heals on 20 M tokens; the recipe for a cut of this size is 80 M\. In an earlier real run/);
  // Balanced says it can fail the thinking checks (a plain sentence, not the red one)
  b.run(`acApplyPreset("balanced")`);
  html = card(b);
  assert.match(html, /Runs that healed below the recipe have failed the thinking checks before/);
  // a model that does not think has no such warning, and its thinking knobs do not make the preset Custom
  b.run(`ac.thinking = false; acApplyPreset("fast")`);
  assert.doesNotMatch(card(b), /data-acspeed-warn/);
  b.run(`acApplyPreset("thorough"); ac.sp.retry_heal = false; ac.sp.check_cut = false; ac.gate = false`);
  assert.equal(b.run("acPresetNow()"), "thorough");
});

test("a small model has the recipe chosen, and the time limit shows what was given up or the refusal", () => {
  const b = browser();
  b.run(`state = ${JSON.stringify(baseState())}; page = "reshape"; rs.state = ${JSON.stringify(pageState())}; rs.model = "small"; rs.options = { inspect: ${JSON.stringify(insp({ total_params: 8e9 }))}, options: [], width: null, width_options: null };`);
  card(b);
  assert.equal(b.run("acSpec().preset"), "thorough");
  assert.equal(b.run("acSpec().max_hours"), undefined, "no limit unless asked");
  b.run(`ac.maxOn = true; ac.maxH = "30"`);
  assert.equal(b.run("acSpec().max_hours"), 30);
  const dropped = [{ knob: "think_quick", label: "Thinking checks", from: "Full (32 questions)", to: "Quick (16 questions)", saves_h: [0.6, 1.1], risk: "A close call can go either way." },
                   { knob: "retry_heal", label: "Retry heal after a failed check", from: "on", to: "off", saves_h: [0, 0], risk: "A failed check goes straight to cutting one step less." }];
  b.run(`ac.plan = ${JSON.stringify(speedPlan({ budget: { max_hours: 30, fits: true, asked_h: [57, 108], total_h: [20, 29.5], dropped } }))}`);
  let html = card(b);
  assert.match(html, /id="ac-budget-list"[\s\S]*These settings are given up|id="ac-budget-list"[\s\S]*given up/);
  assert.match(html, /data-acdropped="think_quick"[\s\S]*Quick \(16 questions\) instead of Full \(32 questions\)[\s\S]*A close call can go either way/);
  assert.match(html, /data-acdropped="retry_heal"/);
  b.run(`ac.plan = ${JSON.stringify(speedPlan({ ok: false, blockers: ["Even the fastest settings take about 12 h to 24 h for this cut, and you have 3 h. Choose more time or a larger size."], budget: { max_hours: 3, fits: false, asked_h: [57, 108], total_h: [12, 24], dropped } }))}`);
  html = card(b);
  assert.match(html, /class="err">Even the fastest settings take about 12 h to 24 h/); assert.match(html, /id="ac-go"[^>]*disabled/);
  b.run(`ac.maxOn = false`);
  assert.doesNotMatch(card(b), /id="ac-budget-list"/);
  b.run(`ac.maxOn = true; ac.maxH = ""`);
  assert.match(card(b), /id="ac-maxwhy"/);
});

const prog = (over = {}) => ({ kind: "auto_progress", phase: "auto", run: "20261117-100000-qwen3-8-27b-21b", pct: 42.3, state: "live", elapsed_s: 76800, left_s: [50400, 61200], stage: { n: 1, of: 2 },
  step: { id: "s1.heal", label: "Stage 1 of 2: heal", index: 7, count: 15, pct: 63, part: "cache", sub: "Teacher cache: 12.4 M of 30.0 M tokens · 3,100 tokens/s · 41 min left" }, steps: [], retry_added_min: null, calibration: null, ...over });

test("an auto_progress event shows the overall bar, the stage chip, the time left and the sub-status line with the step's own bar", () => {
  const b = browser();
  open(b, {}, `ac.run = ${JSON.stringify(run("running"))}; ac.running = true; ac.plan = ${JSON.stringify(plan())}`);
  b.run(`onAutoEvent(${JSON.stringify(prog())})`);
  const html = card(b);
  assert.match(html, /id="ac-overall" aria-label="Auto-cut overall"/);
  assert.match(html, /value="42.3"/);
  assert.match(html, /42 % · 21 h 30 min so far · about 14 h to 17 h left \(done around/);
  assert.match(html, /Stage 1 of 2/);
  assert.match(html, /Teacher cache: 12\.4 M of 30\.0 M tokens · 3,100 tokens\/s · 41 min left/);
  assert.match(html, /id="ac-stepbar"[^>]*>|value="63" id="ac-stepbar"/);
  assert.match(html, /step 7 of 15/);
  assert.doesNotMatch(html, /id="ac-eta"/, "the older estimate is only the fallback");
});

test("while the speed is not known the card says estimating, a retry explains the step back", () => {
  const b = browser();
  open(b, {}, `ac.run = ${JSON.stringify(run("running"))}; ac.running = true; ac.plan = ${JSON.stringify(plan())}`);
  b.run(`onAutoEvent(${JSON.stringify(prog({ state: "estimating", step: { id: "s1.heal", label: "heal", index: 1, count: 2, pct: 1, part: null, sub: null }, retry_added_min: 190 }))})`);
  const html = card(b);
  assert.match(html, /estimating the time left…/);
  assert.match(html, /id="ac-retrynote"/);
  assert.match(html, /stepped back once/);
});

test("a run without any auto_progress keeps the older hours estimate", () => {
  const b = browser();
  open(b, {}, `ac.run = ${JSON.stringify(run("running", { steps: [step("s1.heal", "heal", "running")] }))}; ac.running = true; ac.plan = ${JSON.stringify(plan())}; ac.eta = { s: 7200, step: "s1.heal", at: Date.now() }`);
  const html = card(b);
  assert.doesNotMatch(html, /id="ac-overall"/);
  assert.match(html, /id="ac-eta"/);
});

test("a stopped run shows where it stopped and every step its status, estimate and time taken", () => {
  const b = browser();
  const r = run("stopped", { progress: prog({ pct: 42.3 }), steps: [{ ...step("plan", "Check", "done"), est: [1, 1], started: "2026-11-17T10:00:00Z", finished: "2026-11-17T10:05:00Z" }, { ...step("s1.heal", "heal", "pending"), est: [180, 300] }] });
  open(b, {}, `ac.run = ${JSON.stringify(r)}; ac.running = false; ac.plan = ${JSON.stringify(plan())}`);
  const html = card(b);
  assert.match(html, /Stopped at 42 %/);
  assert.match(html, /ac-chip-done">done</);
  assert.match(html, /ac-chip-pending">waiting</);
  assert.match(html, /est\. 3 h to 5 h/);
  assert.match(html, /took 5 min/);
});

test("the stage notification is on by default, can be switched off and goes into the spec", () => {
  const b = browser();
  open(b);
  assert.match(card(b), /id="ac-notify" checked/);
  assert.equal(b.run("acSpec().notify_stages"), true);
  b.run("ac.notifyStages = false");
  assert.equal(b.run("acSpec().notify_stages"), false);
  assert.doesNotMatch(card(b), /id="ac-notify" checked/);
});
