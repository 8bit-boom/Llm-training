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
