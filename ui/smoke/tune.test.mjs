// Tune: presets you can see and select (the bar on the Heal card and the Tune page), the preview before anything changes, the typed-name delete, and why buttons are off. The stub checks the invoke keys.
import test from "node:test";
import assert from "node:assert/strict";
import { makeBrowser, plain, tick } from "./_browser.mjs";

const view = (over = {}) => ({ model: { name: "slim-w", shape_id: "s-1", weights_id: "w-1", layers: 48, hidden: 5120, params: 20.1e9, arch: "qwen3_5", lineage: [] }, machine: { key: "m-1", gpu: "NVIDIA GeForce RTX 5090", vram_mb: 32607, driver: "621.24", software: {} },
  presets: [
    { id: "t-aaa", scope: "project", name: "Tested 2026-11-21", created: "2026-11-21T14:02:00Z", kind: "exact", note: "tested for this model", whole: true, model: "slim-w", params: 20.1e9, machine: { gpu: "NVIDIA GeForce RTX 5090", vram_mb: 32607 }, gains: [{ function: "heal.student_batch", gain: 2.41 }], stale: [{ function: "heal.student_batch", reasons: ["torch 2.14 -> 2.15 since the test"] }], report: "", notes: [], classes: [] },
    { id: "t-bbb", scope: "library", name: "Other card", created: "2026-11-20T10:00:00Z", kind: "other_pc", note: "another PC", whole: false, model: "slim-w", params: 20.1e9, machine: { gpu: "NVIDIA GeForce RTX 4090", vram_mb: 24564 }, gains: [], stale: [], report: "", notes: [], classes: [] }],
  unreadable: [], ...over });
const pv = { changes: [{ setting: "student_batch_tokens", label: "Tokens per forward pass", from: 0, to: 8192, gain: 2.41, function: "heal.student_batch", class: "tol", why: "" }, { setting: "kl_chunk", label: "Loss chunk", from: 128, to: 256, gain: 1.08, function: "heal.loss", class: "exact", why: "" }], whole: true, kind: "exact", note: "tested for this model", stale: [], token: "a1", gain: 2.41, name: "Tested 2026-11-21", context: [] };

function heal(answers = {}) {
  const b = makeBrowser({ answers: { tune_presets: view(), memory_status: { gpu: null, ram: null, containers: [], busy: false, wsl: false }, ...answers } });
  b.run(`rs.hl = { name: "slim-w", opts: {} }; rs.hlSpec = { seq: 4096, batch: 1, per_update: 32, checkpointing: "auto", kl_chunk: 128, head_checkpoint: null, student_batch_tokens: null };`);
  return b;
}

test("the preset bar lists the presets with their match, gains and a stale chip, and another PC is not selectable", async () => {
  const b = heal();
  await b.run(`tuneLoad("slim-w")`);
  const html = b.run(`tuneBar("heal", "slim-w")`);
  assert.match(html, /Tested 2026-11-21 · tested for this model · stale, re-test\?/);
  assert.match(html, /<option value="t-bbb"[^>]*disabled>Other card · another PC · library/);
  assert.match(html, /Use…/); assert.match(html, /Save current as preset…/);
  b.run(`tn.sel = "t-aaa"`);
  const sel = b.run(`tuneBar("heal", "slim-w")`);
  assert.match(sel, /measured: student_batch ×2\.41/); assert.match(sel, /stale: torch 2\.14 -&gt; 2\.15 since the test/);
});

test("Use… shows what would change with the measured gain, and Apply all fills the card; nothing else", async () => {
  const b = heal({ tune_apply_preview: pv });
  await b.run(`tuneLoad("slim-w")`);
  b.run(`tn.sel = "t-aaa"`);
  await b.run(`tuneAction("heal", "use")`);
  const call = b.calls.find((c) => c[0] === "tune_apply_preview");
  assert.deepEqual(plain(call[1]).presetId, "t-aaa"); assert.equal(plain(call[1]).scope, "project"); assert.equal(plain(call[1]).model, "slim-w");
  assert.equal(plain(call[1]).current.student_batch_tokens, 0, "the card's values as they are now");
  const html = b.run(`tuneBar("heal", "slim-w")`);
  assert.match(html, /Tokens per forward pass<\/td><td>0<\/td><td><strong>8192<\/strong><\/td><td>×2\.41/);
  assert.match(html, /Loss chunk/); assert.match(html, /same results within tolerance/); assert.match(html, /id="tn-apply-heal"/);
  assert.equal(b.run("rs.hlSpec.student_batch_tokens"), null, "not applied yet");
  b.run(`wireTune()`);
  b.els["tn-apply-heal"].onclick();
  assert.equal(b.run("rs.hlSpec.student_batch_tokens"), 8192); assert.equal(b.run("rs.hlSpec.kl_chunk"), 256); assert.equal(b.run("rs.hlSpec.seq"), 4096);
});

test("a quality knob is marked as a short trial, not a full heal", () => {
  const b = heal();
  b.run(`tn.preview = { where: "heal", data: ${JSON.stringify({ ...pv, changes: [{ setting: "rank", label: "LoRA size", from: 64, to: 32, gain: 1.3, function: "heal.quality", class: "quality", why: "" }] })} }`);
  assert.match(b.run(`tuneStageHtml("heal")`), /quality knob: based on a short trial, not a full heal/);
});

test("Save current as preset sends the card with the real keys; delete needs the typed name", async () => {
  const b = heal({ tune_preset_save: { id: "t-new", name: "Mine" }, tune_preset_delete: null });
  await b.run(`tuneLoad("slim-w")`);
  await b.run(`tuneAction("heal", "save")`);
  assert.match(b.run(`tuneBar("heal", "slim-w")`), /Name of the new preset/);
  b.run(`wireTune()`);
  b.els["tn-ask-input-heal"] = { value: "Mine" };
  await b.run(`tuneAskOk("heal")`);
  const save = plain(b.calls.find((c) => c[0] === "tune_preset_save")[1]);
  assert.deepEqual([save.model, save.name, save.scope], ["slim-w", "Mine", "project"]);
  assert.equal(save.card.seq, 4096); assert.equal(save.card.student_batch_tokens, 0);
  b.run(`tn.sel = "t-aaa"`);
  await b.run(`tuneAction("heal", "del")`);
  assert.match(b.run(`tuneBar("heal", "slim-w")`), /Type the preset&#39;s exact name \(Tested 2026-11-21\) to delete it|Type the preset's exact name \(Tested 2026-11-21\) to delete it/);
  b.els["tn-ask-input-heal"] = { value: "Tested 2026-11-21" };
  await b.run(`tuneAskOk("heal")`);
  assert.deepEqual(plain(b.calls.find((c) => c[0] === "tune_preset_delete")[1]), { id: "t-aaa", scope: "project", confirm: "Tested 2026-11-21" });
});

test("while a job runs the buttons are off and every one says why", async () => {
  const b = heal({ memory_status: { gpu: null, ram: null, containers: [], busy: true, wsl: false } });
  await b.run("memPoll()");
  await b.run(`tuneLoad("slim-w")`);
  b.run(`tn.sel = "t-aaa"`);
  const html = b.run(`tuneBar("heal", "slim-w")`);
  assert.match(html, /id="tn-use-heal"[^>]*disabled[^>]*title="Off because a job is running"/);
  assert.match(html, /Some buttons are off because a job is running/);
  b.run(`rs.hlSpec.vllm = true`);
});

test("the Tune page has the model picker, the bar and the list", async () => {
  const b = makeBrowser({ answers: { tune_presets: view(), memory_status: { gpu: null, ram: null, containers: [], busy: false, wsl: false } } });
  b.run(`state = ${JSON.stringify({ project: { name: "P", folder: "/p" }, steps: [], bundle: { world: "W" }, settings: { theme: "light", expert_mode: false, show_help_panel: true }, worlds: [], recent: [] })}; rs.state = { made: [{ name: "slim-w", status: "done", size_gb: 43 }], models: [] }; page = "tune";`);
  const html = b.run("renderTune()");
  assert.match(html, /Tune this PC for a model/); assert.match(html, /<option value="slim-w" selected>slim-w · 43 GB/); assert.match(html, /Looking…|Tested settings for this model/);
  await b.run(`tuneLoad("slim-w")`);
  const again = b.run("renderTune()");
  assert.match(again, /id="tn-list"/); assert.match(again, /48 layers, hidden size 5120, 20\.1 B parameters on NVIDIA GeForce RTX 5090 \(32 GB\)/);
  assert.match(b.run("renderSteps(), $('steps').innerHTML"), /🎛 Tune/);
});

// ---- Auto: test and choose the best settings
const plan = { tests: [
  { id: "heal.student_batch", label: "Tokens per forward pass", what: "w1", tier: 1, class: "tol", default_on: true, chosen: true, built: true, unavailable: null, minutes: [15, 25], knobs: ["student_batch_tokens"] },
  { id: "heal.loss", label: "Loss chunk and output-layer logits", what: "w2", tier: 1, class: "exact", default_on: true, chosen: true, built: true, unavailable: null, minutes: [20, 35], knobs: ["kl_chunk"] },
  { id: "heal.quality", label: "Quality settings (short trial heals)", what: "w3", tier: 2, class: "quality", default_on: true, chosen: false, built: false, unavailable: "not built in this version yet", minutes: [60, 100], knobs: [] },
  { id: "disk", label: "Disk speed", what: "w4", tier: 3, class: "exact", default_on: false, chosen: false, built: false, unavailable: "not built in this version yet", minutes: [2, 3], knobs: [] }],
  total_minutes: [35, 60], done_between: [1000, 4600], model: { name: "slim-w", params_b: 20.1, layers: 48, hidden: 5120 }, card_mb: 32768 };
const stateOf = (over = {}) => ({ id: "20261122-1000", status: "done", preset_id: "t-new", note: "", spec: { model: "slim-w" }, tests: [
  { id: "heal.student_batch", status: "done", decision: { why: "4096 tokens per forward pass is 2.41 times as fast as one row" }, note: "", result: { gain: 2.41, class: "tol" } },
  { id: "heal.loss", status: "done", decision: { why: "Within noise (±3.0 %): your loss settings stay." }, note: "", result: null }], ...over });

function tunePage(answers = {}) {
  const b = makeBrowser({ answers: { tune_presets: view(), tune_plan: plan, memory_status: { gpu: null, ram: null, containers: [], busy: false, wsl: false }, tune_start: { run: "20261122-1000" }, tune_state: stateOf(), ...answers } });
  b.run(`state = ${JSON.stringify({ project: { name: "P", folder: "/p" }, steps: [], bundle: { world: "W" }, settings: { theme: "light", expert_mode: false, show_help_panel: true }, worlds: [], recent: [] })}; rs.state = { made: [{ name: "slim-w", status: "done", size_gb: 43 }], models: [] }; page = "tune"; tn.pick = "slim-w"; rs.hlSpec = { seq: 4096 };`);
  return b;
}

test("the Tune page lists every test in its tier with the class, the estimate and why one is not available, and one big button", async () => {
  const b = tunePage();
  await b.run("tuLoadPlan()");
  const html = b.run("tuAutoHtml()");
  assert.match(html, /Auto: test and choose the best settings/);
  assert.match(html, /Speed and memory: the same results, tested in minutes/); assert.match(html, /Quality settings: short trial heals on your rows \(a short trial, not a full heal\)/);
  assert.match(html, /same results within tolerance/); assert.match(html, /quality knob: based on a short trial, not a full heal/);
  assert.match(html, /not available: not built in this version yet/);
  assert.match(html, /About 35 to 60 min/); assert.match(html, /id="tu-go"(?![^>]*disabled)/);
  assert.match(html, /data-tu-test="heal.student_batch" checked/); assert.match(html, /data-tu-test="heal.quality"\s+disabled/);
});

test("one click starts the run with the ticked tests, the cap and the card's settings, with the real keys", async () => {
  const b = tunePage();
  await b.run("tuLoadPlan()");
  b.run("tu.maxH = '2'");
  await b.run("tuGo()");
  const call = plain(b.calls.find((c) => c[0] === "tune_start")[1]);
  assert.deepEqual(Object.keys(call), ["spec"]);
  assert.deepEqual(call.spec.tests, ["heal.student_batch", "heal.loss"]);
  assert.equal(call.spec.model, "slim-w"); assert.equal(call.spec.max_hours, 2); assert.equal(call.spec.trial_tokens_k, 400);
  assert.equal(b.run("tu.running"), true);
});

test("the events fill the progress and the live rows, and the end shows Apply and save preset with every change and its gain", async () => {
  const b = tunePage({ tune_apply_preview: (a) => (a.target === "heal_card" ? pv : { ...pv, changes: [] }) });
  await b.run("tuLoadPlan()");
  await b.run("tuGo()");
  await b.run(`onTuneEvent({ kind: "tune_test", index: 0, total: 2, id: "heal.student_batch", label: "Tokens per forward pass", status: "running" })`);
  await b.run(`onTuneEvent({ kind: "tune_row", id: "heal.student_batch", row: { batch_tokens: 4096, tokens_per_s: 780, peak_vram_mb: 24576 } })`);
  let html = b.run("tuAutoHtml()");
  assert.match(html, /Test 1 of 2: Tokens per forward pass \(running\)/); assert.match(html, /780 tokens\/s/); assert.match(html, /24\.0 GB/);
  assert.match(html, /4096 tokens per forward pass is 2\.41 times as fast as one row/);
  await b.run(`onTuneEvent({ kind: "done", data: { run: "20261122-1000", status: "done", preset_id: "t-new" } })`);
  html = b.run("tuAutoHtml()");
  assert.match(html, /Apply and save preset/); assert.match(html, /already saved/); assert.match(html, /Tokens per forward pass<\/td><td class="dim small">Heal card<\/td><td>0<\/td><td><strong>8192<\/strong><\/td><td>×2\.41/);
  assert.match(html, /id="tu-apply-all"(?![^>]*disabled)/);
  const prev = plain(b.calls.filter((c) => c[0] === "tune_apply_preview")[0][1]);
  assert.deepEqual([prev.presetId, prev.scope, prev.target], ["t-new", "project", "heal_card"]);
  assert.deepEqual(b.calls.filter((c) => c[0] === "tune_apply_preview").map((c) => plain(c[1]).target), ["heal_card", "settings_teacher", "settings_vllm", "settings_llama", "settings_forward"], "the saved settings are previewed too");
  b.run("wireTune()");
  b.els["tu-apply-all"].onclick();
  assert.equal(b.run("rs.hlSpec.student_batch_tokens"), 8192); assert.equal(b.run("rs.hlSpec.kl_chunk"), 256);
  assert.match(b.run("renderTune()"), /Applied 2 changes to the Heal card/);
});

test("a stopped run offers Continue, a run that waits for the card says why, and a running job switches the button off with the reason", async () => {
  const b = tunePage({ tune_state: stateOf({ status: "waiting_gpu", preset_id: null, note: "Tokens per forward pass needs about 17.9 GB of the card and only 9.0 GB are free. Held by: chrome.exe 900 MB. Free the memory (Memory panel, Free GPU), then Continue." }) });
  await b.run("tuLoadPlan()");
  b.run(`tu.runId = "20261122-1000"`);
  await b.run("tuRefreshState()");
  const html = b.run("tuAutoHtml()");
  assert.match(html, /Continue the tests/); assert.match(html, /Held by: chrome\.exe 900 MB/);
  await b.run("tuGo()");
  assert.ok(b.calls.some((c) => c[0] === "tune_continue" && plain(c[1]).runId === "20261122-1000"));
  const busy = tunePage({ memory_status: { gpu: null, ram: null, containers: [], busy: true, wsl: false } });
  await busy.run("memPoll()");
  await busy.run("tuLoadPlan()");
  const off = busy.run("tuAutoHtml()");
  assert.match(off, /id="tu-go"[^>]*disabled[^>]*title="Off because a job is running"/); assert.match(off, /The button is off because a job is running/);
});

test("the Heal card and the Auto page have the Auto button, and Auto can use a tested preset as a snapshot", async () => {
  const b = heal();
  await b.run(`tuneLoad("slim-w")`);
  assert.match(b.run(`tuneBar("heal", "slim-w")`), /id="tn-auto-heal"[^>]*>Auto: test and choose the best settings/);
  const a = makeBrowser({ answers: { tune_presets_here: [{ id: "t-aaa", scope: "project", name: "Tested", model: "slim-w", created: "x", gains: [] }], tune_snapshot: { preset_id: "t-aaa", scope: "project", name: "Tested", values: { student_batch_tokens: 8192, kl_chunk: 256, seq: 4096 }, tokens_per_s: 780, measured_student_b: 20.1 } } });
  a.run(`ac.tunedList = [{ id: "t-aaa", scope: "project", name: "Tested", model: "slim-w" }]; ac.tunedSnap = ${JSON.stringify({ preset_id: "t-aaa", values: { student_batch_tokens: 8192, kl_chunk: 256, seq: 4096 }, tokens_per_s: 780 })}`);
  const html = a.run(`acTunedHtml("")`);
  assert.match(html, /Use tested settings for this PC/); assert.match(html, /student batch tokens 8192, kl chunk 256/); assert.doesNotMatch(html, /seq 4096/); assert.match(html, /measured 780 tokens\/s/);
});

test("the saved settings a test decided are in the same confirmation and are applied with the token of their own preview", async () => {
  const spv = { changes: [{ setting: "teacher_batch_tokens", label: "Teacher tokens per forward pass", from: 16384, to: 32768, gain: 1.4, function: "teacher.standard", class: "tol", why: "" }], token: "a-teacher", name: "Tested", whole: true, kind: "exact", note: "", stale: [], gain: 1.4, context: [] };
  const b = tunePage({ tune_apply_preview: (a) => (a.target === "heal_card" ? pv : a.target === "settings_teacher" ? spv : { ...pv, changes: [] }), tune_apply: { applied: [] } });
  await b.run("tuLoadPlan()");
  await b.run("tuGo()");
  await b.run(`onTuneEvent({ kind: "done", data: { run: "20261122-1000", status: "done", preset_id: "t-new" } })`);
  const html = b.run("tuAutoHtml()");
  assert.match(html, /Teacher tokens per forward pass<\/td><td class="dim small">Settings<\/td><td>16384<\/td><td><strong>32768<\/strong>/);
  assert.match(html, /Tokens per forward pass<\/td><td class="dim small">Heal card<\/td>/);
  b.run("wireTune()");
  await b.els["tu-apply-all"].onclick();
  const applied = b.calls.filter((c) => c[0] === "tune_apply").map((c) => plain(c[1]));
  assert.deepEqual(applied.map((a) => [a.target, a.token, a.presetId, a.scope, a.model]), [["settings_teacher", "a-teacher", "t-new", "project", "slim-w"]], "only the target with changes, with its own token");
  assert.equal(b.run("rs.hlSpec.student_batch_tokens"), 8192);
  assert.match(b.run("renderTune()"), /Applied 2 changes to the Heal card and saved 1 program setting\./);
});

test("the Auto page says what Auto-cut takes from a preset and what it does not, and its button runs the one-click Auto for the latest made model", async () => {
  const b = tunePage();
  b.run(`page = "auto"; ac.tunedList = []; ac.tunedSnap = ${JSON.stringify({ preset_id: "t-aaa", values: { student_batch_tokens: 8192, teacher_batch_tokens: 32768, rank: 32, optimizer: "adamw", seq: 4096 }, chosen_keys: ["student_batch_tokens", "teacher_batch_tokens", "rank", "think_batch"], tokens_per_s: 780 })}`);
  const html = b.run(`acTunedHtml("")`);
  assert.match(html, /heals of this run use: student batch tokens 8192, teacher batch tokens 32768/);
  assert.match(html, /Not taken by Auto-cut: rank 32/); assert.doesNotMatch(html, /Not taken by Auto-cut:[^<]*seq/);
  assert.match(html, /short trial heals, not a full heal/); assert.match(html, /Program settings in this preset \(think batch\) are saved from the Tune page/);
  assert.match(html, /id="ac-tune-auto"(?![^>]*disabled)[^>]*>Auto: test and choose the best settings for slim-w/);
  b.run(`ac.tunedSnap = ${JSON.stringify({ preset_id: "t-bbb", values: { teacher_batch_tokens: 32768, kl_chunk: 256 }, chosen_keys: [] })}`);
  assert.doesNotMatch(b.run(`acTunedHtml("")`), /teacher batch tokens/, "the card's untouched copy of a setting is not a test result");
  const orig = b.run("globalThis.confirm = () => true; 1");
  await b.run(`tuneAutoFor(acTuneModel())`);
  const start = b.calls.find((c) => c[0] === "tune_start");
  assert.ok(start, "the one-click Auto is started"); assert.equal(plain(start[1]).spec.model, "slim-w");
  assert.equal(orig, 1);
});

test("the live table shows the number each kind of test measures: speed, the held-out loss of a short trial, layers on the card, the disk", async () => {
  const b = tunePage();
  await b.run("tuLoadPlan()");
  b.run(`tu.runId = "20261122-1000"; tu.state = ${JSON.stringify(stateOf())}; tu.rows = { q: [{ id: "rank=32", label: "LoRA size 32", ok: true, held_after: 1.8123, peak_vram_mb: 20480 }], l: [{ ngl: 48, ok: true, tokens_per_s: 180 }], d: [{ write_mb_s: 1800, read_mb_s: 3200, mb: 2048 }] }`);
  const html = b.run("tuLiveHtml()");
  assert.match(html, /LoRA size 32<\/td><td>held-out loss 1\.8123 \(short trial\)<\/td><td>20\.0 GB/);
  assert.match(html, /48 layers on the card<\/td><td>180 tokens\/s/); assert.match(html, /Disk<\/td><td>writes 1800 MB\/s, reads 3200 MB\/s/);
});
