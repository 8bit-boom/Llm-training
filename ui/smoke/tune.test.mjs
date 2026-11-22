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
