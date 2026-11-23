// Smoke test of the "Save a model as GGUF (no training needed)" card of the Slim or grow page (docs/RESHAPE.md): the three sources, the export name of each, the
// sizes and options it sends to export_model_start, the queue button, and what the card says when the job ends. app.js runs in a stand-in browser that records HTML.
// Run with `node --test ui/smoke`.
import test from "node:test";
import assert from "node:assert/strict";
import { makeBrowser, tick } from "./_browser.mjs";

const baseState = () => ({ project: { name: "P", folder: "/p" }, steps: [], bundle: { world: "W" }, settings: { theme: "light", expert_mode: false, show_help_panel: true }, worlds: [], recent: [] });
const reshapeState = () => ({ models: [], made: [], dataset_version: 1, env_ready: true, vram_gb: 32, gpu: "RTX 5090", free_disk_gb: 500, unsloth: true });
const quants = [["q4_k_m", "Q4_K_M: smallest, good quality (recommended)"], ["q8_0", "Q8_0: large, very close to the original"], ["f16", "F16: the original precision (very large)"]].map(([id, name]) => ({ id, name, label: id.toUpperCase() }));

const madeEntry = (over = {}) => ({ kind: "made", name: "mycut", sub: "model", label: "mycut · as made", export_name: "mycut", family: "qwen3_5", params_b: 21.6, size_gb: 43.1, thinking: false, source: { kind: "made", name: "mycut", sub: "model" }, ...over });
const modelState = (over = {}) => ({
  made: [madeEntry(), madeEntry({ sub: "healed", label: "mycut · healed copy", export_name: "mycut-healed", size_gb: 42.0, thinking: true, source: { kind: "made", name: "mycut", sub: "healed" } })],
  models: [
    { id: "qwen3.5-4b", name: "Qwen 3.5 4B", family: "qwen3_5", params_b: 4, local: true, download_gb: 0, source: { kind: "hub", repo: "unsloth/Qwen3.5-4B" } },
    { id: "qwen3.5-27b", name: "Qwen 3.5 27B", family: "qwen3_5", params_b: 27.8, local: false, download_gb: 55.6, source: { kind: "hub", repo: "unsloth/Qwen3.5-27B" } },
  ],
  quants, mixed_quants: ["mixed_4", "mixed_3", "mixed_2", "mixed_fit", "mixed_measured"].map((id, i) => ({ id, name: id + " · Mixed precision (our own recipe)", label: "Mixed " + ["4-bit (balanced)", "3-bit (small)", "2-bit (smallest: short answers only)", "fit (experimental)", "measured"][i], mixed: true })),
  imatrix: { chunks: [64, 128, 256], default_chunks: 128, ctx: 512, general_share: 0.2, saved: {}, library: [], mtp: {} },
  env_ready: true, world: "W", ...over,
});

// the page the way the program leaves it after rsRefresh: the state is in and the pick of the card points at the first choice of its kind (rsxEnsurePick).
// init() runs on a microtask and lands on the welcome page, so the page is set again before rendering and wiring.
async function browser(st = modelState(), wired = true) {
  const b = makeBrowser({ state: baseState });
  b.run(`state = ${JSON.stringify(baseState())}; page = "reshape"; rs.state = ${JSON.stringify(reshapeState())}; rs.model = "qwen3.5-27b"; rs.options = null; rsx.state = ${JSON.stringify(st)};`);
  await tick();
  if (wired) {
    b.run(`page = "reshape"; rsxEnsurePick(); renderReshape(); wireReshape();`);
  }
  return b;
}

const html = (b) => b.run("renderReshape()");

test("the card offers the three sources, the made models and the list, and says where the files go", async () => {
  const b = await browser();
  const h = html(b);
  assert.match(h, /Save a model as GGUF \(no training needed\)/);
  assert.match(h, /A model made here \(2\)/);
  assert.match(h, /A model of the list/);
  assert.match(h, /A folder on this computer/);
  assert.match(h, /mycut · as made · 21\.6 B · 43\.1 GB/);
  assert.match(h, /mycut · healed copy · 21\.6 B · 42 GB/);
  assert.match(h, /exports\/mycut-&lt;size&gt;\//);
  assert.match(h, /Save the GGUF/);
  assert.match(h, /Add to queue/);
});

test("the models of the list are offered when that source is chosen, with the download note for the ones not here", async () => {
  const b = await browser();
  b.run(`rsx.src = "hub"; rsx.pick = "unsloth/Qwen3.5-4B";`);
  const h = html(b);
  assert.match(h, /Qwen 3\.5 4B · 4\.0 B/);
  assert.match(h, /Qwen 3\.5 27B · 27\.8 B · downloads first \(≈ 55\.6 GB\)/);
});

test("a model of the list that is on the computer says nothing about downloading", async () => {
  const b = await browser();
  b.run(`rsx.src = "hub"; rsx.pick = "unsloth/Qwen3.5-4B";`);
  assert.doesNotMatch(html(b), /downloads it first/);
});

test("the made model is the default source and the args name it, its sizes and its context", async () => {
  const b = await browser();
  await b.run("rsxRun('make', 'export_model_start', rsxMakeArgs())");
  await tick();
  const call = b.calls.find(([c]) => c === "export_model_start");
  assert.ok(call, "export_model_start was invoked");
  const args = call[1];
  assert.equal(JSON.stringify(args.source), JSON.stringify({ kind: "made", name: "mycut", sub: "model" }));
  assert.equal(args.quant, "q4_k_m");
  assert.equal(JSON.stringify(args.quants), JSON.stringify(["q4_k_m"]));
  assert.equal(args.numCtx, 8192);
  assert.equal(args.thinking, false);
  assert.equal(args.paramsB, 21.6);
  assert.equal(args.downloadGb, 0);
  assert.equal(args.imatrix, null);
  assert.equal(args.mixed, null);
});

test("the healed copy exports under its own name and keeps thinking", async () => {
  const b = await browser();
  b.els["rsx-pick"].value = "mycut||healed";
  b.els["rsx-pick"].onchange();
  assert.match(html(b), /exports\/mycut-healed-&lt;size&gt;\//);
  const args = JSON.parse(b.run("JSON.stringify(rsxMakeArgs())"));
  assert.equal(args.source.kind, "made");
  assert.equal(args.source.name, "mycut");
  assert.equal(args.source.sub, "healed");
  assert.equal(args.thinking, true);
});

test("a model of the list carries its download size for the queue's checklist", async () => {
  const b = await browser();
  b.run(`rsx.src = "hub"; rsx.pick = "unsloth/Qwen3.5-27B";`);
  const args = JSON.parse(b.run("JSON.stringify(rsxMakeArgs())"));
  assert.deepEqual(args.source, { kind: "hub", repo: "unsloth/Qwen3.5-27B" });
  assert.equal(args.paramsB, 27.8);
  assert.equal(args.downloadGb, 55.6);
  assert.equal(args.thinking, false);
});

test("a folder source sends the typed path and the folder's name is the export name", async () => {
  const b = await browser();
  b.run(`rsx.src = "folder"; rsx.folder = "C:\\\\models\\\\Qwen3.5-9B";`);
  const args = JSON.parse(b.run("JSON.stringify(rsxMakeArgs())"));
  assert.deepEqual(args.source, { kind: "folder", dir: "C:\\models\\Qwen3.5-9B", name: null });
  assert.equal(b.run("rsxExportName()"), "Qwen3.5-9B");
});

test("the importance matrix is off by default and sends its options when ticked", async () => {
  const b = await browser();
  assert.equal(JSON.parse(b.run("JSON.stringify(rsxMakeArgs())")).imatrix, null);
  b.run(`rsx.im = true; rsx.imChunks = 256;`);
  const args = JSON.parse(b.run("JSON.stringify(rsxMakeArgs())"));
  assert.equal(args.imatrix.enabled, true);
  assert.equal(args.imatrix.chunks, 256);
  assert.equal(args.imatrix.reuse, false);
  assert.equal(args.imatrix.nextn, false);
});

test("a mixed size with a target of its own needs it before the button lights up", async () => {
  const b = await browser();
  b.run(`rsx.quants = ["mixed_fit"]; rsx.mx.target = "";`);
  const h = html(b);
  assert.match(h, /Give the size the fit should aim for\./);
  assert.match(h, /id="rsx-go" disabled/);
  b.run(`rsx.mx.target = "12.5";`);
  const h2 = html(b);
  assert.doesNotMatch(h2, /Give the size the fit should aim for\./);
  assert.doesNotMatch(h2, /id="rsx-go" disabled/);
});

test("a mixed recipe size without a target needs nothing (only the fits do)", async () => {
  const b = await browser();
  b.run(`rsx.quants = ["mixed_4"];`);
  const h = html(b);
  assert.doesNotMatch(h, /id="rsx-go" disabled/);
});

test("the queue button sends the same args as the start button", async () => {
  const b = await browser();
  await b.run(`quAdd("export_model", { dataset: { quadd: "export_model" } })`);
  await tick();
  const call = b.calls.find(([c]) => c === "queue_add");
  assert.ok(call, "queue_add was invoked");
  assert.equal(call[1].kind, "export_model");
  assert.equal(JSON.stringify(call[1].args.source), JSON.stringify({ kind: "made", name: "mycut", sub: "model" }));
  assert.equal(call[1].args.numCtx, 8192);
});

test("the end event names the exports and points at Step 5", async () => {
  const b = await browser();
  b.run(`rsx.running = true; rsx.text = "Converting";`);
  b.send("reshape-event", { kind: "progress", phase: "export", step: "gguf", value: 0.5, text: "Converting to a 16-bit GGUF" });
  assert.equal(b.run("rsx.progress"), 0.5);
  b.send("reshape-event", { kind: "done", phase: "export", data: { name: "mycut-q4_k_m, mycut-q8_0", run: "mycut", exports: [{ name: "mycut-q4_k_m" }, { name: "mycut-q8_0" }] } });
  assert.equal(b.run("rsx.running"), false);
  assert.match(b.run("rsx.ok"), /Saved: mycut-q4_k_m, mycut-q8_0/);
  assert.match(b.run("rsx.ok"), /Step 5/);
});

test("an error of the job lands in the card", async () => {
  const b = await browser();
  b.run(`rsx.running = true;`);
  b.send("reshape-event", { kind: "error", phase: "export", text: "The model to export is missing." });
  assert.equal(b.run("rsx.running"), false);
  assert.match(b.run("rsx.error"), /The model to export is missing\./);
});
