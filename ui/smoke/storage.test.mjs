// The Storage page: nav, rows arriving, sizes, totals, the confirmation with the typed word, cancel, the Delete buttons of the other pages. Run with `node --test ui/smoke`.
import test from "node:test";
import assert from "node:assert/strict";
import { makeBrowser, source } from "./_browser.mjs";

const baseState = (over = {}) => ({ project: { name: "P", folder: "/p" }, steps: [{ key: "test", label: "Step 5", unlocked: true, done: false }], bundle: { world: "W" }, settings: { theme: "light", expert_mode: false, show_help_panel: true }, worlds: [], recent: [], ...over });
const item = (id, over = {}) => ({ id, group: "exports", kind: "export_f16", label: `run1: the 16-bit GGUF ${id}`, path_short: `~/P/exports/run1/${id}`, size: null, size_cached: false, modified: "2026-11-20 10:00", from: null, status: "safe", refs: [], note: null, suggest: null, model: false, link: false, ...over });
const model = (id, over = {}) => item(id, { group: "models", kind: "model_entry", label: `cut-a (cut and healed)`, path_short: "~/P/models/cut-a", model: true, from: "made from org/Qwen3.8-27B", ...over });
const nav = (b) => { b.run("renderSteps()"); return b.els["steps"].innerHTML; };
const open = (b) => b.run(`state = ${JSON.stringify(baseState())}; page = "storage"; sg.scanned = true; sg.free = 100;`);
const ev = (b, p) => b.run(`onStorageEvent(${JSON.stringify(p)})`);

test("the list has Storage after Compare quants and before Training setup, locked without a project", () => {
  const b = makeBrowser({ state: baseState });
  b.run(`state = ${JSON.stringify(baseState())}; page = "welcome"`);
  const html = nav(b);
  const [q, s, t] = [html.indexOf('data-key="quantcheck"'), html.indexOf('data-key="storage"'), html.indexOf('data-key="setup"')];
  assert.ok(q > 0 && s > q && t > s, "between Compare quants and Training setup");
  assert.match(html, /data-key="storage"[^>]*>🧹 Storage</);
  b.run(`state = ${JSON.stringify(baseState({ project: null, bundle: null }))}`);
  assert.match(nav(b), /data-key="storage" disabled>🔒 🧹 Storage/);
});

test("rows arrive first and the sizes follow, with a placeholder until they are measured", () => {
  const b = makeBrowser({ state: baseState });
  open(b);
  ev(b, { kind: "item", item: item("s1") }); ev(b, { kind: "item", item: item("s2", { status: "in_use", refs: ["Auto-cut run r1 (stopped) still needs it to go on"] }) });
  let html = b.run("renderStorage()");
  assert.match(html, /size not measured yet/); assert.match(html, /in use/); assert.match(html, /still needs it to go on/);
  assert.match(html, /<input type="checkbox" data-stsel="s2" [^>]*disabled/, "an item in use cannot be ticked");
  ev(b, { kind: "size", id: "s1", size: 9_000_000_000 });
  html = b.run("renderStorage()");
  assert.match(html, /9\.0 GB/);
  ev(b, { kind: "size", id: "s2", size: null });
  assert.match(b.run("renderStorage()"), /size not available/);
});

test("the total follows the ticks, shows the disk before and after, and leaves out what lies inside another ticked item", () => {
  const b = makeBrowser({ state: baseState });
  open(b);
  ev(b, { kind: "item", item: item("e", { kind: "export", label: "run1-Q4", path_short: "~/P/exports/run1" }) });
  ev(b, { kind: "item", item: item("f", { path_short: "~/P/exports/run1/model-f16.gguf" }) });
  ev(b, { kind: "item", item: item("g", { path_short: "~/P/exports/run2/model-f16.gguf" }) });
  ["e", "f", "g"].forEach((id, i) => ev(b, { kind: "size", id, size: [20e9, 9e9, 5e9][i] }));
  assert.match(b.run("renderStorage()"), /Nothing is ticked/);
  b.run(`sg.sel = { e: true, f: true, g: true }`);
  const html = b.run("renderStorage()");
  assert.match(html, /Frees <strong>25 GB<\/strong>/, "the 9 GB file inside the ticked export is not counted twice");
  assert.match(html, /disk free now 100\.0 GB → 125\.0 GB/);
  assert.match(html, /id="st-del"(?![^>]*disabled)/);
});

test("the confirmation says it is permanent, lists the items and needs the typed word for a model", async () => {
  const dry = { dry_run: true, items: [{ id: "m", label: "cut-a (cut and healed)", size: 7e9, path_short: "~/P/models/cut-a" }], total_bytes: 7e9, needs_delete_word: true, warnings: ["cut-a: the export run1 was made from it"] };
  const b = makeBrowser({ state: baseState, answers: { storage_delete: (a) => (a.dry_run ? dry : { started: true }) } });
  open(b);
  ev(b, { kind: "item", item: model("m") }); ev(b, { kind: "size", id: "m", size: 7e9 });
  b.run(`sg.sel = { m: true }`);
  await b.run("stAsk(false)");
  assert.ok(b.calls.some((c) => c[0] === "storage_delete" && c[1].dry_run === true && c[1].ids[0] === "m"), "the dry run asks what would go");
  let html = b.run("renderStorage()");
  assert.match(html, /id="st-confirm"/); assert.match(html, /This is permanent\./); assert.match(html, /Recycle Bin/); assert.match(html, /~\/P\/models\/cut-a/); assert.match(html, /the export run1 was made from it/);
  assert.match(html, /id="st-typed"/); assert.match(html, /id="st-go" disabled/);
  b.run("wireStorage()");
  b.els["st-typed"].value = "delet"; b.els["st-typed"].oninput();
  assert.equal(b.els["st-go"].disabled, true);
  b.els["st-typed"].value = "DELETE"; b.els["st-typed"].oninput();
  assert.equal(b.els["st-go"].disabled, false);
  await b.run("stGo()");
  const real = b.calls.filter((c) => c[0] === "storage_delete").pop();
  assert.equal(real[1].dry_run, false); assert.equal(real[1].confirm, "DELETE"); assert.deepEqual(JSON.parse(JSON.stringify(real[1].ids)), ["m"]);
  assert.ok(real[1].ids.every((i) => typeof i === "string" && !/[\\/]/.test(i)), "only opaque ids are sent, never a path");
});

test("a small plain item needs no typed word; a typed word that is wrong sends nothing", async () => {
  const dry = { dry_run: true, items: [{ id: "f", label: "x", size: 1e6, path_short: "~/x" }], total_bytes: 1e6, needs_delete_word: false, warnings: [] };
  const b = makeBrowser({ state: baseState, answers: { storage_delete: (a) => (a.dry_run ? dry : { started: true }) } });
  open(b);
  ev(b, { kind: "item", item: item("f") }); b.run(`sg.sel = { f: true }`);
  await b.run("stAsk(false)");
  assert.doesNotMatch(b.run("renderStorage()"), /id="st-typed"/); assert.doesNotMatch(b.run("renderStorage()"), /id="st-go" disabled/);
  b.run(`sg.conf.dry.needs_delete_word = true; sg.conf.typed = "nope"`);
  const n = b.calls.length;
  await b.run("stGo()");
  assert.equal(b.calls.length, n, "nothing is sent without the word");
});

test("the deletion shows progress with a Cancel button and the result lists freed bytes and failures", () => {
  const b = makeBrowser({ state: baseState, answers: {} });
  open(b);
  ev(b, { kind: "item", item: item("f") });
  ev(b, { kind: "progress", text: "Deleting cut-a (1 of 2)", value: 0.5 });
  let html = b.run("renderStorage()");
  assert.match(html, /Deleting cut-a \(1 of 2\)/); assert.match(html, /id="st-cancel"/); assert.match(html, /<input type="checkbox" data-stsel="f" [^>]*disabled/);
  b.run("wireStorage()"); b.els["st-cancel"].onclick();
  assert.ok(b.calls.some((c) => c[0] === "storage_cancel"));
  ev(b, { kind: "done", data: { phase: "delete", freed_bytes: 7e9, deleted: [{ id: "a", label: "cut-a", freed: 7e9 }], failed: [{ id: "b", label: "run1", errors: [{ path: "~/P/exports/run1/x.gguf", error: "Another program is using it" }] }], cancelled: false, free_gb: 120 } });
  html = b.run("renderStorage()");
  assert.match(html, /Done: 7\.0 GB freed/); assert.match(html, /✓ cut-a/); assert.match(html, /✕ run1: ~\/P\/exports\/run1\/x\.gguf - Another program is using it/);
  assert.ok(b.calls.some((c) => c[0] === "storage_scan"), "the list is made again after a deletion");
});

test("the suggestions are listed with their reason, never ticked, and the preset ticks only the intermediates", () => {
  const b = makeBrowser({ state: baseState });
  open(b);
  const sug = { ids: ["a", "b"], total_bytes: 14e9, preset: { id: "keep_final", label: "After a finished Auto-cut keep only the final healed model and the GGUF files", ids: ["a"], total_bytes: 9e9 } };
  b.run(`onStorageEvent(${JSON.stringify({ kind: "done", phase: "scan", data: { items: [item("a", { suggest: "The quantized GGUF files exist." }), item("b", { suggest: "Only needed to compare more sizes." }), model("m")], suggestions: sug, free_gb: 50, busy: false, total_bytes: 1 } })})`);
  const html = b.run("renderStorage()");
  assert.match(html, /<h2>Safe to delete<\/h2>/); assert.match(html, /The quantized GGUF files exist\./); assert.match(html, /about <strong>14 GB<\/strong>/);
  assert.doesNotMatch(html, /data-stsel="[ab]" checked/, "nothing is preselected");
  b.run("wireStorage()"); b.els["st-preset"].onclick();
  assert.deepEqual(JSON.parse(b.run("JSON.stringify(sg.sel)")), { a: true });
  assert.doesNotMatch(b.run("JSON.stringify(Object.keys(sg.sel))"), /"m"/, "a whole model is never in the preset");
});

test("while a job runs the page says nothing can be deleted", () => {
  const b = makeBrowser({ state: baseState });
  open(b);
  b.run(`onStorageEvent(${JSON.stringify({ kind: "done", phase: "scan", data: { items: [item("a")], suggestions: null, free_gb: 1, busy: true } })})`);
  const html = b.run("renderStorage()");
  assert.match(html, /A job is running\. Nothing can be deleted/); assert.match(html, /id="st-del" disabled/);
});

test("the Delete buttons of the other pages open the Storage page with that model or export ticked, and delete nothing", async () => {
  const b = makeBrowser({ state: baseState, answers: { help: "<p>h</p>", storage_scan: null, storage_delete: { dry_run: true, items: [], total_bytes: 0, needs_delete_word: true, warnings: [] } } });
  b.run(`state = ${JSON.stringify(baseState())}; page = "reshape"; rs.state = { models: [], made: [], dataset_version: 3, env_ready: true }`);
  b.run(`stDeleteFor("model_entry", "cut-a")`);
  assert.equal(b.run("page"), "storage");
  assert.ok(b.calls.some((c) => c[0] === "storage_scan"));
  b.run(`onStorageEvent(${JSON.stringify({ kind: "done", phase: "scan", data: { items: [model("m")], suggestions: null, free_gb: 10, busy: false } })})`);
  await new Promise((r) => setTimeout(r, 20));
  assert.deepEqual(JSON.parse(b.run("JSON.stringify(sg.sel)")), { m: true }, "the model is ticked");
  assert.ok(b.calls.some((c) => c[0] === "storage_delete" && c[1].dry_run === true), "only a dry run so far");
  assert.ok(!b.calls.some((c) => c[0] === "storage_delete" && c[1].dry_run === false));
  // the Slim page's list and the export list carry the button
  b.run(`page = "reshape"`);
  assert.match(source, /data-stdel="model_entry\|/); assert.match(source, /data-stdel="export\|/);
});
