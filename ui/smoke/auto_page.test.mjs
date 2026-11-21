// The Auto page: Auto-cut has its own entry in the left list, its card is on that page and no longer on the Slim page. Run with `node --test ui/smoke`.
import test from "node:test";
import assert from "node:assert/strict";
import { makeBrowser } from "./_browser.mjs";

const baseState = (over = {}) => ({ project: { name: "P", folder: "/p" }, steps: [{ key: "data", label: "Step 1", unlocked: true, done: true }, { key: "test", label: "Step 5", unlocked: true, done: false }], bundle: { world: "W" },
  settings: { theme: "light", expert_mode: false, show_help_panel: true }, worlds: [], recent: [], ...over });
const pageState = () => ({ models: [{ id: "qwen3.8-27b", name: "Qwen3.8 27B", params_b: 27.8, reshape: null }], made: [], dataset_version: 3, env_ready: true, vram_gb: 32, gpu: "RTX 5090", free_disk_gb: 500 });
const insp = () => ({ supported: true, model_type: "qwen3_5", total_params: 27.78e9 });
const setup = (b, page, extra = "") => b.run(`state = ${JSON.stringify(baseState())}; page = "${page}"; rs.state = ${JSON.stringify(pageState())}; rs.model = "qwen3.8-27b"; rs.options = { inspect: ${JSON.stringify(insp())}, options: [], width: null, width_options: null }; ${extra}`);
const nav = (b) => { b.run("renderSteps()"); return b.els["steps"].innerHTML; };

test("the list on the left has Auto after the steps and before Slim or grow, and it is locked without a project", () => {
  const b = makeBrowser({ state: baseState });
  setup(b, "welcome");
  const html = nav(b);
  const [a, s, r] = [html.indexOf('data-key="auto"'), html.indexOf('data-key="test"'), html.indexOf('data-key="reshape"')];
  assert.ok(s > 0 && a > s && r > a, "Auto sits between the steps and Slim or grow");
  assert.match(html, /data-key="auto"[^>]*>🚀 Auto</);
  assert.doesNotMatch(html.slice(html.indexOf('data-key="auto"'), html.indexOf('data-key="auto"') + 120), /disabled/);
  b.run(`state = ${JSON.stringify(baseState({ bundle: null }))}`);
  assert.match(nav(b), /data-key="auto" disabled>🔒 🚀 Auto/);
});

test("the card is on the Auto page and not on the Slim page, which points to Auto", () => {
  const b = makeBrowser({ state: baseState });
  setup(b, "auto");
  const auto = b.run("renderAuto()");
  assert.match(auto, /<h1>🚀 Auto<\/h1>/); assert.match(auto, /id="ac-card"/); assert.match(auto, /<h2>Auto-cut \(one click\)<\/h2>/); assert.match(auto, /id="ac-model"/);
  setup(b, "reshape");
  const slim = b.run("renderReshape()");
  assert.doesNotMatch(slim, /id="ac-card"/); assert.doesNotMatch(slim, /Auto-cut \(one click\)/);
  assert.match(slim, /Want it all in one click\? <button class="link" id="rs-open-auto">Open Auto<\/button>/);
  b.run("wireReshape()"); b.els["rs-open-auto"].onclick();
  assert.equal(b.run("page"), "auto");
});

test("the render of a page is the Auto page for the key auto, and a running run shows its progress there", () => {
  const b = makeBrowser({ state: baseState });
  setup(b, "auto", `ac.run = { id: "r1", status: "running", error: null, steps: [{ id: "measure", label: "Measure the model", status: "running", note: null }], spec: {}, stages: [] }; ac.running = true;`);
  b.run("render()");
  assert.match(b.els["page"].innerHTML, /id="ac-card"/);
  b.run(`onAutoEvent(${JSON.stringify({ kind: "auto_progress", phase: "auto", run: "r1", pct: 42.3, state: "live", elapsed_s: 3600, left_s: [3600, 7200], stage: { n: 1, of: 2 }, step: { id: "measure", label: "Measure the model", index: 2, count: 15, pct: 10, part: null, sub: null }, steps: [], retry_added_min: null })})`);
  assert.match(b.els["page"].innerHTML, /id="ac-overall"/, "the live progress is drawn on the Auto page");
  assert.match(b.els["page"].innerHTML, /value="42.3"/);
});

test("opening the Auto page loads what the card needs", async () => {
  const b = makeBrowser({ state: baseState, answers: { reshape_state: pageState(), auto_status: { run: null, running: false } } });
  b.run(`state = ${JSON.stringify(baseState())}; ac.loaded = true;`);
  await b.run('go("auto")');
  await new Promise((r) => setTimeout(r, 30));
  assert.ok(b.calls.some((c) => c[0] === "reshape_state"), "the Slim state is loaded");
  assert.ok(b.calls.some((c) => c[0] === "auto_status"), "the run is asked for");
  assert.ok(b.calls.some((c) => c[0] === "help" && c[1].key === "auto"), "its help text");
});

test("Use this cut in Auto-cut lands on the Auto page with the pin box", () => {
  const b = makeBrowser({ state: baseState });
  setup(b, "reshape", `rs.tr.se.id = "x"; rs.tr.se.result = { rows: [] };`);
  b.run(`seChosen = () => ({ blocks: [3, 7, 11], intermediate: 10752, loss: 2.04, search_id: "x", think: null })`);
  b.run("sePin(null)");
  assert.equal(b.run("page"), "auto");
  assert.match(b.run("rs.ok"), /pinned for Auto-cut/);
  const html = b.run("renderAuto()");
  assert.match(html, /id="ac-pin"/); assert.match(html, /blocks 3, 7, 11 out/); assert.match(html, /pinned for Auto-cut/);
});
