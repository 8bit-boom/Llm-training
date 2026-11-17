// Smoke test of the preflight checklist above a Start button (here: the Thinking check card), without a window. Run with `node --test ui/smoke`.
import test from "node:test";
import assert from "node:assert/strict";
import { makeBrowser, source } from "./_browser.mjs";


function browser(answers = {}) {
  return makeBrowser({ answers, state: baseState });
}

const baseState = () => ({ project: { name: "P", folder: "/p" }, steps: [], bundle: { world: "W" }, settings: { theme: "light", expert_mode: false, show_help_panel: true }, worlds: [], recent: [] });
const NAME = "20261113-150000-qwen3.5-27b-slim-w10752";
const model = (over = {}) => ({ name: NAME, status: "done", model_id: "custom-w", healed: false, healed_model_id: null, heal: null, test: null, test_healed: null, size_gb: 42, cache_gb: 0, healed_size_gb: 0, scores: null,
  summary: { mode: "prune", shape: "width", layers_before: 64, layers_after: 64, dropped: [], duplicated: [], params_before: 27.8e9, params_after: 21.1e9, loss_before: 2.0, loss_after: 3.1, notes: [], chosen_blocks: [], boundary_distance: null }, ...over });
const pageState = (made) => ({ models: [], made, dataset_version: 3, env_ready: true, vram_gb: 32, gpu: "RTX 5090", free_disk_gb: 500 });
const main = (closed, acc, loops, median) => ({ n: 16, think_rate: 1, closed_rate: closed, accuracy: acc, loop_rate: loops, garbage_rate: 0, truncated_rate: 0.0625, median_think_tokens: median, p90_think_tokens: median * 2 });
const row = (m, over = {}) => ({ suite_version: 1, backend: "hf", main: m, control: { n: 8, accuracy: 0.875 }, created: "2026-11-17T10:00:00Z", ...over });
const gate = (verdict, failed = [], warned = []) => ({ verdict, failed, warned, checks: [] });
const results = (over = {}) => ({ suite_version: 1, targets: { original: true, cut: true, healed: false }, ggufs: [], modes: { full: { rows: {}, gates: {} }, quick: { rows: {}, gates: {} } }, ...over });

function open(b, made, data, extra = "") {
  b.run(`state = ${JSON.stringify(baseState())}; page = "reshape"; rs.state = ${JSON.stringify(pageState(made))}; rs.model = "qwen3.5-27b"; rs.view = ${JSON.stringify(made[0].name)};
    rs.options = null; rs.tk.name = ${JSON.stringify(made[0].name)}; rs.tk.data = ${JSON.stringify(data)}; ${extra}`);
}
const plain = (x) => JSON.parse(JSON.stringify(x));
const card = (b) => b.run("renderReshape()");


const FAIL = { checks: [{ id: "env", label: "Training software", light: "fail", detail: "Not installed yet.", fix: "Open Training setup." }, { id: "disk", label: "Free disk space", light: "ok", detail: "100 GB free.", fix: "" }], can_start: false };
const GREEN = { checks: [{ id: "env", label: "Training software", light: "ok", detail: "Installed.", fix: "" }, { id: "selftest", label: "Graphics card test", light: "warn", detail: "Not run yet.", fix: "Press Test." }], can_start: true };
const asked = (b, kind) => b.calls.filter((c) => c[0] === "preflight" && c[1].kind === kind).length;
const tick = () => new Promise((r) => setTimeout(r, 20));

test("the list is asked for once, shown above the button, and a red light disables every Check button", async () => {
  const b = browser({ preflight: FAIL });
  open(b, [model()], results());
  let html = card(b);
  assert.doesNotMatch(html, /Before you start/, "nothing is shown until the program has answered");
  await tick();
  html = card(b);
  assert.match(html, /<div class="pf pf-bad" role="status" aria-live="polite" data-pf="think_check">/);
  assert.match(html, /<li class="pf-fail">[^]*Training software<\/strong> Not installed yet\. <span class="dim">Open Training setup\.<\/span>/);
  assert.match(html, /<li class="pf-ok">[^]*Free disk space/);
  assert.match(html, /data-tkgo="original"[^>]*disabled/);
  assert.equal(asked(b, "think_check"), 1, "the same facts are not asked again on every drawing");
  assert.deepEqual(plain(b.calls.find((c) => c[0] === "preflight" && c[1].kind === "think_check")[1]), { kind: "think_check", spec: {} });
});

test("yellow lights are shown but do not stop the job, and a program without the command changes nothing", async () => {
  const b = browser({ preflight: GREEN });
  open(b, [model()], results());
  card(b);
  await tick();
  const html = card(b);
  assert.match(html, /<li class="pf-warn">[^]*Graphics card test<\/strong> Not run yet\. <span class="dim">Press Test\.<\/span>/);
  assert.doesNotMatch(html, /pf-bad/);
  assert.doesNotMatch(html, /data-tkgo="original"[^>]*disabled/);
  const old = browser({ preflight: () => { throw new Error("unknown command"); } });
  open(old, [model()], results());
  card(old);
  await tick();
  assert.doesNotMatch(card(old), /Before you start/);
  assert.doesNotMatch(card(old), /data-tkgo="original"[^>]*disabled/);
});

test("a finished install or llama.cpp check asks again", async () => {
  const b = browser({ preflight: FAIL });
  open(b, [model()], results());
  card(b);
  await tick();
  assert.match(card(b), /pf-bad/);
  b.run("pfReset()");
  assert.doesNotMatch(card(b), /pf-bad/, "forgotten until the program has answered again");
  await tick();
  assert.equal(asked(b, "think_check"), 2);
});
