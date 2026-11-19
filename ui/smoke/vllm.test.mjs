// Smoke test of the experimental vLLM engine in the window: the Settings card (off by default, honest labels) and the engine choice of the Thinking check card.
import test from "node:test";
import assert from "node:assert/strict";
import { makeBrowser } from "./_browser.mjs";

const baseState = (vllm) => ({ project: { name: "P", folder: "/p" }, steps: [], bundle: { world: "W" }, settings: { theme: "light", expert_mode: false, show_help_panel: true, ...(vllm ? { vllm } : {}) }, worlds: [], recent: [] });
const NAME = "20261113-150000-qwen3.5-27b-slim-w10752";
const model = () => ({ name: NAME, status: "done", model_id: "custom-w", healed: false, healed_model_id: null, heal: null, test: null, test_healed: null, size_gb: 42, cache_gb: 0, healed_size_gb: 0, scores: null,
  summary: { mode: "prune", shape: "width", layers_before: 64, layers_after: 64, dropped: [], duplicated: [], params_before: 27.8e9, params_after: 21.1e9, loss_before: 2.0, loss_after: 3.1, notes: [], chosen_blocks: [], boundary_distance: null } });
const pageState = () => ({ models: [], made: [model()], dataset_version: 3, env_ready: true, vram_gb: 32, gpu: "RTX 5090", free_disk_gb: 500 });
const main = (closed, acc) => ({ n: 16, think_rate: 1, closed_rate: closed, accuracy: acc, loop_rate: 0, garbage_rate: 0, truncated_rate: 0, median_think_tokens: 800, p90_think_tokens: 1600 });
const row = (m, backend) => ({ suite_version: 1, backend, main: m, control: { n: 8, accuracy: 0.875 }, created: "2026-11-17T10:00:00Z" });
const results = () => ({ suite_version: 1, targets: { original: true, cut: true, healed: false }, ggufs: [], modes: { full: { rows: {}, gates: {} }, quick: { rows: { original: row(main(1, 0.75), "hf") }, gates: {} } },
  vllm_modes: { full: { rows: {}, gates: {} }, quick: { rows: { original: row(main(0.5, 0.25), "vllm") }, gates: {} } } });

function open(b, vllm) {
  b.run(`state = ${JSON.stringify(baseState(vllm))}; page = "reshape"; rs.state = ${JSON.stringify(pageState())}; rs.model = "qwen3.5-27b"; rs.view = ${JSON.stringify(NAME)};
    rs.options = null; rs.tk.name = ${JSON.stringify(NAME)}; rs.tk.data = ${JSON.stringify(results())};`);
}

test("the Settings card is off by default and says it is untested", () => {
  const b = makeBrowser({ answers: {}, state: () => baseState() });
  b.run(`state = ${JSON.stringify(baseState())}`);
  const html = b.run("renderVllm()");
  assert.match(html, /vLLM engine \(experimental\)/);
  assert.match(html, /not yet run on this PC/);
  assert.doesNotMatch(html, /id="vl-enabled"[^>]*checked/);
  assert.doesNotMatch(html, /id="vl-adapter"[^>]*checked/);
  assert.match(html, /Docker inside WSL/); assert.match(html, /WSLc/); assert.match(html, /vllm\/vllm-openai:v0\.34\.0/); assert.match(html, /cyankiwi\/Qwen3\.8-27B-AWQ-INT4/);
  assert.match(html, /nothing is started or installed/);
});

test("the vLLM choice is greyed out while the engine is off and the standard engine stays the default", () => {
  const b = makeBrowser({ answers: {}, state: () => baseState() });
  open(b, null);
  const html = b.run("renderReshape()");
  assert.match(html, /<option value="hf" selected>Transformers \(default\)/);
  assert.match(html, /<option value="vllm"[^>]*disabled>vLLM \(experimental: switched off/);
  assert.doesNotMatch(html, /tk-vllm-note/);
  assert.match(html, /<td>Right answers<\/td><td>75 %<\/td>/);
  b.run(`rs.tk.backend = "vllm"`);
  assert.equal(b.run("tkBackend()"), "hf", "a switched-off engine is never used");
  assert.equal(b.run("tkSpecBackend()"), null);
});

test("with the engine on, the vLLM table is shown with its warning and the job spec says vllm", () => {
  const b = makeBrowser({ answers: {}, state: () => baseState({ enabled: true, kind: "auto" }) });
  open(b, { enabled: true, kind: "auto" });
  b.run(`rs.tk.backend = "vllm"`);
  const html = b.run("renderReshape()");
  assert.match(html, /<option value="vllm" selected\s*>vLLM \(experimental\)/);
  assert.match(html, /id="tk-vllm-note"/); assert.match(html, /experimental/);
  assert.match(html, /<td>Right answers<\/td><td>25 %<\/td>/); assert.doesNotMatch(html, /<td>Right answers<\/td><td>75 %<\/td>/);
  assert.match(html, /\(vLLM, experimental\)/);
  assert.equal(b.run("tkSpecBackend()"), "vllm");
  b.run(`rs.tk.backend = "llama"`);
  assert.match(b.run("renderReshape()"), /data-tkgo="cut"[^>]*disabled/);
});
