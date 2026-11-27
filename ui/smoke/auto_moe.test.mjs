// The Auto page for a mixture of experts (Inkling-2-Mini): the goal, the size in GB, what the plan says, the request that is sent, and that a dense model's page is unchanged.
import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import { makeBrowser } from "./_browser.mjs";

const inspect = JSON.parse(fs.readFileSync(new URL("../../worker/tests/data/inkling2_inspect.json", import.meta.url), "utf8"));
const baseState = (over = {}) => ({ project: { name: "P", folder: "/p" }, steps: [], bundle: { world: "W" }, settings: { theme: "light", expert_mode: false, show_help_panel: true }, worlds: [], recent: [], ...over });
const pageState = () => ({ models: [{ id: "inkling-2-mini", name: "Inkling-2-Mini", params_b: 26.39 }], made: [], dataset_version: 3, env_ready: true, vram_gb: 32, gpu: "RTX 5090", free_disk_gb: 500 });
const planOf = (over = {}) => ({ ok: true, blockers: [], warnings: ["Auto-cut never takes loop passes out"], original_b: 26.39, stages: [], candidates: [], moe: { goal: "size", target_gb: 12, experts_before: 128, experts_after: 96, top_k_before: 8, top_k_after: 8, passes: 3, q4_gb: 12.4, packed_gb: 15.6, active_b: 4.6,
  table: [{ experts: 128, q4_gb: 16.1, packed_gb: 19.5, params_b: 26.4, stages: 0 }, { experts: 96, q4_gb: 12.4, packed_gb: 15.6, params_b: 20.3, stages: 1 }], passes_note: "Choose thinkingmachines/TML-d24-varloop-research for fewer passes.", scoring: "reap", rebalance: true }, ...over });

function open(answers = {}) {
  const b = makeBrowser({ state: baseState, answers });
  b.run(`state = ${JSON.stringify(baseState())}; page = "auto"; ac.loaded = true; rs.state = ${JSON.stringify(pageState())}; rs.model = "inkling-2-mini"; rs.options = { inspect: ${JSON.stringify(inspect)}, options: [] };`);
  return b;
}

test("the card offers the three goals, the size in GB and says Auto-cut never takes passes out", () => {
  const b = open();
  b.run(`ac.plan = ${JSON.stringify(planOf())};`);
  const html = b.run("renderAuto()");
  assert.match(html, /id="ac-moe"/);
  assert.match(html, /128 experts in each of 24 layers, 8 used per token; the middle layers run 3 times/);
  assert.match(html, /name="ac-goal" value="size" checked/);                     // "a smaller file" is the default
  assert.match(html, /A smaller file<\/strong>/); assert.match(html, /Faster answers/); assert.match(html, /A better model for my project/);
  assert.match(html, /<strong>96<\/strong> of 128 experts per layer, about 12\.4 GB, 4\.6 B parameters work per token/);
  assert.match(html, /Auto-cut never takes them out/); assert.match(html, /TML-d24-varloop-research/);
  assert.match(html, /<tr class="ac-pick"><td>96 ◀/);
  assert.doesNotMatch(html, /Aim for <strong id="ac-target-n">/, "the dense size slider (billions of parameters) is replaced");
  assert.doesNotMatch(html, /passes" value|name="ac-passes"/, "there is no control for the loop passes");
});

test("the request carries the goal, the file size and the choices, and never loop passes", () => {
  const b = open();
  b.run("acDefaults()");
  let spec = JSON.parse(b.run("JSON.stringify(acSpec())"));
  assert.equal(spec.moe.goal, "size");
  assert.ok(spec.moe.target_gb > 10 && spec.moe.target_gb < 14, `the default file is about the 96-expert one: ${spec.moe.target_gb}`);
  assert.equal(spec.moe.scoring, "reap"); assert.equal(spec.moe.rebalance, true); assert.equal(spec.keep_thinking, true);
  assert.ok(spec.target_b >= 1, "the dense field stays a valid number");
  assert.ok(!JSON.stringify(spec).includes("passes"));
  b.run(`ac.moe.goal = "speed"; ac.moe.topK = "6"; ac.moe.gb = 9;`);
  spec = JSON.parse(b.run("JSON.stringify(acSpec())"));
  assert.deepEqual([spec.moe.goal, spec.moe.top_k, spec.moe.target_gb], ["speed", 6, 9]);
});

test("choosing a goal re-plans, and the quality goal says where to go", async () => {
  const b = open({ auto_plan: planOf() });
  b.run("acDefaults()");
  b.run("wireAcMoe()");
  b.run(`ac.moe.goal = "quality";`);
  const html = b.run("renderAuto()");
  assert.match(html, /id="ac-moe-quality"/); assert.match(html, /Train page/);
  assert.doesNotMatch(html, /id="ac-moe-gb"/);
});

test("a dense model's Auto page is the one it always was", () => {
  const b = makeBrowser({ state: baseState });
  b.run(`state = ${JSON.stringify(baseState())}; page = "auto"; ac.loaded = true; rs.state = ${JSON.stringify({ ...pageState(), models: [{ id: "qwen3.8-27b", name: "Qwen3.8 27B", params_b: 27.8 }] })}; rs.model = "qwen3.8-27b"; rs.options = { inspect: { supported: true, model_type: "qwen3_5", total_params: 27.78e9, layers: 64 }, options: [] };`);
  const html = b.run("renderAuto()");
  assert.doesNotMatch(html, /id="ac-moe"/); assert.match(html, /Aim for <strong id="ac-target-n">/);
  assert.equal(b.run("acMoeOn()"), false);
  assert.ok(!("moe" in JSON.parse(b.run("JSON.stringify(acSpec())"))));
});
