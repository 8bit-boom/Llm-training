// The Heal card's "heal in rounds" controls (experimental): the request, the 8-step rule, the table, the warnings and "Heal more". Run with `node --test ui/smoke`.
import test from "node:test";
import assert from "node:assert/strict";
import { makeBrowser } from "./_browser.mjs";

const baseState = () => ({ project: { name: "P", folder: "/p" }, steps: [], bundle: { world: "W" }, settings: { theme: "light", expert_mode: false, show_help_panel: true }, worlds: [], recent: [] });
const inspectOf = () => ({ model_type: "qwen3_5", supported: true, why: null, layers: 64, layer_types: Array(64).fill("x"), period: 4, blocks: 16, total_params: 27.8e9, has_vision: true, local: true, source_bytes: 0 });
const model = (over = {}) => ({ name: "slim-w", status: "done", model_id: "custom-w", size_gb: 43, tested: false, test: null, healed: false, healed_model_id: null, heal: null, test_healed: null, cache_gb: 0, healed_size_gb: 0,
  summary: { mode: "prune", shape: "width", layers_before: 64, layers_after: 64, dropped: [], duplicated: [], params_before: 27.8e9, params_after: 21.1e9, loss_before: 2.0, loss_after: 3.1, stable: null, chosen_blocks: [], boundary_distance: null, notes: [], load: "4bit", width: { before: { intermediate_size: 17408 }, after: { intermediate_size: 10752 }, stable_overlap: 0.93 } }, width_scores: null, ...over });
const opts = (over = {}) => ({ name: "slim-w", plan: { mode: "cache", student_method: "qlora", est_gb: 18.7, fits: true, why: null }, defaults: { seq: 8192, batch: 1, grad_accum: 32, checkpointing: true, rank: 32, kl_chunk: 128, qlora: true }, safer: null, default_tokens_m: 40, general_dataset: "x/y", has_data: true, vram_gb: 32, healed: false, heal: null, metrics: [], loss: { original: 2.0, thinned: 3.1 },
  recipe: { id: "thinking", tokens_m: 40, general_share: 0.5, general: "x/traces", seq: 8192, rank: 32, default_on: true },
  tokens: [10, 20, 40, 80].map((t) => ({ tokens_m: t, cache_gb: t / 5, hours: [t / 3, t / 2] })), ...over });
const est = { plan: { mode: "cache", student_method: "qlora", est_gb: 18.7, fits: true }, knobs: { seq: 8192, batch: 1, grad_accum: 32, checkpointing: true, rank: 32, kl_chunk: 128, qlora: true },
  rounds: { steps_per_round: 38.1, min_steps: 8, too_few_steps: false, warn_overhead: false, normal_heal: [2400, 4000], time: { round_m: 5, first_check_m: 10, cap_m: 40, rounds: [2, 8], per_round: [250, 400], overhead_share: [0.08, 0.15], best: [600, 1000], worst: [3000, 4600] } } };

function open(answers = {}, hl = opts()) {
  const b = makeBrowser({ answers, state: baseState });
  const m = model();
  b.run(`state = ${JSON.stringify(baseState())}; page = "reshape"; rs.state = ${JSON.stringify({ models: [{ id: "qwen3.5-27b", name: "Q", params_b: 27.8, custom: false }], made: [m], dataset_version: 3, env_ready: true, vram_gb: 32, free_disk_gb: 500 })}; rs.model = "qwen3.5-27b"; rs.mode = "prune"; rs.options = ${JSON.stringify({ model: { id: "qwen3.5-27b" }, mode: "prune", inspect: inspectOf(), options: [] })}; rs.blocks = 4;`);
  b.run(`rs.hl = { name: ${JSON.stringify(m.name)}, opts: ${JSON.stringify(hl)} }; rs.hlEst = { name: ${JSON.stringify(m.name)}, key: "", est: ${JSON.stringify(est)} };`);
  return b;
}

test("rounds are off by default and the request has no rounds key", () => {
  const b = open();
  const html = b.run("renderReshape()");
  assert.match(html, /id="hl-rounds"/);
  assert.match(html, /Heal in rounds and check after each \(stop early when it is good enough\) — experimental/);
  assert.doesNotMatch(html, /id="hl-round"/);
  assert.equal("rounds" in JSON.parse(JSON.stringify(b.run("rsHealSpec()"))), false);
});

test("ticking it shows the controls, the estimate line and the three warnings verbatim; a 1 M round is disabled with its reason", () => {
  const b = open();
  b.run("rs.hlSpec.rounds = true");
  const html = b.run("renderReshape()");
  assert.match(html, /At most <select id="hl-tokens"/);
  assert.match(html, /<option value="5" selected >5 million \(recommended\)<\/option>/);
  assert.match(html, /<option value="1" +disabled>1 million — only about 7 steps, at least 8 are needed<\/option>/);       // 8192 x 32 per update: 7.6 steps
  assert.match(html, /<option value="2" +>2 million<\/option>/);
  assert.match(html, /About 38 training steps per round;/);
  assert.match(html, /If the first check passes: about 10-17 h\. If it runs to the most tokens: about 50-77 h \(a normal heal: 40-67 h\)/);
  assert.match(html, /id="hl-firstcheck"/);
  assert.match(html, /Quick \(16 questions, about 10-15 min\)/);
  assert.match(html, /Never stop early \(record the checks; to compare with a normal heal\)/);
  assert.ok(html.includes("A Quick check asks 16 questions: one or two answers decide pass or fail. That is why the merged model is checked again at the end; if it fails there, healing goes on and then needs two passed checks in a row."));
  assert.ok(html.includes("The checks during healing look at the model as it is trained (4-bit with the learning on top); the merged model can score a little differently. The merged check decides."));
  assert.ok(html.includes("Experimental: whether healing in rounds ends as good as one long heal is not known yet."));
  assert.match(html, /The original&#39;s answers are always saved first|The original's answers are always saved first/);
});

test("a student that does not think gets the loss wording and no check controls", () => {
  const b = open({}, opts({ recipe: null }));
  b.run("rs.hlSpec.rounds = true");
  const html = b.run("renderReshape()");
  assert.match(html, /Heal in rounds and stop when the error stops falling — experimental/);
  assert.doesNotMatch(html, /id="hl-roundcheck"|id="hl-quick-noise"/);
});

test("the request carries the rounds with the cap, the first check and the stop rule", async () => {
  const sent = [];
  const b = open({ heal_start: (a) => { sent.push(a); return null; } });
  b.run("rs.hlSpec.rounds = true; rs.hlSpec.roundM = 10; rs.hlSpec.firstCheck = 20; rs.hlSpec.roundCheck = 'full'; rs.hlSpec.stopRule = 'never'; renderReshape(); wireReshape();");
  await b.els["hl-go"].onclick();
  assert.deepEqual(JSON.parse(JSON.stringify(sent[0].spec.rounds)), { round_m: 10, max_tokens_m: 40, first_check_m: 20, check: "full", stop: "never" });
  assert.equal(sent[0].spec.tokens_m, 40);
  // the cap follows the choice of the tokens and a round never exceeds it
  b.run("rs.hlSpec.tokens_m = 10; rs.hlSpec.roundM = 20");
  const r = JSON.parse(JSON.stringify(b.run("rsHealSpec().rounds")));
  assert.deepEqual([r.max_tokens_m, r.round_m, r.first_check_m], [10, 10, 10]);
  // a change of the controls asks the program for a new estimate
  const calls = [];
  const c = open({ heal_estimate: (a) => { calls.push(a); return est; } });
  c.run("renderReshape(); wireReshape();");
  c.els["hl-rounds"].checked = true;
  c.els["hl-rounds"].onchange();
  assert.equal(calls.length, 1);
  assert.equal(JSON.parse(JSON.stringify(calls[0].spec)).rounds.round_m, 5);
});

test("the table fills from heal.json and from live heal_round events", () => {
  const rows = [{ n: 1, tokens_m: 5, calib_loss: 2.5, decision: "continue", cache_s: 600, train_s: 1200, check_s: 0, check: null },
                { n: 2, tokens_m: 10, calib_loss: 2.2, decision: "continue", cache_s: 600, train_s: 1200, check_s: 600, check: { mode: "quick", verdict: "fail", score: 0.56, main: { closed_rate: 0.69, accuracy: 0.5, loop_rate: 0.13 } } }];
  const b = open({}, opts({ heal: { status: "running", rounds: { state: "rounds", rows, merged_checks: [] } } }));
  b.run("rs.hlSpec.rounds = true");
  let html = b.run("renderReshape()");
  assert.match(html, /id="hl-rounds-table"/);
  assert.match(html, /<td>2<\/td><td>10<\/td><td>69 %<\/td><td>50 %<\/td><td>13 %<\/td><td>0.56<\/td><td>2.200<\/td><td>goes on<\/td><td>40 min<\/td>/);
  b.run(`rs.hlRun = { running: true, step: "train", text: "", progress: null, live: [] }`);
  b.run(`onHealEvent({ kind: "heal_round", phase: "heal", round: 3, tokens_m: 15, calib_loss: 2.0, decision: "stop_pass", check: { mode: "quick", verdict: "pass", score: 0.95, main: { closed_rate: 0.97, accuracy: 0.8, loop_rate: 0 } }, note: "Round 3 passed" })`);
  html = b.run("renderReshape()");
  assert.match(html, /<td>3<\/td><td>15<\/td><td>97 %<\/td>.*<td>passed: stop<\/td>/);
});

test("after the most tokens without a pass the card offers 10 M more on the same run", async () => {
  const sent = [];
  const rounds = { state: "ended", cap_m: 40, spec: { round_m: 5, check: "quick" }, rows: [{ n: 8, tokens_m: 40, decision: "stop_cap", check: null }], final: { verdict: "fail", mode: "quick" }, merged_checks: [] };
  const b = open({ heal_start: (a) => { sent.push(a); return null; } }, opts({ healed: true, heal: { status: "done", rounds } }));
  b.run(`rs.state.made[0].healed = true; rs.state.made[0].healed_model_id = "custom-w-healed"; renderReshape(); wireReshape();`);
  const html = b.run("renderReshape()");
  assert.match(html, /did not pass its thinking check after 40 million tokens/);
  assert.match(html, /id="hl-more"[^>]*>Heal 10 M more/);
  b.run("wireReshape()");
  await b.els["hl-more"].onclick();
  const r = JSON.parse(JSON.stringify(sent[0].spec.rounds));
  assert.deepEqual([r.max_tokens_m, r.extend, r.round_m], [50, true, 5]);
  // a pass offers nothing
  const c = open({}, opts({ healed: true, heal: { status: "done", rounds: { ...rounds, final: { verdict: "pass" } } } }));
  c.run(`rs.state.made[0].healed = true`);
  assert.doesNotMatch(c.run("renderReshape()"), /id="hl-more"/);
});
