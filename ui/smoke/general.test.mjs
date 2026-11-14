// Smoke test of the general-text chooser (Heal card and importance-matrix box share it) without a window: app.js runs in a stand-in browser.
import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const source = fs.readFileSync(path.join(here, "..", "app.js"), "utf8");

function browser(answers = {}) {
  const els = {};
  const el = (id) => (els[id] ??= { id, innerHTML: "", textContent: "", checked: false, value: "", classList: { toggle() {} }, querySelectorAll: () => [], onclick: null, onchange: null, oninput: null, onkeydown: null, scrollTop: 0 });
  const calls = [];
  const invoke = async (cmd, args) => { calls.push([cmd, JSON.parse(JSON.stringify(args ?? {}))]); if (cmd in answers) { const a = answers[cmd]; return typeof a === "function" ? a(args) : a; } if (cmd === "get_state") return { project: { name: "P", folder: "/p" }, steps: [], settings: { theme: "light" }, bundle: null, worlds: [], recent: [] }; return cmd === "help" ? "<p>help</p>" : null; };
  const win = { __TAURI__: { core: { invoke }, dialog: {}, event: { listen: async () => {} }, webview: { getCurrentWebview: () => ({ onDragDropEvent: async () => {} }) } }, matchMedia: () => ({ matches: false }) };
  const doc = { getElementById: el, querySelectorAll: () => [], querySelector: () => null, createElement: () => el("tmp"), body: { appendChild() {} }, documentElement: { setAttribute() {}, removeAttribute() {} } };
  const ctx = vm.createContext({ window: win, document: doc, console, confirm: () => true, setTimeout, clearTimeout });
  vm.runInContext(source, ctx);
  return { els, calls, run: (code) => vm.runInContext(code, ctx) };
}

const P = (key, over = {}) => ({ key, id: key.split("#")[0], name: key, license: "MIT", license_note: "", kind: "chat", note: "A note.", config: null, split: "train", gated: false, needs_ack: false, url: "https://huggingface.co/datasets/" + key.split("#")[0], ...over });
const MULTI = "r0b0tlab/qwen3.8-max-glm5.2-kimi-k3-distillation#sft_balanced";
const genState = (over = {}) => ({
  default: "mlabonne/open-perfectblend", has_token: false, acknowledged: [], custom: [], choice: {},
  presets: [P("mlabonne/open-perfectblend", { name: "Open PerfectBlend (recommended)", license: "Apache-2.0" }),
    P("HuggingFaceH4/ultrachat_200k", { split: "train_sft" }),
    P("HuggingFaceFW/fineweb-edu#sample-10BT", { id: "HuggingFaceFW/fineweb-edu", kind: "text", config: "sample-10BT", license: "ODC-By" }),
    P(MULTI, { id: MULTI.split("#")[0], config: "sft_balanced", license: "other (noncommercial research)", license_note: "Noncommercial research use; mixed licences; teacher-service terms. Read the dataset's LICENSE.", needs_ack: true }),
    P("nvidia/Nemotron-Pretraining-SFT-v1#Nemotron-SFT-General", { id: "nvidia/Nemotron-Pretraining-SFT-v1", gated: true, license: "other (NVIDIA)", config: "Nemotron-SFT-General" })],
  ...over,
});
const use = (b, st) => { b.run(`state = {project: {name: "P", folder: "/p"}}; gcUse(${JSON.stringify(st)});`); };
const draw = (b, scope = "heal") => { const h = b.run(`gcHtml(${JSON.stringify(scope)})`); b.run(`gcWire(${JSON.stringify(scope)}, () => {})`); return h; };

test("the list shows name, licence, kind and a gated mark; the default is chosen; My own... is last", () => {
  const b = browser();
  use(b, genState());
  const html = draw(b);
  assert.match(html, /<option value="mlabonne\/open-perfectblend" selected>Open PerfectBlend \(recommended\) · Apache-2.0 · chat<\/option>/);
  assert.match(html, /fineweb-edu#sample-10BT" >HuggingFaceFW\/fineweb-edu#sample-10BT · ODC-By · text</);
  assert.match(html, /Nemotron-SFT-General · other \(NVIDIA\) · chat · needs access \+ your token/);
  assert.match(html, /<option value="own" >My own…<\/option><\/select>/);
  assert.match(html, /<strong>Licence:<\/strong> Apache-2\.0/);
  assert.match(html, /id="gc-heal-check"/);
  assert.doesNotMatch(html, /gc-heal-mode/);
});

test("choosing a preset saves it as the project's choice with its own config and split; the last choice comes back", async () => {
  const b = browser();
  use(b, genState());
  draw(b);
  b.els["gc-heal-sel"].value = "HuggingFaceH4/ultrachat_200k"; b.els["gc-heal-sel"].onchange();
  assert.deepEqual(b.calls.at(-1), ["general_choose", { scope: "heal", source: { id: "HuggingFaceH4/ultrachat_200k", config: null, split: "train_sft", column: null } }]);
  assert.match(b.run(`gcHtml("heal")`), /ultrachat_200k" selected/);
  b.els["gc-heal-sel"].value = "HuggingFaceFW/fineweb-edu#sample-10BT"; b.els["gc-heal-sel"].onchange();
  assert.deepEqual(b.calls.at(-1)[1].source, { id: "HuggingFaceFW/fineweb-edu", config: "sample-10BT", split: "train", column: null });
  // the other place has its own choice; a new project starts from what was remembered
  const b2 = browser();
  use(b2, genState({ choice: { heal: { id: "HuggingFaceH4/ultrachat_200k", split: "train_sft" }, imatrix: { id: "HuggingFaceFW/fineweb-edu", config: "sample-10BT" } } }));
  assert.match(b2.run(`gcHtml("heal")`), /ultrachat_200k" selected/);
  assert.match(b2.run(`gcHtml("imatrix")`), /fineweb-edu#sample-10BT" selected/);
  assert.equal(b2.run(`gcSel("heal")`), "HuggingFaceH4/ultrachat_200k");
});

test("a dataset that needs an acknowledgement shows its licence note and cannot be used until the box is ticked", async () => {
  const acked = genState({ acknowledged: [MULTI] });
  const b = browser({ general_ack: () => acked });
  use(b, genState());
  draw(b);
  b.els["gc-heal-sel"].value = MULTI; b.els["gc-heal-sel"].onchange();
  const html = b.run(`gcHtml("heal")`);
  assert.match(html, /Noncommercial research use; mixed licences; teacher-service terms/);
  assert.match(html, /id="gc-heal-ack" >/);                                        // not ticked
  assert.match(b.run(`gcBlocked("heal")`), /Tick the box/);
  b.run(`gcWire("heal", () => {})`);
  b.els["gc-heal-ack"].checked = true; await b.els["gc-heal-ack"].onchange();
  assert.deepEqual(b.calls.at(-1), ["general_ack", { key: MULTI }]);
  assert.equal(b.run(`gcBlocked("heal")`), "");
  assert.match(b.run(`gcHtml("heal")`), /id="gc-heal-ack" checked>/);
});

test("My own... opens the fields; Check it shows what was found; failures get their own words", async () => {
  const answers = { general_check: { verdict: "works", message: "This works: column 'conversations' as chat (from/value messages); 20 of 20 rows read had usable text.", detected: { kind: "conversations", column: "conversations" }, sample: ["Hello\nHi there"], license: "apache-2.0", gated: false } };
  const b = browser(answers);
  use(b, genState());
  draw(b);
  b.els["gc-heal-sel"].value = "own"; b.els["gc-heal-sel"].onchange();
  let html = draw(b);
  assert.match(html, /id="gc-heal-id"/); assert.match(html, /id="gc-heal-config"/); assert.match(html, /id="gc-heal-split"/); assert.match(html, /id="gc-heal-column"/);
  assert.match(html, /The licence of a dataset you add is yours to check/);
  assert.match(b.run(`gcBlocked("heal")`), /Fill in the dataset/);
  // the check needs something to read
  await b.els["gc-heal-check"].onclick();
  assert.match(b.run(`gcScope("heal").error`), /Fill in the dataset or the file first/);
  b.els["gc-heal-id"].value = "acme/stories"; b.els["gc-heal-id"].onchange();
  b.els["gc-heal-split"].value = "dev"; b.els["gc-heal-split"].onchange();
  b.els["gc-heal-column"].value = "conversations"; b.els["gc-heal-column"].onchange();
  assert.equal(b.run(`gcBlocked("heal")`), "");
  await b.els["gc-heal-check"].onclick();
  assert.deepEqual(b.calls.filter((c) => c[0] === "general_check").at(-1)[1], { source: { id: "acme/stories", config: null, split: "dev", column: "conversations" } });
  html = b.run(`gcHtml("heal")`);
  assert.match(html, /<strong>Works\.<\/strong> This works: column &#39;conversations&#39; as chat/);
  assert.match(html, /Hello\nHi there/); assert.match(html, /Licence on the Hub: apache-2\.0/);
  // each failure says what to do
  for (const [verdict, head] of [["needs_access", "Needs access."], ["not_found", "Not found."], ["no_column", "No usable column."], ["offline", "Offline."]]) {
    answers.general_check = { verdict, message: "plain words for " + verdict, sample: [] };
    await b.els["gc-heal-check"].onclick();
    html = b.run(`gcHtml("heal")`);
    assert.ok(html.includes(`<strong>${head}</strong> plain words for ${verdict}`), verdict);
    assert.equal(html.includes("<details open><summary>Hugging Face token"), verdict === "needs_access", verdict);   // "needs access" opens the token box
  }
  // a file on this computer
  b.els["gc-heal-mode"].value = "file"; b.els["gc-heal-mode"].onchange();
  b.run(`gcWire("heal", () => {})`);
  assert.match(b.run(`gcHtml("heal")`), /id="gc-heal-file"/);
  b.els["gc-heal-file"].value = "C:\\texts\\a.jsonl"; b.els["gc-heal-file"].onchange();
  assert.deepEqual(JSON.parse(JSON.stringify(b.run(`gcSource("heal")`))), { file: "C:\\texts\\a.jsonl", column: "conversations" });
  // a problem of the check itself (the training software is missing) is shown, not lost
  const b2 = browser({ general_check: () => { throw "The training software is not installed yet."; } });
  use(b2, genState()); draw(b2);
  b2.els["gc-heal-sel"].value = "own"; b2.els["gc-heal-sel"].onchange(); draw(b2);
  b2.els["gc-heal-id"].value = "a/b"; b2.els["gc-heal-id"].onchange();
  await b2.els["gc-heal-check"].onclick();
  assert.match(b2.run(`gcHtml("heal")`), /The training software is not installed yet/);
});

test("Add to my list saves it and selects it; Remove takes it out again", async () => {
  const withOwn = genState({ custom: [{ id: "acme/stories", name: "acme/stories", split: "dev" }], added: "acme/stories" });
  const b = browser({ general_add: withOwn, general_remove: genState() });
  use(b, genState());
  draw(b);
  b.els["gc-heal-sel"].value = "own"; b.els["gc-heal-sel"].onchange(); draw(b);
  b.els["gc-heal-id"].value = "acme/stories"; b.els["gc-heal-id"].onchange();
  await b.els["gc-heal-add"].onclick();
  assert.deepEqual(b.calls.find((c) => c[0] === "general_add")[1], { source: { id: "acme/stories", config: null, split: null, column: null } });
  let html = b.run(`gcHtml("heal")`);
  assert.match(html, /<option value="acme\/stories" selected>acme\/stories · licence: yours to check · \?<\/option>/);
  assert.match(html, /Added by you \(acme\/stories\); the licence is yours to check/);
  assert.match(html, /id="gc-heal-remove"/);
  b.run(`gcWire("heal", () => {})`);
  await b.els["gc-heal-remove"].onclick();
  assert.deepEqual(b.calls.find((c) => c[0] === "general_remove")[1], { key: "acme/stories" });
  html = b.run(`gcHtml("heal")`);
  assert.doesNotMatch(html, /acme\/stories/);
  assert.match(html, /open-perfectblend" selected/);
});

test("the Hugging Face token is sent once and never shown again", async () => {
  const tok = "hf_" + "a1B2".repeat(8);
  const b = browser({ hf_token_set: () => genState({ has_token: true }), hf_token_clear: () => genState() });
  use(b, genState());
  let html = draw(b);
  assert.match(html, /Hugging Face token \(none\)/); assert.match(html, /environment variable/); assert.doesNotMatch(html, /Remove token/);
  b.els["gc-token"].value = tok; b.els["gc-token"].onchange();
  await b.els["gc-token-save"].onclick();
  assert.deepEqual(b.calls.find((c) => c[0] === "hf_token_set")[1], { token: tok });
  html = b.run(`gcHtml("heal")`);
  assert.match(html, /Hugging Face token \(saved\)/); assert.match(html, /Remove token/); assert.ok(!html.includes(tok) && !html.includes("a1B2"));
  b.run(`gcWire("heal", () => {})`);
  await b.els["gc-token-clear"].onclick();
  assert.ok(b.calls.some((c) => c[0] === "hf_token_clear"));
  // a gated dataset opens the token box and says it needs approval
  b.els["gc-heal-sel"].value = "nvidia/Nemotron-Pretraining-SFT-v1#Nemotron-SFT-General"; b.els["gc-heal-sel"].onchange();
  html = b.run(`gcHtml("heal")`);
  assert.match(html, /needs access approval on Hugging Face/); assert.match(html, /<details open>/);
});

test("the importance-matrix box uses the same chooser and sends the choice with Make the files", async () => {
  const est = (c) => ({ chunks: c, ctx: 512, tokens: c * 512, gpu_minutes: 1, cpu_minutes: 2 });
  const st = { runs: [{ name: "run1", model: "Q", status: "done", loss: 0.8 }], exports: [], servers: [], suggested_name: "w", env_ready: true, quants: [{ id: "q4_k_m", name: "Q4_K_M" }],
    imatrix: { chunks: [64, 128, 256], default_chunks: 128, ctx: 512, general_share: 0.2, no_matrix_for: ["q8_0", "f16"], saved: {}, estimates: { run1: { 64: est(64), 128: est(128), 256: est(256) } }, general: genState() } };
  const b = browser();
  b.run(`state = {project: {name: "P", folder: "/p"}, steps: [], bundle: {world: "W"}, settings: {theme: "light"}, worlds: [], recent: []}; page = "use"; us.state = ${JSON.stringify(st)};`);
  assert.doesNotMatch(b.run("renderUse()"), /gc-imatrix-sel/);                       // only when the matrix is ticked
  b.run(`us.im = true`);
  let html = b.run("renderUse()");
  assert.match(html, /id="gc-imatrix-sel"/); assert.match(html, /About 20 % of the calibration text is general text/);
  b.run("wireUse()");
  b.els["gc-imatrix-sel"].value = MULTI; b.els["gc-imatrix-sel"].onchange();
  html = b.run("renderUse()");
  assert.match(html, /id="us-make" disabled/); assert.match(html, /Tick the box that says you read the licence note/);
  b.run(`gc.state.acknowledged = [${JSON.stringify(MULTI)}]`);
  html = b.run("renderUse()");
  assert.match(html, /id="us-make" >/);
  b.run("wireUse()");
  await b.els["us-make"].onclick();
  const sent = b.calls.find((c) => c[0] === "export_start")[1].imatrix;
  assert.deepEqual(sent.general, { id: "r0b0tlab/qwen3.8-max-glm5.2-kimi-k3-distillation", config: "sft_balanced", split: "train", column: null });
});

test("the Heal card sends the chosen dataset and offers the default after a failure", async () => {
  const b = browser();
  use(b, genState({ choice: { heal: { id: "HuggingFaceH4/ultrachat_200k", split: "train_sft" } } }));
  b.run(`rs.hlSpec.general = true; rs.hl = { name: "m", opts: { tokens: [] } }`);
  assert.deepEqual(JSON.parse(JSON.stringify(b.run("rsHealSpec().general"))), { id: "HuggingFaceH4/ultrachat_200k", config: null, split: "train_sft", column: null });
  b.run(`rs.hlSpec.general = false`);
  assert.equal(b.run("rsHealSpec().general"), null);
});
