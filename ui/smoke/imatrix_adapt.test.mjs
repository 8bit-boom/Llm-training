// Smoke test of "Adapt to my cut model" (the Adapt card, the weights of the Merge card, the Create tick for a cut model), without a window: app.js runs in a
// stand-in browser that records HTML and every call to the program.
import test from "node:test";
import assert from "node:assert/strict";
import { makeBrowser, source } from "./_browser.mjs";


function browser(answers = {}, dialogs = []) {
  return makeBrowser({ answers, dialogs, cloneArgs: true, state: null });
}

const est = (c) => ({ chunks: c, ctx: 512, tokens: c * 512, gpu_minutes: c / 32, cpu_minutes: c });
const P = (key, over = {}) => ({ key, id: key, name: key, license: "Apache-2.0", license_note: "", kind: "chat", note: "A note.", config: null, split: "train", gated: false, needs_ack: false, url: "", ...over });
const general = () => ({ default: "mlabonne/open-perfectblend", has_token: false, acknowledged: [], custom: [], choice: {}, presets: [P("mlabonne/open-perfectblend", { name: "Open PerfectBlend (recommended)" })] });
const entry = (over = {}) => ({ id: "20261101-1000-run1", label: "my first", model: "run1", size_mb: 12.5, chunks: 64, ctx: 512, general_used: true, general_source: "mlabonne/open-perfectblend", nextn: false, nextn_asked: false,
  created: "2026-11-01T10:00:00Z", via: "llama.cpp", has_gguf: true, has_dat: false, in_use: false, source: "the trained run run1", calibration: "your saved data (version 1)", notes: [], ...over });
const cutModel = (over = {}) => ({ name: "slim16b", origin_repo: "Qwen/Qwen3.8-27B", layers_before: 64, layers_after: 32, intermediate_before: 17408, intermediate_after: 10752, thinner: true, healed: true, refusal: null, refusal_code: null, needs_layer_map: false, kept: [], ...over });
const libState = (over = {}) => ({ entries: [], exports: [], runs: [{ name: "run1", model: "Qwen3.5 9B", status: "done", loss: 0.8, base: "Qwen/Qwen3.8-27B", cut: false }],
  models: [{ id: "qwen3.5-4b", name: "Qwen3.5 4B", repo: "Qwen/Qwen3.5-4B", family: "qwen3_5", params_b: 4, reshape: null, cut_origin: null }, { id: "custom-local-slim16b", name: "Qwen3.8 27B · slim 32 layers", repo: "/p/models/slim16b/model", family: "qwen3_5", params_b: 16, reshape: "slimmed", cut_origin: "Qwen/Qwen3.8-27B" }],
  cut_models: [cutModel()], published_bases: [{ repo: "Qwen/Qwen3.8-27B", label: "Qwen/Qwen3.8-27B (the original of the cut model slim16b)" }, { repo: "Qwen/Qwen3.5-4B", label: "Qwen/Qwen3.5-4B" }],
  data_version: 3, chunks: [64, 128, 256], default_chunks: 128, ctx: 512, general_share: 0.2, estimates: { default: { 64: est(64), 128: est(128), 256: est(256) } }, general: general(), env_ready: true, ...over });
const PREVIEW = { ok: true, model: "slim16b", layers_before: 64, layers_after: 32, ffn_down_before: 17408, ffn_down_after: 10752, exact: false, size_after_mb: 5.7, text: "keeps 32 of 64 layers; ffn_down 17,408 -> 10,752 per layer; about 5.7 MB", notes: [] };

const open = (b, st, extra = "") => { b.run(`state = {project: {name: "P", folder: "/p"}, steps: [], bundle: {world: "W"}, settings: {theme: "light"}, worlds: [], recent: []}; page = "imatrix"; im.state = ${JSON.stringify(st)}; ${extra}`); const h = b.run("renderImatrix()"); b.run("wireImatrix()"); return h; };
const redraw = (b) => { const h = b.run("renderImatrix()"); b.run("wireImatrix()"); return h; };
const J = (b, code) => JSON.parse(b.run(`JSON.stringify(${code})`));
// the stand-in app starts itself asynchronously (it asks for its state and shows the welcome page): after waiting, the matrix page is put back
const settle = async (b) => { await new Promise((r) => setTimeout(r, 10)); b.run(`state = {project: {name: "P", folder: "/p"}, steps: [], bundle: {world: "W"}, settings: {theme: "light"}, worlds: [], recent: []}; page = "imatrix"`); };
const previews = (b) => b.calls.filter((c) => c[0] === "imatrix_adapt_preview");

test("the Adapt card says honestly what it is, and explains the weight with the real numbers", () => {
  const html = open(browser(), libState());
  assert.match(html, /<h2>Adapt to my cut model<\/h2>/); assert.match(html, /without measuring anything/);
  assert.match(html, /still describes the <em>unmodified<\/em> model's activations/); assert.match(html, /<em>approximation<\/em>/); assert.match(html, /best merged with a matrix of your own, measured on the healed model/);
  assert.match(html, /<strong>not Unsloth's Dynamic recipe<\/strong>/); assert.match(html, /Compare quants/);
  assert.match(html, /1,149,614 rows of text/); assert.match(html, /about 65,000/); assert.match(html, /outweighs yours about 17 to 1/); assert.match(html, /multiplies the sums <em>and<\/em> the counts/); assert.match(html, /0\.05 makes the two about even/);
  assert.match(html, /id="im-adapt-go"/);
});

test("without a cut model the card says to make one first and cannot start", () => {
  const html = open(browser(), libState({ cut_models: [] }));
  assert.match(html, /id="im-ad-nomodel"/); assert.match(html, /Slim or grow a model/); assert.match(html, /id="im-adapt-go" disabled/); assert.match(html, /Make a cut model first/);
});

test("the preview shows the numbers of the model, and Adapt sends the choice", async () => {
  const b = browser({ imatrix_adapt_preview: PREVIEW, imatrix_list: libState() });
  let html = open(b, libState());
  assert.match(html, /My cut model <select id="im-ad-model">/); assert.match(html, /slim16b · 32 of 64 layers, MLP 17,408 → 10,752 · healed/);
  await settle(b);
  assert.deepEqual(previews(b)[0], ["imatrix_adapt_preview", { spec: { source: { kind: "published", base: "Qwen/Qwen3.8-27B" }, model: "slim16b", weight: null, label: "", kept_layers: null, source_size: null } }], "the original of the cut model is the default base");
  html = b.run("renderImatrix()");
  assert.match(html, /id="im-ad-preview"><strong>keeps 32 of 64 layers; ffn_down 17,408 -&gt; 10,752 per layer; about 5.7 MB<\/strong> \(an estimate until the matrix file is at hand\)/);
  assert.equal(previews(b).length, 1, "drawing again does not ask again");
  b.run("wireImatrix()"); await settle(b);
  assert.equal(previews(b).length, 1);
  b.els["im-ad-weight"].value = "0.1"; b.els["im-ad-weight"].onchange(); await settle(b);
  assert.equal(previews(b).at(-1)[1].spec.weight, 0.1);
  b.els["im-ad-label"].value = "Unsloth's, cut"; b.els["im-ad-label"].onchange();
  b.run("wireImatrix()");
  await b.els["im-adapt-go"].onclick();
  assert.deepEqual(b.calls.at(-1), ["imatrix_adapt", { spec: { source: { kind: "published", base: "Qwen/Qwen3.8-27B" }, model: "slim16b", weight: 0.1, label: "Unsloth's, cut", kept_layers: null, source_size: null } }]);
  b.run(`onImatrixEvent({kind: "progress", step: "imatrix", value: 0.6, text: "Writing the adapted matrix"})`);
  html = b.run("renderImatrix()");
  assert.match(html, /Writing the adapted matrix/); assert.match(html, /<progress max="1" value="0.6">/); assert.equal((html.match(/id="im-stop"/g) || []).length, 1);
  b.run("wireImatrix()"); b.els["im-stop"].onclick(); assert.equal(b.calls.at(-1)[0], "imatrix_stop");
  b.run(`onImatrixEvent({kind: "done", data: {id: "a", entry: {id: "20261116-1000-slim16b-adapted", label: "Unsloth's, cut", size_mb: 5.7}, numbers: {layers_before: 64, layers_after: 32, ffn_down_before: 17408, ffn_down_after: 10752}, adapted: {weight: 0.1}, notes: ["Warning: llama-imatrix did not accept it.", "This matrix still describes the UNMODIFIED model's activations."]}})`);
  await settle(b);
  html = b.run("renderImatrix()");
  assert.match(html, /Adapted matrix ready:<\/strong> Unsloth&#39;s, cut \(5.7 MB\): keeps 32 of 64 layers; ffn_down 17,408 → 10,752 per layer; weight 0.1/); assert.match(html, /class="warn small">Warning: llama-imatrix did not accept it/); assert.match(html, /UNMODIFIED model/);
  assert.ok(b.calls.some((c) => c[0] === "imatrix_list"), "the list is refreshed");
  b.run(`im.pending = "adapt"; onImatrixEvent({kind: "error", text: "Cancelled."})`);
  assert.match(b.run("renderImatrix()"), /Cancelled\. Nothing was kept\./);
});

test("a refusal of the program is shown in its words and stops the button; a model that cannot be adapted says why", async () => {
  const text = "This matrix has data for 12 layers, but the original of the model slim16b (Qwen/Qwen3.8-27B) has 64.";
  const b = browser({ imatrix_adapt_preview: { ok: false, code: "wrong_original", text } });
  open(b, libState({ entries: [entry({ id: "e1", label: "the wrong one" })] }), `im.adSrc = "library"`);
  await settle(b);
  let html = redraw(b);
  assert.match(html, /id="im-ad-preview-refusal">This matrix has data for 12 layers/); assert.match(html, /id="im-adapt-go" disabled/);
  assert.deepEqual(previews(b)[0][1].spec.source, { kind: "library", id: "e1" });
  const heads = "The model slim16b was made thinner in more than its MLP: its attention heads were cut from 24 to 16.";
  html = open(browser({ imatrix_adapt_preview: PREVIEW }), libState({ cut_models: [cutModel({ refusal: heads, refusal_code: "heads" })] }));
  assert.match(html, /id="im-ad-refusal">The model slim16b was made thinner in more than its MLP/); assert.match(html, /id="im-adapt-go" disabled/);
  const bad = open(browser({ imatrix_adapt_preview: PREVIEW }), libState(), `im.adWeight = "0"`);
  assert.match(bad, /The weight must be a number above 0/); assert.match(bad, /id="im-adapt-go" disabled/);
});

test("a model whose record has only counts asks which layers it kept and sends the answer", async () => {
  const b = browser({ imatrix_adapt_preview: { ok: false, code: "needs_layer_map", text: "The record of the model old says that it has 8 layers of the original 16, but not which ones." } });
  let html = open(b, libState({ cut_models: [cutModel({ name: "old", needs_layer_map: true, refusal: "x", refusal_code: "needs_layer_map" })] }));
  assert.match(html, /Layers of the original it kept <input type="text" id="im-ad-kept"/); assert.match(html, /counted from 0/);
  assert.doesNotMatch(html, /id="im-ad-refusal"/, "the box, not the refusal, is how to go on");
  b.els["im-ad-kept"].value = "0, 1, 2 3;12"; b.els["im-ad-kept"].onchange(); await settle(b);
  assert.deepEqual(previews(b).at(-1)[1].spec.kept_layers, [0, 1, 2, 3, 12]);
});

test("the matrix can come from the list (adapted ones are not offered), from a file, or be the published one that is looked up first", async () => {
  const b = browser({ imatrix_adapt_preview: PREVIEW, imatrix_published: { base: "Qwen/Qwen3.8-27B", repo: "unsloth/Qwen3.8-27B-GGUF", file: "imatrix_unsloth.gguf", size: 13642656, page: "https://huggingface.co/unsloth/Qwen3.8-27B-GGUF", gated: false, cached: false } }, ["/f/pub.gguf"]);
  let html = open(b, libState({ entries: [entry({ id: "e1", label: "original" }), entry({ id: "e2", label: "already", source_kind: "adapted" }), entry({ id: "e3", label: "dat only", has_gguf: false })] }), `im.adSrc = "library"`);
  assert.match(html, /<option value="e1" selected>original/); assert.doesNotMatch(html, /<option value="e2"/); assert.doesNotMatch(html, /<option value="e3"/);
  b.run(`im.adSrc = "file"`); html = redraw(b);
  assert.match(html, /id="im-ad-file"/);
  await b.els["im-ad-filebrowse"].onclick();
  assert.equal(J(b, "im.adFile"), "/f/pub.gguf");
  await settle(b);
  assert.deepEqual(previews(b).at(-1)[1].spec.source, { kind: "file", file: "/f/pub.gguf" });
  b.run(`im.adSrc = "published"`); redraw(b);
  await b.els["im-ad-lookup"].onclick();
  html = b.run("renderImatrix()");
  assert.match(html, /id="im-ad-pub-info"><strong>unsloth\/Qwen3.8-27B-GGUF<\/strong> · 13.0 MB/); assert.match(html, /Adapting downloads 13.0 MB from huggingface.co/);
  b.run("wireImatrix()"); await settle(b);
  assert.equal(previews(b).at(-1)[1].spec.source_size, 13642656, "the size the Hub states is sent for the estimate");
});

test("an adapted entry says what it was adapted to", () => {
  const e = entry({ id: "a1", label: "cut", source_kind: "adapted", adapt: { model: "slim16b", layers_before: 64, layers_after: 32, ffn_down_before: 17408, ffn_down_after: 10752, weight: 0.1 }, parents: [{ kind: "published", label: "Unsloth's published matrix for Qwen/Qwen3.8-27B", repo: "unsloth/Qwen3.8-27B-GGUF", chunks: 1251 }] });
  const html = open(browser(), libState({ entries: [e] }));
  assert.match(html, /<span class="tag">adapted<\/span>/); assert.match(html, /Adapted to slim16b: keeps 32 of 64 layers; ffn_down 17,408 → 10,752 per layer; weight 0.1\. It still describes the unmodified model's activations\./);
  assert.match(html, /Parents: Unsloth&#39;s published matrix for Qwen\/Qwen3.8-27B \(1251 chunks\) · from unsloth\/Qwen3.8-27B-GGUF/);
});

// ------------------------------------------------------------------------------------------------------------------ the Merge card

test("the Merge card has weights, sent only when they are not 1, and says which weights were used", async () => {
  const st = libState({ entries: [entry({ id: "a", label: "mine" }), entry({ id: "b", label: "second" })] });
  const b = browser({ imatrix_list: st });
  open(b, st, `im.mergePicked = ["a", "b"]; im.mergePub = true; im.mergePubBase = "Qwen/Qwen3.8-27B"`);
  let html = redraw(b);
  assert.match(html, /<summary>Weights \(optional\)<\/summary>/); assert.match(html, /about 1,149,614 rows of text/); assert.match(html, /outweighs yours about 17 to 1/); assert.match(html, /The result lists the weights that were used/);
  assert.equal((html.match(/class="im-mw"/g) || []).length, 3);
  assert.deepEqual(J(b, "imMergeSpec()").inputs, [{ kind: "library", id: "a" }, { kind: "library", id: "b" }, { kind: "published", base: "Qwen/Qwen3.8-27B" }], "no weights: the spec as it always was");
  b.groups[".im-mw"] = [{ dataset: { key: "pub" }, value: "0.05", onchange: null }]; redraw(b); b.groups[".im-mw"][0].onchange();
  assert.deepEqual(J(b, "imMergeSpec()").inputs[2], { kind: "published", base: "Qwen/Qwen3.8-27B", weight: 0.05 });
  html = b.run("renderImatrix()");
  assert.match(html, /<details id="im-merge-weights" open>/); assert.match(html, /data-key="pub" min="0" step="any" value="0.05"/);
  b.groups[".im-mw"] = [{ dataset: { key: "lib:a" }, value: "0", onchange: null }]; redraw(b); b.groups[".im-mw"][0].onchange();
  assert.match(b.run("renderImatrix()"), /A weight must be a number above 0/); assert.match(b.run("renderImatrix()"), /id="im-merge-go" disabled/);
  b.run(`im.mergeWeights = {pub: "0.05"}`); redraw(b);
  await b.els["im-merge-go"].onclick();
  assert.deepEqual(b.calls.at(-1)[1].spec.inputs.map((i) => i.weight), [undefined, undefined, 0.05]);
  b.run(`onImatrixEvent({kind: "done", data: {id: "m", entry: {id: "m1", label: "mine + published", size_mb: 7}, warnings: [], weights: [{label: "mine", chunks: 128, tokens: 65536, weight: 1, share: 0.53}, {label: "Unsloth's published matrix for Qwen/Qwen3.8-27B (adapted to slim16b)", chunks: 1251, tokens: 57480.7, weight: 0.05, share: 0.47}]}})`);
  await settle(b);
  html = b.run("renderImatrix()");
  assert.match(html, /mine: 65,536 rows of text · about 53 % of the merged statistics/); assert.match(html, /\(adapted to slim16b\): 57,481 rows of text \(weight 0.05 used\) · about 47 %/);
  assert.deepEqual(J(b, "im.mergeWeights"), {}, "reset after a merge");
});

test("a published matrix ticked next to a matrix of a cut model is adapted first, and the card says so", () => {
  const st = libState({ entries: [entry({ id: "own", label: "mine", cut_model: { dir: "/p/models/slim16b", name: "slim16b" } })] });
  const html = open(browser(), st, `im.mergePicked = ["own"]; im.mergePub = true; im.mergePubBase = "Qwen/Qwen3.8-27B"`);
  assert.match(html, /id="im-merge-autoadapt">The published matrix is <strong>adapted first<\/strong> to your cut model <strong>slim16b<\/strong> \(it keeps 32 of 64 layers; ffn_down 17,408 → 10,752 per layer\)/);
  assert.match(html, /It is never merged as it is/);
  const plain = open(browser(), libState({ entries: [entry({ id: "own" })] }), `im.mergePicked = ["own"]; im.mergePub = true; im.mergePubBase = "Qwen/Qwen3.8-27B"`);
  assert.doesNotMatch(plain, /im-merge-autoadapt/);
});

test("the program's refusal to merge a published matrix raw is shown as it is", () => {
  const b = browser();
  open(b, libState());
  b.run(`im.pending = "merge"; onImatrixEvent({kind: "error", text: "Unsloth's published matrix for Qwen/Qwen3.8-27B describes the original model, but mine was measured on a model with 32 layers instead of 64 (a model cut on the Slim or grow page). Adapt the published matrix to your cut model first (the button Adapt to my cut model in Importance matrices)."})`);
  assert.match(b.run("renderImatrix()"), /describes the original model, but mine was measured on a model with 32 layers instead of 64/);
});

// ------------------------------------------------------------------------------------------------------------------ the Create form

test("the Create tick for a model cut on the Slim page says that the published matrix is adapted first", async () => {
  const b = browser({ imatrix_list: libState() });
  let html = open(b, libState({ runs: [{ name: "runcut", model: "slim", status: "done", loss: 0.9, base: "Qwen/Qwen3.8-27B", cut: true }] }), `im.src = "run"; im.run = "runcut"`);
  assert.match(html, /id="im-mergepub2"[^>]*> Also merge Unsloth's published matrix for the base model \(Qwen\/Qwen3.8-27B\)/);
  b.els["im-mergepub2"].checked = true; b.els["im-mergepub2"].onchange();
  html = b.run("renderImatrix()");
  assert.match(html, /id="im-mergepub2-cut"><strong>This model was cut on the Slim page\.<\/strong> The published matrix describes the original model, so it is first adapted to your cut model/); assert.match(html, /never merged as it is/);
  b.run("wireImatrix()"); await b.els["im-create"].onclick();
  assert.equal(b.calls.at(-1)[1].spec.merge_published, true);
  // a cut model of the model list offers it too, with the original from its record; a model that matches does not get the cut text
  b.run(`im.running = false; im.src = "model"; im.model = "custom-local-slim16b"`);
  html = redraw(b);
  assert.match(html, /for the base model \(Qwen\/Qwen3.8-27B\)/); assert.match(html, /im-mergepub2-cut/);
  b.run(`im.running = false; im.model = "qwen3.5-4b"`);
  html = redraw(b);
  assert.match(html, /for the base model \(Qwen\/Qwen3.5-4B\)/); assert.doesNotMatch(html, /im-mergepub2-cut/);
});
