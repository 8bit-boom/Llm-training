// Smoke test of the "Compare quants" page (docs/QUANT_COMPARE.md) and of the "Compare these sizes" button of the export results, without a window: app.js runs in a
// stand-in browser that records HTML and every call to the program.
import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const source = fs.readFileSync(path.join(here, "..", "app.js"), "utf8");

function browser(answers = {}, dialogs = []) {
  const els = {};
  const el = (id) => (els[id] ??= { id, innerHTML: "", textContent: "", checked: false, value: "", classList: { toggle() {} }, querySelectorAll: () => [], onclick: null, onchange: null, oninput: null, onkeydown: null, scrollTop: 0 });
  const calls = [];
  const invoke = async (cmd, args) => {
    calls.push([cmd, JSON.parse(JSON.stringify(args ?? {}))]);
    if (cmd in answers) { const a = answers[cmd]; if (typeof a === "function") return a(args); if (a instanceof Error) throw a; return a; }
    if (cmd === "help") return "<p>help</p>";
    return null;
  };
  const win = { __TAURI__: { core: { invoke }, dialog: { open: async (o) => { calls.push(["dialog.open", o]); return dialogs.shift() ?? null; } }, event: { listen: async () => {} }, webview: { getCurrentWebview: () => ({ onDragDropEvent: async () => {} }) } }, matchMedia: () => ({ matches: false }) };
  const groups = {};
  const doc = { getElementById: el, querySelectorAll: (sel) => groups[sel] || [], querySelector: () => null, createElement: () => el("tmp"), body: { appendChild() {} }, documentElement: { setAttribute() {}, removeAttribute() {} } };
  let confirmed = true;
  const ctx = vm.createContext({ window: win, document: doc, console, confirm: () => confirmed, setTimeout, clearTimeout });
  vm.runInContext(source, ctx);
  return { els, calls, groups, run: (code) => vm.runInContext(code, ctx), confirm: (v) => { confirmed = v; } };
}

const cand = (over = {}) => ({ path: "/p/exports/run1-q4_k_m/model-q4_k_m.gguf", export: "run1-q4_k_m", run: "run1", quant: "q4_k_m", label: "Q4_K_M, with matrix", group: "Q4_K_M", matrix: true, mixed: null, size_mb: 4100, ...over });
const ref = (over = {}) => ({ path: "/p/exports/run1-f16/model-f16.gguf", label: "run1-f16 (16-bit, from the export)", run: "run1", size_mb: 16200, kind: "export", ...over });
const planOpts = (fits = [true, true, true, true, true, true]) => [4, 8, 16, 32, 64, 128].map((c, i) => ({ chunks: c, tokens: c * 512, scored: c * 255, bytes: c * 126e6, gb: +(c * 0.1266).toFixed(2), fits: fits[i] }));
const plan = (over = {}) => ({ vocab: 248320, vocab_assumed: false, ctx: 512, options: planOpts(), free: 80e9, usable: 72e9, default_chunks: 16, max_chunks: 500, refusal: null, ...over });
const qState = (over = {}) => ({ entries: [], candidates: { references: [ref()], quants: [cand(), cand({ path: "/p/exports/run1-q4n/model-q4_k_m.gguf", export: "run1-q4n", label: "Q4_K_M, no matrix", matrix: false, size_mb: 4100 }), cand({ path: "/p/exports/run1-q8_0/model-q8_0.gguf", export: "run1-q8_0", quant: "q8_0", label: "Q8_0", group: "Q8_0", matrix: null, size_mb: 8600 })] },
  runs: [{ name: "run1", model: "Qwen3.5 9B", status: "done" }], models: [{ id: "qwen3.5-4b", name: "Qwen3.5 4B", repo: "Qwen/Qwen3.5-4B", family: "qwen3_5", params_b: 4 }], data_version: 3, has_test_rows: true, chunks: [4, 8, 16, 32, 64, 128], default_chunks: 16, ctx: 512, free_bytes: 80e9, env_ready: true, ...over });
const row = (id, label, size, kld, over = {}) => ({ id, label, path: "/p/" + id + ".gguf", size, params: 9e9, bpw: size * 8 / 9e9, status: "done", mean_kld: kld, median_kld: kld / 3, p99: kld * 4, max_kld: kld * 10, same_top: 97 - kld * 100, ppl: 6.2, error: null, ...over });
const results = (over = {}) => ({ id: "20261115-1030-ref", state: "done", reference: { name: "ref-f16.gguf", path: "/p/ref-f16.gguf", params: 9e9 }, text: { source: "the held-out questions of the saved data version" }, settings: { chunks: 16, ctx: 512 }, key: { llama_cpp: "version: 5000 (abc)" },
  rows: [row("1", "Q8_0", 8.6e9, 0.002), row("2", "Q4_K_M, no matrix", 4.1e9, 0.04), row("3", "Q4_K_M, with matrix", 4.1e9, 0.025), row("4", "Q2_K", 2.9e9, 0.31)], pareto: ["4", "3", "1"],
  verdict: ["The smallest file is Q2_K (2.90 GB, mean KLD 0.3100).", "KL divergence is a proxy: use it with your own questions."], notes: ["a note"], ...over });

const open = (b, st, extra = "") => { b.run(`state = {project: {name: "P", folder: "/p"}, steps: [], bundle: {world: "W"}, settings: {theme: "light"}, worlds: [], recent: []}; page = "quantcheck"; qk.state = ${JSON.stringify(st)}; ${extra}`); const h = b.run("renderQuantcheck()"); b.run("wireQuantcheck()"); return h; };
const J = (b, code) => JSON.parse(b.run(`JSON.stringify(${code})`));
const redraw = (b) => { const h = b.run("renderQuantcheck()"); b.run("wireQuantcheck()"); return h; };

test("the page is in the navigation after Importance matrices, needs the world data, and says what KLD is and that it is a proxy", () => {
  const b = browser();
  b.run(`state = {project: {name: "P", folder: "/p"}, steps: [], bundle: {world: "W"}, settings: {theme: "light"}, worlds: [], recent: []}; page = "welcome"; renderSteps();`);
  const nav = b.els["steps"].innerHTML;
  assert.match(nav, /data-key="quantcheck"[^>]*>📏 Compare quants/);
  assert.ok(nav.indexOf('data-key="imatrix"') < nav.indexOf('data-key="quantcheck"'));
  assert.doesNotMatch(nav.match(/<button[^>]*data-key="quantcheck"[^>]*>/)[0], /disabled/);
  b.run(`state.bundle = null; renderSteps();`);
  assert.match(b.els["steps"].innerHTML.match(/<button[^>]*data-key="quantcheck"[^>]*>/)[0], /disabled/);
  assert.match(b.run(`renderQuantcheck()`), /Import your world data first/);
  const html = open(browser(), qState());
  assert.match(html, /KL divergence<\/strong> \(KLD\)/); assert.match(html, /KLD is a <strong>proxy<\/strong>/); assert.match(html, /does not replace trying the model on your own questions/);
  assert.match(html, /text the matrix was not measured on/); assert.match(html, /LLAMA_CPP_DIR/); assert.match(html, /No comparison saved in this project yet/);
  assert.doesNotMatch(html, /Dynamic|UD-/);
  assert.match(open(browser(), null), /Loading/);
});

test("opening the page asks for the state and, for the chosen reference, the disk need once", async () => {
  const b = browser({ quantcheck_state: qState(), quantcheck_plan: plan(), export_state: null });
  await new Promise((r) => setTimeout(r, 5));
  b.run(`state = {project: {name: "P", folder: "/p"}, steps: [], bundle: {world: "W"}, settings: {theme: "light"}, worlds: [], recent: []};`);
  await b.run(`go("quantcheck")`);
  await new Promise((r) => setTimeout(r, 20));
  assert.ok(b.calls.some((c) => c[0] === "quantcheck_state"));
  assert.deepEqual(b.calls.filter((c) => c[0] === "quantcheck_plan"), [["quantcheck_plan", { reference: "/p/exports/run1-f16/model-f16.gguf" }]]);
  assert.match(b.els["page"].innerHTML, /id="qk-plan"/); assert.match(b.els["page"].innerHTML, /does not need|temporary file/);
  b.run("render()");
  assert.equal(b.calls.filter((c) => c[0] === "quantcheck_plan").length, 1, "not again for the same reference");
});

test("the form offers the 16-bit files, the exported files with their words, the text choice and the amounts with their disk need", () => {
  const b = browser();
  const html = open(b, qState(), `qk.plan = ${JSON.stringify(plan())}; qk.planRef = "/p/exports/run1-f16/model-f16.gguf";`);
  assert.match(html, /<option value="\/p\/exports\/run1-f16\/model-f16.gguf" selected>run1-f16 \(16-bit, from the export\) · 16200 MB/);
  assert.match(html, /A 16-bit GGUF file I choose/); assert.match(html, /Make a 16-bit file with llama.cpp/);
  assert.match(html, /Q4_K_M, with matrix <span class="dim small">· 4100 MB · run1-q4_k_m/); assert.match(html, /Q4_K_M, no matrix/); assert.match(html, /Q8_0 <span/);
  assert.match(html, /Files made with and without an importance matrix, and the mixed-precision sizes, go in the same table/);
  assert.match(html, /The held-out part of my saved data/); assert.match(html, /held-out questions of data version 3/); assert.match(html, /special markers taken out/);
  assert.match(html, /<option value="16" selected>16 chunks \(2.03 GB temporary file\) · recommended/); assert.match(html, /32 chunks \(4.05 GB temporary file\)/);
  assert.match(html, /<th>Words compared<\/th>/); assert.match(html, /vocabulary has 248,320 words/); assert.match(html, /about 2 bytes per vocabulary word/);
  assert.match(html, /Keep the 16-bit model's answers afterwards/);
  assert.match(html, /id="qk-start" disabled/); assert.match(html, /Choose at least one file to compare/);
  assert.doesNotMatch(open(browser(), qState({ has_test_rows: false })), /held-out questions of data version/);
  assert.match(open(browser(), qState({ has_test_rows: false })), /holds out no test questions, so a sample of its training rows is used \(another sample than the matrix calibration's\)/);
});

test("no 16-bit file in the exports is said plainly, and a project without a data version asks for a text file", () => {
  let html = open(browser(), qState({ candidates: { references: [], quants: [] }, runs: [] }));
  assert.match(html, /No 16-bit file was found in your exports/); assert.match(html, /No exported GGUF files in this project yet/);
  assert.match(html, /<option value="__file" selected>/); assert.match(html, /Choose the 16-bit file\./);
  html = open(browser(), qState({ data_version: null }), `qk.picked = ["${cand().path}"];`);
  assert.match(html, /no saved version of the training data yet/); assert.match(html, /Choose the test text/);
});

test("ticking files and pressing Compare sends the exact choice", async () => {
  const b = browser();
  open(b, qState(), `qk.plan = ${JSON.stringify(plan())}; qk.planRef = "/p/exports/run1-f16/model-f16.gguf";`);
  b.groups[".qk-pick"] = [{ value: cand().path, checked: true, onchange: null }, { value: "/p/exports/run1-q4n/model-q4_k_m.gguf", checked: true, onchange: null }, { value: "/p/exports/run1-q8_0/model-q8_0.gguf", checked: false, onchange: null }];
  redraw(b);
  b.groups[".qk-pick"][0].onchange();
  assert.deepEqual(J(b, "qk.picked"), [cand().path, "/p/exports/run1-q4n/model-q4_k_m.gguf"]);
  let html = redraw(b);
  assert.doesNotMatch(html, /id="qk-start" disabled/);
  b.els["qk-typed"].value = "  D:\\m\\mixed.gguf "; b.els["qk-typed"].onchange(); b.els["qk-typeadd"].onclick();
  assert.deepEqual(J(b, "qk.typed"), ["D:\\m\\mixed.gguf"]); assert.match(redraw(b), /D:\\m\\mixed.gguf/);
  await b.els["qk-start"].onclick();
  const [cmd, args] = b.calls.at(-1);
  assert.equal(cmd, "quantcheck_start");
  assert.deepEqual(args.spec.reference, { kind: "file", file: "/p/exports/run1-f16/model-f16.gguf" });
  assert.deepEqual(args.spec.quants, [{ path: cand().path, label: "Q4_K_M, with matrix", group: "Q4_K_M", matrix: true, mixed: null }, { path: "/p/exports/run1-q4n/model-q4_k_m.gguf", label: "Q4_K_M, no matrix", group: "Q4_K_M", matrix: false, mixed: null },
    { path: "D:\\m\\mixed.gguf", label: "mixed.gguf", group: null, matrix: null, mixed: null }]);
  assert.deepEqual(args.spec.text, { kind: "data", run: "run1", tokenizer: null }); assert.equal(args.spec.chunks, 16); assert.equal(args.spec.keep_logits, false); assert.equal(args.spec.resume, null);
  assert.equal(b.run("qk.running"), true);
});

test("a text file, a typed model, more chunks, keeping the answers, and making the reference are all sent", async () => {
  const b = browser();
  open(b, qState(), `qk.plan = ${JSON.stringify(plan())}; qk.planRef = ""; qk.picked = ["${cand().path}"]; qk.refSel = "__make";`);
  let html = redraw(b);
  assert.match(html, /id="qk-makerun"/); assert.match(html, /merged \(a trained run\) and converted to a 16-bit GGUF file with llama.cpp's converter/); assert.match(html, /a 27 B model: 54 GB/);
  b.els["qk-makesrc"].value = "model"; b.els["qk-makesrc"].onchange(); html = redraw(b); assert.match(html, /id="qk-makemodel"/);
  b.els["qk-makemodel"].value = "qwen3.5-4b"; b.els["qk-makemodel"].onchange();
  b.els["qk-text"].value = "file"; b.els["qk-text"].onchange(); html = redraw(b);
  assert.match(html, /id="qk-textfile"/); assert.match(html, /Choose the text file\./);
  b.els["qk-textfile"].value = "/t/test.txt"; b.els["qk-textfile"].onchange();
  b.els["qk-chunks"].value = "32"; b.els["qk-chunks"].onchange();
  b.els["qk-keep"].checked = true; b.els["qk-keep"].onchange();
  redraw(b);
  await b.els["qk-start"].onclick();
  const { spec } = b.calls.at(-1)[1];
  assert.deepEqual(spec.reference, { kind: "make", source: { kind: "model", model: "qwen3.5-4b" } });
  assert.deepEqual(spec.text, { kind: "file", file: "/t/test.txt" }); assert.equal(spec.chunks, 32); assert.equal(spec.keep_logits, true);
  // a typed model for the chat template instead of a run
  b.run(`qk.running = false; qk.text = "data"; qk.tokMode = "typed"; qk.tokTyped = "owner/model"`);
  redraw(b); assert.match(b.run("renderQuantcheck()"), /id="qk-toktyped"/);
  b.els["qk-start"].onclick();
  assert.deepEqual(b.calls.at(-1)[1].spec.text, { kind: "data", run: null, tokenizer: "owner/model" });
  b.run(`qk.running = false; qk.refSel = "__make"; qk.makeSrc = "run"`); redraw(b); b.els["qk-start"].onclick();
  assert.deepEqual(b.calls.at(-1)[1].spec.reference, { kind: "make", source: { kind: "run", run: "run1" } });
});

test("the disk: a refusal is shown and greyed out the button, a chunk count that does not fit is refused, and the numbers are the program's", () => {
  const b = browser();
  const refusal = "There is not enough free disk space for this comparison: even the smallest test needs about 0.25 GB.";
  let html = open(b, qState(), `qk.plan = ${JSON.stringify(plan({ default_chunks: null, refusal, options: planOpts([false, false, false, false, false, false]) }))}; qk.planRef = "/p/exports/run1-f16/model-f16.gguf"; qk.picked = ["${cand().path}"];`);
  assert.match(html, /id="qk-refusal">There is not enough free disk space/); assert.match(html, /id="qk-start" disabled/); assert.match(html, /too big for the disk/);
  html = open(browser(), qState(), `qk.plan = ${JSON.stringify(plan({ options: planOpts([true, true, false, false, false, false]) }))}; qk.planRef = "/p/exports/run1-f16/model-f16.gguf"; qk.picked = ["${cand().path}"]; qk.chunks = 32;`);
  assert.match(html, /That amount of text needs more free disk than there is\. Choose fewer chunks/); assert.match(html, /id="qk-start" disabled/); assert.match(html, /does not fit/);
  html = open(browser(), qState(), `qk.plan = ${JSON.stringify(plan({ vocab_assumed: true, vocab: 250000 }))}; qk.planRef = ""; qk.picked = ["${cand().path}"];`);
  assert.match(html, /vocabulary is not known yet, so 250,000 words are assumed/);
  html = open(browser(), qState({ env_ready: false }), `qk.picked = ["${cand().path}"];`);
  assert.match(html, /training software is not installed yet/); assert.match(html, /id="qk-start" disabled/);
});

test("progress, Stop, a stopped run and a failure are shown in plain words", async () => {
  const b = browser({ quantcheck_state: qState() });
  open(b, qState(), `qk.picked = ["${cand().path}"]; qk.plan = ${JSON.stringify(plan())}; qk.planRef = "/p/exports/run1-f16/model-f16.gguf";`);
  await b.els["qk-start"].onclick();
  b.run(`onQuantcheckEvent({kind: "progress", step: "quantcheck", value: 0.4, text: "Comparing Q4_K_M, with matrix"})`);
  let html = b.run("renderQuantcheck()");
  assert.match(html, /Comparing Q4_K_M, with matrix/); assert.match(html, /<progress max="1" value="0.4">/); assert.match(html, /id="qk-stop"/);
  b.run("wireQuantcheck()"); b.els["qk-stop"].onclick();
  assert.equal(b.calls.at(-1)[0], "quantcheck_stop");
  b.run(`onQuantcheckEvent({kind: "error", text: "Cancelled."})`);
  html = b.run("renderQuantcheck()");
  assert.match(html, /Stopped\. The files that were already measured are kept/); assert.doesNotMatch(html, /id="qk-stop"/); assert.equal(b.run("qk.running"), false);
  b.run(`onQuantcheckEvent({kind: "error", text: "llama-perplexity was not found, so the quants cannot be compared. Install ... LLAMA_CPP_DIR"})`);
  assert.match(b.run("renderQuantcheck()"), /class="err">llama-perplexity was not found/);
  b.run(`onQuantcheckEvent({kind: "log", text: "$ llama-perplexity -m x"})`);
  assert.match(b.run("renderQuantcheck()"), /<summary>Log<\/summary><pre class="log">\$ llama-perplexity -m x/);
});

test("the result: a table sorted by size, bits per weight, the line marked, failed rows with their reason, the chart and the words", () => {
  const r = results();
  r.rows.push({ id: "5", label: "Broken", path: "/p/b.gguf", status: "failed", error: "Its vocabulary has 20 words, the reference's has 10: not the same model." });
  const b = browser();
  const html = open(b, qState(), `qk.shown = ${JSON.stringify(r)}`);
  assert.match(html, /Result: 20261115-1030-ref/); assert.match(html, /Reference: ref-f16.gguf · 9.0 billion weights · test text: the held-out questions/); assert.match(html, /16 chunks of 512 tokens · version: 5000 \(abc\)/);
  const names = [...html.matchAll(/<tr[^>]*><td>(?:● )?([^<]+?)(?: <span|<\/td>)/g)].map((m) => m[1]);
  assert.deepEqual(names.slice(0, 4), ["Q2_K", "Q4_K_M, no matrix", "Q4_K_M, with matrix", "Q8_0"], "smallest first");
  assert.match(html, /● Q2_K/); assert.match(html, /● Q4_K_M, with matrix/); assert.doesNotMatch(html, /● Q4_K_M, no matrix/); assert.match(html, /● Q8_0/);
  assert.match(html, /<td>2.90<\/td><td>2.58<\/td>\s*<td>0.3100<\/td>/, "size in GB, bits per weight, mean KLD");
  assert.match(html, /title="Its vocabulary has 20 words[^"]*">failed<\/span>/); assert.match(html, /<td colspan="9">Its vocabulary has 20 words, the reference&#39;s has 10: not the same model\.<\/td>/);
  assert.match(html, /<svg id="qk-chart"[^>]*role="img"[^>]*aria-label="Mean KL divergence against file size; lower left is better"/); assert.match(html, /lower left is better\. The line joins the files nothing smaller beats/);
  assert.equal((html.match(/<circle /g) || []).length, 4, "one dot for each measured file"); assert.match(html, /<polyline points="[^"]+"/);
  assert.match(html, /<h3>In words<\/h3><ul><li>The smallest file is Q2_K/); assert.match(html, /KL divergence is a proxy/); assert.match(html, /a note/);
  assert.match(html, /KLD against size \(log scale\)/, "0.002 to 0.31 spans more than a factor of 30");
  // a partial comparison says so, and a running figure is marked
  const p = results({ state: "stopped", rows: [row("1", "Q4_K_M", 4e9, 0.03, { partial: true }), { id: "2", label: "Q5_K_M", path: "/p/2.gguf", status: "pending" }], pareto: ["1"] });
  const h2 = open(browser(), qState(), `qk.shown = ${JSON.stringify(p)}`);
  assert.match(h2, /<strong>stopped<\/strong>/); assert.match(h2, /not measured yet/); assert.match(h2, /\(running figure\)/);
  assert.doesNotMatch(h2, /<polyline/, "a line needs two points");
  assert.equal(b.run("qkChartHtml({rows: []})"), "");
});

test("the end of a run shows its results and refreshes the list; the list offers Show, Continue, delete the answers and Delete", async () => {
  const e = (over = {}) => ({ id: "20261115-1030-ref", state: "stopped", reference: "ref-f16.gguf", rows: 3, rows_done: 1, chunks: 16, created: "2026-11-15T10:30:00Z", base_kept: true, base_mb: 2030, running: false, ...over });
  const b = browser({ quantcheck_state: qState({ entries: [e()] }) });
  open(b, qState({ entries: [e(), e({ id: "20261114-0900-old", state: "done", rows_done: 3, base_kept: false })] }));
  const html = b.run("renderQuantcheck()");
  assert.match(html, /stopped \(the finished files are kept\)/); assert.match(html, /1 of 3 files measured/); assert.match(html, /the 16-bit model's answers are kept \(2030 MB\)/);
  assert.equal((html.match(/class="secondary qk-continue"/g) || []).length, 1, "a finished comparison needs no Continue");
  assert.equal((html.match(/class="secondary qk-dellog"/g) || []).length, 1); assert.match(html, /Delete the saved answers \(2030 MB\)/); assert.equal((html.match(/class="secondary qk-del"/g) || []).length, 2);
  b.run(`onQuantcheckEvent({kind: "done", data: ${JSON.stringify(results())}})`);
  await new Promise((r) => setTimeout(r, 5));
  assert.match(b.run("renderQuantcheck()"), /The comparison is ready\./); assert.match(b.run("renderQuantcheck()"), /id="qk-result"/); assert.ok(b.calls.some((c) => c[0] === "quantcheck_state"));
});

test("Show, Continue, Delete the saved answers and Delete call the program; a refusal in the question changes nothing", async () => {
  const saved = results({ state: "stopped", reference: { name: "ref-f16.gguf", path: "/p/exports/run1-f16/model-f16.gguf" }, rows: [row("1", "Q4_K_M", 4e9, 0.03), { id: "2", label: "Q5_K_M", path: "/p/q5.gguf", group: "Q5_K_M", matrix: false, mixed: null, status: "pending" }] });
  const e = { id: "20261115-1030-ref", state: "stopped", reference: "ref-f16.gguf", rows: 2, rows_done: 1, chunks: 16, created: "2026-11-15T10:30:00Z", base_kept: true, base_mb: 2030, running: false };
  const b = browser({ quantcheck_get: saved, quantcheck_delete: qState(), quantcheck_delete_logits: qState({ entries: [{ ...e, base_kept: false }] }) });
  open(b, qState({ entries: [e] }), `qk.plan = ${JSON.stringify(plan())}; qk.planRef = "/p/exports/run1-f16/model-f16.gguf";`);
  const btn = (cls) => ({ dataset: { id: e.id }, onclick: null, cls });
  for (const c of [".qk-show", ".qk-continue", ".qk-dellog", ".qk-del"]) b.groups[c] = [btn(c)];
  redraw(b);
  await b.groups[".qk-show"][0].onclick();
  assert.deepEqual(b.calls.at(-1), ["quantcheck_get", { id: e.id }]); assert.match(b.run("renderQuantcheck()"), /Result: 20261115-1030-ref/);
  await b.groups[".qk-continue"][0].onclick();
  const start = b.calls.at(-1);
  assert.equal(start[0], "quantcheck_start");
  assert.equal(start[1].spec.resume, e.id); assert.deepEqual(start[1].spec.reference, { kind: "file", file: "/p/exports/run1-f16/model-f16.gguf" }); assert.equal(start[1].spec.chunks, 16);
  assert.deepEqual(start[1].spec.quants.map((q) => q.path), ["/p/1.gguf", "/p/q5.gguf"], "the files the comparison had");
  b.run("qk.running = false");
  await b.groups[".qk-dellog"][0].onclick();
  assert.deepEqual(b.calls.at(-1), ["quantcheck_delete_logits", { id: e.id }]); assert.match(b.run("renderQuantcheck()"), /The saved answers were deleted\./);
  await b.groups[".qk-del"][0].onclick();
  assert.deepEqual(b.calls.at(-1), ["quantcheck_delete", { id: e.id }]); assert.match(b.run("renderQuantcheck()"), /Deleted\./);
  b.confirm(false); b.calls.length = 0;
  await b.groups[".qk-del"][0].onclick(); await b.groups[".qk-dellog"][0].onclick();
  assert.deepEqual(b.calls, [], "no change after a refusal");
});

test("a refused start and a refused delete show the program's plain message", async () => {
  const b = browser({ quantcheck_start: new Error("The file /x.gguf was not found."), quantcheck_delete: new Error("This comparison is running right now. Press Stop first, then delete it.") });
  open(b, qState(), `qk.picked = ["${cand().path}"]; qk.plan = ${JSON.stringify(plan())}; qk.planRef = "/p/exports/run1-f16/model-f16.gguf";`);
  await b.els["qk-start"].onclick();
  assert.match(b.run("renderQuantcheck()"), /The file \/x\.gguf was not found\./); assert.equal(b.run("qk.running"), false);
  b.run(`qk.state.entries = [{id: "a", state: "done", rows: 1, rows_done: 1, chunks: 16, base_kept: false, running: false}]`);
  b.groups[".qk-del"] = [{ dataset: { id: "a" }, onclick: null }]; redraw(b);
  await b.groups[".qk-del"][0].onclick();
  assert.match(b.run("renderQuantcheck()"), /running right now\. Press Stop first/);
});

test("the export results offer Compare these sizes for a run with more than one file, and it fills the page with that run's files", async () => {
  const exp = (name, quant, over = {}) => ({ name, run: "run1", quant, gguf_file: `model-${quant}.gguf`, size_mb: 100, dir: "/p/exports/" + name, installed: [], ...over });
  const st = { runs: [{ name: "run1", model: "M", status: "done", loss: 0.5 }], quants: [], servers: [], exports: [exp("run1-q4_k_m", "q4_k_m"), exp("run1-q8_0", "q8_0")], suggested_name: "x", env_ready: true, imatrix: null };
  const b = browser({ quantcheck_state: qState(), export_state: st });
  await new Promise((r) => setTimeout(r, 5));                       // init() goes to the welcome page first
  b.run(`state = {project: {name: "P", folder: "/p"}, steps: [], bundle: {world: "W"}, settings: {theme: "light"}, worlds: [], recent: []}; page = "use"; us.state = ${JSON.stringify(st)}; us.run = "run1";`);
  const html = b.run("renderUse()");
  assert.match(html, /id="us-compare"[^>]*>Compare these sizes<\/button>/);
  const one = { ...st, exports: [st.exports[0]] };
  b.run(`us.state = ${JSON.stringify(one)}`);
  assert.doesNotMatch(b.run("renderUse()"), /us-compare/, "one file has nothing to be compared with");
  b.run(`us.state = ${JSON.stringify(st)}`); b.run("renderUse()"); b.run("wireUse()");
  await b.els["us-compare"].onclick();
  await new Promise((r) => setTimeout(r, 20));
  assert.equal(b.run("page"), "quantcheck");
  assert.deepEqual(J(b, "qk.picked"), qState().candidates.quants.filter((q) => q.run === "run1").map((q) => q.path));
  assert.equal(b.run("qk.tokRun"), "run1");
  assert.match(b.els["page"].innerHTML, /Q4_K_M, with matrix/);
});
