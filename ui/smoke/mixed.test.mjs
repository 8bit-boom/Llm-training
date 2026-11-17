// Smoke test of the mixed-precision sizes on the Step 5 export page (docs/MIXED_QUANTS.md), without a window: app.js runs in a stand-in browser that records HTML and every call to the program.
import test from "node:test";
import assert from "node:assert/strict";
import { makeBrowser, source } from "./_browser.mjs";


function browser(answers = {}, dialogs = []) {
  return makeBrowser({ answers, dialogs, cloneArgs: true, state: null });
}

const plainQuants = [["q4_k_m", "Q4_K_M: smallest"], ["q8_0", "Q8_0: large"]].map(([id, name]) => ({ id, name, label: name.split(":")[0] }));
const mixedQuants = ([["mixed_4", "Mixed 4-bit (balanced) · Mixed precision (our own recipe): about 5 bits per weight"], ["mixed_3", "Mixed 3-bit (small) · Mixed precision (our own recipe): about 4 bits per weight"],
    ["mixed_2", "Mixed 2-bit (smallest: short answers only) · Mixed precision (our own recipe): about 3 bits per weight"], ["mixed_fit", "Mixed: fit to a size (experimental) · Mixed precision (our own recipe): made to fit the size you give"]]
    .map(([id, name]) => ({ id, name, label: name.split(" · ")[0], mixed: true })));
const exportState = (over = {}) => ({ runs: [{ name: "run1", model: "Qwen3.5 9B", status: "done", loss: 0.8 }], exports: [], servers: [], suggested_name: "w", env_ready: true, quants: plainQuants, mixed_quants: mixedQuants,
  imatrix: { chunks: [64, 128, 256], default_chunks: 128, ctx: 512, general_share: 0.2, no_matrix_for: ["q8_0", "f16"], saved: {}, estimates: {}, library: [{ id: "m1", label: "my matrix", model: "run1", chunks: 128, created: "2026-11-01T10:00:00Z" }] },
  mixed: { recipe: "Mixed precision (our own recipe)", files: [{ path: "/p/exports/other-q4_k_m/model-q4_k_m.gguf", run: "other", label: "other-q4_k_m: model-q4_k_m.gguf" }, { path: "/p/exports/run1-q4_k_m/model-q4_k_m.gguf", run: "run1", label: "run1-q4_k_m: model-q4_k_m.gguf" }] }, ...over });
const preset = (id, label, gb, bpw, base) => ({ id, label, recipe: "Mixed precision (our own recipe)", base, embd: "Q5_K", output: "Q6_K", summary: "", rules: ["blk\\.(?:0|1)\\.(?:attn_q)\\.weight$=Q6_K", "\\.(?:ffn_up|ffn_gate)\\.weight$=Q4_K"], flags: [], estimate: { bytes: gb * 1e9, gb, bpw, unmatched: 0, dead_rules: [] }, notes: id === "mixed_2" ? ["A rule matched no tensor of this model (x)."] : [] });
const plan = (over = {}) => ({ arch: "qwen35", n_layer: 64, params: 27e9, recipe: "Mixed precision (our own recipe)", presets: [preset("mixed_4", "Mixed 4-bit (balanced)", 16.1, 5.1, "Q4_K_M"), preset("mixed_3", "Mixed 3-bit (small)", 13.4, 4.2, "Q3_K_M"), preset("mixed_2", "Mixed 2-bit (smallest: short answers only)", 10.2, 3.2, "Q2_K")],
  plain_bpw: { Q2_K: 3.1593, Q3_K_M: 3.996, Q4_K_M: 4.8944 }, fit: null, ...over });
const page = (b, st, extra = "") => { b.run(`state = {project: {name: "P", folder: "/p"}, steps: [], bundle: {world: "W"}, settings: {theme: "light"}, worlds: [], recent: []}; page = "use"; us.state = ${JSON.stringify(st)}; ${extra}`); const html = b.run("renderUse()"); b.run("wireUse()"); return html; };
const redraw = (b) => { const h = b.run("renderUse()"); b.run("wireUse()"); return h; };
const tick = (b, ids) => { b.run(`us.quants = ${JSON.stringify(ids)}`); return redraw(b); };

test("the mixed sizes are listed after the plain ones under the name of the recipe, and are our own", () => {
  const html = page(browser(), exportState());
  assert.match(html, /File sizes: [^]*Q4_K_M<\/label>[^]*Q8_0<\/label><\/span>/);
  assert.match(html, /<strong>Mixed precision \(our own recipe\):<\/strong> [^]*Mixed 4-bit \(balanced\)<\/label>[^]*Mixed 3-bit \(small\)<\/label>[^]*Mixed 2-bit \(smallest: short answers only\)<\/label>[^]*Mixed: fit to a size \(experimental\)<\/label>/);
  assert.equal((html.match(/class="us-q"/g) || []).length, 6);
  assert.doesNotMatch(html, /Dynamic|UD-|Unsloth Dynamic/i);
  assert.doesNotMatch(html, /id="us-mxwhy"/, "the explanation appears when a mixed size is ticked");
  assert.doesNotMatch(page(browser(), exportState(), `us.what = "merged"`), /Mixed precision/, "no sizes when only the merged model is made");
  assert.doesNotMatch(page(browser(), exportState({ mixed_quants: undefined, mixed: undefined })), /Mixed precision/, "an old state without mixed sizes shows nothing of them");
});

test("ticking a mixed size explains how it is made, who makes it and that nobody has measured yet whether it is better", () => {
  const html = page(browser(), exportState(), `us.quants = ["mixed_4"]`);
  assert.match(html, /id="us-mxwhy"/); assert.match(html, /llama-quantize<\/code> gives single tensors their own types/);
  assert.match(html, /llama.cpp makes the <em>whole batch<\/em> instead \(Unsloth cannot do per-tensor types\) and the result says so/);
  assert.match(html, /More bits go to the embedding and output tensors, the attention k \/ v \/ output, the linear-attention projections, ffn_down and the first and last layers; fewer to the ffn_up \/ gate of the middle layers/);
  assert.match(html, /<strong>Mixed precision \(our own recipe\)<\/strong>: they are our own recipes, not copies of anyone else&#39;s|<strong>Mixed precision \(our own recipe\)<\/strong>: they are our own recipes, not copies of anyone else's/);
  assert.match(html, /nobody has measured yet whether they beat the plain sizes: that is what <strong>Compare quants<\/strong> is for/);
  assert.match(html, /The importance matrix, when you use one, is applied to these sizes too/);
  assert.doesNotMatch(html, /Dynamic|UD-/i);
});

test("the estimated sizes: the file of this run is chosen, the worker is asked for its shapes, and the table, the rules and the notes are shown", async () => {
  const b = browser({ mixed_plan: plan() });
  let html = page(b, exportState(), `us.quants = ["mixed_4"]`);
  assert.match(html, /<option value="\/p\/exports\/run1-q4_k_m\/model-q4_k_m.gguf" selected>run1-q4_k_m: model-q4_k_m.gguf/);
  assert.doesNotMatch(html, /<option value="\/p\/exports\/other-q4_k_m\/model-q4_k_m.gguf" selected>/); assert.match(html, /A GGUF file I choose…/);
  await b.els["us-mxplanbtn"].onclick();
  assert.deepEqual(b.calls.at(-1), ["mixed_plan", { gguf: "/p/exports/run1-q4_k_m/model-q4_k_m.gguf", targetGb: null, matrix: null }]);
  html = b.run("renderUse()");
  assert.match(html, /<th>Mixed precision \(our own recipe\)<\/th><th>Estimated size<\/th><th>Bits per weight<\/th><th>Base type<\/th>/);
  assert.match(html, /<td>Mixed 4-bit \(balanced\)<\/td><td>16.1 GB<\/td><td>5.1<\/td><td>Q4_K_M<\/td>/); assert.match(html, /<td>Mixed 2-bit \(smallest: short answers only\)<\/td><td>10.2 GB<\/td><td>3.2<\/td>/);
  assert.match(html, /27.0 billion weights and shapes in the file&#39;s header|27.0 billion weights and shapes in the file's header/); assert.match(html, /architecture qwen35, 64 layers/);
  assert.match(html, /about 4.8944 \(Q4_K_M\), 3.996 \(Q3_K_M\) and 3.1593 \(Q2_K\) bits per weight/); assert.match(html, /The real size is shown after the export/);
  assert.match(html, /<summary>Mixed 4-bit \(balanced\): the rules llama-quantize gets<\/summary>/); assert.match(html, /--output-tensor-type Q6_K\n--token-embedding-type Q5_K\n--tensor-type blk\\\.\(\?:0\|1\)\\\.\(\?:attn_q\)\\\.weight\$=Q6_K/);
  assert.match(html, /\(base type Q4_K_M\)/); assert.match(html, /A rule matched no tensor of this model \(x\)\./);
});

test("a typed file, no file at all and a file the worker cannot read are each said plainly", async () => {
  const b = browser({ mixed_plan: new Error("The GGUF file could not be read (not a GGUF file). Choose a complete GGUF file of this model (any size: only its tensor shapes are read).") });
  let html = page(b, exportState({ mixed: { recipe: "Mixed precision (our own recipe)", files: [] } }), `us.quants = ["mixed_3"]`);
  assert.match(html, /No GGUF of this model exists yet\. Choose any GGUF file of it/); assert.match(html, /id="us-mxplanbtn" disabled/);
  b.els["us-mxfile"].value = "__typed"; b.els["us-mxfile"].onchange();
  html = redraw(b);
  assert.match(html, /id="us-mxtyped"/); assert.match(html, /id="us-mxplanbtn" disabled/, "nothing typed yet");
  b.els["us-mxtyped"].value = "  D:\\m\\model.gguf "; b.els["us-mxtyped"].onchange();
  assert.doesNotMatch(redraw(b), /id="us-mxplanbtn" disabled/);
  await b.els["us-mxplanbtn"].onclick();
  assert.deepEqual(b.calls.at(-1), ["mixed_plan", { gguf: "D:\\m\\model.gguf", targetGb: null, matrix: null }]);
  assert.match(b.run("renderUse()"), /id="us-mxerror">Error: The GGUF file could not be read|id="us-mxerror">The GGUF file could not be read \(not a GGUF file\)\. Choose a complete GGUF file of this model/);
});

test("Fit to a size is labelled experimental, needs a target, asks the worker for the rules and shows them", async () => {
  const fit = { ok: true, label: "Mixed: fit to a size (experimental)", fits: true, target_bytes: 12.5e9, estimate: { gb: 12.4, bpw: 4.1 }, notes: ["12 tensors had no figure in the matrix; they were given the median one."], rules: ["blk\\.(?:3|7)\\.attn_k\\.weight$=Q6_K", "blk\\.(?:0|1)\\.ffn_up\\.weight$=Q3_K"], upgrades: 40 };
  const b = browser({ mixed_plan: plan({ fit }) });
  let html = page(b, exportState(), `us.quants = ["mixed_fit"]`);
  assert.match(html, /<strong>Fit to a size \(experimental\)<\/strong>/); assert.match(html, /Experimental, and said so/); assert.match(html, /It is a <strong>guess<\/strong> from the matrix&#39;s per-tensor statistics, not a measurement|It is a <strong>guess<\/strong> from the matrix's per-tensor statistics, not a measurement/);
  assert.match(html, /id="us-mxtargetwhy"/); assert.match(html, /id="us-mxfitbtn" disabled/); assert.match(html, /id="us-make" disabled/, "no target: no export");
  assert.match(html, /<option value="">the matrix of this export<\/option><option value="m1" >my matrix<\/option>/);
  b.els["us-mxtarget"].value = " 12.5 "; b.els["us-mxtarget"].onchange();
  b.els["us-mxmatrix"].value = "m1"; b.els["us-mxmatrix"].onchange();
  html = redraw(b);
  assert.doesNotMatch(html, /id="us-mxtargetwhy"/); assert.doesNotMatch(html, /id="us-mxfitbtn" disabled/); assert.doesNotMatch(html, /id="us-make" disabled/);
  await b.els["us-mxfitbtn"].onclick();
  assert.deepEqual(b.calls.at(-1), ["mixed_plan", { gguf: "/p/exports/run1-q4_k_m/model-q4_k_m.gguf", targetGb: 12.5, matrix: "m1" }]);
  html = b.run("renderUse()");
  assert.match(html, /id="us-mxfit"><strong>Mixed: fit to a size \(experimental\):<\/strong> about 12.4 GB \(4.1 bits per weight\) for the 12.50 GB asked/);
  assert.match(html, /12 tensors had no figure in the matrix/); assert.match(html, /<summary>The 2 rules this gives<\/summary>/); assert.match(html, /--tensor-type blk\\\.\(\?:3\|7\)\\\.attn_k\\\.weight\$=Q6_K/);
});

test("a fit that cannot be done is said, and one the lowest types cannot reach says so", async () => {
  let b = browser({ mixed_plan: plan({ fit: { ok: false, label: "x", error: "The matrix old.dat could not be read as a GGUF matrix here (not a GGUF file); a legacy .dat file cannot be used for fitting." } }) });
  page(b, exportState(), `us.quants = ["mixed_fit"]; us.mx.target = "12"`);
  await b.els["us-mxfitbtn"].onclick();
  assert.match(b.run("renderUse()"), /id="us-mxfiterror">The matrix old.dat could not be read as a GGUF matrix here/);
  b = browser({ mixed_plan: plan({ fit: { ok: true, label: "Mixed: fit to a size (experimental)", fits: false, target_bytes: 1e9, estimate: { gb: 9.7, bpw: 3 }, notes: ["Even the lowest types (Q2_K) need 9.70 GB, more than the 1.00 GB asked for; the lowest ones are given."], rules: ["a=Q2_K"] } }) });
  page(b, exportState(), `us.quants = ["mixed_fit"]; us.mx.target = "1"`);
  await b.els["us-mxfitbtn"].onclick();
  assert.match(b.run("renderUse()"), /<strong>which the lowest types cannot reach<\/strong>/); assert.match(b.run("renderUse()"), /Even the lowest types \(Q2_K\) need 9.70 GB/);
});

test("Make the files sends the sizes and, only for the fit, the target and matrix", async () => {
  const b = browser();
  page(b, exportState(), `us.quants = ["mixed_4", "q4_k_m"]`);
  await b.els["us-make"].onclick();
  let [cmd, args] = b.calls.at(-1);
  assert.equal(cmd, "export_start"); assert.deepEqual(args.quants, ["mixed_4", "q4_k_m"]); assert.equal(Object.hasOwn(args, "mixed"), false, "no fit: nothing about it is sent");
  b.run(`us.running = false; us.quants = ["mixed_fit"]; us.mx.target = "12.5"; us.mx.matrix = "m1"`); redraw(b);
  await b.els["us-make"].onclick();
  [cmd, args] = b.calls.at(-1);
  assert.deepEqual(args.quants, ["mixed_fit"]); assert.deepEqual(args.mixed, { fit: { target_gb: 12.5, matrix: "m1" } });
  b.run(`us.running = false; us.mx.matrix = ""`); redraw(b); await b.els["us-make"].onclick();
  assert.deepEqual(b.calls.at(-1)[1].mixed, { fit: { target_gb: 12.5, matrix: null } });
});

test("a mixed size in the list of files says whose recipe it is, its estimate, a fit and a fall-back; the end message carries the llama.cpp note", () => {
  const e = (over) => ({ name: "run1-mixed_4", run: "run1", quant: "mixed_4", gguf_file: "model-mixed_4.gguf", size_mb: 16200, dir: "/p/exports/run1-mixed_4", installed: [], ...over });
  const html = page(browser(), exportState({ exports: [e({ mixed: { id: "mixed_4", label: "Mixed 4-bit (balanced)", recipe: "Mixed precision (our own recipe)", estimate: { gb: 16.1 }, note: "The mixed-precision sizes need llama-quantize's per-tensor types.", notes: [] } }),
    e({ name: "run1-mixed_fit", quant: "mixed_fit", mixed: { id: "mixed_fit", label: "Mixed: fit to a size (experimental)", recipe: "Mixed precision (our own recipe)", estimate: { gb: 12.4 }, fit: { target_bytes: 12.5e9 }, notes: [] } }),
    e({ name: "run1-mixed_fit2", quant: "mixed_fit", mixed: { id: "mixed_3", fell_back_to: "mixed_3", label: "Mixed 3-bit (small)", recipe: "Mixed precision (our own recipe)", estimate: { gb: 13.4 }, fit: null, notes: [] } })] }));
  assert.match(html, /<strong>Mixed precision \(our own recipe\)<\/strong>: Mixed 4-bit \(balanced\), estimated 16.1 GB/);
  assert.match(html, /<strong>Mixed precision \(our own recipe\)<\/strong>: Mixed: fit to a size \(experimental\), estimated 12.4 GB, fitted to 12.50 GB \(experimental\)/);
  assert.match(html, /Mixed 3-bit \(small\), estimated 13.4 GB, the fit fell back to mixed_3/);
  const b = browser();
  page(b, exportState());
  b.run(`onExportEvent({kind: "done", data: {name: "run1-mixed_4", run: "run1", exports: [{imatrix: null, mixed: {note: "The mixed-precision sizes need llama-quantize's per-tensor types, which Unsloth's export cannot do, so this export was made with llama.cpp."}}]}})`);
  assert.match(b.run("renderUse()"), /so this export was made with llama.cpp/);
});
