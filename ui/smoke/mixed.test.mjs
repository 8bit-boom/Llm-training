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

// ------------------------------------------------------------------------------------------------ Mixed: measured fit to a size
const measuredQuants = [...mixedQuants, { id: "mixed_measured", name: "Mixed: measured fit to a size · Mixed precision (our own recipe): each tensor's error measured on your weights, weighted by the importance matrix", label: "Mixed: measured fit to a size", mixed: true }];
const qfState = (over = {}) => ({ references: [{ path: "/p/exports/run1-q4_k_m/model-f16.gguf", label: "run1-q4_k_m: model-f16.gguf" }, { path: "/p/exports/other/model-f16.gguf", label: "other: model-f16.gguf" }], ladder: ["Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_0"],
  plans: [], verdict: null, caveats: "The error is measured with a diagonal approximation ... Compare quants (KL divergence) is the judge, not this number.", ...over });
const estimate = (over = {}) => ({ engine: "llama", llama_quantize: "/l/llama-quantize", llama_version: "version: 5000 (abc)", tensor_type_file: true, minutes_note: "About 2 minutes (an estimate: the numpy speed was measured just now; the 5 times faster llama engine and a disk read speed of 200 MB/s are assumptions).",
  sampled_weights: 280e6, read_bytes: 560e6, disk_need: 2.7e9, min_gb: 8.1, max_gb: 17.2, plain_estimates: { Q4_K_M: 10.6, Q5_K_M: 12.4, Q6_K: 14.3, Q8_0: 18.5 }, matrix_ok: true, notes: [], ...over });
const result = (over = {}) => ({ fits: true, engine: "llama", plan: { total: 10.59e9 }, report: { rows: [{ id: "plan", label: "Measured plan", gb: 10.59, error: 0.07 }, { id: "uniform_q4_k", label: "Every tensor Q4_K", gb: 10.4, error: 0.084 }, { id: "plain_q4_k_m", label: "Plain Q4_K_M (as llama.cpp's mixture is approximated here)", gb: 10.6, error: 0.081 }],
  types: { Q4_K: { tensors: 100, params: 1 }, Q6_K: { tensors: 20, params: 1 } }, bpw_by_role: { ffn_up: 4.2, attn_k: 6.5 }, caveats: "The error is measured with a diagonal approximation ... Compare quants (KL divergence) is the judge, not this number." },
  rules: ["blk\\.(?:0|1)\\.attn_k\\.weight$=Q6_K"], flags: ["--output-tensor-type", "Q6_K", "--token-embedding-type", "Q4_K", "--tensor-type-file", "t.txt"], notes: ["One note."], files: [], ...over });
const qfPage = (b, extra = "") => page(b, exportState({ mixed_quants: measuredQuants }), `qf.open = true; qf.state = ${JSON.stringify(qfState())}; ${extra}`);

test("the measured size is listed with the others and asks for a target and a matrix when ticked", () => {
  const b = browser();
  let html = page(b, exportState({ mixed_quants: measuredQuants }));
  assert.match(html, /Mixed: measured fit to a size<\/label>/); assert.doesNotMatch(html, /id="us-mmbox"/);
  html = page(b, exportState({ mixed_quants: measuredQuants }), `us.quants = ["mixed_measured"]`);
  assert.match(html, /id="us-mmbox"/); assert.match(html, /id="us-mmtargetwhy"/); assert.match(html, /id="us-make" disabled/);
  b.els["us-mmtarget"].value = " 10.6 "; b.els["us-mmtarget"].onchange();
  b.els["us-mmmatrix"].value = "m1"; b.els["us-mmmatrix"].onchange();
  html = redraw(b);
  assert.doesNotMatch(html, /id="us-mmtargetwhy"/); assert.doesNotMatch(html, /id="us-make" disabled/);
  assert.doesNotMatch(html, /Dynamic|UD-/i);
});

test("what Make sends for the measured size, alone and next to Fit to a size", async () => {
  const b = browser({ export_start: null });
  page(b, exportState({ mixed_quants: measuredQuants }), `us.quants = ["mixed_measured"]; us.mm.target = "10.6"; us.mm.matrix = "m1"`);
  b.run(`us.running = false`); redraw(b);
  await b.els["us-make"].onclick();
  const args = b.calls.find((c) => c[0] === "export_start")[1];
  assert.deepEqual(args.quants, ["mixed_measured"]); assert.deepEqual(args.mixed, { measured: { target_gb: 10.6, matrix: "m1" } });
  b.run(`us.running = false; us.quants = ["mixed_fit", "mixed_measured", "q4_k_m"]; us.mx.target = "12"; us.mx.matrix = ""`); redraw(b);
  await b.els["us-make"].onclick();
  const both = b.calls.filter((c) => c[0] === "export_start").at(-1)[1];
  assert.deepEqual(both.mixed, { fit: { target_gb: 12, matrix: null }, measured: { target_gb: 10.6, matrix: "m1" } });
});

test("the measured-fit box: the file, the matrix, the target in GB and GiB, and the caveat before anything runs", () => {
  const b = browser();
  const html = qfPage(b, `qf.target = "10.6"`);
  assert.match(html, /<summary><strong>Fit to a size \(measured\)<\/strong>/); assert.match(html, /<option value="\/p\/exports\/run1-q4_k_m\/model-f16.gguf" selected>run1-q4_k_m: model-f16.gguf/);
  assert.match(html, /A 16-bit GGUF I choose…/); assert.match(html, /every channel counts the same/); assert.match(html, /my matrix/);
  assert.match(html, /id="qf-gib">= 9.87 GiB/); assert.match(html, /measurement of the weights, not of the model<\/strong>: <strong>Compare quants<\/strong> \(KL divergence\) is the judge/);
  assert.match(html, /id="qf-q4" disabled/); assert.match(html, /id="qf-run" >Measure and plan/); assert.match(html, /id="qf-estbtn" >Estimate/);
  assert.doesNotMatch(html, /Dynamic|UD-/i);
  assert.match(qfPage(browser()), /id="qf-run" disabled>Measure and plan/); assert.match(qfPage(browser()), /id="qf-why">Give the size in GB\./);
});

test("Estimate shows the minutes as an estimate, the engine, the type-file support, the disk and the range; the same-size buttons fill the target", async () => {
  const b = browser({ quantfit_estimate: estimate() });
  qfPage(b, `qf.target = "10.6"`);
  await b.els["qf-estbtn"].onclick();
  assert.deepEqual(b.calls.at(-1)[0], "quantfit_estimate");
  const sent = b.calls.at(-1)[1].spec;
  assert.equal(sent.gguf, "/p/exports/run1-q4_k_m/model-f16.gguf"); assert.equal(sent.target_gb, 10.6); assert.equal(sent.make, false); assert.equal(sent.engine, "auto"); assert.equal(sent.floor, "Q3_K"); assert.equal(sent.sample_rows, 128);
  const html = b.run("renderUse()");
  assert.match(html, /About 2 minutes \(an estimate:/); assert.match(html, /Engine: llama \(llama-quantize found, version: 5000 \(abc\); <code>--tensor-type-file<\/code> yes\)/);
  assert.match(html, /Looks at 280 million weights \(0.56 GB read\), needs about 2.7 GB of free disk/); assert.match(html, /Smallest 8.10 GB, largest 17.20 GB/); assert.match(html, /Q5_K_M about 12.4 GB/);
  assert.doesNotMatch(html, /id="qf-q4" disabled/);
  b.run("wireUse()"); b.els["qf-q5"].onclick();
  assert.match(b.run("qf.target"), /12.4/); assert.match(b.run("renderUse()"), /id="qf-gib">= 11.55 GiB/);
  // a target below the smallest is said before anything runs
  b.run(`qf.target = "5"`);
  assert.match(b.run("renderUse()"), /id="qf-below">5 GB is below the smallest this model can be with Q3_K as the lowest type \(8.10 GB\)/);
});

test("an estimate that fails (a matrix of another model) is shown in the worker's words", async () => {
  const b = browser({ quantfit_estimate: new Error("The matrix does not fit this model: blk.1.attn_k.weight has 5120 input channels. This matrix is not of this model; adapt it first.") });
  qfPage(b, `qf.target = "10.6"`);
  await b.els["qf-estbtn"].onclick();
  assert.match(b.run("renderUse()"), /id="qf-esterror">[^]*not of this model; adapt it first/);
});

test("Measure and plan sends the choice, shows progress with a Stop button, and the table, the rules and the caveat when it is done", async () => {
  const b = browser({ quantfit_start: null, quantfit_stop: null, quantfit_state: qfState() });
  qfPage(b, `qf.target = "10.6"; qf.matrix = "m1"; qf.rows = 256; qf.floor = "Q2_K"; qf.engine = "reference"`);
  await b.els["qf-run"].onclick();
  const spec = b.calls.find((c) => c[0] === "quantfit_start")[1].spec;
  assert.deepEqual(spec, { gguf: "/p/exports/run1-q4_k_m/model-f16.gguf", matrix: "m1", target_gb: 10.6, floor: "Q2_K", ceiling: "Q8_0", sample_rows: 256, engine: "reference", make: false, neighbours: false, compare: null });
  b.run(`onQuantfitEvent(${JSON.stringify({ kind: "progress", value: 0.4, text: "Measuring (40 of 100): blk.3.attn_q.weight" })})`);
  let html = b.run("renderUse()");
  assert.match(html, /id="qf-step">Measuring \(40 of 100\)/); assert.match(html, /id="qf-stop">Stop/); assert.match(html, /id="qf-run" disabled>/);
  b.run("wireUse()"); b.els["qf-stop"].onclick();
  assert.equal(b.calls.at(-1)[0], "quantfit_stop");
  b.run(`onQuantfitEvent(${JSON.stringify({ kind: "done", data: result() })})`);
  html = b.run("renderUse()");
  assert.match(html, /<td>Measured plan<\/td><td>10.59 GB<\/td><td>0.07<\/td>/); assert.match(html, /<td>Every tensor Q4_K<\/td>/); assert.match(html, /Plain Q4_K_M \(as llama.cpp&#39;s mixture is approximated here\)|Plain Q4_K_M \(as llama.cpp's mixture is approximated here\)/);
  assert.match(html, /Types chosen: Q4_K: 100 tensors, Q6_K: 20 tensors/); assert.match(html, /ffn_up 4.2 · attn_k 6.5/); assert.match(html, /The 1 rules this gives/); assert.match(html, /--tensor-type blk\\\.\(\?:0\|1\)\\\.attn_k\\\.weight\$=Q6_K/);
  assert.match(html, /One note\./); assert.match(html, /id="qf-caveat">The error is measured with a diagonal approximation/); assert.match(html, /id="qf-ok">The plan is ready\./);
  assert.match(html, /id="qf-run" >Measure and plan/);
});

test("Stop keeps what was measured and the button says Resume; a refusal is shown as it is", async () => {
  const b = browser({ quantfit_start: null });
  qfPage(b, `qf.target = "10.6"`);
  await b.els["qf-run"].onclick();
  b.run(`onQuantfitEvent(${JSON.stringify({ kind: "error", text: "Cancelled." })})`);
  let html = b.run("renderUse()");
  assert.match(html, /id="qf-error">Stopped\. What was measured is kept: press Resume to go on\./); assert.match(html, /id="qf-run" >Resume/);
  b.run(`onQuantfitEvent(${JSON.stringify({ kind: "error", text: "This matrix is not of this model; adapt it first." })})`);
  assert.match(b.run("renderUse()"), /id="qf-error">This matrix is not of this model/);
  const b2 = browser({ quantfit_start: new Error("Give the size in GB that the measured fit should aim for.") });
  qfPage(b2, `qf.target = "10.6"`); await b2.els["qf-run"].onclick();
  assert.match(b2.run("renderUse()"), /id="qf-error">(Error: )?Give the size in GB/);
});

test("Make the file and Make and compare send the files options, and the promotion verdict is shown", async () => {
  const b = browser({ quantfit_start: null, quantfit_state: qfState() });
  qfPage(b, `qf.target = "10.6"; us.run = "run1"`);
  b.els["qf-make"].checked = true; b.els["qf-make"].onchange();
  let html = b.run("renderUse()");
  assert.doesNotMatch(html, /id="qf-compare" checked/); assert.doesNotMatch(html, /id="qf-compare" ><|id="qf-compare"  disabled/);
  await b.els["qf-run"].onclick();
  let spec = b.calls.filter((c) => c[0] === "quantfit_start").at(-1)[1].spec;
  assert.equal(spec.make, true); assert.equal(spec.neighbours, true); assert.equal(spec.compare, null);
  b.run(`qf.running = false`); b.run("renderUse()"); b.run("wireUse()");
  b.els["qf-compare"].checked = true; b.els["qf-compare"].onchange();
  b.run("renderUse()"); b.run("wireUse()");
  b.els["qf-text"].value = "general"; b.els["qf-text"].onchange();
  await b.els["qf-run"].onclick();
  spec = b.calls.filter((c) => c[0] === "quantfit_start").at(-1)[1].spec;
  assert.deepEqual(spec.compare, { kind: "general", run: "run1" });
  b.run(`onQuantfitEvent(${JSON.stringify({ kind: "done", data: result({ files: [{ path: "/q/a.gguf" }], comparison: { promotion: [{ id: "1", promoted: true, reason: "Mixed: measured fit to a size (0.00 GB smaller than Q4_K_M) is 25.0 % closer to the 16-bit model, more than the measurement noise: it is worth keeping." }], verdict: ["Mixed is better."] } }) })})`);
  html = b.run("renderUse()");
  assert.match(html, /id="qf-promotion">Mixed: measured fit to a size \(0.00 GB smaller than Q4_K_M\) is 25.0 % closer[^<]*worth keeping/); assert.match(html, /id="qf-verdict"><li>Mixed is better\.<\/li>/); assert.match(html, /id="qf-ok">The files are made\./);
  assert.doesNotMatch(html, /Dynamic|UD-/i);
});

test("the page loads the box's state when it is opened, and a refusal below the minimum is shown", async () => {
  const b = browser({ quantfit_state: qfState() });
  page(b, exportState({ mixed_quants: measuredQuants }), `qf.open = true; qf.state = null`);
  await b.run("qfLoad()");
  assert.equal(b.calls.at(-1)[0], "quantfit_state");
  assert.match(b.run("renderUse()"), /<option value="\/p\/exports\/other\/model-f16.gguf"/);
  b.run(`qf.state.verdict = {target_gb: 10.6, verdict: ["Mixed is better."]}`);
  assert.match(b.run("renderUse()"), /id="qf-lastverdict">Last verdict \(10.6 GB\): Mixed is better\./);
});
