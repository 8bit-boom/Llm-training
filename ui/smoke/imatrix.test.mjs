// Smoke test of the "Importance matrices" card (the matrix library) and of "Use a saved matrix" in the export box, without a window: app.js runs in a
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
const entry = (over = {}) => ({ id: "20261101-1000-run1", label: "my first", model: "run1", size_mb: 12.5, chunks: 128, ctx: 512, general_used: true, general_source: "mlabonne/open-perfectblend", nextn: false, nextn_asked: false,
  created: "2026-11-01T10:00:00Z", via: "llama.cpp", has_gguf: true, has_dat: false, in_use: false, source: "the trained run run1", calibration: "your saved data (version 1)", notes: [], ...over });
const libState = (over = {}) => ({ entries: [], exports: [], runs: [{ name: "run1", model: "Qwen3.5 9B", status: "done", loss: 0.8 }], models: [{ id: "qwen3.5-4b", name: "Qwen3.5 4B", repo: "Qwen/Qwen3.5-4B", family: "qwen3_5", params_b: 4, reshape: null }, { id: "thin", name: "Qwen thin", repo: "/m/thin", family: "qwen3_5", params_b: 3, reshape: "slimmed: 36 → 28 layers" }],
  data_version: 3, chunks: [64, 128, 256], default_chunks: 128, ctx: 512, general_share: 0.2, estimates: { default: { 64: est(64), 128: est(128), 256: est(256) }, "run:run1": { 64: est(64), 128: est(128), 256: est(256) } }, general: general(), env_ready: true, ...over });

const open = (b, st, extra = "") => { b.run(`state = {project: {name: "P", folder: "/p"}, steps: [], bundle: {world: "W"}, settings: {theme: "light"}, worlds: [], recent: []}; page = "imatrix"; im.state = ${JSON.stringify(st)}; ${extra}`); const h = b.run("renderImatrix()"); b.run("wireImatrix()"); return h; };
const redraw = (b) => { const h = b.run("renderImatrix()"); b.run("wireImatrix()"); return h; };

test("the card is in the navigation next to Slim or grow a model and says what a matrix is and when it helps", () => {
  const b = browser();
  b.run(`state = {project: {name: "P", folder: "/p"}, steps: [], bundle: {world: "W"}, settings: {theme: "light"}, worlds: [], recent: []}; page = "welcome"; renderSteps();`);
  const nav = b.els["steps"].innerHTML;
  assert.match(nav, /data-key="imatrix"[^>]*>🧮 Importance matrices/);
  assert.ok(nav.indexOf("data-key=\"reshape\"") < nav.indexOf("data-key=\"imatrix\""));
  assert.doesNotMatch(nav.match(/<button[^>]*data-key="imatrix"[^>]*>/)[0], /disabled/);
  b.run(`state.bundle = null; renderSteps();`);
  assert.match(b.els["steps"].innerHTML.match(/<button[^>]*data-key="imatrix"[^>]*>/)[0], /disabled/);
  const html = open(browser(), libState());
  assert.match(html, /importance matrix<\/strong> is a short measurement/);
  assert.match(html, /about 4 or 5 bits/); assert.match(html, /For Q8_0 and F16 it does nothing/);
  assert.match(html, /LLAMA_CPP_DIR/); assert.match(html, /No matrix saved in this project yet/);
  assert.match(open(browser(), null), /Loading/);
  assert.match(b.run(`state.bundle = null; renderImatrix()`), /Import your world data first/);
});

test("opening the page asks for the library, and Step 5 links to the card", async () => {
  const b = browser({ imatrix_list: libState({ entries: [entry()] }), export_state: null });
  await new Promise((r) => setTimeout(r, 5));                       // init() goes to the welcome page first
  b.run(`state = {project: {name: "P", folder: "/p"}, steps: [], bundle: {world: "W"}, settings: {theme: "light"}, worlds: [], recent: []};`);
  await b.run(`go("imatrix")`);
  await new Promise((r) => setTimeout(r, 5));
  assert.ok(b.calls.some((c) => c[0] === "imatrix_list"));
  assert.match(b.els["page"].innerHTML, /my first/);
});

test("the list shows label, model, size, chunks, general text, head, date and who made it", () => {
  const html = open(browser(), libState({ entries: [entry(), entry({ id: "20261102-0900-x", label: "", model: "", general_used: false, nextn: true, has_dat: true, chunks: null, via: "unsloth", size_mb: 3, created: "2026-11-02T09:00:00Z" })] }));
  assert.match(html, /<strong>my first<\/strong>/); assert.match(html, /Model: run1 · 12.5 MB · 128 chunks · with general text \(mlabonne\/open-perfectblend\) · no MTP head · 2026-11-01 · made by llama.cpp/);
  assert.match(html, /Model: not stated · 3 MB · unknown amount of text · your own text only · MTP head included · 2026-11-02 · made by unsloth · legacy .dat too/);
  assert.match(html, /Made from: the trained run run1 · text: your saved data \(version 1\)/);
});

test("each entry offers the copy, folder, legacy and delete actions; .dat only when there is none; delete is greyed while a job uses it", () => {
  const html = open(browser(), libState({ entries: [entry(), entry({ id: "b", has_dat: true }), entry({ id: "c", in_use: true })] }));
  assert.equal((html.match(/class="secondary im-copy"/g) || []).length, 3); assert.equal((html.match(/im-open/g) || []).length, 3);
  assert.equal((html.match(/class="secondary im-dat"/g) || []).length, 2, "not for the one that has a .dat");
  assert.match(html, /im-del" data-id="c" disabled title="A job uses this matrix right now"/); assert.match(html, /in use right now/);
});

test("Save a copy asks for a folder, sends it, and says where the copy went; Open folder and Delete call the program", async () => {
  const b = browser({ imatrix_save_copy: "/out/20261101-1000-run1-2", imatrix_delete: libState(), imatrix_open_folder: null }, ["/picked"]);
  open(b, libState({ entries: [entry()] }));
  const btn = (cls, id) => ({ dataset: { id }, onclick: null, cls });
  b.groups[".im-copy"] = [btn("copy", "20261101-1000-run1")]; b.groups[".im-open"] = [btn("open", "20261101-1000-run1")]; b.groups[".im-del"] = [btn("del", "20261101-1000-run1")];
  redraw(b);
  b.groups[".im-copy"][0].onclick();
  let html = redraw(b);
  assert.match(html, /id="im-copydir"/); assert.match(html, /A new folder named <code>20261101-1000-run1<\/code> is made inside it \(a number is added if that name is taken: nothing is overwritten\)/);
  await b.els["im-copybrowse"].onclick();
  assert.equal(b.run("im.copyDir"), "/picked");
  redraw(b);
  b.els["im-copydir"].value = "/picked";                                // what the person sees in the box
  await b.els["im-copygo"].onclick();
  assert.deepEqual(b.calls.at(-1), ["imatrix_save_copy", { id: "20261101-1000-run1", dest: "/picked" }]);
  assert.match(b.run("renderImatrix()"), /Saved a copy in \/out\/20261101-1000-run1-2/); assert.equal(b.run("im.copyFor"), "");
  await b.groups[".im-open"][0].onclick();
  assert.deepEqual(b.calls.at(-1), ["imatrix_open_folder", { id: "20261101-1000-run1" }]);
  await b.groups[".im-del"][0].onclick();
  assert.deepEqual(b.calls.at(-1), ["imatrix_delete", { id: "20261101-1000-run1" }]); assert.match(b.run("renderImatrix()"), /Deleted\./);
  b.confirm(false); b.calls.length = 0;
  await b.groups[".im-del"][0].onclick();
  assert.deepEqual(b.calls, [], "a refusal in the question deletes nothing");
});

test("a refused delete or copy shows the plain message", async () => {
  const b = browser({ imatrix_delete: new Error("This matrix is being used by a job right now."), imatrix_save_copy: new Error("There is not enough free space in /x: 5.0 MB are needed.") });
  open(b, libState({ entries: [entry()] }));
  b.groups[".im-del"] = [{ dataset: { id: "20261101-1000-run1" }, onclick: null }];
  redraw(b);
  await b.groups[".im-del"][0].onclick();
  assert.match(b.run("renderImatrix()"), /class="err">Error: This matrix is being used|This matrix is being used by a job right now/);
  b.run(`im.copyFor = "20261101-1000-run1"; im.copyDir = "/x"`); redraw(b);
  await b.els["im-copygo"].onclick();
  assert.match(b.run("renderImatrix()"), /not enough free space/);
});

test("Make a .dat runs a job with progress and Stop", async () => {
  const b = browser();
  open(b, libState({ entries: [entry()] }));
  b.groups[".im-dat"] = [{ dataset: { id: "20261101-1000-run1" }, onclick: null }];
  redraw(b);
  await b.groups[".im-dat"][0].onclick();
  assert.deepEqual(b.calls.at(-1), ["imatrix_make_dat", { id: "20261101-1000-run1" }]);
  b.run(`onImatrixEvent({kind: "progress", step: "imatrix", value: 0.5, text: "Making the legacy .dat file"})`);
  assert.match(b.run("renderImatrix()"), /Making the legacy .dat file/); assert.match(b.run("renderImatrix()"), /id="im-stop"/);
  b.run(`onImatrixEvent({kind: "done", data: {}})`);
  assert.match(b.run("renderImatrix()"), /The legacy .dat file was made\./);
});

test("the three model sources each show their own fields and notes", () => {
  const b = browser();
  let html = open(b, libState());
  assert.match(html, /<option value="run" selected>A trained run/); assert.match(html, /id="im-run"/); assert.match(html, /merged into its base model/);
  assert.doesNotMatch(html, /id="im-model"|id="im-gguf"/);
  b.els["im-src"].value = "model"; b.els["im-src"].onchange();
  html = redraw(b);
  assert.match(html, /id="im-model"/); assert.match(html, /Qwen thin · 3.0 B · slimmed: 36 → 28 layers/); assert.match(html, /A folder or a Hugging Face id…/); assert.match(html, /Made, slimmed and healed models are in the list/);
  b.els["im-model"].value = "__typed"; b.els["im-model"].onchange();
  html = redraw(b);
  assert.match(html, /id="im-modelpath"/); assert.match(html, /owner\/name/);
  b.els["im-src"].value = "gguf"; b.els["im-src"].onchange();
  html = redraw(b);
  assert.match(html, /id="im-gguf"/); assert.match(html, /A more precise file gives a better matrix/); assert.match(html, /F16 or BF16\) or Q8_0/);
  assert.match(html, /id="im-tok"/, "the chat template is needed for the saved data");
  b.els["im-cal"].value = "text"; b.els["im-cal"].onchange();
  html = redraw(b);
  assert.doesNotMatch(html, /id="im-tok"/); assert.match(html, /id="im-text"/); assert.match(html, /used as it is/);
});

test("the calibration chooser: saved data with the shared general-text chooser, share default 20 %, or the person's own text", () => {
  const b = browser();
  let html = open(b, libState());
  assert.match(html, /My saved training data \(version 3\) and general text/); assert.match(html, /<option value="20" selected>about 20 % \(recommended\)/);
  assert.match(html, /data-scope="imatrix"/); assert.match(html, /Open PerfectBlend \(recommended\)/); assert.match(html, /about 80 % of the text/);
  b.els["im-share"].value = "0"; b.els["im-share"].onchange();
  html = redraw(b);
  assert.doesNotMatch(html, /data-scope="imatrix"/); assert.match(html, /<option value="0" selected>none \(only my data\)/);
  b.els["im-cal"].value = "text"; b.els["im-cal"].onchange();
  html = redraw(b);
  assert.doesNotMatch(html, /data-scope/); assert.match(html, /Plain text, used as it is/);
  html = open(browser(), libState({ data_version: null }));
  assert.match(html, /There is no saved version of the training data yet/); assert.match(html, /id="im-create" disabled/);
});

test("the options: 64 / 128 / 256 chunks (128 chosen), a time guess from the export's formula, the head, the legacy .dat and a label", () => {
  const b = browser();
  let html = open(b, libState());
  assert.match(html, /<option value="64" >64 chunks \(quick\)/); assert.match(html, /<option value="128" selected>128 chunks \(recommended\)/); assert.match(html, /<option value="256" >256 chunks \(thorough\)/);
  assert.match(html, /65,536 tokens, about 4 min on a graphics card \(up to 128 min on the processor\), plus the time to convert the model/);
  b.els["im-chunks"].value = "256"; b.els["im-chunks"].onchange();
  assert.match(redraw(b), /131,072 tokens, about 8 min/);
  html = b.run("renderImatrix()");
  assert.match(html, /id="im-nextn" checked>/); assert.match(html, /id="im-dat" >/); assert.match(html, /id="im-label"/);
});

test("Create the matrix sends the choices; progress and Stop show; the result lists the notes and the statistics lines", async () => {
  const b = browser();
  open(b, libState());
  b.els["im-share"].value = "30"; b.els["im-share"].onchange(); redraw(b);
  b.els["im-label"].value = " Q4 test "; b.els["im-label"].onchange();
  b.els["im-dat"].checked = true; b.els["im-dat"].onchange();
  b.els["im-nextn"].checked = false; b.els["im-nextn"].onchange();
  await b.els["im-create"].onclick();
  const [cmd, args] = b.calls.at(-1);
  assert.equal(cmd, "imatrix_create");
  assert.deepEqual(args.spec, { source: { kind: "run", run: "run1" }, calibration: "data", text_file: null, general: { id: "mlabonne/open-perfectblend", config: null, split: "train", column: null }, general_share: 0.3, chunks: 128, nextn: false, dat: true, label: "Q4 test" });
  b.run(`onImatrixEvent({kind: "progress", step: "imatrix", value: 0.4, text: "Chunk 51 of 128"})`);
  let html = b.run("renderImatrix()");
  assert.match(html, /Chunk 51 of 128/); assert.match(html, /<progress max="1" value="0.4">/); assert.match(html, /id="im-stop"/); assert.match(html, /id="im-create" disabled/);
  redraw(b);
  b.els["im-stop"].onclick();
  assert.equal(b.calls.at(-1)[0], "imatrix_stop");
  b.run(`onImatrixEvent({kind: "done", data: {entry: ${JSON.stringify(entry({ label: "Q4 test" }))}, notes: ["The MTP head was left out: no head."], statistics: {lines: ["Tensors measured: 320", "Largest importance value: 8"], entries: 320}}})`);
  html = b.run("renderImatrix()");
  assert.match(html, /Matrix ready:<\/strong> Q4 test \(12.5 MB\)/); assert.match(html, /The MTP head was left out: no head\./); assert.match(html, /<li>Tensors measured: 320<\/li>/); assert.match(html, /<li>Largest importance value: 8<\/li>/);
  assert.doesNotMatch(html, /id="im-stop"/);
  assert.equal(b.calls.at(-1)[0], "imatrix_list", "the list is read again");
});

test("a result without statistics says so quietly, and Cancelled and failures are shown in plain words", () => {
  const b = browser();
  open(b, libState());
  b.run(`im.running = true; im.pending = "create"; onImatrixEvent({kind: "done", data: {entry: ${JSON.stringify(entry())}, notes: [], statistics: null}})`);
  assert.match(b.run("renderImatrix()"), /This llama-imatrix gave no statistics/);
  b.run(`im.running = true; onImatrixEvent({kind: "error", text: "Cancelled."})`);
  assert.match(b.run("renderImatrix()"), /Cancelled\. Nothing was kept\./);
  b.run(`im.running = true; onImatrixEvent({kind: "error", text: "llama.cpp was not found, so no importance matrix can be made. ... LLAMA_CPP_DIR"})`);
  assert.match(b.run("renderImatrix()"), /class="err">llama.cpp was not found/);
});

test("the other sources send their own fields", async () => {
  const b = browser();
  open(b, libState());
  b.els["im-src"].value = "model"; b.els["im-src"].onchange(); redraw(b);
  b.els["im-model"].value = "thin"; b.els["im-model"].onchange(); redraw(b);
  await b.els["im-create"].onclick();
  assert.deepEqual(b.calls.at(-1)[1].spec.source, { kind: "model", model: "thin" });
  b.run(`im.running = false`);
  b.els["im-model"].value = "__typed"; b.els["im-model"].onchange(); redraw(b);
  assert.match(b.run("renderImatrix()"), /id="im-create" disabled/); assert.match(b.run("renderImatrix()"), /Choose the model first/);
  b.els["im-modelpath"].value = " owner/name "; b.els["im-modelpath"].onchange(); redraw(b);
  await b.els["im-create"].onclick();
  assert.deepEqual(b.calls.at(-1)[1].spec.source, { kind: "model", path: "owner/name" });
  b.run(`im.running = false`);
  b.els["im-src"].value = "gguf"; b.els["im-src"].onchange(); redraw(b);
  b.els["im-gguf"].value = "/m/model-q8_0.gguf"; b.els["im-gguf"].onchange(); b.els["im-tok"].value = "owner/name"; b.els["im-tok"].onchange(); redraw(b);
  b.els["im-cal"].value = "text"; b.els["im-cal"].onchange(); redraw(b);
  b.els["im-text"].value = "/t/mine.txt"; b.els["im-text"].onchange(); redraw(b);
  await b.els["im-create"].onclick();
  const spec = b.calls.at(-1)[1].spec;
  assert.deepEqual(spec.source, { kind: "gguf", file: "/m/model-q8_0.gguf", tokenizer: "owner/name" });
  assert.deepEqual([spec.calibration, spec.text_file, spec.general, spec.general_share], ["text", "/t/mine.txt", null, null]);
});

test("a missing training software disables a model source that needs it but not a GGUF file", () => {
  const b = browser();
  const html = open(b, libState({ env_ready: false }));
  assert.match(html, /The training software is not installed yet/); assert.match(html, /id="im-create" disabled/);
  b.els["im-src"].value = "gguf"; b.els["im-src"].onchange();
  assert.doesNotMatch(redraw(b), /training software is not installed/);
});

test("adding a file from the computer and from an earlier export", async () => {
  const b = browser({ imatrix_import: (a) => ({ added: { id: "n", warning: a.file ? "A legacy .dat matrix cannot be looked into here; it was added as it is." : null }, state: libState({ entries: [entry({ id: "n" })] }) }) }, ["/home/me/old.dat"]);
  open(b, libState({ exports: [{ name: "run1-q4_k_m", model: "run1", chunks: 128, added: false }, { name: "run1-q5", added: true }] }));
  b.els["im-addopen"].onclick();
  let html = redraw(b);
  assert.match(html, /id="im-addlabel"/); assert.match(html, /legacy .dat file/); assert.match(html, /<option value="run1-q4_k_m" >run1-q4_k_m · 128 chunks/); assert.doesNotMatch(html, /value="run1-q5"/);
  b.els["im-addlabel"].value = "from my friend"; b.els["im-addlabel"].onchange();
  await b.els["im-addfile"].onclick();
  assert.deepEqual(b.calls.at(-1), ["imatrix_import", { file: "/home/me/old.dat", label: "from my friend" }]);
  assert.match(b.run("renderImatrix()"), /Added to the library\. A legacy .dat matrix cannot be looked into here/);
  open(b, libState({ exports: [{ name: "run1-q4_k_m", model: "run1", chunks: 128, added: false }] }));
  b.els["im-addexp"].value = "run1-q4_k_m"; b.els["im-addexp"].onchange(); redraw(b);
  await b.els["im-addexpgo"].onclick();
  assert.deepEqual(b.calls.at(-1), ["imatrix_import", { export: "run1-q4_k_m", label: "" }]);
});

test("a refused import shows the plain message", async () => {
  const b = browser({ imatrix_import: new Error("Choose an importance matrix file: imatrix.gguf (the current format) or a legacy .dat file.") }, ["/x/notes.txt"]);
  open(b, libState());
  b.els["im-addopen"].onclick(); redraw(b);
  await b.els["im-addfile"].onclick();
  assert.match(b.run("renderImatrix()"), /Choose an importance matrix file/);
});

// ------------------------------------------------------------------------------------------------------------------ the export page

const exportState = (library = [], over = {}) => ({
  runs: [{ name: "run1", model: "Qwen3.5 9B", status: "done", loss: 0.8 }], exports: [], servers: [], suggested_name: "w", env_ready: true,
  quants: [["q4_k_m", "Q4_K_M: smallest"], ["q8_0", "Q8_0: large"]].map(([id, name]) => ({ id, name })),
  imatrix: { chunks: [64, 128, 256], default_chunks: 128, ctx: 512, general_share: 0.2, no_matrix_for: ["q8_0", "f16"], saved: {}, estimates: { run1: { 64: est(64), 128: est(128), 256: est(256) } }, general: general(), library, ...over },
});
const brief = (id, over = {}) => ({ id, label: "my first", model: "run1", chunks: 128, size_mb: 12, created: "2026-11-01T10:00:00Z", nextn: false, has_dat: false, source: "x", ...over });
const usPage = (b, st) => { b.run(`state = {project: {name: "P", folder: "/p"}, steps: [], bundle: {world: "W"}, settings: {theme: "light"}, worlds: [], recent: []}; page = "use"; us.state = ${JSON.stringify(st)}; us.im = true;`); const h = b.run("renderUse()"); b.run("wireUse()"); return h; };
const usRedraw = (b) => { const h = b.run("renderUse()"); b.run("wireUse()"); return h; };

test("the export box offers 'Use a saved matrix' with the library, an add button and a link to the card; empty library says so", () => {
  const b = browser();
  let html = usPage(b, exportState([brief("20261101-1000-run1"), brief("20261102-0900-x", { label: "", model: "", chunks: null, created: "2026-11-02T09:00:00Z" })]));
  assert.match(html, /Use a saved matrix <select id="us-imsaved"><option value="">Measure a new one now<\/option>/);
  assert.match(html, /<option value="20261101-1000-run1" >my first · run1 · 128 chunks · 2026-11-01/); assert.match(html, /<option value="20261102-0900-x" >20261102-0900-x · other model · unknown amount · 2026-11-02/);
  assert.match(html, /id="us-imadd">Add a matrix file from my computer…/); assert.match(html, /id="us-imlib">Open the matrix library/);
  assert.match(html, /Calibration text/, "measuring is the default: the old box is unchanged");
  assert.match(usPage(browser(), exportState([])), /The library is empty/);
  const c = browser();
  usPage(c, exportState([]));
  c.run(`state.bundle = {world: "W"}`);
  c.els["us-imlib"].onclick();
  assert.equal(c.run("page"), "imatrix");
});

test("choosing a saved matrix hides the measuring options, checks it against the run and shows a yellow warning, then Make the files sends use_saved", async () => {
  const b = browser({ imatrix_check: { readable: true, warnings: ["This matrix has data for 2 layers, but the model has 4. It was probably made from a different model; llama-quantize decides whether it can use it."] } });
  usPage(b, exportState([brief("20261101-1000-run1")]));
  b.els["us-imsaved"].value = "20261101-1000-run1"; b.els["us-imsaved"].onchange();
  let html = usRedraw(b);
  await new Promise((r) => setTimeout(r, 5));
  assert.deepEqual(b.calls.find((c) => c[0] === "imatrix_check"), ["imatrix_check", { id: "20261101-1000-run1", run: "run1" }]);
  html = usRedraw(b);
  assert.match(html, /Nothing is measured: this matrix is copied into the export and given to <code>llama-quantize<\/code>/);
  assert.doesNotMatch(html, /id="us-imchunks"|data-scope="imatrix"|id="us-imreuse"/);
  assert.match(html, /<div class="warn small" id="us-imwarn"><strong>Check before you export\.<\/strong> This matrix has data for 2 layers, but the model has 4\./);
  assert.match(html, /id="us-make" >/, "a warning never blocks the export");
  await b.els["us-make"].onclick();
  const a = b.calls.find((c) => c[0] === "export_start")[1];
  assert.equal(a.imatrix.use_saved, "20261101-1000-run1"); assert.equal(a.imatrix.enabled, true);
});

test("an unreadable matrix and a failed check are warnings too; going back to measuring shows the old box; a deleted entry is forgotten", async () => {
  const b = browser({ imatrix_check: new Error("That saved matrix no longer exists.") });
  usPage(b, exportState([brief("a")]));
  b.els["us-imsaved"].value = "a"; b.els["us-imsaved"].onchange(); usRedraw(b);
  await new Promise((r) => setTimeout(r, 5));
  assert.match(usRedraw(b), /id="us-imwarn">Error: That saved matrix no longer exists\.|id="us-imwarn">That saved matrix no longer exists\./);
  b.els["us-imsaved"].value = ""; b.els["us-imsaved"].onchange();
  assert.match(usRedraw(b), /Calibration text/);
  b.run(`us.imSaved = "gone"`);
  const b2 = browser({ export_state: exportState([brief("a")]) });
  usPage(b2, exportState([brief("a")]));
  b2.run(`us.imSaved = "gone"`);
  await b2.run("usRefresh()");
  assert.equal(b2.run("us.imSaved"), "");
});

test("changing the run checks the saved matrix again for that run", async () => {
  const b = browser({ imatrix_check: { readable: true, warnings: [] } });
  const st = exportState([brief("a")]); st.runs.push({ name: "run2", model: "Qwen3.5 4B", status: "done" });
  usPage(b, st);
  b.els["us-imsaved"].value = "a"; b.els["us-imsaved"].onchange(); usRedraw(b);
  b.els["us-run"].value = "run2"; b.els["us-run"].onchange(); usRedraw(b);
  await new Promise((r) => setTimeout(r, 5));
  assert.deepEqual(b.calls.filter((c) => c[0] === "imatrix_check").map((c) => c[1].run), ["run1", "run2"]);
});

test("adding a matrix file from the export page imports it and selects it", async () => {
  const b = browser({ imatrix_import: { added: { id: "new1", warning: null }, state: libState() }, export_state: exportState([brief("new1", { label: "from a friend" })]) }, ["/x/m.gguf"]);
  usPage(b, exportState([]));
  b.els["us-imadd"].onclick();
  let html = usRedraw(b);
  assert.match(html, /id="us-imaddlabel"/); assert.match(html, /A \.gguf matrix \(the current format\) or a legacy \.dat file, for example one made on another computer/);
  b.els["us-imaddlabel"].value = "from a friend"; b.els["us-imaddlabel"].onchange();
  await b.els["us-imaddfile"].onclick();
  assert.deepEqual(b.calls.find((c) => c[0] === "imatrix_import"), ["imatrix_import", { file: "/x/m.gguf", label: "from a friend" }]);
  assert.equal(b.run("us.imSaved"), "new1");
  assert.match(b.run("renderUse()"), /Added to the library\./);
});

test("without a saved matrix chosen the export options are exactly as before (no use_saved key)", async () => {
  const b = browser();
  usPage(b, exportState([brief("a")]));
  await b.els["us-make"].onclick();
  const a = b.calls.find((c) => c[0] === "export_start")[1];
  assert.equal("use_saved" in a.imatrix, false);
});
