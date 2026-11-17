// Smoke test of the llama.cpp installer in the window (docs/LLAMA_CPP_INSTALL.md): the section of the Training setup page in each state (absent, confirming, installing with
// progress, installed, update available, failed with the CPU offer), the "Install llama.cpp now" button wherever llama.cpp is needed (Step 5, Importance matrices, the merge,
// Compare quants) and on an error with the code mark, and the confirmation before the first download. app.js runs in a stand-in browser that records HTML and every call.
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

const FOUND = { folder: "C:/Users/me/AppData/Roaming/NDLoreForge/llama.cpp", how: "program_install", how_label: "installed by this program", convert: "x", quantize: "y" };
const absent = (over = {}) => ({ install_dir: "C:/d/llama.cpp", platform: { os: "windows", arch: "x64" }, installed: null, install_error: null, layout_ok: false, folder_without_record: false, found: null, tools: { imatrix: null, perplexity: null }, converter_found: false, last_failure: null, pinned_tag: null, search_order: [], ...over });
const installed = (over = {}) => absent({ installed: { tag: "b6000", backend: "cuda", cuda_version: "13.5", installed_at: "2026-11-15T10:00:00+00:00", versions: { "llama-imatrix": "version: 6000 (abc1234)" }, converter_check: "ok" }, layout_ok: true, found: FOUND, converter_found: true,
  tools: { imatrix: { path: "i", how: "program_install", how_label: "installed by this program" }, perplexity: { path: "p", how: "program_install", how_label: "installed by this program" } }, ...over });
const PLAN = { tag: "b6000", backend: "cuda", cuda_version: "13.5", reason: "The CUDA 13.5 build: NVIDIA GeForce RTX 5090 and its driver can use it.", notes: [], assets: [{ name: "llama-b6000-bin-win-cuda-13.5-x64.zip", size: 270e6 }, { name: "cudart-llama-bin-win-cuda-13.5-x64.zip", size: 380e6 }], total_bytes: 680e6, total_mb: 680, options: [], gpu: { name: "NVIDIA GeForce RTX 5090", driver_cuda: "12.9", blackwell: true } };

const base = (b, extra = "") => b.run(`state = {project: {name: "P", folder: "/p"}, steps: [], bundle: {world: "W"}, settings: {theme: "light", expert_mode: false}, worlds: [], recent: []}; ${extra}`);
const setup = (b, status, extra = "") => { base(b, `page = "setup"; env.status = {checks: [], can_install: true, env_installed: true}; lc.status = ${JSON.stringify(status)}; ${extra}`); const h = b.run("renderSetup()"); wired(b); return h; };
const section = (html) => html.slice(html.indexOf('id="lc-section"'));
// the stand-in finds buttons made by a list through `groups`: every class of the section gets stand-in buttons before wireLlama() runs
const wired = (b, spec = {}) => { for (const sel of [".lc-start", ".lc-go", ".lc-notnow", ".lc-cancel", ".lc-cpu", ".lc-check", ".lc-remove"]) b.groups[sel] = Array.from({ length: (spec[sel] || [{}]).length }, (_, i) => ({ onclick: null, dataset: (spec[sel] || [{}])[i] })); b.run("wireLlama()"); };
const click = (b, sel, i = 0) => b.groups[sel][i].onclick();
const qstate = () => ({ entries: [], candidates: { references: [], quants: [] }, runs: [], models: [], data_version: 3, has_test_rows: true, chunks: [4, 8, 16], default_chunks: 16, ctx: 512, free_bytes: 80e9, env_ready: true });
const imstate = () => ({ entries: [], exports: [], runs: [{ name: "run1", model: "Qwen3.5 9B", status: "done", loss: 0.8 }], models: [], data_version: 3, chunks: [64, 128, 256], default_chunks: 128, ctx: 512, general_share: 0.2, estimates: { default: {} }, general: { default: "x", has_token: false, acknowledged: [], custom: [], choice: {}, presets: [] }, env_ready: true, published_bases: [] });

test("absent: the section says so, offers the install, shows nothing downloaded yet and the honest note", () => {
  const b = browser();
  const html = section(setup(b, absent()));
  assert.match(html, /<h2>llama.cpp<\/h2>/); assert.match(html, /Not installed\.<\/strong> The program can install it for you/);
  assert.match(html, /class="primary lc-start"[^>]*>Install llama.cpp<\/button>/);
  assert.doesNotMatch(html, /lc-remove|lc-cpu|lc-cancel|<progress/);
  assert.match(html, /SmartScreen or an antivirus may warn/); assert.match(html, /Nothing needs administrator rights/); assert.match(html, /LLAMA_CPP_DIR<\/code> always wins/);
  assert.match(html, /about 30 MB for a processor-only build, several hundred MB for a CUDA build/);
});

test("the first download is confirmed with its size first: one click, no typing, and nothing is installed before it", async () => {
  const b = browser({ llamacpp_plan: PLAN });
  setup(b, absent());
  await click(b, ".lc-start");
  assert.deepEqual(b.calls.filter((c) => c[0].startsWith("llamacpp")), [["llamacpp_plan", { backend: "auto", tag: null }]]);
  let html = b.run("renderSetup()"); wired(b);
  assert.match(html, /Install llama.cpp b6000: NVIDIA \(CUDA 13.5\)\?/); assert.match(html, /downloads about <strong>680 MB<\/strong> from github.com/);
  assert.match(html, /llama-b6000-bin-win-cuda-13.5-x64.zip<\/code> \(270 MB\)/); assert.match(html, /RTX 5090/);
  assert.match(html, /class="primary lc-go">Download and install \(680 MB\)<\/button>/);
  assert.match(html, /nothing needs administrator rights/);
  assert.equal(b.calls.filter((c) => c[0] === "llamacpp_install").length, 0, "no download before the click");
  click(b, ".lc-notnow");
  assert.equal(b.run("lc.confirm"), null); assert.doesNotMatch(b.run("renderSetup()"), /lc-go/);
  await click(b, ".lc-start"); b.run("renderSetup()"); wired(b);
  await click(b, ".lc-go");
  assert.deepEqual(b.calls.filter((c) => c[0] === "llamacpp_install"), [["llamacpp_install", { backend: "cuda", tag: "b6000" }]], "what was shown is what is installed");
  assert.equal(b.run("lc.running"), true);
});

test("a plan that cannot be made shows its plain message", async () => {
  const b = browser({ llamacpp_plan: () => { throw "GitHub could not be reached. Check the internet connection"; } });
  setup(b, absent());
  await click(b, ".lc-start");
  assert.match(b.run("renderSetup()"), /class="err lc-error">GitHub could not be reached/);
  assert.equal(b.run("lc.planning"), false);
});

test("installing: the phase, the bytes, a progress bar and Cancel; the events of the environment channel are routed here", () => {
  const b = browser();
  setup(b, absent(), `lc.running = true; lc.step = "";`);
  b.run(`onEnvEvent({kind: "step", task: "llamacpp", index: 2, total: 6, phase: "download", text: "Downloading the program files (270 MB)"})`);
  b.run(`onEnvEvent({kind: "progress", task: "llamacpp", step: "download", value: 0.25, text: "Downloading the program files: 120 MB of 270 MB", bytes: 120e6, total: 680e6})`);
  b.run(`onEnvEvent({kind: "log", task: "llamacpp", text: "Release b6000"})`);
  const html = section(b.run("renderSetup()"));
  assert.match(html, /Step 2 of 6: Downloading the program files \(270 MB\)|Downloading the program files: 120 MB of 270 MB/);
  assert.match(html, /<progress max="1" value="0.25">/); assert.match(html, /120 MB of about 680 MB/); assert.match(html, /class="secondary lc-cancel">Cancel/); assert.match(html, /Release b6000/);
  assert.match(html, /lc-start" disabled/, "no second install while one runs");
  assert.equal(b.run("env.log.length"), 0, "not mixed into the environment's own log");
  wired(b); click(b, ".lc-cancel");
  assert.deepEqual(b.calls.filter((c) => c[0] === "llamacpp_cancel"), [["llamacpp_cancel", {}]]);
  // the done / error events carry no task: the page knows by lc.running
  b.run(`onEnvEvent({kind: "error", text: "Cancelled."})`);
  assert.equal(b.run("lc.running"), false); assert.equal(b.run("lc.error"), "Cancelled. Nothing was changed.");
  assert.equal(b.run("env.error"), null);
});

test("installed: where it is, how it was found, the release and the versions, and Remove; the result after an install", async () => {
  const b = browser({ llamacpp_status: installed() });
  let html = section(setup(b, installed()));
  assert.match(html, /Found<\/strong> in <code>C:\/Users\/me\/AppData\/Roaming\/NDLoreForge\/llama.cpp<\/code> \(installed by this program\)/);
  assert.match(html, /Release <strong>b6000<\/strong>, cuda 13.5, installed 2026-11-15/);
  assert.match(html, /llama-imatrix: version: 6000 \(abc1234\)/); assert.match(html, /Converter: ok/);
  assert.match(html, /class="secondary lc-remove"/); assert.match(html, /Install again/);
  // a folder of the person's own: found, but not ours: no Remove
  const own = installed({ installed: null, found: { ...FOUND, how: "env_var", how_label: "your LLAMA_CPP_DIR folder", folder: "D:/llama.cpp" } });
  html = section(setup(browser(), own));
  assert.match(html, /\(your LLAMA_CPP_DIR folder\)/); assert.doesNotMatch(html, /lc-remove/);
  // after an install
  b.run(`onEnvEvent({kind: "done", data: {task: "llamacpp", tag: "b6000", backend: "cuda", cuda_version: "13.5", versions: {"llama-imatrix": "version: 6000 (abc)"}, converter_check: "ok"}})`);
  await new Promise((r) => setTimeout(r, 5));
  assert.ok(b.calls.some((c) => c[0] === "llamacpp_status"), "the status is read again");
  assert.match(section(b.run("renderSetup()")), /llama.cpp b6000 is installed<\/strong> \(cuda 13.5\)/);
  // remove asks first
  wired(b); b.confirm(false); await click(b, ".lc-remove");
  assert.equal(b.calls.filter((c) => c[0] === "llamacpp_remove").length, 0);
  b.confirm(true); await click(b, ".lc-remove");
  assert.equal(b.calls.filter((c) => c[0] === "llamacpp_remove").length, 1);
});

test("an update is a hint with its own button; the check says when there is none", async () => {
  const b = browser({ llamacpp_check_update: { installed: "b6000", latest: "b6100", newer: true }, llamacpp_plan: { ...PLAN, tag: "b6100" } });
  setup(b, installed(), `lc.backend = "auto"`);
  await click(b, ".lc-check");
  let html = section(b.run("renderSetup()")); wired(b);
  assert.match(html, /A newer release is available: <strong>b6100<\/strong> \(installed: b6000\)/); assert.match(html, /class="secondary lc-start" data-backend="cuda">Update<\/button>/);
  wired(b, { ".lc-start": [{}, { backend: "cuda" }] });
  await click(b, ".lc-start", 1);   // Update: the same backend as the installed one
  assert.deepEqual(b.calls.at(-1), ["llamacpp_plan", { backend: "cuda", tag: null }]);
  const b2 = browser({ llamacpp_check_update: { installed: "b6000", latest: "b6000", newer: false } });
  setup(b2, installed()); await click(b2, ".lc-check");
  html = section(b2.run("renderSetup()"));
  assert.match(html, /This is the newest release \(b6000\)/); assert.doesNotMatch(html, />Update</);
});

test("a failed CUDA install offers the CPU build with one click (no second question)", async () => {
  const fail = absent({ last_failure: { code: "cuda_start_failed", message: "x", offer_cpu: true } });
  const b = browser({ llamacpp_plan: { ...PLAN, backend: "cpu", cuda_version: null, assets: [{ name: "llama-b6000-bin-win-cpu-x64.zip", size: 30e6 }], total_bytes: 60e6 } });
  setup(b, fail, `lc.error = "The CUDA build of llama.cpp does not start on this computer (llama-imatrix: missing DLL)."`);
  const html = section(b.run("renderSetup()")); wired(b);
  assert.match(html, /class="err lc-error">The CUDA build of llama.cpp does not start/); assert.match(html, /class="primary lc-cpu">Install the CPU build instead<\/button>/);
  await click(b, ".lc-cpu");
  assert.deepEqual(b.calls.filter((c) => c[0].startsWith("llamacpp")), [["llamacpp_plan", { backend: "cpu", tag: null }], ["llamacpp_install", { backend: "cpu", tag: "b6000" }]]);
  // a cancel is not a failure: no offer
  const b2 = browser();
  setup(b2, absent(), `lc.error = "Cancelled. Nothing was changed."`);
  assert.doesNotMatch(section(b2.run("renderSetup()")), /lc-cpu/);
});

test("Expert mode adds the choice of build and the CPU button; the choice is what is planned", async () => {
  const b = browser({ llamacpp_plan: PLAN });
  setup(b, absent(), `state.settings.expert_mode = true;`);
  let html = section(b.run("renderSetup()")); wired(b);
  assert.match(html, /<select id="lc-backend"[^>]*>[^]*value="vulkan"[^]*value="cpu"[^]*value="build"/); assert.match(html, /lc-cpu/);
  b.els["lc-backend"].value = "vulkan"; b.els["lc-backend"].onchange();
  await click(b, ".lc-start");
  assert.deepEqual(b.calls.at(-1), ["llamacpp_plan", { backend: "vulkan", tag: null }]);
  assert.doesNotMatch(section(setup(browser(), absent())), /id="lc-backend"/, "not shown in the simple mode");
});

test("a damaged record and a folder that is not ours are said plainly", () => {
  assert.match(section(setup(browser(), absent({ install_error: "install.json is damaged" }))), /record of the installed copy is damaged \(install.json is damaged\)/);
  assert.match(section(setup(browser(), absent({ folder_without_record: true }))), /exists but was not made by this program or is incomplete/);
});

// ---------------------------------------------------------------------------------------------------------------- the button wherever llama.cpp is needed

test("Importance matrices: a notice with the button when llama-imatrix is missing, nothing when it is there", () => {
  const b = browser();
  base(b, `page = "imatrix"; im.state = ${JSON.stringify(imstate())}; lc.status = ${JSON.stringify(absent())}`);
  let html = b.run("renderImatrix()"); wired(b);
  assert.match(html, /class="warn llama-notice" data-need="imatrix"><strong>llama.cpp is not installed \(or not found\)/); assert.match(html, /It is needed to measure an importance matrix/);
  assert.match(html, /class="primary lc-start">Install llama.cpp now<\/button>/);
  assert.match(html, /then in the folder it installed llama.cpp into, then in a folder called llama.cpp in your home folder, then next to the program/);
  assert.match(html, /<code>llama-imatrix<\/code> and <code>llama-perplexity<\/code> are also looked for on the PATH \(the converter script is not\)/);
  assert.doesNotMatch(html, /then on the PATH\./, "the old inaccurate sentence is gone");
  assert.match(html, /LLAMA_CPP_DIR/);
  assert.match(html, /Merge matrices<\/h2><div class="warn llama-notice" data-need="merge">/, "the merge card has its own button");
  b.run(`lc.status = ${JSON.stringify(installed())}`);
  assert.doesNotMatch(b.run("renderImatrix()"), /llama-notice/);
  // llama-imatrix is there but there is no converter and the model must be converted
  b.run(`lc.status = ${JSON.stringify(installed({ converter_found: false, found: null }))}; im.src = "run"`);
  assert.match(b.run("renderImatrix()"), /data-need="imatrix"/);
  b.run(`im.src = "gguf"`);
  assert.doesNotMatch(b.run("renderImatrix()").replace(/data-need="merge"/g, ""), /data-need="imatrix"/, "a GGUF source needs only llama-imatrix");
  b.run(`lc.status = null`);
  assert.doesNotMatch(b.run("renderImatrix()"), /llama-notice/, "unknown status: no false alarm");
});

test("an error that ends with the code mark loses the mark and gets the install button, the page is read again after the install", async () => {
  const b = browser({ llamacpp_plan: PLAN, imatrix_list: imstate(), llamacpp_status: installed() });
  base(b, `page = "imatrix"; im.state = ${JSON.stringify(imstate())}; lc.status = ${JSON.stringify(installed())}; im.error = "llama.cpp was not found, so no importance matrix can be made. LLAMA_CPP_DIR [llama_cpp_missing]"`);
  const html = b.run("renderImatrix()"); wired(b);
  assert.match(html, /class="err">llama.cpp was not found, so no importance matrix can be made. LLAMA_CPP_DIR<\/div>/); assert.doesNotMatch(html, /\[llama_cpp_missing\]/);
  assert.match(html, /class="primary lc-start">Install llama.cpp now<\/button>/, "shown although the status says it is there: the error is the witness");
  await click(b, ".lc-start"); b.run("renderImatrix()"); wired(b); await click(b, ".lc-go");
  b.run(`onEnvEvent({kind: "done", data: {task: "llamacpp", tag: "b6000", backend: "cpu"}})`);
  await new Promise((r) => setTimeout(r, 10));
  assert.equal(b.run("im.error"), null, "the old error is cleared so the person can retry");
  assert.ok(b.calls.filter((c) => c[0] === "imatrix_list").length >= 1, "the matrix page is read again");
  assert.match(b.run("renderImatrix()"), /llama.cpp b6000 is installed/);
  // an error without the mark has no button
  b.run(`im.error = "The matrix could not be made (disk full)."; lc.result = null`);
  assert.doesNotMatch(b.run("renderImatrix()"), /lc-start/);
});

test("Compare quants: the notice names llama-perplexity; the error with the mark gets the button", () => {
  const b = browser();
  base(b, `page = "quantcheck"; qk.state = ${JSON.stringify(qstate())}; lc.status = ${JSON.stringify(installed({ tools: { imatrix: { how: "x", how_label: "x" }, perplexity: null } }))}`);
  let html = b.run("renderQuantcheck()");
  assert.match(html, /data-need="perplexity"[^]*Compare quants needs its llama-perplexity program/); assert.match(html, /Install llama.cpp now/);
  b.run(`lc.status = ${JSON.stringify(installed())}`);
  assert.doesNotMatch(b.run("renderQuantcheck()"), /llama-notice/);
  b.run(`qk.error = "llama-perplexity was not found, so the quants cannot be compared. [llama_cpp_missing]"`);
  html = b.run("renderQuantcheck()");
  assert.match(html, /class="err">llama-perplexity was not found, so the quants cannot be compared.<\/div>/); assert.match(html, /lc-start/);
});

test("Step 5: ticking the importance matrix or a mixed size shows the button when llama.cpp is missing; the export error with the mark has it too", () => {
  const b = browser();
  base(b, `page = "use"; us.state = ${JSON.stringify(exportState())}; lc.status = ${JSON.stringify(absent())}`);
  assert.doesNotMatch(b.run("renderUse()"), /llama-notice/, "nothing is needed until the matrix or a mixed size is chosen");
  b.run(`us.im = true`);
  let html = b.run("renderUse()");
  assert.equal((html.match(/class="primary lc-start"/g) || []).length, 1); assert.match(html, /Install llama.cpp now/);
  b.run(`us.im = false; us.quants = ["mixed_4"]`);
  html = b.run("renderUse()");
  assert.match(html, /llama-notice[^]*id="us-mxwhy"/); assert.equal((html.match(/class="primary lc-start"/g) || []).length, 1);
  b.run(`us.im = true`);
  assert.equal((b.run("renderUse()").match(/class="primary lc-start"/g) || []).length, 1, "one button, not two");
  b.run(`lc.status = ${JSON.stringify(installed())}; us.error = "The mixed-precision sizes need llama.cpp (convert_hf_to_gguf.py and llama-quantize). [llama_cpp_missing]"`);
  html = b.run("renderUse()");
  assert.match(html, /class="err">The mixed-precision sizes need llama.cpp[^<]*<\/div>/); assert.match(html, /lc-start/); assert.doesNotMatch(html, /\[llama_cpp_missing\]/);
});

test("while it installs, every page that needs llama.cpp shows the progress where the button was", () => {
  const b = browser();
  base(b, `page = "quantcheck"; qk.state = ${JSON.stringify(qstate())}; lc.status = ${JSON.stringify(absent())}; lc.running = true; lc.step = "Checking that it starts"; lc.progress = 0.9`);
  const html = b.run("renderQuantcheck()");
  assert.match(html, /Installing llama.cpp…[^]*Checking that it starts[^]*<progress max="1" value="0.9">/); assert.match(html, /lc-cancel/); assert.doesNotMatch(html, /lc-start/);
});

test("starting the app on these pages reads the status (go)", async () => {
  const b = browser({ llamacpp_status: installed() });
  base(b, `page = "welcome"`);
  await b.run(`go("quantcheck")`);
  assert.ok(b.calls.some((c) => c[0] === "llamacpp_status"));
  assert.equal(b.run("lc.status.installed.tag"), "b6000");
});
