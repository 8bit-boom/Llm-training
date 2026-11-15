// Smoke test of "Merge matrices" (the Merge card, Unsloth's published matrix, the merge after Create), without a window: app.js runs in a
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
  // buttons made by a list ("class" handlers) are found through querySelectorAll: the stand-in answers with the elements named in `groups`
  const groups = {};
  const doc = { getElementById: el, querySelectorAll: (sel) => groups[sel] || [], querySelector: () => null, createElement: () => el("tmp"), body: { appendChild() {} }, documentElement: { setAttribute() {}, removeAttribute() {} } };
  let confirmed = true;
  const ctx = vm.createContext({ window: win, document: doc, console, confirm: () => confirmed, setTimeout, clearTimeout });
  vm.runInContext(source, ctx);
  return { els, calls, groups, run: (code) => vm.runInContext(code, ctx), confirm: (v) => { confirmed = v; } };
}

const est = (c) => ({ chunks: c, ctx: 512, tokens: c * 512, gpu_minutes: c / 32, cpu_minutes: c });
const P = (key, over = {}) => ({ key, id: key, name: key, license: "Apache-2.0", license_note: "", kind: "chat", note: "A note.", config: null, split: "train", gated: false, needs_ack: false, url: "", ...over });
const general = () => ({ default: "mlabonne/open-perfectblend", has_token: false, acknowledged: [], custom: [], choice: {}, presets: [P("mlabonne/open-perfectblend", { name: "Open PerfectBlend (recommended)" })] });
const entry = (over = {}) => ({ id: "20261101-1000-run1", label: "my first", model: "run1", size_mb: 12.5, chunks: 64, ctx: 512, general_used: true, general_source: "mlabonne/open-perfectblend", nextn: false, nextn_asked: false,
  created: "2026-11-01T10:00:00Z", via: "llama.cpp", has_gguf: true, has_dat: false, in_use: false, source: "the trained run run1", calibration: "your saved data (version 1)", notes: [], ...over });
const libState = (over = {}) => ({ entries: [], exports: [], runs: [{ name: "run1", model: "Qwen3.5 9B", status: "done", loss: 0.8, base: "Qwen/Qwen3.8-27B" }], models: [{ id: "qwen3.5-4b", name: "Qwen3.5 4B", repo: "Qwen/Qwen3.5-4B", family: "qwen3_5", params_b: 4, reshape: null }, { id: "thin", name: "Qwen thin", repo: "/m/thin", family: "qwen3_5", params_b: 3, reshape: "slimmed" }],
  published_bases: [{ repo: "Qwen/Qwen3.8-27B", label: "Qwen/Qwen3.8-27B (the base of the run run1)" }, { repo: "Qwen/Qwen3.5-4B", label: "Qwen/Qwen3.5-4B" }],
  data_version: 3, chunks: [64, 128, 256], default_chunks: 128, ctx: 512, general_share: 0.2, estimates: { default: { 64: est(64), 128: est(128), 256: est(256) }, "run:run1": { 64: est(64), 128: est(128), 256: est(256) } }, general: general(), env_ready: true, ...over });
const two = () => libState({ entries: [entry(), entry({ id: "20261102-1000-b", label: "second", chunks: 128 })] });
const pubInfo = (over = {}) => ({ base: "Qwen/Qwen3.8-27B", repo: "unsloth/Qwen3.8-27B-GGUF", file: "imatrix_unsloth.gguf", local_name: "imatrix_unsloth.gguf", size: 157286400, url: "https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/imatrix_unsloth.gguf", page: "https://huggingface.co/unsloth/Qwen3.8-27B-GGUF", gated: false, cached: false, cache_file: "/p/imatrix_downloads/unsloth--Qwen3.8-27B-GGUF/imatrix_unsloth.gguf", ...over });

const open = (b, st, extra = "") => { b.run(`state = {project: {name: "P", folder: "/p"}, steps: [], bundle: {world: "W"}, settings: {theme: "light"}, worlds: [], recent: []}; page = "imatrix"; im.state = ${JSON.stringify(st)}; ${extra}`); const h = b.run("renderImatrix()"); b.run("wireImatrix()"); return h; };
const redraw = (b) => { const h = b.run("renderImatrix()"); b.run("wireImatrix()"); return h; };
const J = (b, code) => JSON.parse(b.run(`JSON.stringify(${code})`));
const tick = (b, ids) => { b.groups[".im-pick"] = ids.map((id) => ({ value: id, checked: true, onchange: null })); redraw(b); b.groups[".im-pick"][0].onchange(); };

test("every entry has a tick to merge it; a merged entry says so and names its parents", () => {
  const merged = entry({ id: "20261103-1000-merged", label: "mine + Unsloth's", source_kind: "merged", parents: [{ kind: "library", label: "my first", chunks: 64 }, { kind: "published", label: "Unsloth's published matrix for Qwen/Qwen3.8-27B", repo: "unsloth/Qwen3.8-27B-GGUF", chunks: 1000 }], via: "llama.cpp", source: "a merge of my first, Unsloth's" });
  const html = open(browser(), libState({ entries: [entry(), merged] }));
  assert.equal((html.match(/class="im-pick"/g) || []).length, 2);
  assert.match(html, /<span class="tag">merged<\/span>/); assert.match(html, /Parents: my first \(64 chunks\) \+ Unsloth&#39;s published matrix for Qwen\/Qwen3.8-27B \(1000 chunks\) · from unsloth\/Qwen3.8-27B-GGUF/);
  assert.match(html, /Made from: a merge of my first/);
});

test("the Merge card says honestly what merging does, and needs two things", () => {
  const b = browser();
  let html = open(b, two());
  assert.match(html, /<h2>Merge matrices<\/h2>/); assert.match(html, /llama-imatrix --in-file a --in-file b/); assert.match(html, /Merging <em>adds<\/em> the statistics/);
  assert.match(html, /A matrix measured on more text therefore outweighs a smaller one/); assert.match(html, /a large published matrix can drown a small one of your own/);
  assert.match(html, /describes the <em>unmodified base model<\/em>, not your fine-tune/); assert.match(html, /a mismatch is a warning, and if llama-imatrix refuses, its own message is shown/);
  assert.match(html, /id="im-merge-go" disabled/); assert.match(html, /a merge needs at least two things/);
  tick(b, ["20261101-1000-run1", "20261102-1000-b"]);
  assert.deepEqual(J(b, "im.mergePicked"), ["20261101-1000-run1", "20261102-1000-b"]);
  html = redraw(b);
  assert.doesNotMatch(html, /id="im-merge-go" disabled/); assert.match(html, /2 ticked in the list/);
  assert.match(open(browser(), libState({ entries: [entry()], env_ready: false }), `im.mergePicked = ["20261101-1000-run1"]; im.mergePub = true`), /not set up yet \(open Training setup\)/);
});

test("matrix files can be added and removed, by path or by browsing", async () => {
  const b = browser({}, ["/picked/imatrix.gguf"]);
  open(b, two());
  b.els["im-mergefile"].value = "  C:\\m\\friend.gguf "; b.els["im-mergefile"].onchange(); b.els["im-mergefileadd"].onclick();
  assert.deepEqual(J(b, "im.mergeFiles"), ["C:\\m\\friend.gguf"]);
  await b.els["im-mergefilebrowse"].onclick();
  assert.deepEqual(J(b, "im.mergeFiles"), ["C:\\m\\friend.gguf", "/picked/imatrix.gguf"]);
  assert.match(redraw(b), /C:\\m\\friend.gguf/);
  b.groups[".im-unmergefile"] = [{ dataset: { i: "0" }, onclick: null }]; redraw(b);
  b.groups[".im-unmergefile"][0].onclick();
  assert.deepEqual(J(b, "im.mergeFiles"), ["/picked/imatrix.gguf"]);
  assert.match(b.run("renderImatrix()"), /1 file\(s\)/);
});

test("Unsloth's published matrix: the choice of base model, the lookup and what it will download", async () => {
  const b = browser({ imatrix_published: pubInfo() });
  let html = open(b, two());
  assert.match(html, /Also use Unsloth's published matrix for the base model/); assert.doesNotMatch(html, /id="im-pub-lookup"/);
  b.els["im-mergepub"].checked = true; b.els["im-mergepub"].onchange();
  html = redraw(b);
  assert.match(html, /id="im-mergepubbase"/); assert.match(html, /<option value="Qwen\/Qwen3.8-27B" selected>Qwen\/Qwen3.8-27B \(the base of the run run1\)/);
  assert.match(html, /unsloth\/&lt;model name&gt;-GGUF/); assert.match(html, /so this one step needs the internet/);
  await b.els["im-pub-lookup"].onclick();
  assert.deepEqual(b.calls.at(-1), ["imatrix_published", { base: "Qwen/Qwen3.8-27B" }]);
  html = b.run("renderImatrix()");
  assert.match(html, /id="im-pub-info"/); assert.match(html, /<strong>unsloth\/Qwen3.8-27B-GGUF<\/strong> · file <code>imatrix_unsloth.gguf<\/code> · 150.0 MB · not gated/);
  assert.match(html, /This merge downloads 150.0 MB from huggingface.co<\/strong> into this project's <code>imatrix_downloads<\/code> folder, once/); assert.match(html, /Source: <code>https:\/\/huggingface.co\/unsloth\/Qwen3.8-27B-GGUF<\/code>/);
  b.run(`im.mergePubInfo = ${JSON.stringify(pubInfo({ cached: true }))}`);
  assert.match(b.run("renderImatrix()"), /Already downloaded: <code>\/p\/imatrix_downloads[^<]*<\/code> \(nothing will be downloaded\)/);
  b.run(`im.mergePubInfo = ${JSON.stringify(pubInfo({ gated: true, size: null }))}`);
  assert.match(b.run("renderImatrix()"), /<strong>gated<\/strong>/); assert.match(b.run("renderImatrix()"), /size not stated/);
  // changing the base forgets the lookup
  b.els["im-mergepubbase"].value = "Qwen/Qwen3.5-4B"; b.els["im-mergepubbase"].onchange();
  assert.equal(b.run("im.mergePubInfo"), null); assert.equal(b.run("im.mergePubBase"), "Qwen/Qwen3.5-4B");
  assert.match(open(browser(), libState({ published_bases: [] })), /id="im-mergepub"[^>]* disabled/);
  assert.match(open(browser(), libState({ published_bases: [] })), /No base model on the Hugging Face Hub is known in this project yet/);
});

test("every Hub problem is shown in the program's plain words", async () => {
  const msgs = ["huggingface.co could not be reached (URLError). Check the internet connection; only this download needs it, everything else in this program works offline.",
    "There is no published matrix for Qwen/Unknown: the Hub has no repository unsloth/Unknown-GGUF (or it is private).", "The repository unsloth/X-GGUF needs you to accept its terms on its Hugging Face page and a Hugging Face token to download it. (HTTP 401)",
    "Qwen/x is a folder on this computer (a model made by Slim or grow, or a healed one), so it has no published matrix on the Hub."];
  for (const m of msgs) {
    const b = browser({ imatrix_published: new Error(m) });
    open(b, two(), `im.mergePub = true; im.mergePubBase = "Qwen/Qwen3.8-27B"`);
    await b.els["im-pub-lookup"].onclick();
    const html = b.run("renderImatrix()");
    assert.match(html, /id="im-pub-error"/); assert.ok(html.includes(esc(m)) || html.includes(m.replace(/'/g, "&#39;")), m);
  }
});
function esc(s) { return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#39;"); }

test("Merge sends the ticked matrices, the files, the published base and the label; progress, Stop and the result with its warnings and weights", async () => {
  const b = browser({ imatrix_list: two() });
  open(b, two(), `im.mergePub = true; im.mergePubBase = "Qwen/Qwen3.8-27B"; im.mergeFiles = ["/f/x.gguf"]; im.mergeLabel = "  mine + theirs "`);
  tick(b, ["20261101-1000-run1", "20261102-1000-b"]);
  b.els["im-mergelabel"].value = "mine + theirs"; b.els["im-mergelabel"].onchange();
  await b.els["im-merge-go"].onclick();
  assert.deepEqual(b.calls.at(-1), ["imatrix_merge", { spec: { inputs: [{ kind: "library", id: "20261101-1000-run1" }, { kind: "library", id: "20261102-1000-b" }, { kind: "file", file: "/f/x.gguf" }, { kind: "published", base: "Qwen/Qwen3.8-27B" }], label: "mine + theirs" } }]);
  b.run(`onImatrixEvent({kind: "progress", step: "imatrix", value: 0.3, text: "Downloading Unsloth's published matrix: 40 of 150 MB"})`);
  let html = b.run("renderImatrix()");
  assert.match(html, /Downloading Unsloth&#39;s published matrix: 40 of 150 MB/); assert.match(html, /<progress max="1" value="0.3">/); assert.equal((html.match(/id="im-stop"/g) || []).length, 1, "one Stop button");
  b.run("wireImatrix()"); b.els["im-stop"].onclick();
  assert.equal(b.calls.at(-1)[0], "imatrix_stop");
  b.run(`onImatrixEvent({kind: "done", data: {id: "m", entry: {id: "20261103-1000-merged", label: "mine + theirs", size_mb: 12.5}, warnings: ["The published matrix carries about 94 % of the merged statistics: make yours with more chunks.", "Unsloth's published matrix describes the UNMODIFIED base model, not your fine-tune."],
    weights: [{label: "my first", chunks: 64, share: 0.06}, {label: "Unsloth's published matrix for Qwen/Qwen3.8-27B", chunks: 1000, share: 0.94}]}})`);
  await new Promise((r) => setTimeout(r, 5));
  html = b.run("renderImatrix()");
  assert.match(html, /Merged matrix ready:<\/strong> mine \+ theirs \(12.5 MB\)/); assert.match(html, /can be used in exports and in Compare quants like any other/);
  assert.match(html, /class="warn small">The published matrix carries about 94 %/); assert.match(html, /UNMODIFIED base model/);
  assert.match(html, /my first: 64 chunks · about 6 % of the merged statistics/); assert.match(html, /: 1000 chunks · about 94 % of the merged statistics/);
  assert.deepEqual(J(b, "im.mergePicked"), []); assert.deepEqual(J(b, "im.mergeFiles"), []);
  assert.ok(b.calls.some((c) => c[0] === "imatrix_list"), "the list is refreshed");
  b.run(`im.pending = "merge"; onImatrixEvent({kind: "error", text: "Cancelled."})`);
  assert.match(b.run("renderImatrix()"), /Cancelled\. Nothing was kept; the matrices you merge are unchanged\./);
  b.run(`onImatrixEvent({kind: "error", text: "llama-imatrix refused to combine them: its message is inconsistent expert count."})`);
  assert.match(b.run("renderImatrix()"), /llama-imatrix refused to combine them/);
});

test("the Create form offers the merge only when the model has a base on the Hub, and sends it only when ticked", async () => {
  const b = browser({ imatrix_list: libState() });
  let html = open(b, libState());
  assert.match(html, /id="im-mergepub2"[^>]*> Also merge Unsloth's published matrix for the base model \(Qwen\/Qwen3.8-27B\)/);
  await b.els["im-create"].onclick();
  assert.equal(Object.hasOwn(b.calls.at(-1)[1].spec, "merge_published"), false, "off: the spec is as before");
  b.run("im.running = false");
  b.els["im-mergepub2"].checked = true; b.els["im-mergepub2"].onchange();
  html = b.run("renderImatrix()");
  assert.match(html, /A second entry is made after yours \(yours is kept\)/); assert.match(html, /a large published matrix can outweigh a small one of your own/); assert.match(html, /If the download or the merge fails, your matrix is kept and the reason is shown/);
  b.run("wireImatrix()"); await b.els["im-create"].onclick();
  assert.equal(b.calls.at(-1)[1].spec.merge_published, true);
  // no base on the Hub: not offered, and not sent even when it was ticked before
  b.run(`im.running = false; im.src = "model"; im.model = "thin"`);
  html = redraw(b);
  assert.doesNotMatch(html, /id="im-mergepub2"/);
  await b.els["im-create"].onclick();
  assert.equal(Object.hasOwn(b.calls.at(-1)[1].spec, "merge_published"), false);
  b.run(`im.running = false; im.src = "model"; im.model = "__typed"; im.modelPath = "owner/name"`);
  assert.match(redraw(b), /for the base model \(owner\/name\)/);
  b.run(`im.running = false; im.src = "gguf"; im.tokenizer = ""`);
  assert.doesNotMatch(redraw(b), /id="im-mergepub2"/);
});

test("a creation that also merged shows both entries, and a merge that failed shows why next to the kept matrix", () => {
  const b = browser();
  open(b, libState());
  b.run(`im.result = ${JSON.stringify({ entry: { id: "a", label: "mine", size_mb: 10 }, notes: [], merged: { entry: { id: "b", label: "mine + Unsloth's published matrix", size_mb: 11 }, warnings: ["Unsloth's published matrix describes the UNMODIFIED base model."] } })}`);
  let html = b.run("renderImatrix()");
  assert.match(html, /Matrix ready:<\/strong> mine \(10 MB\)/); assert.match(html, /id="im-merged-result"/); assert.match(html, /Also merged with Unsloth's published matrix:<\/strong> mine \+ Unsloth&#39;s published matrix \(11 MB\)\. Your own matrix is kept as well/);
  assert.match(html, /UNMODIFIED base model/);
  b.run(`im.result = ${JSON.stringify({ entry: { id: "a", label: "mine", size_mb: 10 }, notes: [], merge_error: "huggingface.co could not be reached (URLError). The matrix you made is kept." })}`);
  html = b.run("renderImatrix()");
  assert.match(html, /Matrix ready:<\/strong> mine/); assert.match(html, /class="warn" id="im-merge-error">huggingface.co could not be reached \(URLError\)\. The matrix you made is kept\./);
});
