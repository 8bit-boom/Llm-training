// Smoke test of the job queue panel on the Home page and the "Add to queue" buttons, without a window. Run with `node --test ui/smoke`.
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
  const el = (id) => (els[id] ??= { id, innerHTML: "", textContent: "", checked: false, value: "", classList: { toggle() {} }, querySelectorAll: () => [], onclick: null, onchange: null, oninput: null, onkeydown: null, style: {}, dataset: {} });
  const calls = [];
  const invoke = async (cmd, args) => { calls.push([cmd, args]); if (cmd in answers) { const a = answers[cmd]; return typeof a === "function" ? a(args) : a; } if (cmd === "get_state") return { project: { name: "P", folder: "/p" }, steps: [], bundle: null, settings: { theme: "light", expert_mode: false, show_help_panel: true }, worlds: [], recent: [] }; return cmd === "help" ? "<p>help</p>" : null; };
  const win = { __TAURI__: { core: { invoke }, dialog: {}, event: { listen: async () => {} }, webview: { getCurrentWebview: () => ({ onDragDropEvent: async () => {} }) } }, matchMedia: () => ({ matches: false }) };
  const doc = { getElementById: el, querySelectorAll: () => [], querySelector: () => null, createElement: () => el("tmp"), body: { appendChild() {} }, documentElement: { setAttribute() {}, removeAttribute() {} } };
  const ctx = vm.createContext({ window: win, document: doc, console, confirm: () => true, alert: () => {}, setTimeout, clearTimeout });
  vm.runInContext(source, ctx);
  return { ctx, calls, run: (code) => vm.runInContext(code, ctx) };
}
const plain = (x) => JSON.parse(JSON.stringify(x));
const view = (items, paused = false) => ({ items, paused, running: (items.find((i) => i.status === "running") || {}).id || null, queued: items.filter((i) => i.status === "queued").length });
const item = (id, status, over = {}) => ({ id, kind: "export", status, label: `Export ${id}`, error: null, ...over });

test("the Home page lists the queue with what each entry can do", () => {
  const b = browser();
  b.run(`state = ${JSON.stringify({ project: null, recent: [], settings: {} })}; qu.view = ${JSON.stringify(view([item("a", "running"), item("b", "queued"), item("c", "failed", { error: "Disk full. (details: logs/jobs/x.log) [fix:disk_full]" }), item("d", "skipped", { error: "Training software: Not installed." })]))}`);
  const html = b.run("renderWelcome()");
  assert.match(html, /<h2>Job queue<\/h2>/);
  assert.match(html, /Export a<\/strong> · running/);
  assert.doesNotMatch(html, /data-quremove="a"/, "a running entry cannot be removed");
  assert.match(html, /data-quremove="b"/); assert.match(html, /data-qumove="b" data-to="0"/); assert.match(html, /data-qumove="b" data-to="2"/);
  assert.match(html, /Export c<\/strong> · failed <span class="dim">Disk full\. \(details: logs\/jobs\/x\.log\)<\/span>/, "the fix mark is not shown raw");
  assert.match(html, /Export d<\/strong> · skipped <span class="dim">Training software: Not installed\./);
  assert.match(html, /id="qu-pause"/); assert.doesNotMatch(html, /id="qu-run"/);
});

test("a paused queue offers Run, and an empty one shows nothing", () => {
  const b = browser();
  b.run(`state = ${JSON.stringify({ project: null, recent: [], settings: {} })}; qu.view = ${JSON.stringify(view([item("b", "queued")], true))}`);
  assert.match(b.run("renderWelcome()"), /id="qu-run" >Run the queue<\/button><span class="dim small">The queue is waiting: nothing starts until you press Run/);
  b.run(`qu.view = ${JSON.stringify(view([], false))}`);
  assert.doesNotMatch(b.run("renderWelcome()"), /Job queue/);
});

test("adding, removing and moving go through the queue commands and show their answer", async () => {
  const b = browser({ queue_add: (a) => view([item("n1", "queued", { kind: a.kind })]), queue_remove: () => view([]), queue_move: () => view([item("n1", "queued")]) });
  b.run(`state = ${JSON.stringify({ project: { name: "P", folder: "/p" }, recent: [], settings: {} })}; page = "welcome"; us.run = "run1"; us.quant = "q4_k_m"; us.quants = ["q4_k_m"]; us.what = "gguf"; us.numCtx = 8192; us.im = false; us.mx = {}`);
  await b.run(`quAdd("export", null)`);
  const call = b.calls.find((c) => c[0] === "queue_add");
  assert.equal(call[1].kind, "export");
  assert.deepEqual(plain(call[1].args), { run: "run1", quant: "q4_k_m", numCtx: 8192, what: "gguf", quants: ["q4_k_m"], imatrix: null });
  assert.equal(plain(b.run("qu.view")).items[0].id, "n1");
  assert.match(b.run("qu.note"), /Added to the queue/);
  await b.run(`quDo("queue_remove", { id: "n1" })`);
  assert.deepEqual(plain(b.run("qu.view")).items, []);
  await b.run(`quDo("queue_move", { id: "n1", to: 0 })`);
  assert.deepEqual(plain(b.calls.find((c) => c[0] === "queue_move")[1]), { id: "n1", to: 0 });
});

test("a refusal from the program is shown, not lost", async () => {
  const b = browser({ queue_add: () => { throw "Open a project first."; } });
  b.run(`state = ${JSON.stringify({ project: null, recent: [], settings: {} })}; us.quants = []; us.what = "gguf"; us.mx = {}`);
  await b.run(`quAdd("export", null)`);
  assert.equal(b.run("qu.note"), "Open a project first.");
});

test("the Home page has the notification and webhook settings once a project is open", () => {
  const b = browser();
  b.run(`state = ${JSON.stringify({ project: { name: "P", folder: "/p" }, recent: [], settings: { notifications: true, webhook_url: "https://ntfy.sh/t" } })}`);
  const html = b.run("renderWelcome()");
  assert.match(html, /id="notify-on" checked/);
  assert.match(html, /id="webhook-url" value="https:\/\/ntfy\.sh\/t"/);
  assert.match(html, /never a name or a folder/);
  b.run(`state.project = null`);
  assert.doesNotMatch(b.run("renderWelcome()"), /webhook-url/);
});
