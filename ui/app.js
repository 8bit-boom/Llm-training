// ND Lore Forge window. All logic lives in Rust (crates/ndforge-core); this file only draws the state it returns.
const tauri = window.__TAURI__;
const invoke = tauri.core.invoke;
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const TITLES = {
  welcome: ["Welcome to ND Lore Forge", "Teach a small AI model your nd-world world. Start by creating a project (one project = one world)."],
  train: ["Step 3 · Train", "Teaches the model your world."],
  test: ["Step 4 · Test", "Checks that the trained model knows your world better than the original."],
  use: ["Step 5 · Use in nd-world", "Makes the model usable in nd-world."],
};

let state = null;
let page = "welcome";
let pending = null;      // a checked file waiting for the user's confirmation: {path, info}
let message = null;      // {kind: "err"|"ok", text}
let env = { status: null, running: false, log: [], step: "", progress: null, result: null, error: null };
const $ = (id) => document.getElementById(id);

async function call(cmd, args) {
  try { return await invoke(cmd, args); }
  catch (e) { message = { kind: "err", text: String(e) }; render(); return null; }
}

// night mode: "auto" follows the system until the user chooses
const systemDark = () => window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
function applyTheme() {
  const t = state.settings.theme || "auto";
  if (t === "auto") document.documentElement.removeAttribute("data-theme"); else document.documentElement.setAttribute("data-theme", t);
  $("night").checked = t === "dark" || (t === "auto" && systemDark());
}

function setState(s) { if (s) { state = s; } }

function renderSteps() {
  const items = [{ key: "welcome", label: "Welcome", unlocked: true, done: false }, ...state.steps];
  items.push({ key: "setup", label: "⚙ Training setup", unlocked: true, done: false });
  $("steps").innerHTML = items.map((s) =>
    `<button class="step ${s.key === page ? "current" : ""}" data-key="${s.key}" ${s.unlocked ? "" : "disabled"}>${s.done ? "✓ " : s.unlocked ? "" : "🔒 "}${esc(s.label)}</button>`).join("");
  $("steps").querySelectorAll("button").forEach((b) => b.onclick = () => go(b.dataset.key));
}

function msgHtml() { return message ? `<div class="${message.kind}">${esc(message.text)}</div>` : ""; }

function renderWelcome() {
  const recent = state.recent.map((r) => `<li><button data-folder="${esc(r.folder)}">${esc(r.name)}<small>${esc(r.folder)}</small></button></li>`).join("");
  return `<h1>${TITLES.welcome[0]}</h1><p class="lead">${TITLES.welcome[1]}</p>
    <div class="row"><input type="text" id="pname" placeholder="Project name, for example the name of your world" maxlength="80">
      <button class="primary" id="create">Create a new project</button></div>
    ${msgHtml()}
    <div class="row"><button class="secondary" id="openp">Open an existing project…</button></div>
    ${recent ? `<h2>Recent projects</h2><ul class="recent">${recent}</ul>` : ""}`;
}

function bundleCard() {
  if (!state.bundle) return "";
  return `<div class="card"><h2>World: ${esc(state.bundle.world)}</h2>This project contains:<ul>${state.bundle.lines.map((l) => `<li>${esc(l)}</li>`).join("") || "<li>(no records)</li>"}</ul></div>`;
}

function pendingCard() {
  if (!pending) return "";
  const { info } = pending;
  const kind = info.check.kind;
  const lines = `<ul>${info.lines.map((l) => `<li>${esc(l)}</li>`).join("")}</ul>`;
  if (kind === "refused") return `<div class="card"><h2>This file cannot be used</h2><div class="err">${esc(info.check.text)}</div><button class="secondary" id="cancel">OK</button></div>`;
  const title = kind === "update" ? "Update the world data?" : "Use this file?";
  return `<div class="card"><h2>${title} (${esc(info.world)})</h2>${lines}<p>${kind === "update" ? "<strong>What changed:</strong> " : ""}${esc(info.check.text)}</p>
    <div class="row"><button class="primary" id="apply">${kind === "update" ? "Use this file" : "Import"}</button><button class="secondary" id="cancel">Cancel</button></div></div>`;
}

function renderData() {
  return `<h1>Step 1 · Get world data</h1>
    <p class="lead">Export a <strong>Training bundle</strong> from nd-world (Tools → Export &amp; Backup), then choose the file here or drag it onto this page. The help panel explains how.</p>
    <div class="drop" id="drop">Drag a <code>.ndtrain</code> file here</div>
    <div class="row"><button class="primary" id="choose">${state.bundle ? "Choose a newer training file…" : "Choose training file…"}</button></div>
    ${msgHtml()}${pendingCard()}${bundleCard()}`;
}

let ds = { state: null, filter: { builder: "", search: "", status: "" }, offset: 0, total: 0, items: [], editing: null, busy: false, error: null, ok: null };
const PAGE = 20;

function recipeFromForm() {
  const r = ds.state.recipe;
  const builders = [...document.querySelectorAll(".bld:checked")].map((c) => c.value);
  return { ...r, world_book: $("rb-book").checked, builders, phrasings: Number($("rb-phr").value), test_percent: Number($("rb-test").value) };
}

function renderDataset() {
  if (!state.bundle) return `<h1>Step 2 · Make training data</h1><p class="lead">Import your world data first (Step 1).</p>`;
  const st = ds.state;
  if (!st) return `<h1>Step 2 · Make training data</h1><p class="lead">Loading…</p>`;
  const r = st.recipe, sum = st.summary;
  const bl = st.builders.map((b) => `<label class="opt"><input type="checkbox" class="bld" value="${b.id}" ${r.builders.includes(b.id) ? "checked" : ""}> ${esc(b.name)}</label>`).join("");
  const sel = (id, vals, cur) => `<select id="${id}">${vals.map((v) => `<option ${v == cur ? "selected" : ""}>${v}</option>`).join("")}</select>`;
  const recipe = `<div class="card wide"><h2>Recipe</h2>
    <label class="opt"><input type="checkbox" id="rb-book" ${r.world_book ? "checked" : ""}> <strong>World book</strong> (the lore as text)</label>
    ${bl}
    <div class="row" style="margin-top:10px"><label>Different wordings ${sel("rb-phr", [1, 2, 3], r.phrasings)}</label>
      <label>Kept for testing ${sel("rb-test", [0, 10, 20, 30], r.test_percent)} %</label></div>
    <button class="primary" id="make" ${ds.busy ? "disabled" : ""}>${sum ? "Make the examples again" : "Make the examples"}</button></div>`;
  let summary = "";
  if (sum) {
    const rows = sum.builders.map((b) => `<tr><td>${esc(b.name)}</td><td>${b.kept}</td><td>${b.dropped}</td><td><button class="link" data-show="${b.id}">Show</button></td></tr>`).join("");
    summary = `<div class="card wide"><h2>What was made</h2>${sum.stale ? `<div class="err">The world data changed since these were made. Press "Make the examples again" (your drops and edits stay).</div>` : ""}
      <p>${sum.kept} examples kept of ${sum.total} · about ${sum.tokens.toLocaleString()} tokens · ${sum.edited} edited</p>
      <table class="tbl"><tr><th>Kind</th><th>Kept</th><th>Dropped</th><th></th></tr>${rows}</table></div>`;
  }
  let review = "";
  if (sum) {
    const f = ds.filter;
    const opts = [`<option value="">All kinds</option>`, ...sum.builders.map((b) => `<option value="${b.id}" ${f.builder === b.id ? "selected" : ""}>${esc(b.name)}</option>`)].join("");
    const items = ds.items.map((e) => {
      const q = e.text != null ? `<em>${esc(e.source_title)} (world book)</em>` : esc(e.messages.find((m) => m.role === "user")?.content);
      const a = e.text != null ? e.text : e.messages.filter((m) => m.role === "assistant").map((m) => m.content).pop();
      const body = ds.editing === e.id
        ? `<textarea id="edit-box" rows="8">${esc(a)}</textarea><div class="row"><button class="primary" data-saveedit="${e.id}">Save the change</button><button class="secondary" id="canceledit">Cancel</button></div>`
        : `<div class="ans">${esc(a)}</div>`;
      return `<div class="ex ${e.status === "drop" ? "dropped" : ""}"><div class="q"><span class="tag">${esc(e.builder)}</span> ${q}${e.edited ? ' <span class="tag">edited</span>' : ""}</div>${body}
        <div class="row small"><button class="secondary" data-toggle="${e.id}" data-to="${e.status === "drop" ? "keep" : "drop"}">${e.status === "drop" ? "Keep" : "Drop"}</button>
        <button class="secondary" data-edit="${e.id}">Edit</button><span class="dim">from: ${esc(e.source_title)}</span></div></div>`;
    }).join("") || `<p class="dim">Nothing matches.</p>`;
    const pages = Math.max(1, Math.ceil(ds.total / PAGE)), cur = Math.floor(ds.offset / PAGE) + 1;
    review = `<div class="card wide"><h2>Review</h2>
      <div class="row"><select id="f-builder">${opts}</select>
        <select id="f-status"><option value="">Kept and dropped</option><option value="keep" ${f.status === "keep" ? "selected" : ""}>Kept only</option><option value="drop" ${f.status === "drop" ? "selected" : ""}>Dropped only</option></select>
        <input type="text" id="f-search" placeholder="Search…" value="${esc(f.search)}" style="min-width:160px">
        <button class="secondary" id="f-go">Search</button></div>
      <div class="row small"><span class="dim">${ds.total} shown</span><button class="secondary" id="drop-all">Drop all shown</button><button class="secondary" id="keep-all">Keep all shown</button></div>
      ${items}
      <div class="row"><button class="secondary" id="prev" ${ds.offset === 0 ? "disabled" : ""}>Previous</button><span>Page ${cur} of ${pages}</span><button class="secondary" id="next" ${ds.offset + PAGE >= ds.total ? "disabled" : ""}>Next</button></div></div>`;
  }
  const vers = st.versions.length ? `<ul>${st.versions.map((v) => `<li><strong>Version ${v.version}</strong> · ${v.counts.world_book_texts} texts, ${v.counts.train_chat} practice, ${v.counts.test_chat} for testing · ${v.created}</li>`).join("")}</ul>` : "<p class=\"dim\">No version saved yet.</p>";
  const save = sum ? `<div class="card wide"><h2>Save</h2><p>Training uses a saved version. Dropped examples are left out.</p><button class="primary" id="save" ${ds.busy ? "disabled" : ""}>Save this version</button>${vers}</div>` : "";
  return `<h1>Step 2 · Make training data</h1><p class="lead">Turns your world into practice material. The help panel explains every choice.</p>
    ${ds.error ? `<div class="err">${esc(ds.error)}</div>` : ""}${ds.ok ? `<div class="ok">${esc(ds.ok)}</div>` : ""}${recipe}${summary}${review}${save}`;
}

async function dsCall(cmd, args) {
  ds.error = null;
  try { return await invoke(cmd, args); } catch (e) { ds.error = String(e); return null; }
}

async function dsRefresh(keepPage) {
  const st = await dsCall("dataset_state");
  if (st) ds.state = st;
  if (ds.state && ds.state.summary) {
    if (!keepPage) ds.offset = 0;
    const r = await dsCall("dataset_examples", { filter: ds.filter, offset: ds.offset, limit: PAGE });
    if (r) { ds.items = r.items; ds.total = r.total; }
  }
  if (page === "dataset") render();
}

const LIGHT = { ok: "🟢", warn: "🟡", fail: "🔴" };

function renderSetup() {
  const st = env.status;
  const checks = st ? st.checks.map((c) => `<li class="check lt-${c.light}"><span class="lamp">${LIGHT[c.light]}</span><div><strong>${esc(c.label)}</strong><div class="detail">${esc(c.detail)}</div>${c.fix ? `<div class="fix">${esc(c.fix)}</div>` : ""}</div></li>`).join("") : "<li>Checking…</li>";
  const busy = env.running;
  const log = env.log.slice(-200).map(esc).join("\n");
  const result = env.result ? `<div class="ok">Test passed: ${esc(env.result.device_name)} · ${esc(env.result.precision)} · tiny model trained in ${esc(env.result.train_seconds)} s (peak memory ${esc(env.result.peak_vram_mb)} MB).</div>` : "";
  return `<h1>Training setup</h1>
    <p class="lead">Gets this computer ready to train: checks the graphics card and driver, installs the training software, and tests the card. The help panel explains each step.</p>
    <ul class="checks">${checks}</ul>
    <div class="row">
      <button class="secondary" id="recheck" ${busy ? "disabled" : ""}>Check again</button>
      <button class="primary" id="install" ${busy || !st || !st.can_install ? "disabled" : ""}>Install the training software</button>
      <button class="primary" id="selftest" ${busy || !st || !st.env_installed ? "disabled" : ""}>Test the graphics card</button>
      <button class="secondary" id="cancelenv" ${busy ? "" : "disabled"}>Cancel</button>
    </div>
    ${env.step ? `<div class="stepline">${esc(env.step)}</div>` : ""}
    ${env.progress != null ? `<progress max="1" value="${env.progress}"></progress>` : ""}
    ${env.error ? `<div class="err">${esc(env.error)}</div>` : ""}${result}
    ${env.log.length ? `<pre class="log" id="log">${log}</pre>` : ""}`;
}

async function refreshEnv() {
  env.status = await invoke("env_status");
  if (page === "setup") render();
}

async function startEnv(cmd) {
  env.running = true; env.log = []; env.step = ""; env.progress = null; env.result = null; env.error = null;
  render();
  try { await invoke(cmd); } catch (e) { env.running = false; env.error = String(e); render(); }
}

function onEnvEvent(p) {
  if (p.kind === "log") env.log.push(p.text);
  else if (p.kind === "step") { env.step = `Step ${p.index} of ${p.total}: ${p.text}`; env.progress = (p.index - 1) / p.total; }
  else if (p.kind === "progress") { env.step = p.text || p.step; env.progress = p.value; }
  else if (p.kind === "done") { env.running = false; env.progress = 1; env.step = "Finished."; if (p.data && p.data.device_name) env.result = p.data; refreshEnv(); }
  else if (p.kind === "error") { env.running = false; env.progress = null; env.error = p.text; refreshEnv(); }
  if (page === "setup") { render(); const l = $("log"); if (l) l.scrollTop = l.scrollHeight; }
}

function renderSoon(key) {
  const [t, lead] = TITLES[key];
  return `<h1>${t}</h1><p class="lead">${lead}</p><span class="soon">Coming in a later version. The help panel shows what this step will do.</span>`;
}

function render() {
  if (!state) return;
  renderSteps();
  const el = $("page");
  el.innerHTML = page === "welcome" ? renderWelcome() : page === "data" ? renderData() : page === "setup" ? renderSetup() : page === "dataset" ? renderDataset() : renderSoon(page);
  $("status").textContent = state.project ? `Project: ${state.project.name}  (${state.project.folder})` : "No project open";
  applyTheme();
  $("expert").checked = state.settings.expert_mode;
  $("helptoggle").checked = state.settings.show_help_panel;
  $("layout").classList.toggle("nohelp", !state.settings.show_help_panel);
  wire();
}

function wire() {
  const on = (id, fn) => { const e = $(id); if (e) e.onclick = fn; };
  on("create", createProject);
  const pn = $("pname"); if (pn) pn.onkeydown = (e) => { if (e.key === "Enter") createProject(); };
  on("openp", async () => {
    const dir = await tauri.dialog.open({ directory: true, title: "Choose a project folder" });
    if (dir) openProject(dir);
  });
  document.querySelectorAll(".recent button").forEach((b) => b.onclick = () => openProject(b.dataset.folder));
  on("choose", async () => {
    const f = await tauri.dialog.open({ multiple: false, title: "Choose a training file", filters: [{ name: "Training bundle", extensions: ["ndtrain"] }, { name: "All files", extensions: ["*"] }] });
    if (f) checkFile(f);
  });
  wireDataset();
  on("recheck", refreshEnv);
  on("install", () => startEnv("env_install"));
  on("selftest", () => startEnv("env_selftest"));
  on("cancelenv", () => invoke("env_cancel"));
  on("apply", applyFile);
  on("cancel", () => { pending = null; render(); });
}

function wireDataset() {
  if (page !== "dataset" || !ds.state) return;
  const on = (id, fn) => { const e = $(id); if (e) e.onclick = fn; };
  on("make", async () => {
    ds.busy = true; ds.ok = null; render();
    const r = await dsCall("dataset_generate", { recipe: recipeFromForm() });
    ds.busy = false;
    if (r) ds.ok = `Made ${r.total} examples (${r.kept} kept).`;
    await dsRefresh(false);
  });
  on("save", async () => {
    ds.busy = true; render();
    const m = await dsCall("dataset_save");
    ds.busy = false; ds.ok = m ? `Saved version ${m.version}.` : null;
    setState(await invoke("get_state"));
    await dsRefresh(true);
  });
  const readFilter = () => { ds.filter = { builder: $("f-builder").value, status: $("f-status").value, search: $("f-search").value }; };
  on("f-go", () => { readFilter(); dsRefresh(false); });
  const fs = $("f-search"); if (fs) fs.onkeydown = (e) => { if (e.key === "Enter") { readFilter(); dsRefresh(false); } };
  ["f-builder", "f-status"].forEach((id) => { const e = $(id); if (e) e.onchange = () => { readFilter(); dsRefresh(false); }; });
  on("prev", () => { ds.offset = Math.max(0, ds.offset - PAGE); dsRefresh(true); });
  on("next", () => { ds.offset += PAGE; dsRefresh(true); });
  on("drop-all", async () => { readFilter(); await dsCall("dataset_set_status_matching", { filter: ds.filter, status: "drop" }); dsRefresh(false); });
  on("keep-all", async () => { readFilter(); await dsCall("dataset_set_status_matching", { filter: ds.filter, status: "keep" }); dsRefresh(false); });
  on("canceledit", () => { ds.editing = null; render(); });
  document.querySelectorAll("[data-show]").forEach((b) => b.onclick = () => { ds.filter = { builder: b.dataset.show, status: "", search: "" }; dsRefresh(false); });
  document.querySelectorAll("[data-toggle]").forEach((b) => b.onclick = async () => { await dsCall("dataset_set_status", { ids: [b.dataset.toggle], status: b.dataset.to }); dsRefresh(true); });
  document.querySelectorAll("[data-edit]").forEach((b) => b.onclick = () => { ds.editing = b.dataset.edit; render(); });
  document.querySelectorAll("[data-saveedit]").forEach((b) => b.onclick = async () => {
    const r = await dsCall("dataset_edit", { id: b.dataset.saveedit, text: $("edit-box").value });
    if (r !== null || !ds.error) ds.editing = null;
    dsRefresh(true);
  });
}

async function createProject() {
  const name = $("pname").value;
  message = null;
  const s = await call("create_project", { name });
  if (s) { setState(s); go("data"); }
}

async function openProject(folder) {
  message = null;
  const s = await call("open_project", { folder });
  if (s) { setState(s); go("data"); }
}

async function checkFile(path) {
  if (page !== "data") return;
  message = null; pending = null;
  const info = await call("check_bundle", { path });
  if (info) pending = { path, info };
  render();
}

async function applyFile() {
  const r = await call("apply_bundle", { path: pending.path });
  pending = null;
  if (r) { setState(r.state); message = { kind: "ok", text: r.changes ? "Imported. Changes: " + r.changes : "Imported." }; }
  render();
}

async function go(key) {
  page = key;
  message = message && message.kind === "err" && key !== "welcome" && key !== "data" ? null : message;
  render();
  if (key === "setup") refreshEnv();
  if (key === "dataset" && state.bundle) { ds.ok = null; dsRefresh(false); }
  $("help").innerHTML = await invoke("help", { key });
  $("help").scrollTop = 0;
}

async function init() {
  setState(await invoke("get_state"));
  $("expert").onchange = async (e) => setState(await invoke("set_setting", { key: "expert_mode", value: e.target.checked })) || render();
  $("night").onchange = async (e) => { setState(await invoke("set_theme", { theme: e.target.checked ? "dark" : "light" })); render(); };
  $("helptoggle").onchange = async (e) => { setState(await invoke("set_setting", { key: "show_help_panel", value: e.target.checked })); render(); };
  try {
    await tauri.webview.getCurrentWebview().onDragDropEvent((e) => {
      const d = $("drop");
      if (e.payload.type === "over" && d) d.classList.add("over");
      else if (e.payload.type === "drop") { if (d) d.classList.remove("over"); if (e.payload.paths?.length) checkFile(e.payload.paths[0]); }
      else if (d) d.classList.remove("over");
    });
  } catch (_) { /* drag and drop is a convenience; the button always works */ }
  try { await tauri.event.listen("env-event", (e) => onEnvEvent(e.payload)); } catch (_) { /* events are optional in previews */ }
  go("welcome");
}
init();
