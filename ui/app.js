// ND Lore Forge window. All logic lives in Rust (crates/ndforge-core); this file only draws the state it returns.
const tauri = window.__TAURI__;
const invoke = tauri.core.invoke;
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const TITLES = {
  welcome: ["Welcome to ND Lore Forge", "Teach a small AI model your nd-world world. Start by creating a project (one project = one world)."],
  dataset: ["Step 2 · Make training data", "Turns your world into practice questions and answers."],
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
  el.innerHTML = page === "welcome" ? renderWelcome() : page === "data" ? renderData() : page === "setup" ? renderSetup() : renderSoon(page);
  $("status").textContent = state.project ? `Project: ${state.project.name}  (${state.project.folder})` : "No project open";
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
  on("recheck", refreshEnv);
  on("install", () => startEnv("env_install"));
  on("selftest", () => startEnv("env_selftest"));
  on("cancelenv", () => invoke("env_cancel"));
  on("apply", applyFile);
  on("cancel", () => { pending = null; render(); });
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
  $("help").innerHTML = await invoke("help", { key });
  $("help").scrollTop = 0;
}

async function init() {
  setState(await invoke("get_state"));
  $("expert").onchange = async (e) => setState(await invoke("set_setting", { key: "expert_mode", value: e.target.checked })) || render();
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
