// ND Lore Forge window. All logic lives in Rust (crates/ndforge-core); this file only draws the state it returns.
const tauri = window.__TAURI__;
const invoke = tauri.core.invoke;
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const TITLES = {
  welcome: ["Welcome to ND Lore Forge", "Teach a small AI model your nd-world world. Start by creating a project (one project = one world)."],
  test: ["Step 4 · Test", "Checks that the trained model knows your world better than the original."],
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
    ${msgHtml()}${pendingCard()}${retrainCard()}${bundleCard()}`;
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

// ------------------------------------------------------------------------------------------------ step 3: train
let tr = { server: "", srvCheck: null, adding: false, state: null, model: null, method: "lora", preset: "normal", plan: null, over: {}, running: false, run: null, metrics: [], samples: [], log: [], progress: null, stepText: "", error: null, notice: null };
const FIT = { yes: ["✓", "fits"], tight: ["⚠", "tight"], no: ["✗", "too big"], unknown: ["?", "unknown card"] };

function fmtEta(s) { if (s == null) return "–"; const m = Math.floor(s / 60); return m >= 60 ? `${Math.floor(m / 60)} h ${m % 60} min` : m >= 1 ? `${m} min` : `${s} s`; }

function lossChart(points) {
  if (points.length < 2) return `<div class="dim">The loss chart appears after the first steps.</div>`;
  const W = 560, H = 190, P = 30;
  const xs = points.map((p) => p.step), ys = points.map((p) => p.loss);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  const sx = (x) => P + (x1 === x0 ? 0 : ((x - x0) / (x1 - x0)) * (W - P - 8));
  const sy = (y) => H - P - (y1 === y0 ? 0.5 : (y - y0) / (y1 - y0)) * (H - P - 10);
  const d = points.map((p, i) => `${i ? "L" : "M"}${sx(p.step).toFixed(1)},${sy(p.loss).toFixed(1)}`).join(" ");
  return `<svg viewBox="0 0 ${W} ${H}" class="chart" role="img" aria-label="Training loss"><path d="${d}" fill="none" stroke="var(--accent)" stroke-width="2"/>
    <line x1="${P}" y1="${H - P}" x2="${W - 8}" y2="${H - P}" stroke="var(--line)"/><line x1="${P}" y1="8" x2="${P}" y2="${H - P}" stroke="var(--line)"/>
    <text x="${P + 4}" y="18" class="axis">${y1.toFixed(2)}</text><text x="${P + 4}" y="${H - P - 4}" class="axis">${y0.toFixed(2)}</text>
    <text x="${P}" y="${H - 8}" class="axis">step ${x0}</text><text x="${W - 60}" y="${H - 8}" class="axis">step ${x1}</text></svg>`;
}

function samplesHtml() {
  const byQ = {};
  tr.samples.forEach((s) => (byQ[s.question] = byQ[s.question] || []).push(s));
  const qs = Object.keys(byQ);
  if (!qs.length) return "";
  return `<h3>Sample answers</h3>` + qs.map((q) => {
    const list = byQ[q], first = list[0], last = list[list.length - 1];
    return `<div class="sample"><div class="q">${esc(q)}</div><div class="cols"><div><span class="tag">before training</span><div class="ans">${esc(first.answer) || "<em>(empty)</em>"}</div></div>
      ${list.length > 1 ? `<div><span class="tag">now (step ${last.step})</span><div class="ans">${esc(last.answer) || "<em>(empty)</em>"}</div></div>` : ""}</div></div>`;
  }).join("");
}

function renderTrain() {
  if (!state.bundle) return `<h1>Step 3 · Train</h1><p class="lead">Import your world data first (Step 1).</p>`;
  const st = tr.state;
  if (!st) return `<h1>Step 3 · Train</h1><p class="lead">Loading…</p>`;
  if (!st.version) return `<h1>Step 3 · Train</h1><p class="lead">Save a version of the training data first (Step 2).</p>`;
  const v = st.version;
  const gpu = st.gpu ? `${esc(st.gpu.name)} · ${st.gpu.vram_gb} GB · ${esc(st.gpu.precision)}` : esc(st.gpu_problem || "no graphics card found");
  const cards = st.models.map((c) => {
    const m = c.model, f = c[tr.method], fit = FIT[f.fit] || FIT.unknown;
    return `<button class="mcard ${tr.model === m.id ? "sel" : ""} fit-${f.fit}" data-model="${m.id}"><strong>${esc(m.name)}</strong>${c.recommended === tr.method ? "" : ""}
      <span class="dim">${m.params_b.toFixed(1)} B parameters · ${esc(m.license)}</span><span>${fit[0]} ${fit[1]} · ≈ ${f.measured != null ? f.measured + " GB measured" : f.est_vram_gb + " GB"}</span><span class="dim">${esc(m.good_for)}</span></button>`;
  }).join("");
  const srv = tr.server ? st.servers.find((x) => x.url === tr.server) : null;
  const srvStatus = srv && srv.status;
  const srvLine = !srv ? `Training on this computer: ${gpu}.` : srvStatus
    ? (srvStatus.gpu ? `${esc(srv.name)}: ${esc(srvStatus.gpu.name)} · ${srvStatus.gpu.vram_gb} GB · ${esc(srvStatus.gpu.precision)} · ${srvStatus.busy ? "busy with " + esc(srvStatus.job) : "free"}${srvStatus.ollama ? " · Ollama there will be paused while training" : ""}` : `${esc(srv.name)}: no graphics card (${esc(srvStatus.gpu_problem || "none found")})`)
    : "Not checked yet. Press “Test the connection”.";
  const where = `<div class="card wide"><h2>Where to train</h2><div class="row"><label>Computer <select id="tr-where"><option value="">This computer</option>${st.servers.map((x) => `<option value="${esc(x.url)}" ${tr.server === x.url ? "selected" : ""}>${esc(x.name)} (${esc(x.url)})</option>`).join("")}</select></label>
      ${srv ? `<button class="secondary" id="srv-test" ${tr.running ? "disabled" : ""}>Test the connection</button><button class="secondary" id="srv-remove" ${tr.running ? "disabled" : ""}>Remove this server</button>` : ""}
      <button class="secondary" id="srv-add">Add a server…</button></div>
    ${tr.adding ? `<div class="row"><input type="text" id="s-name" placeholder="Name, e.g. TrueNAS"><input type="text" id="s-url" placeholder="Address, e.g. http://truenas:8765"><input type="password" id="s-token" placeholder="Access token (NDFORGE_TOKEN)" style="min-width:220px"><button class="primary" id="s-save">Add</button><button class="secondary" id="s-cancel">Cancel</button></div>` : ""}
    <div class="dim">${srvLine}</div>${tr.srvCheck ? `<div class="${tr.srvCheck.ok ? "ok" : "err"}">${esc(tr.srvCheck.text)}</div>` : ""}
    ${srv ? `<p class="dim">The server's graphics card is used only by the training until it ends: AI chat, transcription and image generation on that server wait or are unavailable meanwhile. Only LoRA / QLoRA is available there. The trained model is copied back to this computer.</p>` : ""}</div>`;
  const pl = tr.plan;
  const sel = st.models.find((c) => c.model.id === tr.model);
  const exp = state.settings.expert_mode && pl ? `<div class="card wide"><h2>Settings (Expert)</h2><div class="grid">${[
    ["rank", "LoRA rank", pl.hyper.rank], ["lr", "Learning rate", pl.hyper.lr], ["epochs", "Epochs", pl.hyper.epochs], ["max_steps", "Fixed steps (empty = use epochs)", pl.hyper.max_steps ?? ""],
    ["batch", "Batch size", pl.hyper.batch], ["grad_accum", "Gradient accumulation", pl.hyper.grad_accum], ["max_seq", "Sequence length (tokens)", pl.hyper.max_seq]].map(([k, l, val]) =>
      `<label>${l}<input type="text" class="ov" data-key="${k}" value="${tr.over[k] ?? val}"></label>`).join("")}</div></div>` : "";
  const planHtml = pl ? `<div class="card wide"><h2>The plan</h2><p>About <strong>${pl.steps}</strong> steps · memory ≈ <strong>${pl.est_vram_gb} GB</strong> (${(FIT[pl.fit] || FIT.unknown)[1]}) · sequence ${pl.hyper.max_seq} tokens · batch ${pl.hyper.batch}×${pl.hyper.grad_accum}</p>
    ${pl.notes.length ? `<ul>${pl.notes.map((n) => `<li>${esc(n)}</li>`).join("")}</ul>` : ""}</div>` : "";
  const last = st.runs[0];
  const canResume = last && (last.status === "stopped" || last.status === "failed") && !tr.running;
  const ctrl = `<div class="row"><button class="primary" id="tr-start" ${tr.running || !tr.model || (!tr.server && !st.env_ready) || (tr.server && !(srvStatus && srvStatus.gpu && !srvStatus.busy)) ? "disabled" : ""}>${tr.server && srv ? "Start training on " + esc(srv.name) : "Start training"}</button>
    <button class="secondary" id="tr-stop" ${tr.running ? "" : "disabled"}>Stop</button>
    ${canResume ? `<button class="secondary" id="tr-resume" data-run="${esc(last.name)}">Continue the last run (${esc(last.model)})</button>` : ""}</div>`;
  const env = !st.env_ready ? `<div class="err">The training software is not installed yet. Open "⚙ Training setup" first.</div>` : "";
  const live = (tr.running || tr.metrics.length || tr.error || tr.notice) ? `<div class="card wide"><h2>${tr.running ? "Training…" : "Last run"}</h2>
    ${tr.stepText ? `<div class="stepline">${esc(tr.stepText)}</div>` : ""}${tr.progress != null ? `<progress max="1" value="${tr.progress}"></progress>` : ""}
    ${tr.error ? `<div class="err">${esc(tr.error)}</div>` : ""}${tr.notice ? `<div class="ok">${esc(tr.notice)}</div>` : ""}
    ${(() => { const m = tr.metrics.at(-1); return m ? `<div class="stats"><span>Step <strong>${m.step}/${m.total_steps}</strong></span><span>Loss <strong>${(+m.loss).toFixed(3)}</strong></span><span>Memory peak <strong>${m.vram_mb ? (m.vram_mb / 1024).toFixed(1) + " GB" : "–"}</strong></span><span>GPU <strong>${m.gpu_temp != null ? m.gpu_temp + " °C" : "–"}</strong></span><span>Time left <strong>${fmtEta(m.eta_s)}</strong></span></div>` : ""; })()}
    ${lossChart(tr.metrics.filter((m) => m.kind === "metric" || m.type === "metric"))}${samplesHtml()}
    ${tr.log.length ? `<details><summary>Log</summary><pre class="log">${tr.log.slice(-120).map(esc).join("\n")}</pre></details>` : ""}</div>` : "";
  const runs = st.runs.length ? `<div class="card wide"><h2>Earlier runs</h2><table class="tbl"><tr><th>Run</th><th>Status</th><th>Step</th><th>Loss</th><th>Memory</th><th>Where</th><th></th></tr>${st.runs.map((r) =>
    `<tr><td>${esc(r.name)}</td><td>${esc(r.status)}</td><td>${r.step ?? ""}${r.total_steps ? "/" + r.total_steps : ""}</td><td>${r.loss != null ? (+r.loss).toFixed(3) : ""}</td><td>${r.peak_vram_mb ? (r.peak_vram_mb / 1024).toFixed(1) + " GB" : ""}</td><td>${esc(r.server || "this computer")}</td><td><button class="link" data-showrun="${esc(r.name)}">Show</button></td></tr>`).join("")}</table></div>` : "";
  return `<h1>Step 3 · Train</h1><p class="lead">Teaches the model your world with LoRA (a small add-on) and practice answers (SFT). The help panel explains each choice.</p>
    <div class="dim">Training data: version ${v.number} · ${v.counts.world_book_texts} world-book texts, ${v.counts.train_chat} practice examples · Graphics card: ${gpu}</div>${tr.server ? "" : env}${where}
    <div class="card wide"><h2>1 · Choose the model</h2><div class="mcards">${cards}</div></div>
    <div class="card wide"><h2>2 · How to train</h2><div class="row"><label>Type <select id="tr-method"><option value="lora" ${tr.method === "lora" ? "selected" : ""}>LoRA, 16-bit (recommended)</option><option value="qlora" ${tr.method === "qlora" ? "selected" : ""}>QLoRA, 4-bit (needs less memory)</option></select></label>
      <label>Preset <select id="tr-preset"><option value="quick" ${tr.preset === "quick" ? "selected" : ""}>Quick test (minutes)</option><option value="normal" ${tr.preset === "normal" ? "selected" : ""}>Normal</option><option value="best" ${tr.preset === "best" ? "selected" : ""}>Best quality (slower)</option></select></label></div>
      <p class="dim">Full fine-tuning (changing every weight) comes in a later version.</p></div>
    ${planHtml}${exp}<div class="card wide"><h2>3 · Train</h2>${ctrl}</div>${live}${runs}`;
}

async function trRefresh() {
  try {
    tr.state = await invoke("train_state", { server: tr.server || null });
    if (!tr.model && tr.state.models.length) {
      const first = tr.state.models.find((c) => c.lora.fit === "yes" || c.lora.fit === "tight") || tr.state.models[0];
      tr.model = tr.state.models.find((c) => c.model.id === "qwen3.5-4b")?.model.id ?? first.model.id;
      tr.method = (tr.state.models.find((c) => c.model.id === tr.model) || first).recommended;
    }
    if (tr.model) tr.plan = await invoke("train_plan", { model: tr.model, method: tr.method, preset: tr.preset, server: tr.server || null });
  } catch (e) { tr.error = String(e); }
  if (page === "train") render();
}

function onServerEvent(p) {
  if (p.kind === "done") tr.srvCheck = { ok: true, text: p.data.gpu ? `Connected. ${p.data.gpu.name}, ${p.data.gpu.vram_gb} GB, ${p.data.busy ? "busy" : "free"}.` : "Connected, but the server found no graphics card." };
  else tr.srvCheck = { ok: false, text: p.text };
  trRefresh();
}

function onTrainEvent(p) {
  if (p.kind === "metric") { tr.metrics.push(p); tr.progress = p.step / Math.max(1, p.total_steps); tr.stepText = `Step ${p.step} of ${p.total_steps}`; }
  else if (p.kind === "sample") tr.samples.push(p);
  else if (p.kind === "stage") { tr.stepText = p.text; tr.progress = null; }
  else if (p.kind === "progress") { if (p.step !== "train" || tr.progress == null) { tr.stepText = p.text || p.step; tr.progress = p.value; } }
  else if (p.kind === "log") tr.log.push(p.text);
  else if (p.kind === "done") { tr.running = false; tr.notice = p.data.stopped ? `Stopped at step ${p.data.steps}. You can continue this run later.` : `Finished: ${p.data.steps} steps, final loss ${p.data.final_loss != null ? (+p.data.final_loss).toFixed(3) : "–"}.`; trRefresh(); }
  else if (p.kind === "error") { tr.running = false; tr.error = p.text; trRefresh(); }
  if (page === "train") { render(); const l = document.querySelector("details .log"); if (l) l.scrollTop = l.scrollHeight; }
}

async function trStart(resume) {
  tr.metrics = []; tr.samples = []; tr.log = []; tr.progress = null; tr.stepText = "Starting…"; tr.error = null; tr.notice = null; tr.running = true; render();
  const over = {};
  Object.entries(tr.over).forEach(([k, v]) => { if (v !== "" && v != null) over[k] = k === "lr" || k === "epochs" ? Number(v) : Number(v); });
  try { await invoke("train_start", { model: tr.model, method: tr.method, preset: tr.preset, overrides: over, resume: resume || null, server: tr.server || null }); }
  catch (e) { tr.running = false; tr.error = String(e); render(); }
}

function wireTrain() {
  if (page !== "train" || !tr.state) return;
  const on = (id, fn) => { const e = $(id); if (e) e.onclick = fn; };
  document.querySelectorAll("[data-model]").forEach((b) => b.onclick = () => { tr.model = b.dataset.model; tr.over = {}; trRefresh(); });
  const wh = $("tr-where"); if (wh) wh.onchange = () => { tr.server = wh.value; tr.srvCheck = null; tr.over = {}; trRefresh(); if (tr.server && !(tr.state.servers.find((x) => x.url === tr.server) || {}).status) invoke("train_server_check", { url: tr.server }).catch((e) => { tr.srvCheck = { ok: false, text: String(e) }; render(); }); };
  on("srv-test", () => { tr.srvCheck = null; invoke("train_server_check", { url: tr.server }).catch((e) => { tr.srvCheck = { ok: false, text: String(e) }; render(); }); });
  on("srv-remove", async () => { tr.state = await invoke("train_server_remove", { url: tr.server }); tr.server = ""; tr.srvCheck = null; trRefresh(); });
  on("srv-add", () => { tr.adding = true; render(); });
  on("s-cancel", () => { tr.adding = false; render(); });
  on("s-save", async () => {
    try { tr.state = await invoke("train_server_add", { name: $("s-name").value, url: $("s-url").value, token: $("s-token").value }); tr.server = tr.state.servers.at(-1).url; tr.adding = false; tr.srvCheck = null; await trRefresh();
      invoke("train_server_check", { url: tr.server }).catch(() => {}); }
    catch (e) { tr.srvCheck = { ok: false, text: String(e) }; render(); }
  });
  const m = $("tr-method"); if (m) m.onchange = () => { tr.method = m.value; tr.over = {}; trRefresh(); };
  const pr = $("tr-preset"); if (pr) pr.onchange = () => { tr.preset = pr.value; tr.over = {}; trRefresh(); };
  document.querySelectorAll(".ov").forEach((i) => i.onchange = () => { tr.over[i.dataset.key] = i.value; });
  on("tr-start", () => trStart(null));
  on("tr-resume", () => trStart($("tr-resume").dataset.run));
  on("tr-stop", () => { tr.stepText = "Stopping after this step…"; invoke("train_stop"); render(); });
  document.querySelectorAll("[data-showrun]").forEach((b) => b.onclick = async () => {
    const ev = await invoke("train_metrics", { run: b.dataset.showrun });
    tr.metrics = ev.filter((e) => e.type === "metric").map((e) => ({ ...e, kind: "metric" }));
    tr.samples = ev.filter((e) => e.type === "sample");
    tr.notice = `Run ${b.dataset.showrun}`; tr.error = null; tr.progress = null; tr.stepText = ""; render();
  });
}

// ------------------------------------------------------------------------------------------------ step 5: use in nd-world
let us = { state: null, run: "", quant: "q4_k_m", numCtx: 8192, server: "", modelName: "", running: false, step: "", progress: null, log: [], error: null, ok: null, probe: null, check: null, adding: false };
const copyBtn = (text) => `<button class="link" data-copy="${esc(text)}">Copy</button>`;

function renderUse() {
  if (!state.bundle) return `<h1>Step 5 · Use in nd-world</h1><p class="lead">Import your world data first (Step 1).</p>`;
  const st = us.state;
  if (!st) return `<h1>Step 5 · Use in nd-world</h1><p class="lead">Loading…</p>`;
  if (!st.runs.length) return `<h1>Step 5 · Use in nd-world</h1><p class="lead">Train a model first (Step 3). A run that was stopped can be exported too once it has saved its model.</p>`;
  if (!us.run) us.run = st.runs[0].name;
  if (!us.server && st.servers.length) us.server = st.servers[0].url;
  if (!us.modelName) us.modelName = st.suggested_name;
  const runOpts = st.runs.map((r) => `<option value="${esc(r.name)}" ${us.run === r.name ? "selected" : ""}>${esc(r.name)} · ${esc(r.model)} · ${esc(r.status)}${r.loss != null ? " · loss " + (+r.loss).toFixed(2) : ""}</option>`).join("");
  const quantOpts = st.quants.map((q) => `<option value="${q.id}" ${us.quant === q.id ? "selected" : ""}>${esc(q.name)}</option>`).join("");
  const busy = us.running;
  const files = `<div class="card wide"><h2>1 · Make the files</h2>
    <div class="row"><label>Run <select id="us-run">${runOpts}</select></label><label>File size <select id="us-quant">${quantOpts}</select></label>
      <label>Context <select id="us-ctx">${[4096, 8192, 16384, 32768].map((n) => `<option ${us.numCtx === n ? "selected" : ""}>${n}</option>`).join("")}</select></label></div>
    <p class="dim">Joins the trained add-on into the model and makes a <strong>GGUF</strong> file with a <strong>Modelfile</strong> (a few minutes; the file is 1 to 6 GB).</p>
    <button class="primary" id="us-make" ${busy || !st.env_ready ? "disabled" : ""}>Make the files</button>${!st.env_ready ? ` <span class="err">Open "⚙ Training setup" first.</span>` : ""}</div>`;
  const ex = st.exports.length ? st.exports.map((e) => `<li><strong>${esc(e.name)}</strong> · ${e.size_mb != null ? e.size_mb + " MB" : "file missing"}<br><span class="dim">${esc(e.dir)}</span> ${copyBtn(e.dir)}
      ${e.installed && e.installed.length ? `<br><span class="dim">Installed: ${e.installed.map((i) => esc(i.model) + " @ " + esc(i.url)).join(", ")}</span>` : ""}</li>`).join("") : "";
  const list = ex ? `<div class="card wide"><h2>Your files</h2><ul class="exlist">${ex}</ul></div>` : "";
  const srvOpts = st.servers.map((x) => `<option value="${esc(x.url)}" ${us.server === x.url ? "selected" : ""}>${esc(x.name)} (${esc(x.url)})</option>`).join("");
  const exOpts = st.exports.map((e) => `<option value="${esc(e.name)}" ${us.exp === e.name ? "selected" : ""}>${esc(e.name)}</option>`).join("");
  if (!us.exp && st.exports.length) us.exp = st.exports[0].name;
  const install = st.exports.length ? `<div class="card wide"><h2>2 · Put it into Ollama</h2>
    <div class="row"><label>Files <select id="us-exp">${exOpts}</select></label><label>Ollama server <select id="us-server">${srvOpts}</select></label>
      <button class="secondary" id="us-addsrv">Add a server…</button></div>
    ${us.adding ? `<div class="row"><input type="text" id="srv-name" placeholder="Name, e.g. TrueNAS"><input type="text" id="srv-url" placeholder="Address, e.g. http://truenas:11434"><button class="primary" id="srv-save">Add</button><button class="secondary" id="srv-cancel">Cancel</button></div>` : ""}
    <div class="row"><label>Model name in Ollama <input type="text" id="us-name" value="${esc(us.modelName)}" style="min-width:220px"></label></div>
    <div class="row"><button class="secondary" id="us-check" ${busy ? "disabled" : ""}>Is Ollama running?</button><button class="primary" id="us-install" ${busy ? "disabled" : ""}>Install into Ollama</button><button class="secondary" id="us-probe" ${busy ? "disabled" : ""}>Test it with a question</button></div>
    ${us.check ? `<div class="${us.check.ok ? "ok" : "err"}">${esc(us.check.text)}</div>` : ""}
    ${us.probe ? `<div class="sample"><div class="q">${esc(us.probe.question)}</div><div class="ans">${esc(us.probe.answer) || "<em>(no answer)</em>"}</div><div class="dim">One of the training's own questions. If the answer is empty or nonsense, something is wrong with how Ollama runs the model: tell me what you see.</div></div>` : ""}</div>` : "";
  const name = us.modelName || st.suggested_name;
  const env = `${name}`;
  const how = st.exports.length ? `<div class="card wide"><h2>3 · Use it in nd-world</h2>
    <p><strong>nd-world with Ollama</strong> (the legacy AI back-end):</p>
    <ol><li>On the computer where nd-world runs, set <code>OLLAMA_MODEL=${esc(env)}</code> ${copyBtn("OLLAMA_MODEL=" + env)} in <code>.env</code> (or in the compose file), and <code>OLLAMA_URL</code> to the Ollama server you installed it into.</li>
    <li>Restart nd-world. Or, without restarting: AI → Models tab → pick <strong>${esc(env)}</strong> → ✓ Use.</li>
    <li>Keep nd-world's <strong>lore search</strong> switched on: the model was trained to use the context nd-world gives it.</li></ol>
    <p><strong>On your TrueNAS server:</strong> add the server above (its Ollama must listen on the network: <code>OLLAMA_HOST=0.0.0.0</code>), install the model there, then set <code>OLLAMA_MODEL</code> as above. The model uses the server's graphics card like any other model.</p>
    <p><strong>nd-world with Unsloth Studio</strong>: the GGUF file in the folder above can be loaded in Studio's Model Hub (local model); then set <code>UNSLOTH_MODEL</code> to it and <code>LLM_CONTEXT_TOKENS</code> to the context length it was loaded with (${us.numCtx}). The chat template inside the file answers without thinking, as the model was trained.</p>
    <div class="warn"><strong>For the game master only.</strong> This model was trained on <em>all</em> of your world's text, including GM secrets, and it will repeat them. If players can use AI in this world ("players can use Ask AI"), turn that off, or keep players on another model.</div></div>` : "";
  const prog = (us.running || us.step) ? `<div class="stepline">${esc(us.step)}</div>${us.progress != null ? `<progress max="1" value="${us.progress}"></progress>` : ""}` : "";
  const msg = `${us.error ? `<div class="err">${esc(us.error)}</div>` : ""}${us.ok ? `<div class="ok">${esc(us.ok)}</div>` : ""}`;
  const log = us.log.length ? `<details><summary>Log</summary><pre class="log">${us.log.slice(-100).map(esc).join("\n")}</pre></details>` : "";
  return `<h1>Step 5 · Use in nd-world</h1><p class="lead">Turns the trained model into a file nd-world can use, installs it into Ollama and shows what to set. The help panel explains each part.</p>
    ${files}${prog}${msg}${log}${list}${install}${how}`;
}

async function usRefresh() {
  try { us.state = await invoke("export_state"); } catch (e) { us.error = String(e); }
  if (page === "use") render();
}

function onExportEvent(p) {
  if (p.kind === "log") us.log.push(p.text);
  else if (p.kind === "step") { us.step = `Step ${p.index} of ${p.total}: ${p.text}`; us.progress = (p.index - 1) / p.total; }
  else if (p.kind === "progress") { if (p.text) us.step = p.text; if (p.value != null) us.progress = p.value; }
  else if (p.kind === "done") {
    us.running = false; us.progress = null; us.step = "";
    const d = p.data || {};
    if (d.probe) us.probe = d.probe;
    else if (d.ollama_version) us.check = { ok: true, text: `Ollama ${d.ollama_version} is running. Models there: ${d.models.length ? d.models.join(", ") : "none yet"}.` };
    else if (d.installed) us.ok = `Installed as "${d.installed}" at ${d.url}.`;
    else if (d.gguf_file) us.ok = `Files made: ${d.name}.`;
    usRefresh(); setState_(); 
  } else if (p.kind === "error") {
    us.running = false; us.progress = null; us.step = "";
    if (us.pending === "check") us.check = { ok: false, text: p.text }; else us.error = p.text;
    usRefresh();
  }
  if (page === "use") render();
}

async function setState_() { try { setState(await invoke("get_state")); } catch (_) { /* ignore */ } }

async function usRun(kind, cmd, args) {
  us.running = true; us.error = null; us.ok = null; us.log = []; us.step = "Starting…"; us.progress = null; us.pending = kind;
  if (kind === "probe") us.probe = null; if (kind === "check") us.check = null;
  render();
  try { await invoke(cmd, args); } catch (e) { us.running = false; us.error = String(e); us.step = ""; render(); }
}

function wireUse() {
  if (page !== "use" || !us.state) return;
  const on = (id, fn) => { const e = $(id); if (e) e.onclick = fn; };
  const bind = (id, key, num) => { const e = $(id); if (e) e.onchange = () => { us[key] = num ? Number(e.value) : e.value; }; };
  bind("us-run", "run"); bind("us-quant", "quant"); bind("us-ctx", "numCtx", true); bind("us-exp", "exp"); bind("us-server", "server");
  const nm = $("us-name"); if (nm) nm.onchange = () => { us.modelName = nm.value.trim(); render(); };
  document.querySelectorAll("[data-copy]").forEach((b) => b.onclick = async () => { try { await navigator.clipboard.writeText(b.dataset.copy); b.textContent = "Copied"; setTimeout(() => (b.textContent = "Copy"), 1500); } catch (_) { b.textContent = "Select and copy by hand"; } });
  on("us-make", () => usRun("make", "export_start", { run: us.run, quant: us.quant, numCtx: us.numCtx }));
  on("us-check", () => usRun("check", "ollama_check", { url: us.server }));
  on("us-install", () => usRun("install", "ollama_install", { export: us.exp, url: us.server, name: us.modelName || us.state.suggested_name }));
  on("us-probe", () => usRun("probe", "ollama_probe", { url: us.server, name: us.modelName || us.state.suggested_name }));
  on("us-addsrv", () => { us.adding = true; render(); });
  on("srv-cancel", () => { us.adding = false; render(); });
  on("srv-save", async () => {
    try { us.state = await invoke("ollama_server_add", { name: $("srv-name").value, url: $("srv-url").value }); us.server = us.state.servers.at(-1).url; us.adding = false; us.error = null; }
    catch (e) { us.error = String(e); }
    render();
  });
}

// ------------------------------------------------------------------------------------------- retrain with the new export (step 1)
let rt = { info: null, confirm: false, fromPrev: false, exportAfter: false, installAfter: false, error: null };

async function rtRefresh() {
  try { rt.info = await invoke("retrain_info"); } catch (e) { rt.info = null; }
  if (page === "data") render();
}

function retrainCard() {
  const i = rt.info;
  if (!i || !i.available) return rt.error ? `<div class="err">${esc(rt.error)}</div>` : "";
  const l = i.last_run;
  if (!rt.confirm) return `<div class="card wide"><h2>Your world has changed</h2><p>${esc(i.summary)}</p>
    <p class="dim">Your trained model (${esc(l.model)}) does not know about this yet.</p><button class="primary" id="rt-open">Retrain with the new export</button>${rt.error ? `<div class="err">${esc(rt.error)}</div>` : ""}</div>`;
  const hasEnv = us.state && us.state.servers.length;
  return `<div class="card wide"><h2>Retrain with the new export</h2><p>${esc(i.summary)}</p><p>This will:</p><ol>${i.steps.map((x) => `<li>${esc(x)}</li>`).join("")}</ol>
    <label class="opt"><input type="checkbox" id="rt-prev" ${rt.fromPrev || (i.continue_suggested && rt.fromPrev !== false) ? "checked" : ""}> Continue from the previous model (shorter and faster; best for small changes${i.continue_suggested ? "" : ". More than a quarter of your world changed, so training again from the original is cleaner"})</label>
    <label class="opt"><input type="checkbox" id="rt-export" ${rt.exportAfter ? "checked" : ""}> Then make the files for nd-world (same size as before, Q4_K_M if none yet)</label>
    <label class="opt"><input type="checkbox" id="rt-install" ${rt.installAfter ? "checked" : ""} ${hasEnv ? "" : "disabled"}> Then install it into Ollama on this computer</label>
    <div class="row"><button class="primary" id="rt-go">Start</button><button class="secondary" id="rt-cancel">Cancel</button></div></div>`;
}

function wireRetrain() {
  if (page !== "data") return;
  const on = (id, fn) => { const e = $(id); if (e) e.onclick = fn; };
  on("rt-open", () => { rt.confirm = true; rt.fromPrev = rt.info.continue_suggested; render(); });
  on("rt-cancel", () => { rt.confirm = false; render(); });
  on("rt-go", async () => {
    rt.fromPrev = $("rt-prev").checked; rt.exportAfter = $("rt-export").checked; rt.installAfter = $("rt-install").checked;
    rt.error = null;
    try {
      const url = rt.installAfter && us.state ? (us.state.servers.find((x) => x.url.includes("localhost")) || us.state.servers[0]).url : null;
      tr.metrics = []; tr.samples = []; tr.log = []; tr.progress = null; tr.stepText = "Making the new training data…"; tr.error = null; tr.notice = null; tr.running = true;
      await invoke("retrain_start", { fromPrevious: rt.fromPrev, exportAfter: rt.exportAfter, installUrl: url, ollamaName: null });
      rt.confirm = false; rt.info = null; go("train");
    } catch (e) { tr.running = false; rt.error = String(e); render(); }
  });
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
  el.innerHTML = page === "welcome" ? renderWelcome() : page === "data" ? renderData() : page === "setup" ? renderSetup() : page === "dataset" ? renderDataset() : page === "train" ? renderTrain() : page === "use" ? renderUse() : renderSoon(page);
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
  wireTrain();
  wireUse();
  wireRetrain();
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
  if (r) { setState(r.state); message = { kind: "ok", text: r.changes ? "Imported. Changes: " + r.changes : "Imported." }; rtRefresh(); }
  render();
}

async function go(key) {
  page = key;
  message = message && message.kind === "err" && key !== "welcome" && key !== "data" ? null : message;
  render();
  if (key === "setup") refreshEnv();
  if (key === "dataset" && state.bundle) { ds.ok = null; dsRefresh(false); }
  if (key === "train" && state.bundle) trRefresh();
  if (key === "use" && state.bundle) usRefresh();
  if (key === "data" && state.project) { if (!us.state) usRefresh(); rtRefresh(); }
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
  try { await tauri.event.listen("server-event", (e) => onServerEvent(e.payload)); } catch (_) { /* previews */ }
  try { await tauri.event.listen("export-event", (e) => onExportEvent(e.payload)); } catch (_) { /* previews */ }
  try { await tauri.event.listen("train-event", (e) => onTrainEvent(e.payload)); } catch (_) { /* previews */ }
  try { await tauri.event.listen("env-event", (e) => onEnvEvent(e.payload)); } catch (_) { /* events are optional in previews */ }
  go("welcome");
}
init();
