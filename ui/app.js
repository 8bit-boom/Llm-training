// ND Lore Forge window. All logic lives in Rust (crates/ndforge-core); this file only draws the state it returns.
const tauri = window.__TAURI__;
const invoke = tauri.core.invoke;
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const TITLES = {
  welcome: ["Welcome to ND Lore Forge", "Teach a small AI model your nd-world world. Start by creating a project (one project = one world)."],
  test: ["Step 4 · Test", "Checks that the trained model knows your world better than the original."],
  reshape: ["Slim or grow a model", "Make a smaller or a bigger version of a model by taking whole blocks of layers out or copying some in."],
};

let state = null;
let page = "welcome";
let pending = null;      // a checked file waiting for the user's confirmation: {path, info}
let message = null;      // {kind: "err"|"ok", text}
let env = { status: null, running: false, log: [], step: "", progress: null, result: null, error: null, noUnsloth: false };
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
  items.push({ key: "reshape", label: "🧬 Slim or grow a model", unlocked: !!state.bundle, done: false });
  items.push({ key: "setup", label: "⚙ Training setup", unlocked: true, done: false });
  items.push({ key: "guide", label: "📖 Guide & troubleshooting", unlocked: true, done: false });
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
  const ws = state.worlds && state.worlds.length ? state.worlds : [{ world: state.bundle.world, slug: "", lines: state.bundle.lines }];
  const many = ws.length > 1;
  const cards = ws.map((w) => `<div class="card"><h2>World: ${esc(w.world)}</h2>This project contains:<ul>${w.lines.map((l) => `<li>${esc(l)}</li>`).join("") || "<li>(no records)</li>"}</ul>${many ? `<button class="secondary rmworld" data-slug="${esc(w.slug)}">Remove this world from the project</button>` : ""}</div>`).join("");
  const note = many ? `<p class="lead">The model will be trained on all ${ws.length} worlds at once. Every example carries its own world's name, so the model learns which world a question is about. After adding or removing a world, make the examples again in Step 2.</p>` : "";
  return note + cards;
}

function pendingCard() {
  if (!pending) return "";
  const { info } = pending;
  const kind = info.check.kind;
  const lines = `<ul>${info.lines.map((l) => `<li>${esc(l)}</li>`).join("")}</ul>`;
  if (kind === "refused") return `<div class="card"><h2>This file cannot be used</h2><div class="err">${esc(info.check.text)}</div><button class="secondary" id="cancel">OK</button></div>`;
  const title = kind === "update" ? "Update the world data?" : kind === "add" ? "Add another world?" : "Use this file?";
  return `<div class="card"><h2>${title} (${esc(info.world)})</h2>${lines}<p>${kind === "update" ? "<strong>What changed:</strong> " : ""}${esc(info.check.text)}</p>
    <div class="row"><button class="primary" id="apply">${kind === "update" ? "Use this file" : kind === "add" ? "Add this world" : "Import"}</button><button class="secondary" id="cancel">Cancel</button></div></div>`;
}

function renderData() {
  return `<h1>Step 1 · Get world data</h1>
    <p class="lead">Export a <strong>Training bundle</strong> from nd-world (Tools → Export &amp; Backup), then choose the file here or drag it onto this page. The help panel explains how.</p>
    <div class="drop" id="drop">Drag a <code>.ndtrain</code> file here</div>
    <div class="row"><button class="primary" id="choose">${state.bundle ? "Choose a training file (another world, or a newer file)…" : "Choose training file…"}</button></div>
    ${msgHtml()}${pendingCard()}${retrainCard()}${bundleCard()}`;
}

let ds = { state: null, filter: { builder: "", search: "", status: "" }, offset: 0, total: 0, items: [], editing: null, busy: false, error: null, ok: null, tc: { running: false, text: "", progress: null, report: null, opts: null, trial: false, open: false } };
const PAGE = 20;

function recipeFromForm() {
  const r = ds.state.recipe;
  const builders = [...document.querySelectorAll(".bld:checked")].map((c) => c.value);
  return { ...r, world_book: $("rb-book").checked, builders, phrasings: Number($("rb-phr").value), test_percent: Number($("rb-test").value) };
}

const CHECKS = { supported: ["✓ checked", "ok"], partly: ["⚠ partly supported", "warn"], unsupported: ["✗ not supported", "bad"], unchecked: ["not checked", ""] };
function badge(e) { const c = CHECKS[e.check]; return c ? ` <span class="tag chk ${c[1]}">${c[0]}</span>` : ""; }

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
  let teacher = "";
  if (sum && st.teacher) {
    const T = st.teacher, tc = ds.tc, o = tc.opts || T.options, s0 = T.settings;
    const kinds = T.kinds.map((k) => `<label class="opt"><input type="checkbox" class="tk" value="${k.id}" ${o.kinds.includes(k.id) ? "checked" : ""}> <strong>${esc(k.name)}</strong> <span class="dim">— ${esc(k.help)}</span></label>`).join("");
    const rep = tc.report ? `<div class="ok">${tc.report.stopped ? "Stopped. " : ""}Made ${tc.report.added} new examples from ${tc.report.jobs_done} of ${tc.report.jobs} pieces of text. Checked: ${tc.report.supported} supported, ${tc.report.partly} partly (kept, marked), ${tc.report.unsupported} unsupported (dropped), ${tc.report.unchecked} not checked.${tc.report.failed_calls ? ` ${tc.report.failed_calls} requests failed${tc.report.first_error ? ": " + esc(tc.report.first_error) : ""}.` : ""}</div>` : "";
    teacher = `<div class="card wide"><h2>Better examples with a teacher model <span class="tag">optional</span></h2>
      <p>A bigger model (for example Gemma 4 on your TrueNAS) reads your world and writes deeper questions, rules practice, NPC talk and more. A second pass checks every answer against the lore and drops what the lore does not support.</p>
      ${sum.nd_world_examples ? `<p class="ok">nd-world already made ${sum.nd_world_examples} of these with its own AI model and put them in your world file; they are in the list below (look for kinds starting with “Teacher:”). You can add more here.</p>` : ""}
      <div class="row"><label>Address <input type="text" id="t-url" value="${esc(s0.url)}" style="min-width:230px"></label>
        <label>Model name <input type="text" id="t-model" value="${esc(s0.model)}" placeholder="for example gemma4:e4b" style="min-width:170px"></label>
        <label>Access key <input type="password" id="t-key" placeholder="${T.has_key ? "(saved — leave empty to keep)" : "(none needed for Ollama)"}" style="min-width:150px"></label></div>
      <label class="opt"><input type="checkbox" id="t-unload" ${s0.unload_after ? "checked" : ""}> The teacher is an Ollama server: free its memory when finished, so training can use the graphics card</label>
      <div class="row"><button class="secondary" id="t-save">Save</button><button class="secondary" id="t-check" ${tc.running ? "disabled" : ""}>Test the connection</button></div>
      <details ${tc.open ? "open" : ""} id="t-more"><summary>Choose what to make</summary>${kinds}
        <div class="row" style="margin-top:8px"><label>Examples per piece of text ${sel("t-per", [1, 2, 3, 4], o.per_source)}</label>
          <label>General chat ${sel("t-gen", [0, 5, 10, 20, 30], o.general_percent)} %</label></div>
        <label class="opt"><input type="checkbox" id="t-judge" ${o.judge ? "checked" : ""}> Check every answer against the lore (recommended)</label>
        <label class="opt"><input type="checkbox" id="t-partly" ${o.drop_partly ? "checked" : ""}> Also drop answers that are only partly supported</label></details>
      <div class="row"><button class="primary" id="t-run" ${tc.running ? "disabled" : ""}>Make with the teacher</button>
        <button class="secondary" id="t-trial" ${tc.running ? "disabled" : ""}>Try on 5 pieces first</button>
        ${tc.running ? `<button class="secondary" id="t-stop">Stop</button>` : ""}
        <button class="secondary" id="t-clear" ${tc.running ? "disabled" : ""}>Remove all teacher examples</button></div>
      ${tc.running || tc.text ? `<div class="stepline">${esc(tc.text)}</div>${tc.progress != null ? `<progress max="1" value="${tc.progress}"></progress>` : ""}` : ""}${rep}</div>`;
  }
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
      return `<div class="ex ${e.status === "drop" ? "dropped" : ""}"><div class="q"><span class="tag">${esc(e.builder)}</span>${badge(e)} ${q}${e.edited ? ' <span class="tag">edited</span>' : ""}</div>${body}
        <div class="row small"><button class="secondary" data-toggle="${e.id}" data-to="${e.status === "drop" ? "keep" : "drop"}">${e.status === "drop" ? "Keep" : "Drop"}</button>
        <button class="secondary" data-edit="${e.id}">Edit</button><span class="dim">from: ${esc(e.source_title)}</span></div>${e.note && e.check !== "supported" ? `<div class="dim small">Checker: ${esc(e.note)}</div>` : ""}</div>`;
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
  const vers = st.versions.length ? `<ul>${st.versions.map((v) => `<li><strong>Version ${v.version}</strong> · ${v.counts.world_book_texts} texts, ${v.counts.train_chat} practice, ${v.counts.test_chat} for testing · ${v.created} <button class="secondary studio-export" data-ver="${v.version}" title="Writes train.jsonl (and eval.jsonl) in the format Unsloth Studio's Train page reads, with a README of the settings to choose">Export for Unsloth Studio…</button></li>`).join("")}</ul>` : "<p class=\"dim\">No version saved yet.</p>";
  const save = sum ? `<div class="card wide"><h2>Save</h2><p>Training uses a saved version. Dropped examples are left out.</p><button class="primary" id="save" ${ds.busy ? "disabled" : ""}>Save this version</button>
    <label class="opt" title="Studio has no plain-text mode: the long world-book texts become answers to 'Tell me what the lore says about …'"><input type="checkbox" id="studio-book" ${ds.studioBook === false ? "" : "checked"}> Include the world book when exporting for Unsloth Studio</label>${vers}</div>` : "";
  return `<h1>Step 2 · Make training data</h1><p class="lead">Turns your world into practice material. The help panel explains every choice.</p>
    ${ds.error ? `<div class="err">${esc(ds.error)}</div>` : ""}${ds.ok ? `<div class="ok">${esc(ds.ok)}</div>` : ""}${recipe}${teacher}${summary}${review}${save}`;
}

function onTeacherEvent(p) {
  const tc = ds.tc;
  if (p.kind === "progress") { tc.text = p.text; tc.progress = p.total ? p.done / p.total : null; }
  else if (p.kind === "error") { tc.running = false; tc.text = ""; ds.error = p.text; }
  else if (p.kind === "done") {
    tc.running = false; tc.text = ""; tc.progress = null;
    if (p.data && p.data.checked) ds.ok = "The teacher answered. Everything is ready.";
    else tc.report = p.data;
    dsRefresh(true); return;
  }
  if (page === "dataset") render();
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
let tr = { shutdownAfter: false, stopping: false, bench: { running: false, text: "", progress: null, result: null, deep: false, error: null, used: null }, speed: null, hf: { open: false, input: "", busy: false, found: null, error: null }, stages: "mixed", tryAnyway: false, server: "", srvCheck: null, adding: false, state: null, model: null, method: "lora", preset: "normal", plan: null, over: {}, running: false, run: null, metrics: [], samples: [], log: [], progress: null, stepText: "", error: null, notice: null };
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

function benchHtml(st) {
  const b = tr.bench, r = b.result;
  const local = !tr.server;
  const tied = (r && r.tied) || [];
  let sawRound2 = false;
  const rows = r ? r.rows.map((x) => {
    const head = x.round === 2 && !sawRound2 ? (sawRound2 = true, `<tr><td colspan="4"><strong>Round 2</strong> <span class="dim small">· the best batch size of round 1 with each attention kernel, every setting run twice (both speeds shown) so that noise is visible</span></td></tr>`) : "";
    const runs = x.tok_s_runs && x.tok_s_runs.length > 1 ? `<div class="dim small">${x.tok_s_runs.map((n) => n.toLocaleString()).join(" / ")}</div>` : "";
    const tie = tied.includes(x.id) ? `<div class="dim small">within noise of the best</div>` : "";
    return head + `<tr class="${r.best === x.id ? "best" : ""}"><td>${r.best === x.id ? "★ " : ""}${esc(x.name)}${x.notes && x.notes.length ? `<div class="dim small">${x.notes.map(esc).join(" ")}</div>` : ""}</td>
      ${x.ok ? `<td>${x.tok_s ? x.tok_s.toLocaleString() : "–"}${runs}${tie}</td><td>${x.vs_current != null ? "×" + x.vs_current : ""}</td><td>${x.peak_gb != null ? x.peak_gb + " GB" : "–"}</td>` : `<td colspan="3" class="err">${esc(x.error || "failed")}</td>`}</tr>`;
  }).join("") : "";
  const best = r && r.rows.find((x) => x.id === r.best);
  const useless = !best || best.id === "current";
  return `<div class="card wide"><h2>Find the fastest settings <span class="tag">optional</span></h2>
    <p>Runs a few short real training runs (${16} steps each) with different settings and shows tokens per second and memory for each. Every run loads the model again, so it takes several minutes. Nothing from these runs is kept except the table.</p>
    ${local ? `<div class="row"><label class="opt"><input type="checkbox" id="bn-deep" ${b.deep ? "checked" : ""} ${b.running ? "disabled" : ""}> Also try the attention kernels (with Unsloth too) in a second round with the best batch size, each run twice, and torch.compile, Liger kernels and padding-free batching (much slower; the last three are ignored with Unsloth)</label></div>
    <div class="row"><button class="secondary" id="bn-start" ${b.running || !st.env_ready || tr.running ? "disabled" : ""}>Run the benchmark</button>${b.running ? `<button class="secondary" id="bn-stop">Stop</button>` : ""}</div>` : `<p class="dim">Choose "This computer" under "Where to train": a server's card is tried by training there.</p>`}
    ${b.running || b.text ? `<div class="stepline">${esc(b.text)}</div>${b.progress != null ? `<progress max="1" value="${b.progress}"></progress>` : ""}` : ""}${b.error ? `<div class="err">${esc(b.error)}</div>` : ""}
    ${r ? `<table class="tbl"><tr><th>Setting</th><th>Tokens/s</th><th>vs current</th><th>Memory peak</th></tr>${rows}</table>
      <p class="dim small">${esc(r.model)} · ${r.method} · ${r.steps} steps each · ${esc(r.created)}${r.stopped ? " · stopped early" : ""}. Speeds are measured after the start-up (the first quarter of the steps is left out). A short run can only show big differences${tied.length ? `; ${tied.length} setting${tied.length > 1 ? "s are" : " is"} within noise of the ★` : ""}. ${r.unsloth ? "Unsloth is installed here, so compile, Liger and padding-free were not tried." : ""}</p>
      <div class="row"><button class="primary" id="bn-use" ${useless ? "disabled" : ""}>Use the fastest settings</button>${useless ? `<span class="dim">${best ? "Your current settings are already the fastest." : "No run finished."}</span>` : ""}${b.used ? `<span class="ok">Will be used for the next run: ${esc(b.used)}</span>` : ""}</div>` : ""}</div>`;
}

function speedHtml() {
  const sp = tr.speed;
  if (!sp) return "";
  const batching = sp.bucketed ? `Batches are made of examples of similar length: padding is about ${sp.pad_bucketed_pct} % instead of ${sp.pad_random_pct} %.`
    : sp.pad_random_pct >= 8 ? `Padding is about ${sp.pad_random_pct} % of the computed tokens.` : `Little padding (${sp.pad_random_pct} %).`;
  const extra = [sp.attn ? `attention: ${sp.attn}` : null, sp.checkpointing ? `checkpointing: ${sp.checkpointing}` : null].filter(Boolean).join(" · ");
  return `<div class="dim small">${esc(batching)} Examples are ${sp.mean_len} tokens long on average (longest ${sp.max_len}).${extra ? " " + esc(extra) + "." : ""}${sp.checkpointing_note ? " " + esc(sp.checkpointing_note) : ""}</div>${(sp.notes || []).map((n) => `<div class="warn">⚠ ${esc(n)}</div>`).join("")}`;
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
    return `<button class="mcard ${tr.model === m.id ? "sel" : ""} fit-${f.fit}" data-model="${m.id}"><strong>${esc(m.name)}${m.custom ? ' <span class="tag">added</span>' : ""}</strong>${m.reshape ? `<span class="dim">${esc(m.reshape)}${m.derived_from ? " · from " + esc(String(m.derived_from).split("/").pop()) : ""}</span>` : ""}
      <span class="dim">${m.params_b.toFixed(1)} B parameters · ${esc(m.license)}</span><span>${fit[0]} ${fit[1]} · ≈ ${f.measured != null ? f.measured + " GB measured" : f.est_vram_gb + " GB"}</span><span class="dim">${esc(m.good_for)}</span></button>`;
  }).join("");
  const hf = tr.hf, fm = hf.found && hf.found.model;
  const addModel = `<div class="card wide"><div class="row"><button class="secondary" id="hf-open">＋ Add a model from Hugging Face…</button>${st.models.filter((c) => c.model.custom).map((c) => `<button class="secondary" data-hfremove="${esc(c.model.id)}" title="Remove ${esc(c.model.name)} from the list" ${tr.running ? "disabled" : ""}>✕ ${esc(c.model.name)}</button>`).join("")}</div>
    ${hf.open ? `<p class="dim">Type the model's name as it is on Hugging Face (like <code>unsloth/Qwen3-4B</code>), paste its address, or choose a <strong>folder on this computer</strong> that already holds the model (the one with <code>config.json</code> and the <code>.safetensors</code> files). It must be an open chat/instruct model; nothing is downloaded for a folder.</p>
      <div class="row"><input type="text" id="hf-input" value="${esc(hf.input)}" placeholder="organisation/model-name" style="min-width:320px" ${hf.busy ? "disabled" : ""}><button class="secondary" id="hf-folder" ${hf.busy ? "disabled" : ""}>Choose a folder…</button><button class="primary" id="hf-look" ${hf.busy ? "disabled" : ""}>${hf.busy ? "Looking…" : "Look it up"}</button><button class="secondary" id="hf-cancel">Cancel</button></div>
      ${hf.error ? `<div class="err">${esc(hf.error)}</div>` : ""}
      ${fm ? `<div class="card"><h3>${esc(fm.name)}</h3><p>${esc(fm.repo)} · ${fm.params_b.toFixed(1)} B parameters · licence: ${esc(fm.license)} · chat format: ${fm.family === "generic" ? "the model's own" : esc(fm.family)}<br>Memory for training (estimate): LoRA ≈ ${fm.vram_lora_gb} GB, QLoRA ≈ ${fm.vram_qlora_gb} GB</p>
        ${hf.found.warnings.length ? `<ul>${hf.found.warnings.map((w) => `<li>${esc(w)}</li>`).join("")}</ul>` : ""}
        <div class="row"><button class="primary" id="hf-add">Add it to the list</button></div></div>` : ""}` : ""}</div>`;
  const isUn = tr.server === "unsloth" && st.unsloth;
  const srv = isUn ? { name: st.unsloth.name, url: st.unsloth.url, status: st.unsloth.status, unsloth: true } : tr.server ? st.servers.find((x) => x.url === tr.server) : null;
  const srvStatus = srv && srv.status;
  const srvLine = !srv ? `Training on this computer: ${gpu}.` : srvStatus
    ? (srvStatus.gpu ? `${esc(srv.name)}: ${esc(srvStatus.gpu.name)} · ${srvStatus.gpu.vram_gb} GB · ${esc(srvStatus.gpu.precision)} · ${srvStatus.busy ? "busy with " + esc(srvStatus.job) : "free"}${srvStatus.ollama ? " · Ollama there will be paused while training" : ""}` : `${esc(srv.name)}: no graphics card (${esc(srvStatus.gpu_problem || "none found")})`)
    : "Not checked yet. Press “Test the connection”.";
  const where = `<div class="card wide"><h2>Where to train</h2><div class="row"><label>Computer <select id="tr-where"><option value="">This computer</option>${st.unsloth ? `<option value="unsloth" ${tr.server === "unsloth" ? "selected" : ""}>Unsloth Desktop (${esc(st.unsloth.name)})</option>` : ""}${st.servers.map((x) => `<option value="${esc(x.url)}" ${tr.server === x.url ? "selected" : ""}>${esc(x.name)} (${esc(x.url)})</option>`).join("")}</select></label>
      ${srv ? `<button class="secondary" id="srv-test" ${tr.running ? "disabled" : ""}>Test the connection</button>${srv.unsloth ? "" : `<button class="secondary" id="srv-remove" ${tr.running ? "disabled" : ""}>Remove this server</button>`}` : ""}
      <button class="secondary" id="srv-add">Add a server…</button></div>
    ${tr.adding ? `<div class="row"><input type="text" id="s-name" placeholder="Name, e.g. TrueNAS"><input type="text" id="s-url" placeholder="Address, e.g. http://truenas:8765"><input type="password" id="s-token" placeholder="Access token (NDFORGE_TOKEN)" style="min-width:220px"><button class="primary" id="s-save">Add</button><button class="secondary" id="s-cancel">Cancel</button></div>` : ""}
    <div class="dim">${srvLine}</div>${tr.srvCheck ? `<div class="${tr.srvCheck.ok ? "ok" : "err"}">${esc(tr.srvCheck.text)}</div>` : ""}
    ${srv && srv.unsloth ? `<p class="dim">Unsloth Desktop must stay open while it trains. Unsloth chooses the speed settings itself; Lore Forge sends the rank, learning rate, epochs and context length. Only LoRA / QLoRA, one stage. When Unsloth is on this computer the trained adapter is copied back here; otherwise it stays in Unsloth's folder. A stop in Unsloth cannot be continued.</p>` : srv ? `<p class="dim">The server's graphics card is used only by the training until it ends: AI chat, transcription and image generation on that server wait or are unavailable meanwhile. Only LoRA / QLoRA is available there. The trained model is copied back to this computer.</p>` : ""}</div>`;
  const pl = tr.plan;
  const sel = st.models.find((c) => c.model.id === tr.model);
  const exp = state.settings.expert_mode && pl ? `<div class="card wide"><h2>Settings (Expert)</h2><div class="grid">${[
    ...(tr.method === "full" ? [] : [["rank", "LoRA rank", pl.hyper.rank]]), ["lr", "Learning rate", pl.hyper.lr], ...(pl.hyper.stages === "book_first" ? [["book_epochs", "Passes over the world book", pl.hyper.book_epochs]] : []), ["epochs", "Epochs", pl.hyper.epochs], ["max_steps", "Fixed steps (empty = use epochs)", pl.hyper.max_steps ?? ""],
    ["batch", "Batch size", pl.hyper.batch], ["grad_accum", "Gradient accumulation", pl.hyper.grad_accum], ["max_seq", "Sequence length (tokens)", pl.hyper.max_seq]].map(([k, l, val]) =>
      `<label>${l}<input type="text" class="ov" data-key="${k}" value="${tr.over[k] ?? val}"></label>`).join("")}
    ${[["engine", "Training engine", ["auto", "unsloth", "standard"], "auto: Unsloth if it is installed, otherwise the standard libraries. standard: Transformers + PEFT + TRL only (no Unsloth), even when Unsloth is installed."],
        ["bucket", "Batches by length", ["auto", "on", "off"], "Examples of similar length go into the same batch: less wasted padding, faster. Batches then hold examples of one kind more often, so the loss moves more from step to step."],
        ["compile", "torch.compile", ["off", "on"], "Experimental, plain-libraries path with a graphics card and Triton only. The first steps are slow."],
        ["liger", "Liger kernels", ["off", "on"], "Plain-libraries path, if the liger-kernel package is installed and supports the model."],
        ["attention", "Attention kernel", ["auto", "sdpa", "cudnn", "flash_attention_2", "flex_attention"], "auto = the library's choice (Unsloth picks its own). cudnn = PyTorch's sdpa with NVIDIA's cuDNN attention tried first (PyTorch 2.5+; others behind it if it cannot do a case). FlashAttention-2 needs the flash-attn package and an RTX 30 series or newer; flex attention needs PyTorch 2.5+. One that cannot work is dropped with a reason. Use \"Find the fastest settings\" to compare."],
        ["padding_free", "Padding-free batching", ["off", "on"], "Needs the flash-attn package and a model whose attention layers support it (not Qwen3.5 or Gemma 4 yet)."]].map(([k, l, opts, tip]) =>
      `<label title="${esc(tip)}">${l}<select class="ovs" data-key="${k}">${opts.map((o) => `<option ${(tr.over[k] ?? pl.hyper[k] ?? opts[0]) === o ? "selected" : ""}>${o}</option>`).join("")}</select></label>`).join("")}</div>
    ${sel && sel.model.reshape && sel.model.reshape.startsWith("grown") ? `<label class="opt" title="LoRA is attached only to the layers that were added when the model was grown. Faster and gentler, but the old layers do not adapt."><input type="checkbox" id="tr-newonly" ${tr.over.only_new_layers ? "checked" : ""}> Train only the new layers (this model was grown)</label>` : ""}</div>` : "";
  const planHtml = pl ? `<div class="card wide"><h2>The plan</h2>${pl.hyper.stages === "book_first" ? `<p>Two stages: the <strong>world book</strong> (${v.counts.world_book_texts} texts, ${pl.hyper.book_epochs} passes), then the <strong>practice answers</strong> (${v.counts.train_chat} examples, ${pl.hyper.epochs} passes).</p>` : ""}<p>About <strong>${pl.steps}</strong> steps · memory ≈ <strong>${pl.est_vram_gb} GB</strong> (${(FIT[pl.fit] || FIT.unknown)[1]}) · sequence ${pl.hyper.max_seq} tokens · batch ${pl.hyper.batch}×${pl.hyper.grad_accum}</p>
    ${pl.notes.length ? `<ul>${pl.notes.map((n) => `<li>${esc(n)}</li>`).join("")}</ul>` : ""}</div>` : "";
  const TYPES = [["lora", "LoRA, 16-bit", "Trains a small add-on; the original model stays safe. Recommended whenever it fits."],
    ["qlora", "QLoRA, 4-bit", "The same with the model shrunk to 4-bit: needs less memory, a little less quality."],
    ["full", "Full fine-tune", "Changes every weight of the model: the strongest learning, but needs much more memory and disk, can make the model forget how to chat, and gives a full-size copy."]];
  const typeCards = sel ? TYPES.map(([t, name, desc]) => {
    const info = sel[t], fit = FIT[info.fit] || FIT.unknown;
    const blocked = t === "full" && !info.available && !(tr.tryAnyway && info.reasons.length === 1 && info.reasons[0].includes("needs about") && !tr.server);
    return `<button class="mcard ${tr.method === t ? "sel" : ""} ${blocked ? "off" : ""}" data-type="${t}" ${blocked ? "disabled" : ""}><strong>${name}${sel.recommended === t ? " · recommended" : ""}</strong>
      <span>${fit[0]} ${fit[1]} · ≈ ${info.measured != null ? info.measured + " GB measured" : info.est_vram_gb + " GB"}${t === "full" ? ` · disk ≈ ${info.disk_gb} GB` : ""}</span><span class="dim">${desc}</span></button>`;
  }).join("") : "";
  const fullInfo = sel && sel.full;
  const typeWhy = fullInfo && !fullInfo.available ? `<div class="dim">Full fine-tune is not available here: ${fullInfo.reasons.map(esc).join(" ")}</div>
    ${state.settings.expert_mode && !tr.server && fullInfo.reasons.length === 1 && fullInfo.reasons[0].includes("needs about") ? `<label class="opt"><input type="checkbox" id="tr-anyway" ${tr.tryAnyway ? "checked" : ""}> Try anyway (the memory figure is an estimate; it may run out of memory)</label>` : ""}` : "";
  const last = st.runs[0];
  const canResume = last && last.resumable && !tr.running;
  const ctrl = `<div class="row"><button class="primary" id="tr-start" ${tr.running || !tr.model || (!tr.server && !st.env_ready) || (tr.server && !(srvStatus && srvStatus.gpu && !srvStatus.busy)) ? "disabled" : ""}>${tr.server && srv ? "Start training on " + esc(srv.name) : "Start training"}</button>
    <button class="secondary" id="tr-stop" title="Finishes the current step, saves a checkpoint and stops. The run can be continued later, even after closing the program." ${tr.running && !tr.stopping ? "" : "disabled"}>${tr.stopping ? "Stopping…" : "Stop and save"}</button>
    ${canResume ? `<button class="secondary" id="tr-resume" data-run="${esc(last.name)}">Continue the last run (${esc(last.model)}${last.step ? `, from step ${last.step}${last.total_steps ? " of " + last.total_steps : ""}` : ""})</button>` : ""}</div>
    <div class="row power"><label class="opt" title="Stops the computer from going to sleep while a training runs (the screen may still turn off). Applies from the next start."><input type="checkbox" id="tr-awake" ${state.settings.keep_awake ? "checked" : ""} ${tr.running ? "disabled" : ""}> Keep this computer awake during training</label>
      <label class="opt" title="When a training has finished normally, the computer is switched off after a 60-second warning that you can cancel. Not after a stop or an error. Switches itself off again after use."><input type="checkbox" id="tr-shutdown" ${tr.shutdownAfter ? "checked" : ""}> Shut down this computer when training has finished</label></div>`;
  const env = !st.env_ready ? `<div class="err">The training software is not installed yet. Open "⚙ Training setup" first.</div>` : "";
  const live = (tr.running || tr.metrics.length || tr.error || tr.notice) ? `<div class="card wide"><h2>${tr.running ? "Training…" : "Last run"}</h2>
    ${tr.stepText ? `<div class="stepline">${esc(tr.stepText)}</div>` : ""}${tr.progress != null ? `<progress max="1" value="${tr.progress}"></progress>` : ""}
    ${tr.error ? `<div class="err">${esc(tr.error)}</div>` : ""}${tr.notice ? `<div class="ok">${esc(tr.notice)}</div>` : ""}
    ${(() => { const m = tr.metrics.at(-1); return m ? `<div class="stats"><span>Step <strong>${m.step}/${m.total_steps}</strong></span><span>Loss <strong>${(+m.loss).toFixed(3)}</strong></span><span>Memory peak <strong>${m.vram_mb ? (m.vram_mb / 1024).toFixed(1) + " GB" : "–"}</strong></span>${m.tok_s ? `<span>Speed <strong>${m.tok_s.toLocaleString()} tokens/s</strong></span>` : ""}${m.pad_pct != null ? `<span title="Share of the computed tokens that are padding (wasted)">Padding <strong>${m.pad_pct} %</strong></span>` : ""}<span>GPU <strong>${m.gpu_temp != null ? m.gpu_temp + " °C" : "–"}</strong></span><span>Time left <strong>${fmtEta(m.eta_s)}</strong></span></div>` : ""; })()}
    ${speedHtml()}${lossChart(tr.metrics.filter((m) => m.kind === "metric" || m.type === "metric"))}${samplesHtml()}
    ${tr.log.length ? `<details><summary>Log</summary><pre class="log">${tr.log.slice(-120).map(esc).join("\n")}</pre></details>` : ""}</div>` : "";
  const runs = st.runs.length ? `<div class="card wide"><h2>Earlier runs</h2><table class="tbl"><tr><th>Run</th><th>Type</th><th>Status</th><th>Step</th><th>Loss</th><th>Memory</th><th>Where</th><th></th></tr>${st.runs.map((r) =>
    `<tr><td>${esc(r.name)}</td><td>${esc(r.method === "full" ? "full" : r.method === "qlora" ? "QLoRA" : "LoRA")}</td><td>${esc(r.status)}${r.status === "interrupted" ? `<div class="dim small" title="${esc(r.error || "")}">cut off (program closed or crash)</div>` : ""}${r.adapter_remote ? `<div class="dim small">${esc(r.note || "The adapter stays in Unsloth.")}</div>` : ""}</td><td>${r.step ?? ""}${r.total_steps ? "/" + r.total_steps : ""}</td><td>${r.loss != null ? (+r.loss).toFixed(3) : ""}</td><td>${r.peak_vram_mb ? (r.peak_vram_mb / 1024).toFixed(1) + " GB" : ""}</td><td>${esc(r.server || "this computer")}</td><td><button class="link" data-showrun="${esc(r.name)}">Show</button>${r.resumable && !tr.running ? ` <button class="link" data-continue="${esc(r.name)}" title="Carry on from the last saved point">Continue</button>` : (r.status === "interrupted" || r.status === "stopped" || r.status === "failed") && !r.resumable && !r.server ? ` <span class="dim small" title="It was cut off before the first saved point">no saved point</span>` : ""}</td></tr>`).join("")}</table></div>` : "";
  return `<h1>Step 3 · Train</h1><p class="lead">Teaches the model your world with LoRA (a small add-on) and practice answers (SFT). The help panel explains each choice.</p>
    <div class="dim">Training data: version ${v.number} · ${v.counts.world_book_texts} world-book texts, ${v.counts.train_chat} practice examples · Graphics card: ${gpu}</div>${tr.server ? "" : env}${where}
    <div class="card wide"><h2>1 · Choose the model</h2><div class="mcards">${cards}</div></div>${addModel}
    <div class="card wide"><h2>2 · How to train</h2>
      <p class="dim" style="margin-top:0">How much of the model should change?</p>
      <div class="mcards">${typeCards}</div>
      ${typeWhy}
      <p class="dim" style="margin-bottom:4px">What should it learn from?</p>
      <label class="opt"><input type="radio" name="tr-stages" value="mixed" ${tr.stages === "mixed" || tr.preset === "quick" ? "checked" : ""}> <strong>The world book and the practice answers together</strong> (default)</label>
      <label class="opt"><input type="radio" name="tr-stages" value="book_first" ${tr.stages === "book_first" && tr.preset !== "quick" ? "checked" : ""} ${tr.preset === "quick" ? "disabled" : ""}> <strong>The world book first, then the practice answers</strong> (two stages; better at remembering facts, takes longer${tr.preset === "quick" ? "; not for the quick test" : ""})</label>
      <div class="row" style="margin-top:10px"><label>Preset <select id="tr-preset"><option value="quick" ${tr.preset === "quick" ? "selected" : ""}>Quick test (minutes)</option><option value="normal" ${tr.preset === "normal" ? "selected" : ""}>Normal</option><option value="best" ${tr.preset === "best" ? "selected" : ""}>Best quality (slower)</option></select></label></div></div>
    ${planHtml}${exp}${benchHtml(st)}<div class="card wide"><h2>3 · Train</h2>${ctrl}</div>${live}${runs}`;
}

async function trRefresh() {
  try {
    tr.state = await invoke("train_state", { server: tr.server || null });
    if (!tr.model && tr.state.models.length) {
      const first = tr.state.models.find((c) => c.lora.fit === "yes" || c.lora.fit === "tight") || tr.state.models[0];
      tr.model = tr.state.models.find((c) => c.model.id === "qwen3.5-4b")?.model.id ?? first.model.id;
      tr.method = (tr.state.models.find((c) => c.model.id === tr.model) || first).recommended;
    }
    if (tr.model) {
      // a type that cannot be used here (full fine-tuning on a server or a small card) falls back to the recommended one
      const cur = tr.state.models.find((c) => c.model.id === tr.model);
      if (cur && tr.method === "full" && !cur.full.available && !tr.tryAnyway) tr.method = cur.recommended;
      tr.plan = await invoke("train_plan", { model: tr.model, method: tr.method, preset: tr.preset, server: tr.server || null, stages: tr.preset === "quick" ? null : tr.stages });
      if (!tr.bench.running) { try { tr.bench.result = await invoke("bench_last", { model: tr.model, method: tr.method }); } catch (_) { tr.bench.result = null; } }
    }
  } catch (e) { tr.error = String(e); }
  if (page === "train") render();
}

function onBenchEvent(p) {
  const b = tr.bench;
  if (p.kind === "bench") { b.text = `Setting ${p.index + 1} of ${p.total}: ${p.name}`; b.progress = p.index / Math.max(1, p.total); if (p.row && b.result) { /* shown when the whole run ends */ } }
  else if (p.kind === "error") { b.running = false; b.text = ""; b.progress = null; b.error = p.text; }
  else if (p.kind === "done") { b.running = false; b.text = ""; b.progress = null; b.result = p.data; b.used = null; }
  if (page === "train") render();
}

function onServerEvent(p) {
  if (p.kind === "done") tr.srvCheck = { ok: true, text: p.data.gpu ? `Connected. ${p.data.gpu.name}, ${p.data.gpu.vram_gb} GB, ${p.data.busy ? "busy" : "free"}.` : "Connected, but the server found no graphics card." };
  else tr.srvCheck = { ok: false, text: p.text };
  trRefresh();
}

function onTrainEvent(p) {
  if (p.kind === "metric") { tr.metrics.push(p); tr.progress = p.step / Math.max(1, p.total_steps); tr.stepText = `${p.stage_count > 1 ? (p.stage === "world_book" ? "World book · " : "Practice · ") : ""}Step ${p.step} of ${p.total_steps}`; }
  else if (p.kind === "sample") tr.samples.push(p);
  else if (p.kind === "speed") tr.speed = p;
  else if (p.kind === "stage") { tr.stepText = p.text; tr.progress = null; }
  else if (p.kind === "start" && p.stage_count > 1) tr.stepText = `Stage ${p.stage_index + 1} of ${p.stage_count}: ${p.stage === "world_book" ? "reading the world book" : "practising answers"}`;
  else if (p.kind === "progress") { if (p.step !== "train" || tr.progress == null) { tr.stepText = p.text || p.step; tr.progress = p.value; } }
  else if (p.kind === "log") tr.log.push(p.text);
  else if (p.kind === "done") { tr.running = false; tr.stopping = false; tr.notice = p.data.stopped ? `Stopped and saved at step ${p.data.steps}. Press "Continue" to carry on from here, now or after closing the program.` : `Finished: ${p.data.steps} steps, final loss ${p.data.final_loss != null ? (+p.data.final_loss).toFixed(3) : "–"}.`; trRefresh(); }
  else if (p.kind === "error") { tr.running = false; tr.stopping = false; tr.error = p.text; trRefresh(); }
  if (page === "train") { render(); const l = document.querySelector("details .log"); if (l) l.scrollTop = l.scrollHeight; }
}

// the warning before the computer is switched off: a bar on every page with a Cancel button
function onPowerEvent(p) {
  let bar = document.getElementById("powerbar");
  const gone = () => { if (bar) bar.remove(); };
  if (p.kind === "armed") { tr.shutdownAfter = p.on; if (page === "train") render(); return; }
  if (p.kind === "cancelled") { gone(); tr.notice = "Shutdown cancelled. The computer stays on."; if (page === "train") render(); return; }
  if (p.kind === "failed") { gone(); tr.error = p.text; if (page === "train") render(); return; }
  if (!bar) { bar = document.createElement("div"); bar.id = "powerbar"; document.body.appendChild(bar); }
  if (p.kind === "shutting_down") { bar.textContent = "Shutting down now…"; return; }
  bar.innerHTML = `<span>Training finished. This computer shuts down in <strong>${p.left}</strong> s.</span> <button id="pw-cancel">Cancel shutdown</button>`;
  document.getElementById("pw-cancel").onclick = () => invoke("cancel_shutdown");
}

async function trStart(resume) {
  tr.metrics = []; tr.samples = []; tr.log = []; tr.speed = null; tr.progress = null; tr.stepText = "Starting…"; tr.error = null; tr.notice = null; tr.running = true; tr.stopping = false; render();
  const over = {};
  Object.entries(tr.over).forEach(([k, v]) => { if (v !== "" && v != null) over[k] = typeof v === "boolean" || ["engine", "bucket", "compile", "liger", "padding_free", "attention"].includes(k) ? v : Number(v); });
  if (tr.preset !== "quick") over.stages = tr.stages;
  if (tr.method === "full" && tr.tryAnyway) over.try_anyway = true;
  try { await invoke("train_start", { model: tr.model, method: tr.method, preset: tr.preset, overrides: over, resume: resume || null, server: tr.server || null }); }
  catch (e) { tr.running = false; tr.error = String(e); render(); }
}

function wireTrain() {
  if (page !== "train" || !tr.state) return;
  const on = (id, fn) => { const e = $(id); if (e) e.onclick = fn; };
  const newOnly = $("tr-newonly"); if (newOnly) newOnly.onchange = () => { tr.over.only_new_layers = newOnly.checked; };
  on("hf-open", () => { tr.hf.open = true; tr.hf.error = null; render(); const i = $("hf-input"); if (i) i.focus(); });
  on("hf-cancel", () => { tr.hf = { open: false, input: "", busy: false, found: null, error: null }; render(); });
  const hfLook = async () => {
    const input = $("hf-input").value; tr.hf.input = input; tr.hf.busy = true; tr.hf.found = null; tr.hf.error = null; render();
    try { tr.hf.found = await invoke("model_lookup", { input }); } catch (e) { tr.hf.error = String(e); }
    tr.hf.busy = false; render();
  };
  on("hf-look", hfLook);
  on("hf-folder", async () => { const dir = await tauri.dialog.open({ directory: true, title: "Choose the model's folder" }); if (dir) { tr.hf.input = dir; render(); hfLook(); } });
  const hi = $("hf-input"); if (hi) hi.onkeydown = (e) => { if (e.key === "Enter") hfLook(); };
  on("hf-add", async () => {
    try { const id = await invoke("model_add", { input: tr.hf.input }); tr.model = id; tr.hf = { open: false, input: "", busy: false, found: null, error: null }; tr.over = {}; await trRefresh(); }
    catch (e) { tr.hf.error = String(e); render(); }
  });
  document.querySelectorAll("[data-hfremove]").forEach((b) => b.onclick = async () => {
    if (!confirm("Remove this model from the list? (Runs already made with it stay on disk.)")) return;
    try { await invoke("model_remove", { id: b.dataset.hfremove }); if (tr.model === b.dataset.hfremove) tr.model = null; await trRefresh(); } catch (e) { tr.hf.error = String(e); render(); }
  });
  document.querySelectorAll("[data-model]").forEach((b) => b.onclick = () => { tr.model = b.dataset.model; tr.over = {}; trRefresh(); });
  const srvCheck = () => (tr.server === "unsloth" ? invoke("unsloth_check", { url: (tr.state.unsloth || {}).url || "", key: "", find: false }) : invoke("train_server_check", { url: tr.server }))
    .catch((e) => { tr.srvCheck = { ok: false, text: String(e) }; render(); });
  const wh = $("tr-where"); if (wh) wh.onchange = () => { tr.server = wh.value; tr.srvCheck = null; tr.over = {}; trRefresh(); const known = tr.server === "unsloth" ? tr.state.unsloth : tr.state.servers.find((x) => x.url === tr.server); if (tr.server && !(known || {}).status) srvCheck(); };
  on("srv-test", () => { tr.srvCheck = null; srvCheck(); });
  on("srv-remove", async () => { tr.state = await invoke("train_server_remove", { url: tr.server }); tr.server = ""; tr.srvCheck = null; trRefresh(); });
  on("srv-add", () => { tr.adding = true; render(); });
  on("s-cancel", () => { tr.adding = false; render(); });
  on("s-save", async () => {
    try { tr.state = await invoke("train_server_add", { name: $("s-name").value, url: $("s-url").value, token: $("s-token").value }); tr.server = tr.state.servers.at(-1).url; tr.adding = false; tr.srvCheck = null; await trRefresh();
      invoke("train_server_check", { url: tr.server }).catch(() => {}); }
    catch (e) { tr.srvCheck = { ok: false, text: String(e) }; render(); }
  });
  document.querySelectorAll("[data-type]").forEach((b) => b.onclick = () => { tr.method = b.dataset.type; tr.over = {}; trRefresh(); });
  document.querySelectorAll('input[name="tr-stages"]').forEach((r) => r.onchange = () => { tr.stages = r.value; tr.over = {}; trRefresh(); });
  const any = $("tr-anyway"); if (any) any.onchange = () => { tr.tryAnyway = any.checked; trRefresh(); };
  const pr = $("tr-preset"); if (pr) pr.onchange = () => { tr.preset = pr.value; tr.over = {}; trRefresh(); };
  document.querySelectorAll(".ov").forEach((i) => i.onchange = () => { tr.over[i.dataset.key] = i.value; });
  const bd = $("bn-deep"); if (bd) bd.onchange = () => { tr.bench.deep = bd.checked; };
  on("bn-start", async () => {
    const over = {}; Object.entries(tr.over).forEach(([k, v]) => { if (v !== "" && v != null) over[k] = ["engine", "bucket", "compile", "liger", "padding_free", "attention"].includes(k) ? v : Number(v); });
    tr.bench.running = true; tr.bench.error = null; tr.bench.text = "Starting…"; tr.bench.progress = 0; render();
    try { await invoke("bench_start", { model: tr.model, method: tr.method, preset: tr.preset, overrides: over, deep: tr.bench.deep }); } catch (e) { tr.bench.running = false; tr.bench.error = String(e); render(); }
  });
  on("bn-stop", () => invoke("env_cancel"));
  on("bn-use", () => {
    const r = tr.bench.result, best = r && r.rows.find((x) => x.id === r.best);
    if (!best) return;
    Object.assign(tr.over, best.overrides);
    tr.bench.used = best.name; render();
  });
  document.querySelectorAll(".ovs").forEach((i) => i.onchange = () => { tr.over[i.dataset.key] = i.value; });
  on("tr-start", () => trStart(null));
  const aw = $("tr-awake"); if (aw) aw.onchange = async () => { try { setState(await invoke("set_setting", { key: "keep_awake", value: aw.checked })); } catch (e) { tr.error = String(e); render(); } };
  const sd = $("tr-shutdown"); if (sd) sd.onchange = async () => {
    tr.shutdownAfter = sd.checked;
    try { await invoke("set_shutdown_after", { on: sd.checked }); } catch (e) { tr.shutdownAfter = false; tr.error = String(e); render(); }
  };
  on("tr-resume", () => trStart($("tr-resume").dataset.run));
  on("tr-stop", () => { tr.stopping = true; tr.stepText = "Stopping: finishing this step and saving a checkpoint…"; tr.progress = null; invoke("train_stop"); render(); });
  document.querySelectorAll("[data-continue]").forEach((b) => b.onclick = () => trStart(b.dataset.continue));
  document.querySelectorAll("[data-showrun]").forEach((b) => b.onclick = async () => {
    const ev = await invoke("train_metrics", { run: b.dataset.showrun });
    tr.metrics = ev.filter((e) => e.type === "metric").map((e) => ({ ...e, kind: "metric" }));
    tr.samples = ev.filter((e) => e.type === "sample");
    tr.notice = `Run ${b.dataset.showrun}`; tr.error = null; tr.progress = null; tr.stepText = ""; render();
  });
}

// ------------------------------------------------------------------------------------------------ the general-text chooser (Heal card and importance-matrix box)
// One piece of code for both places: `scope` is "heal" or "imatrix". The program's list (presets + the person's own datasets) comes from the core
// (`general_state`); every change that is saved comes back as the new state.
let gc = { src: null, state: null, scopes: {}, tokenText: "" };
const gcKey = (s) => (s.file || s.id || "") + (s.config ? "#" + s.config : "");
function gcUse(st) { if (st && st !== gc.src) { gc.src = st; gc.state = st; } return gc.state; }
function gcScope(scope) { return gc.scopes[scope] || (gc.scopes[scope] = { sel: null, own: { mode: "hub", id: "", file: "", config: "", split: "", column: "" }, check: null, busy: false, error: null, project: null }); }
function gcEntries() { const st = gc.state || { presets: [], custom: [] }; return [...st.presets.map((p) => ({ ...p, preset: true })), ...st.custom.map((c) => ({ ...c, key: gcKey(c), kind: c.kind || "", custom: true }))]; }
// what is selected: this session's choice, else the project's last choice, else the default
function gcSel(scope) {
  const sc = gcScope(scope), st = gc.state;
  if (!st) return "";
  const project = state && state.project ? state.project.folder : "";
  if (sc.sel === null || sc.project !== project) {
    sc.project = project;
    const last = (st.choice || {})[scope];
    const hit = last && gcEntries().find((e) => e.key === gcKey(last));
    sc.sel = hit ? hit.key : last && (last.id || last.file) ? "own" : st.default;
    if (sc.sel === "own") sc.own = { mode: last.file ? "file" : "hub", id: last.id || "", file: last.file || "", config: last.config || "", split: last.split || "", column: last.column || "" };
  }
  return sc.sel;
}
function gcEntry(scope) { const k = gcSel(scope); return gcEntries().find((e) => e.key === k) || null; }
// the source the job gets (the core fills the config and split of a preset)
function gcSource(scope) {
  const sc = gcScope(scope), e = gcEntry(scope), nz = (v) => (v && String(v).trim() ? String(v).trim() : null);
  if (gcSel(scope) === "own") { const o = sc.own; return o.mode === "file" ? { file: nz(o.file), column: nz(o.column) } : { id: nz(o.id), config: nz(o.config), split: nz(o.split), column: nz(o.column) }; }
  if (!e) return null;
  return e.file ? { file: e.file, column: e.column || null } : { id: e.id, config: e.config || null, split: e.split || null, column: e.column || null };
}
// why the chosen dataset cannot be used yet ("" = it can)
function gcBlocked(scope) {
  const e = gcEntry(scope), st = gc.state;
  if (e && e.needs_ack && !(st.acknowledged || []).includes(e.key)) return "Tick the box that says you read the licence note, or choose another dataset.";
  if (gcSel(scope) === "own") { const o = gcScope(scope).own; if (!(o.mode === "file" ? o.file.trim() : o.id.trim())) return "Fill in the dataset or the file of \"My own...\", or choose another dataset."; }
  return "";
}
const GC_VERDICT = { works: "ok", needs_access: "err", not_found: "err", no_column: "err", offline: "err", empty: "err", error: "err" };
function gcHtml(scope) {
  const st = gc.state;
  if (!st) return "";
  const sc = gcScope(scope), sel = gcSel(scope), e = gcEntry(scope), busy = sc.busy, id = (x) => `gc-${scope}-${x}`;
  const label = (x) => `${x.name || x.id} · ${x.license || "licence: yours to check"} · ${x.kind || "?"}${x.gated ? " · needs access + your token" : ""}`;
  const opts = gcEntries().map((x) => `<option value="${esc(x.key)}" ${sel === x.key ? "selected" : ""}>${esc(label(x))}</option>`).join("") + `<option value="own" ${sel === "own" ? "selected" : ""}>My own…</option>`;
  let info = "";
  if (e) {
    info = `<p class="dim small">${esc(e.note || "")}${e.custom ? ` Added by you (${esc(e.file || e.id)}); the licence is yours to check${sc.check && sc.check.license ? "; the Hub says: " + esc(sc.check.license) : ""}.` : ""}</p>
      <p class="dim small"><strong>Licence:</strong> ${esc(e.license || "yours to check")}${e.license_note ? " · " + esc(e.license_note) : ""}${e.url ? ` · <a href="${esc(e.url)}" target="_blank" rel="noopener">dataset page</a>` : ""}</p>`;
    if (e.needs_ack) info += `<label class="opt"><input type="checkbox" id="${id("ack")}" ${(st.acknowledged || []).includes(e.key) ? "checked" : ""}> I read this licence note and the dataset's LICENSE, and my use fits it.</label>`;
    if (e.gated) info += `<p class="warn small">This dataset needs access approval on Hugging Face. Ask for access on its page and add your Hugging Face token below; without both it cannot be downloaded.</p>`;
  }
  let own = "";
  if (sel === "own") {
    const o = sc.own;
    own = `<div class="grid"><label>Source <select id="${id("mode")}"><option value="hub" ${o.mode === "hub" ? "selected" : ""}>A Hugging Face dataset</option><option value="file" ${o.mode === "file" ? "selected" : ""}>A file on this computer</option></select></label>
      ${o.mode === "file" ? `<label>File (.jsonl .json .txt .csv .parquet)<input type="text" id="${id("file")}" value="${esc(o.file)}" placeholder="C:\\texts\\general.jsonl"></label>`
        : `<label>Dataset (owner/name)<input type="text" id="${id("id")}" value="${esc(o.id)}" placeholder="owner/name"></label><label>Config (optional)<input type="text" id="${id("config")}" value="${esc(o.config)}"></label><label>Split<input type="text" id="${id("split")}" value="${esc(o.split)}" placeholder="train"></label>`}
      <label>Column (optional, found by itself)<input type="text" id="${id("column")}" value="${esc(o.column)}" placeholder="text, messages, conversations…"></label></div>
      <p class="dim small">The licence of a dataset you add is yours to check.</p>`;
  }
  const c = sc.check;
  const res = sc.busy ? `<p class="dim small">Reading about 20 rows…</p>` : c ? `<div class="${GC_VERDICT[c.verdict] === "ok" ? "ok" : "err"} small" id="${id("result")}"><strong>${c.verdict === "works" ? "Works." : { needs_access: "Needs access.", not_found: "Not found.", no_column: "No usable column.", offline: "Offline.", empty: "No rows." }[c.verdict] || "Problem."}</strong> ${esc(c.message)}
      ${c.sample && c.sample.length ? `<pre class="small">${c.sample.map((s) => esc(s)).join("\n———\n")}</pre>` : ""}${c.license ? `<br>Licence on the Hub: ${esc(c.license)}${c.gated ? " (gated)" : ""}` : ""}</div>` : "";
  const token = `<details ${e && e.gated || (sc.check && sc.check.verdict === "needs_access") ? "open" : ""}><summary>Hugging Face token ${st.has_token ? "(saved)" : "(none)"}</summary>
      <div class="row"><input type="password" id="gc-token" placeholder="hf_…" value="${esc(gc.tokenText)}" autocomplete="off"> <button class="secondary" id="gc-token-save">Save token</button> ${st.has_token ? `<button class="link" id="gc-token-clear">Remove token</button>` : ""}</div>
      <p class="dim small">Only for gated datasets (<a href="https://huggingface.co/settings/tokens" target="_blank" rel="noopener">make a read token</a>). It is kept with your other keys on this computer and handed to the worker as an environment variable; it is never put in a job file or the log.</p></details>`;
  return `<div class="chooser" data-scope="${scope}"><div class="row"><label>General text <select id="${id("sel")}" ${busy ? "disabled" : ""}>${opts}</select></label>
      ${e && e.custom ? `<button class="link" id="${id("remove")}">Remove from my list</button>` : ""}</div>${info}${own}
    <div class="row"><button class="secondary" id="${id("check")}" ${busy ? "disabled" : ""}>Check it</button>${sel === "own" ? `<button class="secondary" id="${id("add")}">Add to my list</button>` : ""}</div>
    ${sc.error ? `<p class="err small">${esc(sc.error)}</p>` : ""}${res}${token}</div>`;
}
async function gcCall(scope, cmd, args) {
  const sc = gcScope(scope);
  try { const st = await invoke(cmd, args); if (st && st.presets) { gc.state = st; gc.src = st; } sc.error = null; return st; } catch (e) { sc.error = String(e); return null; }
}
function gcPersist(scope) { const s = gcSource(scope); if (s && gcSel(scope)) invoke("general_choose", { scope, source: s }).then((st) => { if (st && st.presets) { gc.state = st; gc.src = st; } }).catch(() => {}); }
function gcWire(scope, rerender) {
  const sc = gcScope(scope), id = (x) => `gc-${scope}-${x}`, on = (x, fn) => { const el = $(x); if (el) el.onclick = fn; };
  const sel = $(id("sel")); if (sel) sel.onchange = () => { sc.sel = sel.value; sc.check = null; sc.error = null; if (sel.value !== "own") gcPersist(scope); rerender(); };
  const ack = $(id("ack")); if (ack) ack.onchange = async () => { const e = gcEntry(scope); if (e && ack.checked) await gcCall(scope, "general_ack", { key: e.key }); rerender(); };
  [["mode", "mode"], ["id", "id"], ["file", "file"], ["config", "config"], ["split", "split"], ["column", "column"]].forEach(([x, k]) => { const el = $(id(x)); if (el) el.onchange = () => { sc.own[k] = el.value; sc.check = null; if (k === "mode") rerender(); else gcPersist(scope); }; });
  on(id("check"), async () => {
    const src = gcSource(scope);
    if (!src || !(src.id || src.file)) { sc.error = "Fill in the dataset or the file first."; rerender(); return; }
    sc.busy = true; sc.check = null; sc.error = null; rerender();
    try { sc.check = await invoke("general_check", { source: src }); } catch (e) { sc.error = String(e); }
    sc.busy = false; rerender();
  });
  on(id("add"), async () => {
    const src = gcSource(scope), st = await gcCall(scope, "general_add", { source: src });
    if (st && st.added) { sc.sel = st.added; sc.check = null; gcPersist(scope); }
    rerender();
  });
  on(id("remove"), async () => {
    const e = gcEntry(scope); if (!e) return;
    const st = await gcCall(scope, "general_remove", { key: e.key });
    if (st) { Object.values(gc.scopes).forEach((s) => { if (s.sel === e.key) s.sel = st.default; }); sc.check = null; gcPersist(scope); }
    rerender();
  });
  const tk = $("gc-token"); if (tk) tk.onchange = () => { gc.tokenText = tk.value; };
  on("gc-token-save", async () => { const el = $("gc-token"); const v = (el ? el.value : gc.tokenText).trim(); gc.tokenText = ""; if (el) el.value = ""; await gcCall(scope, "hf_token_set", { token: v }); rerender(); });
  on("gc-token-clear", async () => { await gcCall(scope, "hf_token_clear", {}); rerender(); });
}

// ------------------------------------------------------------------------------------------------ step 5: use in nd-world
let us = { state: null, run: "", what: "gguf", quant: "q4_k_m", quants: ["q4_k_m"], numCtx: 8192, im: false, imChunks: 128, imReuse: null, server: "", modelName: "", running: false, step: "", progress: null, log: [], error: null, ok: null, probe: null, check: null, adding: false };
const copyBtn = (text) => `<button class="link" data-copy="${esc(text)}">Copy</button>`;

// the importance-matrix box of the export page (GGUF exports only): off unless ticked; "Reuse the saved matrix" is on when a matrix of this run exists
function imatrixBox(st) {
  const im = st.imatrix;
  if (!im || us.what === "merged") return "";
  gcUse(im.general);
  const saved = (im.saved || {})[us.run];
  const reuse = us.imReuse === null ? !!saved : us.imReuse;
  const helps = us.quants.some((q) => !(im.no_matrix_for || []).includes(q));
  const est = ((im.estimates || {})[us.run] || {})[String(us.imChunks)];
  const names = { 64: "64 chunks (quick)", 128: "128 chunks (recommended)", 256: "256 chunks (thorough)" };
  const detail = us.im ? `<div class="row"><label>Calibration text <select id="us-imchunks">${im.chunks.map((n) => `<option value="${n}" ${us.imChunks === n ? "selected" : ""}>${names[n] || n + " chunks"}</option>`).join("")}</select></label>
      <span class="dim">${est ? `${est.tokens.toLocaleString("en")} tokens, about ${est.gpu_minutes} min on a graphics card (up to ${est.cpu_minutes} min on the processor): a rough guess, the log shows the real progress` : ""}</span></div>
    ${gcHtml("imatrix")}<p class="dim small">About ${Math.round((im.general_share || 0.2) * 100)} % of the calibration text is general text from the dataset above; the rest is your own data. A matrix is only reused for the same dataset.</p>
    ${saved ? `<label class="opt" title="${esc(saved.dir)}"><input type="checkbox" id="us-imreuse" ${reuse ? "checked" : ""}> Reuse the saved matrix (${saved.chunks} chunks, made ${esc(String(saved.created || "").slice(0, 10))}, if it still fits this model and data)</label>` : `<p class="dim">No saved matrix of this run yet: it will be made now and kept next to the files.</p>`}
    ${helps ? "" : `<p class="dim">None of the chosen sizes is below Q8_0, so there is nothing to improve: the matrix would not be made.</p>`}` : "";
  return `<div class="row imatrix"><label class="opt" title="llama.cpp reads about ${im.ctx * im.default_chunks} tokens of your data and some general text and learns which weights matter most; the small sizes then keep those more exactly"><input type="checkbox" id="us-im" ${us.im ? "checked" : ""}> Better small sizes (importance matrix)</label></div>
    <p class="dim">Makes Q6_K, Q5_K_M and Q4_K_M lose less quality (Q8_0 and F16 stay as they are). Needs llama.cpp with <code>llama-imatrix</code>; adds a few minutes. If anything about it fails, the files are made without it and the page says so.</p>${detail}`;
}

function renderUse() {
  if (!state.bundle) return `<h1>Step 5 · Use in nd-world</h1><p class="lead">Import your world data first (Step 1).</p>`;
  const st = us.state;
  if (!st) return `<h1>Step 5 · Use in nd-world</h1><p class="lead">Loading…</p>`;
  if (!st.runs.length) return `<h1>Step 5 · Use in nd-world</h1><p class="lead">Train a model first (Step 3). A run that was stopped can be exported too once it has saved its model.</p>`;
  if (!us.run) us.run = st.runs[0].name;
  if (!us.server && st.servers.length) us.server = st.servers[0].url;
  if (!us.modelName) us.modelName = st.suggested_name;
  const runOpts = st.runs.map((r) => `<option value="${esc(r.name)}" ${us.run === r.name ? "selected" : ""}>${esc(r.name)} · ${esc(r.model)} · ${esc(r.status)}${r.loss != null ? " · loss " + (+r.loss).toFixed(2) : ""}</option>`).join("");
  const busy = us.running;
  const files = `<div class="card wide"><h2>1 · Make the files</h2>
    <div class="row"><label>Run <select id="us-run">${runOpts}</select></label><span class="quants">File sizes: ${st.quants.map((q) => `<label class="opt" title="${esc(q.name)}"><input type="checkbox" class="us-q" value="${q.id}" ${us.quants.includes(q.id) ? "checked" : ""} ${us.what === "merged" ? "disabled" : ""}> ${esc(q.name.split(":")[0])}</label>`).join(" ")}</span>
      <label>Context <select id="us-ctx">${[4096, 8192, 16384, 32768].map((n) => `<option ${us.numCtx === n ? "selected" : ""}>${n}</option>`).join("")}</select></label></div>
    <div class="row"><label>Make <select id="us-what"><option value="gguf" ${us.what === "gguf" ? "selected" : ""}>GGUF + Modelfile (for Ollama)</option><option value="gguf_merged" ${us.what === "gguf_merged" ? "selected" : ""}>GGUF + the merged model (safetensors)</option><option value="merged" ${us.what === "merged" ? "selected" : ""}>Only the merged model (safetensors, no GGUF)</option></select></label></div>
    ${imatrixBox(st)}
    <p class="dim">Joins the trained add-on into the model. ${us.what === "merged" ? "Makes the <strong>merged 16-bit model</strong> as ordinary safetensors files (10 to 20 GB for the bigger models), which loads anywhere without the add-on." : "Makes a <strong>GGUF</strong> file with a <strong>Modelfile</strong> (a few minutes; each file is 1 to 6 GB). Tick several sizes to make them all from one conversion; each becomes its own entry below." + (us.what === "gguf_merged" ? " Also keeps the merged 16-bit model (10 to 20 GB more)." : "")}</p>
    <div class="row"><button class="primary" id="us-make" ${busy || !st.env_ready || (us.what !== "merged" && !us.quants.length) || (us.what !== "merged" && us.im && gcBlocked("imatrix")) ? "disabled" : ""}>Make the files</button><button class="secondary" id="us-lora" ${busy ? "disabled" : ""} title="Copies the small LoRA add-on, with a README on how to load it in Unsloth or PEFT or merge it yourself">Save the LoRA add-on…</button>${!st.env_ready ? ` <span class="err">Open "⚙ Training setup" first (not needed to copy the LoRA add-on).</span>` : ""}</div>${us.what !== "merged" && us.im && gcBlocked("imatrix") ? `<p class="warn small">${esc(gcBlocked("imatrix"))}</p>` : ""}</div>`;
  const ex = st.exports.length ? st.exports.map((e) => `<li><strong>${esc(e.name)}</strong> · ${e.gguf_file === null ? "merged model only" : e.size_mb != null ? e.size_mb + " MB" : "file missing"}${e.merged_exists ? ` · merged model kept` : ""}${e.has_vision ? ` · <span title="${esc(e.vision_note || "")}">vision file included</span> (pictures work in Unsloth Studio and llama.cpp; Ollama uses the text only, see VISION.md in the folder)` : (e.modalities && e.modalities.length ? ` · <span class="dim" title="${esc(e.vision_note || "")}">no vision file</span>` : "")}${e.imatrix && e.imatrix.requested ? (e.imatrix.applied ? ` · <span title="${esc(e.imatrix.note || "")}">importance matrix used (${e.imatrix.chunks} chunks${e.imatrix.reused ? ", saved one" : ""})</span>` : ` · <span class="dim" title="${esc(e.imatrix.reason || e.imatrix.note || "")}">no importance matrix${e.imatrix.reason ? " (" + esc(e.imatrix.reason) + ")" : ""}</span>`) : ""}${e.imatrix && e.imatrix.note ? `<br><span class="dim">Importance matrix: ${esc(e.imatrix.note)}</span>` : ""}<br><span class="dim">${esc(e.dir)}</span> ${copyBtn(e.dir)}
      ${e.installed && e.installed.length ? `<br><span class="dim">Installed: ${e.installed.map((i) => esc(i.model) + " @ " + esc(i.url)).join(", ")}</span>` : ""}</li>`).join("") : "";
  const list = ex ? `<div class="card wide"><h2>Your files</h2><ul class="exlist">${ex}</ul></div>` : "";
  const srvOpts = st.servers.map((x) => `<option value="${esc(x.url)}" ${us.server === x.url ? "selected" : ""}>${esc(x.name)} (${esc(x.url)})</option>`).join("");
  const exOpts = st.exports.filter((e) => e.gguf_file !== null).map((e) => `<option value="${esc(e.name)}" ${us.exp === e.name ? "selected" : ""}>${esc(e.name)}</option>`).join("");
  const ggufExports = st.exports.filter((e) => e.gguf_file !== null);
  if ((!us.exp || !ggufExports.some((e) => e.name === us.exp)) && ggufExports.length) us.exp = ggufExports[0].name;
  const install = ggufExports.length ? `<div class="card wide"><h2>2 · Put it into Ollama</h2>
    <div class="row"><label>Files <select id="us-exp">${exOpts}</select></label><label>Ollama server <select id="us-server">${srvOpts}</select></label>
      <button class="secondary" id="us-addsrv">Add a server…</button></div>
    ${us.adding ? `<div class="row"><input type="text" id="srv-name" placeholder="Name, e.g. TrueNAS"><input type="text" id="srv-url" placeholder="Address, e.g. http://truenas:11434"><button class="primary" id="srv-save">Add</button><button class="secondary" id="srv-cancel">Cancel</button></div>` : ""}
    <div class="row"><label>Model name in Ollama <input type="text" id="us-name" value="${esc(us.modelName)}" style="min-width:220px"></label></div>
    <div class="row"><button class="secondary" id="us-check" ${busy ? "disabled" : ""}>Is Ollama running?</button><button class="primary" id="us-install" ${busy ? "disabled" : ""}>Install into Ollama</button><button class="secondary" id="us-probe" ${busy ? "disabled" : ""}>Test it with a question</button></div>
    ${us.check ? `<div class="${us.check.ok ? "ok" : "err"}">${esc(us.check.text)}</div>` : ""}
    ${us.probe ? `<div class="sample"><div class="q">${esc(us.probe.question)}</div><div class="ans">${esc(us.probe.answer) || "<em>(no answer)</em>"}</div><div class="dim">One of the training's own questions. If the answer is empty or nonsense, something is wrong with how Ollama runs the model: tell me what you see.</div></div>` : ""}</div>` : "";
  const name = us.modelName || st.suggested_name;
  const env = `${name}`;
  const how = ggufExports.length ? `<div class="card wide"><h2>3 · Use it in nd-world</h2>
    <p><strong>nd-world with Ollama</strong> (the legacy AI back-end):</p>
    <ol><li>On the computer where nd-world runs, set <code>OLLAMA_MODEL=${esc(env)}</code> ${copyBtn("OLLAMA_MODEL=" + env)} in <code>.env</code> (or in the compose file), and <code>OLLAMA_URL</code> to the Ollama server you installed it into.</li>
    <li>Restart nd-world. Or, without restarting: AI → Models tab → pick <strong>${esc(env)}</strong> → ✓ Use.</li>
    <li>Keep nd-world's <strong>lore search</strong> switched on: the model was trained to use the context nd-world gives it.</li></ol>
    <p><strong>On your TrueNAS server:</strong> add the server above (its Ollama must listen on the network: <code>OLLAMA_HOST=0.0.0.0</code>), install the model there, then set <code>OLLAMA_MODEL</code> as above. The model uses the server's graphics card like any other model.</p>
    <p><strong>nd-world with Unsloth Studio</strong>: the GGUF file in the folder above can be loaded in Studio's Model Hub (local model); then set <code>UNSLOTH_MODEL</code> to it and <code>LLM_CONTEXT_TOKENS</code> to the context length it was loaded with (${us.numCtx}). The chat template inside the file answers without thinking, as the model was trained.</p>
    <div class="warn"><strong>For the game master only.</strong> This model was trained on <em>all</em> of your world's text, including GM secrets, and it will repeat them. If players can use AI in this world ("players can use Ask AI"), turn that off, or keep players on another model.</div></div>` : "";
  const prog = (us.running || us.step) ? `<div class="stepline">${esc(us.step)}</div>${us.progress != null ? `<progress max="1" value="${us.progress}"></progress>` : ""}${us.running && us.pending === "make" ? ` <button class="secondary" id="us-cancel" title="Stops the export; while the importance matrix is made, llama.cpp is stopped at once">Cancel</button>` : ""}` : "";
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
    else if (d.name && d.run) {
      const notes = [...new Set((d.exports || [d]).map((m) => m.imatrix && m.imatrix.note).filter(Boolean))];
      us.ok = `Files made: ${d.name}.${d.merged_dir ? " Merged model: " + d.merged_dir : ""}${notes.length ? " Importance matrix: " + notes.join(" ") : ""}`;
    }
    usRefresh(); setState_(); 
  } else if (p.kind === "error") {
    us.running = false; us.progress = null; us.step = "";
    if (us.pending === "check") us.check = { ok: false, text: p.text }; else us.error = p.text === "Cancelled." ? "Cancelled. The export was stopped; the files of this attempt may be incomplete." : p.text;
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
  bind("us-run", "run");
  document.querySelectorAll(".us-q").forEach((c) => c.onchange = () => { us.quants = [...document.querySelectorAll(".us-q")].filter((x) => x.checked).map((x) => x.value); render(); }); const wh = $("us-what"); if (wh) wh.onchange = () => { us.what = wh.value; render(); }; bind("us-ctx", "numCtx", true); bind("us-exp", "exp"); bind("us-server", "server");
  const imx = $("us-im"); if (imx) imx.onchange = () => { us.im = imx.checked; render(); };
  const imc = $("us-imchunks"); if (imc) imc.onchange = () => { us.imChunks = Number(imc.value); render(); };
  const imr = $("us-imreuse"); if (imr) imr.onchange = () => { us.imReuse = imr.checked; render(); };
  if (us.im && us.what !== "merged") gcWire("imatrix", render);
  on("us-cancel", () => invoke("env_cancel"));
  const nm = $("us-name"); if (nm) nm.onchange = () => { us.modelName = nm.value.trim(); render(); };
  document.querySelectorAll("[data-copy]").forEach((b) => b.onclick = async () => { try { await navigator.clipboard.writeText(b.dataset.copy); b.textContent = "Copied"; setTimeout(() => (b.textContent = "Copy"), 1500); } catch (_) { b.textContent = "Select and copy by hand"; } });
  on("us-make", () => usRun("make", "export_start", { run: us.run, quant: us.quant, numCtx: us.numCtx, what: us.what, quants: us.what === "merged" ? [us.quants[0] || "q4_k_m"] : us.quants,
    imatrix: us.what !== "merged" && us.im ? { enabled: true, chunks: us.imChunks, reuse: us.imReuse === null ? !!((us.state.imatrix.saved || {})[us.run]) : us.imReuse, general: gcSource("imatrix") } : null }));
  on("us-lora", async () => {
    const dest = await tauri.dialog.open({ directory: true, title: "Choose where to put the LoRA add-on" });
    if (!dest) return;
    us.error = null; us.ok = null;
    try { us.ok = "LoRA add-on saved to " + await invoke("copy_adapter", { run: us.run, dest }) + " (the README there says how to load or merge it)."; } catch (e) { us.error = String(e); }
    render();
  });
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
      tr.metrics = []; tr.samples = []; tr.log = []; tr.speed = null; tr.progress = null; tr.stepText = "Making the new training data…"; tr.error = null; tr.notice = null; tr.running = true;
      await invoke("retrain_start", { fromPrevious: rt.fromPrev, exportAfter: rt.exportAfter, installUrl: url, ollamaName: null });
      rt.confirm = false; rt.info = null; go("train");
    } catch (e) { tr.running = false; rt.error = String(e); render(); }
  });
}

// ------------------------------------------------------------------------------------------------ Unsloth Desktop (Training setup)
let un = { url: "", key: "", name: "Unsloth Desktop", busy: false, result: null, error: null, ack: false, init: false };

// an address that is not this computer, not a private network and not https: the data would cross the internet unencrypted
function unRisk(url) {
  const u = (url || "").trim();
  if (!u || /^https:/i.test(u)) return false;
  const rest = u.replace(/^http:\/\//i, "");
  const host = (rest.startsWith("[") ? rest.slice(0, rest.indexOf("]") + 1) : rest.split(/[\/:]/)[0]).toLowerCase();
  if (host === "localhost" || host === "[::1]" || /^127\./.test(host) || /^10\./.test(host) || /^192\.168\./.test(host) || /^172\.(1[6-9]|2\d|3[01])\./.test(host) || /^169\.254\./.test(host)) return false;
  return host.includes(".") || host.startsWith("[");
}

function renderUnsloth() {
  const saved = state.settings && state.settings.unsloth;
  if (!un.init) { un.init = true; if (saved) { un.url = saved.url; un.name = saved.name; } }
  const risk = unRisk(un.url);
  return `<div class="card wide" id="un-card"><h2>Unsloth Desktop (optional)</h2>
    <p class="dim">Lets you choose <strong>Unsloth Desktop</strong> under "Where to train" in Step 3: Lore Forge sends the training data to Unsloth and follows the training there, instead of using its own training software. Unsloth Desktop must be open while it trains. Create an API key in Unsloth: your picture, Settings, API, a name, Create (Unsloth shows the key only once).</p>
    ${saved ? `<div class="ok">Saved: ${esc(saved.name)} at ${esc(saved.url)}. The API key is kept on this computer.</div>` : ""}
    <div class="row"><input type="text" id="un-url" value="${esc(un.url)}" placeholder="Address, e.g. http://127.0.0.1:8888" style="min-width:260px" ${un.busy ? "disabled" : ""}><button class="secondary" id="un-find" ${un.busy ? "disabled" : ""}>Find</button>
      <input type="password" id="un-key" value="${esc(un.key)}" placeholder="${saved ? "API key (leave empty to keep the saved one)" : "API key (sk-unsloth-…)"}" style="min-width:260px" ${un.busy ? "disabled" : ""}><button class="secondary" id="un-test" ${un.busy ? "disabled" : ""}>Test</button></div>
    ${risk ? `<label class="opt"><input type="checkbox" id="un-ack" ${un.ack ? "checked" : ""}> This address is on the internet and not encrypted: the training data (with the GM's secrets) would cross it in plain text. I understand and want to use it anyway.</label>` : ""}
    <div class="row"><button class="primary" id="un-save" ${un.busy || (!un.key.trim() && !saved) ? "disabled" : ""}>Save</button>${saved ? `<button class="secondary" id="un-remove" ${un.busy ? "disabled" : ""}>Remove</button>` : ""}</div>
    ${un.busy ? `<div class="stepline">Asking Unsloth…</div>` : ""}${un.result ? `<div class="ok">${esc(un.result)}</div>` : ""}${un.error ? `<div class="err">${esc(un.error)}</div>` : ""}</div>`;
}

function wireUnsloth() {
  const on = (id, fn) => { const e = $(id); if (e) e.onclick = fn; };
  const bind = (id, k) => { const e = $(id); if (e) e.oninput = () => { un[k] = e.value; if (k === "url") { const a = $("un-ack"); if (!unRisk(un.url) && a) un.ack = false; } }; };
  bind("un-url", "url"); bind("un-key", "key");
  const ack = $("un-ack"); if (ack) ack.onchange = () => { un.ack = ack.checked; };
  const ask = (find) => { un.busy = true; un.result = null; un.error = null; render(); invoke("unsloth_check", { url: un.url, key: un.key, find }).catch((e) => { un.busy = false; un.error = String(e); render(); }); };
  on("un-find", () => ask(true));
  on("un-test", () => ask(false));
  on("un-save", async () => {
    un.error = null; un.result = null;
    if (unRisk(un.url) && !un.ack) { un.error = "Tick the box about the unencrypted address first, or use an address on this computer or your own network."; render(); return; }
    try { setState(await invoke("unsloth_set", { name: un.name, url: un.url, key: un.key })); un.key = ""; const s = state.settings.unsloth; if (s) un.url = s.url; un.result = "Saved. Choose \"Unsloth Desktop\" under \"Where to train\" in Step 3."; }
    catch (e) { un.error = String(e); }
    render();
  });
  on("un-remove", async () => {
    if (!confirm("Forget the address and the API key of Unsloth?")) return;
    try { setState(await invoke("unsloth_remove")); un.url = ""; un.key = ""; un.result = null; un.init = false; } catch (e) { un.error = String(e); }
    render();
  });
}

function onUnslothEvent(p) {
  if (page === "train" && tr.server === "unsloth") {
    tr.srvCheck = p.kind === "done" ? { ok: true, text: p.data.summary } : { ok: false, text: p.text };
    trRefresh();
    return;
  }
  un.busy = false;
  if (p.kind === "done") { un.result = p.data.summary; if (p.data.url) un.url = p.data.url; un.error = null; }
  else { un.error = p.text; un.result = null; }
  if (page === "setup") render();
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
    <label class="opt"><input type="checkbox" id="no-unsloth" ${env.noUnsloth ? "checked" : ""} ${busy ? "disabled" : ""}> <strong>Install without Unsloth</strong> — only the standard libraries (Transformers, PEFT, TRL). Simpler and it works everywhere, but training is slower and uses more graphics memory.${st && st.engine ? ` <span class="dim">Installed now: ${st.engine === "standard" ? "without Unsloth" : "with Unsloth"}.</span>` : ""}</label>
    <div class="row">
      <button class="secondary" id="recheck" ${busy ? "disabled" : ""}>Check again</button>
      <button class="primary" id="install" ${busy || !st || !st.can_install ? "disabled" : ""}>Install the training software</button>
      <button class="primary" id="selftest" ${busy || !st || !st.env_installed ? "disabled" : ""}>Test the graphics card</button>
      <button class="secondary" id="cancelenv" ${busy ? "" : "disabled"}>Cancel</button>
    </div>
    ${env.step ? `<div class="stepline">${esc(env.step)}</div>` : ""}
    ${env.progress != null ? `<progress max="1" value="${env.progress}"></progress>` : ""}
    ${env.error ? `<div class="err">${esc(env.error)}</div>` : ""}${result}
    ${env.log.length ? `<pre class="log" id="log">${log}</pre>` : ""}
    ${renderUnsloth()}`;
}

async function refreshEnv() {
  env.status = await invoke("env_status");
  if (page === "setup") render();
}

async function startEnv(cmd, args) {
  env.running = true; env.log = []; env.step = ""; env.progress = null; env.result = null; env.error = null;
  render();
  try { await invoke(cmd, args); } catch (e) { env.running = false; env.error = String(e); render(); }
}

function onEnvEvent(p) {
  if (p.kind === "log") env.log.push(p.text);
  else if (p.kind === "step") { env.step = `Step ${p.index} of ${p.total}: ${p.text}`; env.progress = (p.index - 1) / p.total; }
  else if (p.kind === "progress") { env.step = p.text || p.step; env.progress = p.value; }
  else if (p.kind === "done") { env.running = false; env.progress = 1; env.step = "Finished."; if (p.data && p.data.device_name) env.result = p.data; refreshEnv(); }
  else if (p.kind === "error") { env.running = false; env.progress = null; env.error = p.text; refreshEnv(); }
  if (page === "setup") { render(); const l = $("log"); if (l) l.scrollTop = l.scrollHeight; }
}

// ------------------------------------------------------------------------------------------------------------ step 4: test
let te = { against: "base", state: null, run: "", result: null, running: false, text: "", progress: null, judge: false, chat: [], asking: false, system: "", showQ: "", error: null, ok: null };
const CATS = [["lore", "Lore questions", "higher is better"], ["rules", "Rules questions", "higher is better"], ["made_up", "Invents answers about things that do not exist", "lower is better"], ["general", "General chat sensible", "higher is better"]];
const pc = (x) => x == null ? "–" : Math.round(x * 100) + " %";

function renderTest() {
  if (!state.bundle) return `<h1>Step 4 · Test</h1><p class="lead">Import your world data first (Step 1).</p>`;
  const st = te.state;
  if (!st) return `<h1>Step 4 · Test</h1><p class="lead">Loading…</p>`;
  const head = `<h1>Step 4 · Test</h1><p class="lead">Checks that the trained model knows your world better than the original. The help panel explains the scores.</p>${te.error ? `<div class="err">${esc(te.error)}</div>` : ""}${te.ok ? `<div class="ok">${esc(te.ok)}</div>` : ""}`;
  if (!st.runs.length) return head + `<div class="card wide"><h2>Nothing to test yet</h2><p>Train a model in Step 3 first. A run that was stopped before its first save cannot be tested.</p></div>`;
  const opts = st.runs.map((r) => `<option value="${esc(r.name)}" ${r.name === te.run ? "selected" : ""}>${esc(r.name)} · ${esc(r.method || "")}${r.tested ? " · tested" : ""}</option>`).join("");
  const busy = te.running || te.asking;
  const cur = st.runs.find((r) => r.name === te.run);
  const origin = cur && cur.reshaped_from;
  if (!origin) te.against = "base";
  const against = origin ? `<label>Compare with <select id="te-against"><option value="base" ${te.against === "base" ? "selected" : ""}>the slimmed / grown model it was trained from</option><option value="origin" ${te.against === "origin" ? "selected" : ""}>the original: ${esc(String(origin).split("/").pop())}</option></select></label>` : "";
  const pick = `<div class="card wide"><h2>Choose a run</h2><div class="row"><select id="te-run">${opts}</select>${against}
      <button class="primary" id="te-go" ${busy || !st.env_ready ? "disabled" : ""}>Test this run</button>${busy ? `<button class="secondary" id="te-stop">Stop</button>` : ""}</div>
      <label class="opt"><input type="checkbox" id="te-judge" ${te.judge ? "checked" : ""} ${st.teacher_ready ? "" : "disabled"}> Grade lore and rules answers with the teacher model${st.teacher_ready ? ` (${esc(st.teacher_model)})` : " (choose a teacher in Step 2 first)"} — more accurate than counting matching words</label>
      ${!st.env_ready ? `<div class="err">The training software is not installed yet (Training setup).</div>` : ""}
      ${te.running || te.text ? `<div class="stepline">${esc(te.text)}</div>${te.progress != null ? `<progress max="1" value="${te.progress}"></progress>` : ""}` : ""}</div>`;
  let result = "";
  const r = te.result;
  if (r) {
    const row = ([id, name, dir]) => { const o = r.scores.original[id], t = r.scores.trained[id]; return o || t ? `<tr><td>${name} <span class="dim">(${dir})</span></td><td>${(o || t).n}</td><td>${pc(o && o.mean)}</td><td><strong>${pc(t && t.mean)}</strong></td></tr>` : ""; };
    const qs = r.questions.filter((q) => !te.showQ || q.category === te.showQ).map((q) => `<div class="ex"><div class="q"><span class="tag">${esc(q.category)}</span> ${esc(q.question)}</div>
      ${q.reference ? `<div class="dim small">Expected: ${esc(q.reference.slice(0, 400))}</div>` : ""}
      <div class="two"><div><div class="dim small">Original · ${pc(q.score_original)}</div><div class="ans">${esc(q.original)}</div></div><div><div class="dim small">Trained · ${pc(q.score_trained)}</div><div class="ans">${esc(q.trained)}</div></div></div></div>`).join("");
    result = `<div class="card wide"><h2>Result for ${esc(r.run)}</h2>
      <p class="verdict ${r.verdict.better === true ? "ok" : r.verdict.better === false ? "err" : ""}"><strong>${esc(r.verdict.headline)}</strong></p>
      ${r.verdict.lines.map((l) => `<p>${esc(l)}</p>`).join("")}
      <table class="tbl"><tr><th>What was checked</th><th>Questions</th><th>Original</th><th>Trained</th></tr>${CATS.map(row).join("")}</table>
      <p class="dim small">${r.against === "origin" ? "Compared with the original model the base was made from. " : ""}${r.loaded && (r.loaded.original === "4-bit" || r.loaded.trained === "4-bit") ? "A big model was loaded in 4-bit to fit the card (original: " + esc(r.loaded.original) + ", trained: " + esc(r.loaded.trained) + "). " : ""}${r.n} questions, graded by ${r.method === "teacher" ? "the teacher model" : "matching words with the expected answer (a rough guide: read the answers below)"} · data version ${r.dataset_version} · ${esc(r.created)}</p>
      <details><summary>Read the answers</summary><div class="row"><select id="te-cat"><option value="">All</option>${CATS.map((c) => `<option value="${c[0]}" ${te.showQ === c[0] ? "selected" : ""}>${c[1]}</option>`).join("")}</select></div>${qs}</details></div>`;
  }
  const chat = `<div class="card wide"><h2>Ask your own question</h2><p>Both models answer the same question. Loading the models takes a moment each time.</p>
      <details><summary>The instruction given to the model</summary><textarea id="te-sys" rows="4">${esc(te.system || st.default_system)}</textarea></details>
      <div class="row"><input type="text" id="te-q" placeholder="For example: Who is the leader of the Greyfather Pack?" style="min-width:300px" ${busy ? "disabled" : ""}><button class="primary" id="te-ask" ${busy || !st.env_ready ? "disabled" : ""}>Ask both</button></div>
      ${te.asking ? `<div class="stepline">${esc(te.text)}</div>` : ""}
      ${te.chat.map((c) => `<div class="ex"><div class="q">${esc(c.question)}</div><div class="two"><div><div class="dim small">Original</div><div class="ans">${esc(c.original)}</div></div><div><div class="dim small">Trained</div><div class="ans">${esc(c.trained)}</div></div></div></div>`).join("")}
      ${te.chat.length ? `<div class="row"><button class="secondary" id="te-save">Save this chat as a file</button></div>` : ""}</div>`;
  const cmp = st.compare.length ? `<div class="card wide"><h2>Compare runs</h2><table class="tbl"><tr><th>Run</th><th>Model</th><th>Type</th><th>Lore</th><th>Rules</th><th>Invents</th><th>General</th></tr>${st.compare.map((c, i) => `<tr><td>${i === 0 ? "★ " : ""}${esc(c.name)}</td><td>${esc(String(c.model).split("/").pop())}</td><td>${esc(c.method)}${c.stages ? ", " + esc(c.stages) : ""}</td><td>${pc(c.lore)} <span class="dim">(was ${pc(c.lore_original)})</span></td><td>${pc(c.rules)}</td><td>${pc(c.invented)}</td><td>${pc(c.general)}</td></tr>`).join("")}</table>
      <p class="dim small">★ = best lore score. Runs graded by the teacher and runs graded by matching words are not comparable with each other.</p></div>` : "";
  return head + pick + result + chat + cmp;
}

async function teCall(cmd, args) { te.error = null; try { return await invoke(cmd, args); } catch (e) { te.error = String(e); return null; } }

async function teRefresh() {
  const st = await teCall("test_state");
  if (st) {
    te.state = st;
    if (!te.run || !st.runs.some((r) => r.name === te.run)) te.run = st.runs.length ? st.runs[0].name : "";
    te.result = te.run ? await teCall("test_result", { run: te.run }) : null;
  }
  if (page === "test") render();
}

function onTestEvent(p) {
  if (p.kind === "progress") { te.text = p.text || p.step; te.progress = p.value; }
  else if (p.kind === "error") { te.running = false; te.asking = false; te.text = ""; te.progress = null; te.error = p.text; }
  else if (p.kind === "done") {
    const was = te.running; te.running = false; te.asking = false; te.text = ""; te.progress = null;
    if (p.data && p.data.chat) te.chat.push({ question: p.data.question, original: p.data.chat.original, trained: p.data.chat.trained });
    else if (was) te.ok = "Finished. The result is below.";
    setState_(); teRefresh(); return;
  }
  if (page === "test") render();
}

function wireTest() {
  if (page !== "test" || !te.state) return;
  const on = (id, fn) => { const e = $(id); if (e) e.onclick = fn; };
  const sel = $("te-run"); if (sel) sel.onchange = () => { te.run = sel.value; te.chat = []; te.ok = null; te.against = "base"; teRefresh(); };
  const ag = $("te-against"); if (ag) ag.onchange = () => { te.against = ag.value; te.chat = []; };
  const j = $("te-judge"); if (j) j.onchange = () => { te.judge = j.checked; };
  const cat = $("te-cat"); if (cat) cat.onchange = () => { te.showQ = cat.value; render(); };
  on("te-go", async () => { te.running = true; te.ok = null; te.error = null; te.text = "Starting…"; te.progress = null; render(); try { await invoke("test_start", { run: te.run, judge: te.judge, against: te.against }); } catch (e) { te.running = false; te.error = String(e); render(); } });
  on("te-stop", () => invoke("env_cancel"));
  on("te-ask", async () => {
    const q = $("te-q").value; te.system = $("te-sys").value;
    te.asking = true; te.error = null; te.text = "Loading the models…"; render();
    try { await invoke("test_ask", { run: te.run, system: te.system === te.state.default_system ? null : te.system, question: q, against: te.against }); } catch (e) { te.asking = false; te.error = String(e); render(); }
  });
  const q = $("te-q"); if (q) q.onkeydown = (e) => { if (e.key === "Enter") $("te-ask").click(); };
  on("te-save", async () => { const f = await teCall("test_save_transcript", { run: te.run, items: te.chat }); if (f) te.ok = "Saved: " + f; render(); });
}

// ------------------------------------------------------------------------------------------ slim or grow a model
// the healing settings that are "automatic" until the person changes them (docs/HEAL_SETTINGS.md); null = what the program chooses for this model and card
const HEAL_AUTO = { rank: null, seq: null, batch: null, per_update: null, checkpointing: "auto", method: null, optimizer: null, kl_chunk: null, k: null, teacher_load: null, warmup: null, kbit_upcast: null };
let rs = { state: null, model: "", mode: "prune", options: null, optErr: null, loadingOpts: false, token: 0, blocks: null, init: "identity", samples: 128, strategy: "contiguous", pf: 1, pl: 1, load: "auto",
  dropText: "", running: false, testing: null, text: "", progress: null, log: [], error: null, ok: null, view: null,
  // thinner layers (width): the way ("depth" | "width" | "both"), the chosen MLP width, the Expert axes, measuring, and healing
  shape: "depth", wi: null, lv: false, hh: null, wsamples: 256, meas: null, measModel: null,
  // "Try first": what to try, the clicked blocks, the plan (how many cuts, how long), the kept answers and a run in progress
  tr: { what: "blocks", quick: null, sel: [], plan: null, planErr: null, planFor: null, planTok: 0, resFor: null, analysis: null, sps: null, running: false, text: "", progress: null, done: 0, total: 0 },
  hl: null, hlRun: null, hlSpec: { general: true, tokens_m: null, dataset: "", share: 0.5, alpha: 1, lr: 0.0001, mode: "", ...HEAL_AUTO }, hlEst: null };

// Thinning the linear-attention value heads and the attention heads has no published precedent for this kind of model. It is offered in Expert mode only,
// labelled experimental, and stays switched off until the measurements on a real graphics card (docs/RESHAPE.md, "Not verified yet") are done:
// then this is set to true.
const EXPERIMENTAL_AXES_PASSED = false;

// "36–51" from [36, 37, ..., 51]; "3, 7–9" for gaps
function ranges(list) {
  const out = []; let a = null, b = null;
  [...(list || [])].sort((x, y) => x - y).forEach((n) => { if (a === null) { a = b = n; } else if (n === b + 1) { b = n; } else { out.push(a === b ? `${a}` : `${a}–${b}`); a = b = n; } });
  if (a !== null) out.push(a === b ? `${a}` : `${a}–${b}`);
  return out.join(", ");
}

// "3,7-9" -> [3, 7, 8, 9]; null when the text is not a list of layer numbers
function parseLayers(text) {
  const t = text.trim(); if (!t) return [];
  const out = [];
  for (const part of t.split(/[,;\s]+/).filter(Boolean)) {
    const m = part.match(/^(\d+)(?:\s*[-–]\s*(\d+))?$/); if (!m) return null;
    const a = +m[1], b = m[2] === undefined ? a : +m[2]; if (b < a || b - a > 4096) return null;
    for (let i = a; i <= b; i++) out.push(i);
  }
  return out;
}

// one bar per block: how much the text changes inside it (small = the block does little); the blocks that were taken out are coloured
function barChart(dist, chosen) {
  if (!dist || !dist.length) return "";
  const W = 560, H = 150, P = 26, max = Math.max(...dist, 1e-9), bw = (W - P - 8) / dist.length;
  const bars = dist.map((d, i) => {
    const h = (d / max) * (H - P - 22), x = P + i * bw + 1;
    return `<rect x="${x.toFixed(1)}" y="${(H - P - h).toFixed(1)}" width="${Math.max(2, bw - 2).toFixed(1)}" height="${h.toFixed(1)}" fill="${chosen.includes(i) ? "var(--err)" : "var(--accent)"}"><title>block ${i} (layers ${i * 4}–${i * 4 + 3} when the blocks have four layers): ${d.toFixed(3)}${chosen.includes(i) ? " · removed" : ""}</title></rect>`;
  }).join("");
  return `<svg viewBox="0 0 ${W} ${H}" class="chart" role="img" aria-label="How much each block of layers changes the text">${bars}
    <line x1="${P}" y1="${H - P}" x2="${W - 8}" y2="${H - P}" stroke="var(--line)"/><text x="${P}" y="${H - 8}" class="axis">block 0</text><text x="${W - 70}" y="${H - 8}" class="axis">block ${dist.length - 1}</text>
    <text x="${P + 4}" y="14" class="axis">more change ↑ (kept) · red = removed</text></svg>`;
}

function patternText(ins) {
  const types = (ins.layer_types || []).slice(0, ins.period);
  const lin = types.filter((t) => t === "linear_attention").length, full = types.filter((t) => t === "full_attention").length;
  if (ins.period > 1 && lin && full && lin + full === types.length)
    return `${lin} fast linear-attention layers + ${full} full-attention layer${full > 1 ? "s" : ""}, repeated ${ins.blocks}×`;
  return ins.period > 1 ? `blocks of ${ins.period} layers, repeated ${ins.blocks}×` : `${ins.layers} layers of the same kind (a block is one layer)`;
}

function rsOption() { return rs.options && rs.options.options.find((o) => o.blocks === rs.blocks) || null; }

function rsSizeInfo() {
  const o = rsOption(), op = rs.options, v = op && op.vram_gb;
  if (!o) return `<span class="dim">Nothing can be changed in whole blocks for this model.</span>`;
  const tick = (fits) => fits === true ? "✓" : fits === false ? "✗" : "?";
  return `<strong>${o.layers} layers · ${o.params_b} B parameters</strong> · ${o.disk_gb} GB on disk · training: LoRA ≈ ${o.vram_lora_gb} GB ${tick(o.fits_lora)}, QLoRA ≈ ${o.vram_qlora_gb} GB ${tick(o.fits_qlora)}${v ? ` on this card (${v} GB)` : ""}`;
}

const thou = (n) => Number(n).toLocaleString("en-US");

// "How": whole layers, thinner layers, or both (only for models that allow thinning; the others get the reason)
function rsHowUi(op) {
  const w = op.width;
  if (rs.mode !== "prune" || !w) return "";
  if (!w.supported) return w.why ? `<p class="dim small">${esc(w.why)}</p>` : "";
  const r = (v, label) => `<label class="opt"><input type="radio" name="rs-shape" value="${v}" ${rs.shape === v ? "checked" : ""} ${rs.running ? "disabled" : ""}> ${label}</label>`;
  return `<p style="margin:10px 0 4px">How:</p>${r("depth", "<strong>Remove whole layers</strong> — the result runs faster")}${r("width", "<strong>Make every layer thinner</strong> — keeps more quality for the same size (needs healing)")}${r("both", "<strong>Both</strong> — remove a block and make the rest thinner")}`;
}

// the width options of the chosen way: the group for the number of blocks that go (0 = width only), and the stop for the chosen MLP width
function rsWidthOn() { return rs.mode === "prune" && rs.shape !== "depth" && !!(rs.options && rs.options.width_options); }
function rsWidthGroup() {
  const g = rs.options && rs.options.width_options;
  return g ? g.find((x) => x.blocks === (rs.shape === "width" ? 0 : rs.blocks)) || null : null;
}
function rsWidthOption() { const g = rsWidthGroup(); return g ? g.options.find((o) => o.intermediate === rs.wi) || null : null; }

// the stop to start on: the owner's decision is the mild cut (13/17 of the MLP, "Both": one block and 12/17)
function rsPickWidth() {
  const w = rs.options && rs.options.width;
  if (!w || !rs.options.width_options) return;
  const sug = w.suggested || {};
  if (rs.shape === "both" && sug.both && (rs.blocks == null || !rs.options.options.some((x) => x.blocks === rs.blocks))) rs.blocks = sug.both.blocks;
  const g = rsWidthGroup();
  if (g && !g.options.some((o) => o.intermediate === rs.wi)) rs.wi = (rs.shape === "both" && sug.both ? sug.both.intermediate : sug.width) || (g.options.length ? g.options[g.options.length - 1].intermediate : null);
}

function rsWidthInfo() {
  const o = rsWidthOption(), op = rs.options, v = op && op.vram_gb, full = op && op.width && op.width.dims ? op.width.dims.intermediate : null;
  if (!o) return `<span class="dim">Choose how thin the layers become.</span>`;
  const tick = (fits) => fits === true ? "✓" : fits === false ? "✗" : "?";
  const cut = full ? Math.round(100 * (1 - o.intermediate / full)) : null;
  return `MLP ${thou(o.intermediate)}${full ? " of " + thou(full) + " (−" + cut + " %)" : ""} · <strong>${o.layers} layers · ${o.params_b} B parameters${o.exact ? " (exact)" : ""}</strong> · ${o.disk_gb} GB on disk · training: LoRA ≈ ${o.vram_lora_gb} GB ${tick(o.fits_lora)}, QLoRA ≈ ${o.vram_qlora_gb} GB ${tick(o.fits_qlora)}${v ? ` on this card (${v} GB)` : ""}`;
}

// the part of "2 · What to do" for thinner layers: the way, the width slider(s), the Expert axes and the measuring
function rsWidthUi(op, st) {
  const w = op.width || {}, g = rsWidthGroup(), ch = w.choices || {}, d = w.dims || {};
  if (!g) return `<p class="dim">Looking at the sizes…</p>`;
  const idx = Math.max(0, g.options.findIndex((o) => o.intermediate === rs.wi));
  const o = rsWidthOption();
  const warn = o ? (o.warnings || []).map((x) => `<div class="${/worse|Removing|QLoRA|Even|experimental/.test(x) ? "warn" : "dim"}">${esc(x)}</div>`).join("") : "";
  const both = rs.shape === "both";
  const bidx = Math.max(0, op.options.findIndex((x) => x.blocks === rs.blocks));
  const expert = state.settings.expert_mode;
  const lv2 = (ch.lin_v_heads || [])[1], hs = (ch.heads || []).slice(1);
  const axes = expert ? `<details open><summary>Experimental: thin the attention layers as well</summary>
      ${EXPERIMENTAL_AXES_PASSED ? "" : `<p class="dim small">Switched off for now: nobody has published how well this works for this kind of model, and it has not been tried on a real graphics card yet. It is switched on when that has been done (docs/RESHAPE.md).</p>`}
      <label class="opt"><input type="checkbox" id="rs-lv" ${rs.lv ? "checked" : ""} ${EXPERIMENTAL_AXES_PASSED && lv2 ? "" : "disabled"}> Also thin the linear-attention layers (${d.lin_v_heads} → ${lv2 || "–"} value heads) <span class="tag">experimental</span></label>
      <label class="opt">Also remove attention heads <select id="rs-hh" ${EXPERIMENTAL_AXES_PASSED && hs.length ? "" : "disabled"}><option value="">no</option>${hs.map((h) => `<option value="${h}" ${rs.hh === h ? "selected" : ""}>${d.heads} → ${h} heads</option>`).join("")}</select> <span class="tag">experimental</span></label></details>` : "";
  const meas = rs.meas && rs.meas.exists ? `Measured earlier on this data (${thou(rs.meas.tokens || 0)} tokens${rs.meas.load ? ", loaded as " + esc(rs.meas.load) : ""}): <strong>reused</strong>. <button class="link" id="rs-measure-again" ${rs.running ? "disabled" : ""}>Measure again</button>`
    : `Not measured yet: it is measured first (a few minutes), then kept for every other size on this data. <button class="link" id="rs-measure-now" ${rs.running || !st.dataset_version ? "disabled" : ""}>Measure now</button>`;
  return `<div style="margin-top:12px">
    <div>Make every layer thinner: MLP <strong id="rs-w-n">${o ? thou(o.intermediate) : "–"}</strong> of ${thou(d.intermediate || 0)}</div>
    <input type="range" id="rs-wslider" min="0" max="${g.options.length - 1}" step="1" value="${idx}" style="width:100%;max-width:560px" ${rs.running ? "disabled" : ""}>
    <div id="rs-wsizeinfo">${rsWidthInfo()}</div>
    ${both ? `<div style="margin-top:10px">and remove <strong id="rs-blocks-n">${rs.blocks}</strong> block${rs.blocks === 1 ? "" : "s"} of ${op.inspect.period} layers</div>
      <input type="range" id="rs-slider" min="0" max="${op.options.length - 1}" step="1" value="${bidx}" style="width:100%;max-width:560px" ${rs.running ? "disabled" : ""}>
      <div id="rs-sizeinfo" class="dim small">${rsSizeInfo()}</div>` : ""}${warn}</div>
    <p>Measure on: ${st.dataset_version ? `data version v${st.dataset_version} (world book + practice answers), ${expert ? `<select id="rs-wsamples">${[128, 256, 512, 1024].map((n) => `<option ${rs.wsamples === n ? "selected" : ""}>${n}</option>`).join("")}</select>` : rs.wsamples} samples` : `<span class="err">Save a training-data version in Step 2 first: the layers are measured on your own text.</span>`}</p>
    <p class="dim small" id="rs-measinfo">${meas}</p>${axes}`;
}

// a small line chart: the points as a line, the first and the last value written out
function lineChart(points, label, lower) {
  if (!points || points.length < 2) return "";
  const W = 560, H = 120, P = 34, ys = points.map((q) => q[1]), lo = Math.min(...ys), hi = Math.max(...ys, lo + 1e-9);
  const x0 = points[0][0], x1 = Math.max(points[points.length - 1][0], x0 + 1);
  const X = (x) => P + ((x - x0) / (x1 - x0)) * (W - P - 10), Y = (y) => H - 18 - ((y - lo) / (hi - lo)) * (H - 36);
  const pts = points.map((q) => `${X(q[0]).toFixed(1)},${Y(q[1]).toFixed(1)}`).join(" ");
  return `<svg viewBox="0 0 ${W} ${H}" class="chart" role="img" aria-label="${esc(label)}"><polyline fill="none" stroke="var(--accent)" stroke-width="2" points="${pts}"/>
    <line x1="${P}" y1="${H - 18}" x2="${W - 10}" y2="${H - 18}" stroke="var(--line)"/><text x="${P}" y="12" class="axis">${esc(label)}${lower ? " (lower is better)" : ""}: ${points[0][1].toFixed(3)} → ${points[points.length - 1][1].toFixed(3)}</text>
    <text x="${P}" y="${H - 4}" class="axis">step ${points[0][0]}</text><text x="${W - 70}" y="${H - 4}" class="axis">step ${points[points.length - 1][0]}</text></svg>`;
}

const HEAL_STEPS = { cache: "1 of 3 · The original model answers your texts", train: "2 of 3 · The thinner model learns from the original", merge: "3 of 3 · Saving the healed model" };


// the live memory estimate for the settings chosen so far (the numbers come from the core, one source for the page and the job)
async function rsHealEstimate() {
  const name = rs.hl && rs.hl.name;
  if (!name) return;
  const key = JSON.stringify(rsHealSpec());
  try {
    const est = await invoke("heal_estimate", { name, spec: rsHealSpec() });
    if (est && rs.hl && rs.hl.name === name) { rs.hlEst = { name, key, est }; render(); }
  } catch (e) { /* the page keeps the estimate it has */ }
}

// the measured numbers of a healing run (saved ones, then the live ones of this session), as chart points
function healSeries() {
  const ev = ((rs.hl && rs.hl.opts && rs.hl.opts.metrics) || []).concat((rs.hlRun && rs.hlRun.live) || []);
  const seen = new Set(), kd = [], cal = [];
  ev.forEach((e) => { const k = e.kind + ":" + e.step + ":" + (e.kd ?? e.calib_loss); if (seen.has(k)) return; seen.add(k); if (e.kind === "distill_metric") kd.push([e.step, e.kd]); else if (e.kind === "distill_eval") cal.push([e.step, e.calib_loss]); });
  return { kd, cal };
}

// the Expert "Healing settings" section of the Heal card; null in a field of the spec = the automatic value, shown here from `kn`
function healSettings(sp, kn, plan, busy, upcastGb) {
  const d = busy ? "disabled" : "", v = (x, auto) => (x ?? auto);
  const sel = (id, cur, opts) => `<select id="${id}" ${d}>${opts.map(([val, label]) => `<option value="${val}" ${String(cur) === String(val) ? "selected" : ""}>${label}</option>`).join("")}</select>`;
  const pu = v(sp.per_update, (kn.batch || 1) * (kn.grad_accum || 1)), bt = v(sp.batch, kn.batch || 1);
  return `<details id="hl-settings" open><summary>Healing settings</summary>
    <p class="dim small">The safe settings for this model and card are already chosen. If healing runs out of memory, the program tries again by itself with safer ones.</p>
    <div class="grid">
    <label>Sequence length<br>${sel("hl-seq", v(sp.seq, kn.seq), [[512, "512"], [1024, "1024"], [2048, "2048"], [4096, "4096"]])}</label>
    <label>Batch per step<br>${sel("hl-batch", bt, [[1, "1"], [2, "2"], [4, "4"]])}</label>
    <label>Sequences per update<br><input type="text" id="hl-perupdate" value="${pu}" ${d}> <span class="dim small" id="hl-accum">= ${bt} × ${Math.ceil(pu / bt)} steps added up</span></label>
    <label>Gradient checkpointing<br>${sel("hl-ckpt", sp.checkpointing || "auto", [["auto", "Auto" + (kn.checkpointing != null ? " (" + (kn.checkpointing ? "on" : "off") + ")" : "")], ["on", "On (less memory, about a third slower)"], ["off", "Off"]])}</label>
    <label>LoRA size (rank, 8 to 128)<br><input type="text" id="hl-rank" value="${v(sp.rank, kn.rank ?? "")}" ${d}></label>
    <label>Method<br>${sel("hl-method", sp.method || "", [["", "Automatic (" + (plan.student_method === "qlora" ? "QLoRA 4-bit" : "LoRA 16-bit") + ")"], ["qlora", "QLoRA 4-bit"], ["lora", "LoRA 16-bit"]])}</label>
    <label>Optimizer<br>${sel("hl-optim", sp.optimizer || "adamw_8bit", [["adamw_8bit", "8-bit AdamW (less memory)"], ["adamw", "AdamW"]])}</label>
    <label>Loss chunk (64 to 512 tokens)<br><input type="text" id="hl-klchunk" value="${v(sp.kl_chunk, kn.kl_chunk ?? "")}" ${d}></label>
    <label>Answers kept per token (top-k, 8 to 128)<br><input type="text" id="hl-k" value="${v(sp.k, 32)}" ${d}></label>
    <label>The original is loaded as<br>${sel("hl-tload", sp.teacher_load || "auto", [["auto", "Automatic"], ["4bit", "4-bit"], ["16bit", "16-bit"]])}</label>
    <label>Warm-up steps<br><input type="text" id="hl-warmup" value="${v(sp.warmup, 30)}" ${d}></label></div>
    <label class="opt"><input type="checkbox" id="hl-upcast" ${sp.kbit_upcast === true ? "checked" : ""} ${busy || kn.qlora === false ? "disabled" : ""}> Keep the embeddings and output layer in 32-bit (the old behaviour: about <span id="hl-upcast-gb">${upcastGb != null ? upcastGb : "4 to 5"}</span> GB more memory)${kn.qlora === false ? " - only for QLoRA" : ""}</label>
    <p class="dim small">By default the two big word tables stay in 16-bit while healing: they are not changed by the learning, so a 32-bit copy only costs memory. Changing this keeps the saved answers and the learning done so far. If healing runs out of memory with this on, the program turns it off first.</p>
    <p class="dim small">A different sequence length, LoRA size or top-k means the original answers again or the learning starts again; the other settings keep what was saved.</p>
    <div class="row"><button class="secondary" id="hl-reset" ${d}>Reset to safe defaults</button></div></details>`;
}

// "Heal it": the original teaches the thinner model to answer like it did
function healCard(m) {
  const sm = m.summary;
  if (!sm || sm.mode !== "prune") return "";
  const h = rs.hl && rs.hl.name === m.name ? rs.hl.opts : null, run = rs.hlRun;
  const wide = sm.shape === "width" || sm.shape === "both";
  const head = `<h3>Heal it ${wide ? "(recommended)" : "(optional)"}</h3>`;
  if (m.healed) return `<div class="card-in">${head}<p>Healed: <strong>${esc((rs.state.models.find((x) => x.id === m.healed_model_id) || {}).name || m.healed_model_id || "")}</strong> is in the model list of Step 3 (${m.healed_size_gb} GB). ${m.cache_gb ? `The original's saved answers (${m.cache_gb} GB) stay until you delete this model.` : ""}</p></div>`;
  if (!h) return `<div class="card-in">${head}<p class="dim">Loading…</p></div>`;
  const sp = rs.hlSpec, tk = sp.tokens_m || h.default_tokens_m, st = (h.heal && h.heal.status) || null;
  const est = rs.hlEst && rs.hlEst.name === m.name ? rs.hlEst.est : null, plan = (est && est.plan) || h.plan || {}, kn = (est && est.knobs) || h.defaults || plan.knobs || {};
  const t = (h.tokens || []).find((x) => x.tokens_m === tk) || {};
  const expert = state.settings.expert_mode;
  gcUse(h.general);
  const blocked = sp.general ? gcBlocked("heal") : "";
  const verdictP = plan.verdict_text ? `<p class="${plan.verdict === "over" ? "err" : plan.verdict === "tight" ? "warn" : "dim"} small" id="hl-verdict">${esc(plan.verdict_text)}${plan.breakdown ? ` <span class="dim">Weights ${plan.breakdown.weights_gb} GB, LoRA and optimizer ${plan.breakdown.lora_gb} GB, activations ${plan.breakdown.activations_gb} GB, loss ${plan.breakdown.kl_gb} GB, the rest ${(plan.breakdown.overhead_gb + (plan.breakdown.total_gb - plan.breakdown.weights_gb - plan.breakdown.lora_gb - plan.breakdown.activations_gb - plan.breakdown.kl_gb - plan.breakdown.overhead_gb)).toFixed(1)} GB.</span>` : ""}</p>` : "";
  const mem = plan.fits === false ? `<div class="err">${esc(plan.why || "Healing does not fit this card.")}</div>${verdictP}`
    : `<p class="dim small">${plan.mode === "online" ? "The original and the thinner model run at the same time" : "The original answers first and saves its answers to disk, then the thinner model learns (they take turns)"}: about ${plan.est_gb} GB of memory${h.vram_gb ? " (this card has " + h.vram_gb + " GB)" : ""}. The thinner model is trained as ${plan.student_method === "qlora" ? "QLoRA" : "LoRA"}.</p>${verdictP}`;
  const busy = rs.running || rs.testing || (run && run.running);
  const prog = run && run.running ? `<div class="stepline">${esc(HEAL_STEPS[run.step] || "Starting…")}${run.text ? " · " + esc(run.text) : ""}</div>${run.progress != null ? `<progress max="1" value="${run.progress}"></progress>` : ""}` : "";
  const series = healSeries();
  const charts = lineChart(series.kd, "How far the thinner model's answers are from the original's (KL)", true) + lineChart(series.cal, "Text-prediction error on your data", true);
  const loss = h.loss && h.loss.original != null && h.loss.thinned != null ? `<p class="dim small">Before healing: error ${(+h.loss.thinned).toFixed(2)} on your data; the original has ${(+h.loss.original).toFixed(2)}. Healing aims to bring the first close to the second.</p>` : "";
  const used = h.heal && h.heal.general_used && h.heal.general_used.source ? `<p class="dim small">General text: ${h.heal.general_used.skipped ? "skipped" : "used"} <strong>${esc(h.heal.general_used.source)}</strong>${h.heal.general_used.rows ? ` (${h.heal.general_used.rows} rows${h.heal.general_used.detected ? ", read as " + esc(h.heal.general_used.detected.kind) : ""})` : ""}.</p>` : "";
  const other = sp.general && gc.state && gcSel("heal") !== gc.state.default;
  const note = h.heal && h.heal.note ? `<p class="warn">${esc(h.heal.note)}${other ? ` <button class="secondary" id="hl-usedefault">Use ${esc(gc.state.default)} instead</button>` : ""}</p>${used}` : used;
  const resume = st === "stopped" || st === "failed";
  const oomFailed = st === "failed" && h.heal && h.heal.oom === true;
  const tried = oomFailed && (h.heal.retries || []).length ? `<p class="dim small">Tried automatically: ${esc(h.heal.retries.map((r) => r.change).join(", then "))}.</p>` : "";
  const oomBlock = oomFailed ? `<div class="card-in" id="hl-oom">${tried}${h.safer
    ? `<p>The next safer setting that keeps what the model learns: <strong>${esc(h.safer.label)}</strong>. <button class="primary" id="hl-safer">Try again with safer settings</button></p>`
    : `<p class="warn">The automatic steps are used up. Next, in this order: a shorter sequence length, QLoRA, a lower LoRA size (Expert: Healing settings), a smaller model.</p>`}</div>` : "";
  return `<div class="card-in">${head}
    <p>The original model teaches the thinner one to answer like it did, on your world's text and on general text. This is what brings back most of what thinning took away. Your world's facts are still taught afterwards, in Step 3.</p>
    <label class="opt"><input type="checkbox" id="hl-general" ${sp.general ? "checked" : ""} ${busy ? "disabled" : ""}> Also use general text (about half of the healing text). <strong>It is downloaded from Hugging Face</strong> (or read from a file you choose) while the original answers; you need an internet connection for that. Without it, or with this box unticked, healing uses your world's text only (general ability comes back less).</label>
    ${sp.general ? gcHtml("heal") : ""}
    <div class="row">Healing text: <select id="hl-tokens" ${busy ? "disabled" : ""}>${(h.tokens || []).map((x) => `<option value="${x.tokens_m}" ${x.tokens_m === tk ? "selected" : ""}>${x.tokens_m} million tokens</option>`).join("")}</select>
      <span class="dim small">${t.hours ? `about ${t.hours[0]}–${t.hours[1]} hours (an estimate until this card has been measured) · ` : ""}${t.cache_gb != null && plan.mode === "cache" ? `${t.cache_gb} GB of disk for the original's saved answers` : ""}</span></div>
    ${mem}${loss}${note}
    ${expert ? `<details><summary>Expert settings</summary><div class="grid">
      <label>General share<input type="text" id="hl-share" value="${sp.share}"></label>
      <label>Weight of copying the original (1 = only copy it)<input type="text" id="hl-alpha" value="${sp.alpha}"></label>
      <label>Learning rate<input type="text" id="hl-lr" value="${sp.lr}"></label>
      <label>Way<select id="hl-mode"><option value="" ${sp.mode ? "" : "selected"}>what fits (${esc(plan.mode)})</option><option value="cache" ${sp.mode === "cache" ? "selected" : ""}>the original answers first</option><option value="online" ${sp.mode === "online" ? "selected" : ""}>both at once</option></select></label></div></details>` : ""}
    ${expert ? healSettings(sp, kn, plan, busy, est && est.upcast_gb != null ? est.upcast_gb : h.upcast_gb) : ""}${oomBlock}
    <div class="row"><button class="primary" id="hl-go" ${busy || plan.fits === false || !h.has_data || !rs.state.env_ready || blocked ? "disabled" : ""}>${oomFailed ? "Continue with the settings above" : resume ? "Continue healing" : "Start healing"}</button><button class="secondary" id="hl-stop" ${run && run.running ? "" : "disabled"}>Stop</button></div>
    ${blocked ? `<p class="warn small">${esc(blocked)}</p>` : ""}
    ${resume && h.heal ? `<p class="dim small">${oomFailed ? "Healing stopped: " + esc(h.heal.error || "") : st === "failed" ? "Healing stopped with an error: " + esc(h.heal.error || "") : "Healing was stopped"}${!oomFailed && h.heal.step ? " during step " + esc(h.heal.step) : ""}. ${oomFailed ? "The saved answers and the training so far are kept. A changed sequence length, LoRA size or top-k makes the original answer again or starts the learning again; the other settings go on from the last checkpoint." : "Continue goes on where it stopped."}</p>` : ""}
    ${prog}${charts}</div>`;
}

// original · thinned · healed, side by side: what the tests found and the text-prediction error
function sideBySide(m) {
  const t = m.test, th = m.test_healed, sm = m.summary;
  if (!t && !th) return "";
  const hr = (rs.hl && rs.hl.name === m.name && rs.hl.opts && rs.hl.opts.heal && rs.hl.opts.heal.result) || (m.heal && m.heal.result) || {};
  const cell = (side, id, x) => { const v = x && x.scores && x.scores[side] && x.scores[side][id]; return v ? pc(v.mean) : "–"; };
  const base = t || th;
  const rows = CATS.map(([id, name]) => `<tr><td>${name}</td><td>${cell("original", id, base)}</td><td>${t ? cell("trained", id, t) : "–"}</td><td><strong>${th ? cell("trained", id, th) : "–"}</strong></td></tr>`).join("");
  const err = (v) => v == null ? "–" : (+v).toFixed(2);
  return `<h3>Original · thinner · healed</h3><table class="tbl"><tr><th>What was checked</th><th>Original</th><th>Thinner</th><th>Healed</th></tr>${rows}
    <tr><td>Text-prediction error on your data (lower is better)</td><td>${err(sm.loss_before)}</td><td>${err(sm.loss_after)}</td><td><strong>${err(hr.calib_loss_after)}</strong></td></tr></table>
    <p class="dim small">${th ? esc(th.verdict.headline) : "The healed model has not been tested yet."}</p>`;
}

function madeCard(m) {
  const sm = m.summary;
  if (!sm) return `<div class="card wide"><h2>${esc(m.name)}</h2><p class="${m.status === "failed" ? "err" : "dim"}">${esc(m.status)}${m.error ? ": " + esc(m.error) : ""}</p></div>`;
  const slim = sm.mode === "prune";
  const wd = sm.width, wide = !!wd;
  const what = slim && wide && !(sm.dropped || []).length ? `Made every layer thinner (MLP ${thou(wd.before.intermediate_size)} → ${thou(wd.after.intermediate_size)})` : slim && wide ? `Removed layers ${ranges(sm.dropped)} and made every layer thinner (MLP ${thou(wd.before.intermediate_size)} → ${thou(wd.after.intermediate_size)})` : slim ? `Removed layers ${ranges(sm.dropped)} (${(sm.chosen_blocks || []).length || sm.dropped.length} block${(sm.chosen_blocks || []).length === 1 ? "" : "s"})` : `Added ${sm.layers_after - sm.layers_before} layers (copies of layers ${ranges(sm.duplicated)})`;
  const gb = (n) => (n / 1e9).toFixed(2);
  let loss = "";
  if (sm.loss_before != null && sm.loss_after != null) {
    loss = slim ? `Text-prediction error on your data: ${(+sm.loss_before).toFixed(2)} → ${(+sm.loss_after).toFixed(2)} before training it again (lower is better)`
      : (Math.abs(sm.loss_after - sm.loss_before) <= 1e-3 * Math.max(1e-9, Math.abs(sm.loss_before)) ? `Unchanged: it starts as an exact copy of the original (error ${(+sm.loss_after).toFixed(2)} both times, checked)` : `Text-prediction error on your data: ${(+sm.loss_before).toFixed(2)} → ${(+sm.loss_after).toFixed(2)}`);
  } else if (sm.loss_before != null) loss = `Text-prediction error of the original on your data: ${(+sm.loss_before).toFixed(2)}`;
  const t = m.test;
  const row = ([id, name]) => { const o = t.scores.original[id], n = t.scores.trained[id]; return o || n ? `<tr><td>${name}</td><td>${pc(o && o.mean)}</td><td><strong>${pc(n && n.mean)}</strong></td></tr>` : ""; };
  const test = t ? `<h3>Test against the original</h3><p class="verdict ${t.verdict.better === true ? "ok" : t.verdict.better === false ? "err" : ""}"><strong>${esc(t.verdict.headline)}</strong></p>
      ${t.verdict.lines.map((l) => `<p>${esc(l)}</p>`).join("")}
      <table class="tbl"><tr><th>What was checked</th><th>Original</th><th>${slim ? (wide ? "Thinner" : "Slimmed") : "Grown"}</th></tr>${CATS.map(row).join("")}</table>
      <p class="dim small">${t.n} questions · ${t.method === "teacher" ? "graded by the teacher" : "graded by matching words (a rough guide)"} · original loaded in ${esc((t.loaded || {}).original || "16-bit")}, this model in ${esc((t.loaded || {}).trained || "16-bit")} · ${esc(t.created)}</p>` : "";
  const busy = rs.running || rs.testing || (rs.hlRun && rs.hlRun.running);
  const ws = m.width_scores, meanChart = ws && ws.mlp_mean ? `<p class="dim small">How much each layer's MLP does (the average of its measured scores); the neurons that did least were taken out of every layer. ${ws.kept_share != null ? "The neurons that stayed hold " + Math.round(ws.kept_share * 100) + " % of the total score." : ""}</p>${barChart(ws.mlp_mean, [])}` : "";
  const stab = wide && wd.stable_overlap != null ? (wd.stable_overlap >= 0.9 ? `<p class="dim small">The choice of neurons was stable (${Math.round(wd.stable_overlap * 100)} % the same when chosen from either half of your text).</p>` : `<p class="warn">The measurement was not very stable: choosing from either half of your text gave only ${Math.round(wd.stable_overlap * 100)} % the same neurons. Measure on more samples (Expert).</p>`) : "";
  return `<div class="card wide"><h2>${esc(m.name)}</h2>
    ${slim && m.scores && m.scores.boundary_distance ? barChart(m.scores.boundary_distance, m.scores.chosen_blocks || []) : (sm.boundary_distance ? barChart(sm.boundary_distance, sm.chosen_blocks || []) : "")}${meanChart}
    <p><strong>${esc(what)}</strong> · ${sm.layers_before} → ${sm.layers_after} layers · ${gb(sm.params_before)} → ${gb(sm.params_after)} billion parameters · ${m.size_gb} GB on disk</p>
    ${loss ? `<p>${esc(loss)}</p>` : ""}${stab}
    ${slim && sm.stable === false && !wide ? `<p class="warn">The measurement was not stable: two halves of your text chose different layers. Try 256 samples.</p>` : ""}
    ${(sm.notes || []).map((n) => `<p class="dim small">${esc(n)}</p>`).join("")}
    <p class="dim small">The vision part and the multi-token head were copied unchanged. ${sm.load ? "Measured with the model loaded as " + esc(sm.load) + "." : ""} It is in the model list of Step 3 as “${esc((rs.state.models.find((x) => x.id === m.model_id) || {}).name || m.model_id || "")}”.</p>
    <div class="row"><button class="primary" data-rstest="${esc(m.name)}" ${busy || !rs.state.env_ready || !rs.state.dataset_version ? "disabled" : ""} title="${rs.state.dataset_version ? "" : "Save a training-data version in Step 2 first: the questions come from it."}">Test against the original</button>
      ${m.healed ? `<button class="primary" data-rstest-healed="${esc(m.name)}" ${busy || !rs.state.env_ready || !rs.state.dataset_version ? "disabled" : ""}>Test the healed model</button>` : ""}
      <button class="secondary" data-rstrain="${esc(m.healed ? m.healed_model_id || "" : m.model_id || "")}" ${(m.healed ? m.healed_model_id : m.model_id) ? "" : "disabled"}>Train ${m.healed ? "the healed model" : "it"} on your world (Step 3)</button>
      <button class="secondary" data-rsdelete="${esc(m.name)}" ${busy ? "disabled" : ""}>Delete</button></div>
    ${rs.testing === m.name ? `<div class="stepline">${esc(rs.text)}</div>${rs.progress != null ? `<progress max="1" value="${rs.progress}"></progress>` : ""}` : ""}${test}${sideBySide(m)}${healCard(m)}</div>`;
}

// ------------------------------------------------------------------------------------------------------- try before cutting
// A "Try first" card: what a cut will cost, found out in memory on the project's own text BEFORE a model is written (worker job reshape_try). The numbers are the
// same text-prediction error as the loss_before / loss_after of a made model. Colour bands and the stability check come from the worker (one constant there).

const TRY_BAND_WORD = { small: "small", noticeable: "noticeable", large: "large" };
const TRY_WHAT = [["blocks", "Every block on its own"], ["runs", "Every run of the blocks on the slider"], ["widths", "Every width of the slider"], ["choice", "My choice (click blocks in the chart)"]];

function tryDepthOn() { return rs.mode === "prune" && rs.shape !== "width"; }
function tryWidthOnShape() { return rs.mode === "prune" && rs.shape !== "depth" && !!(rs.options && rs.options.width_options); }
function trySamples() {
  const b = rs.options && rs.options.inspect ? rs.options.inspect.total_params / 1e9 : 0;
  const quick = rs.tr.quick == null ? b >= 15 : rs.tr.quick;
  return quick ? 64 : 128;
}
function tryWidths() {
  const g = rs.options && rs.options.width_options && rs.options.width_options.find((x) => x.blocks === 0), full = rs.options && rs.options.width && rs.options.width.dims ? rs.options.width.dims.intermediate : 0;
  return g ? g.options.map((o) => o.intermediate).filter((w) => w < full) : [];
}
// what the person's current choice is: the clicked blocks and/or the chosen width
function tryChoice() { return { blocks: rs.shape === "width" ? [] : [...rs.tr.sel].sort((a, b) => a - b), width: rs.shape === "depth" ? null : rs.wi }; }

// the request for the worker, or null when the card cannot ask for anything now
function trySpec() {
  const op = rs.options, ins = op && op.inspect;
  if (!ins || !ins.supported || rs.mode !== "prune" || !rs.state || !rs.state.dataset_version) return null;
  const spec = { model_id: rs.model, samples: trySamples(), len: 512, load: rs.load, measure_samples: rs.wsamples, strategy: rs.strategy, protect_first: rs.pf, protect_last: rs.pl,
    params_b: ins.total_params / 1e9, blocks: false, runs: [], widths: [], choices: [], force: false };
  const w = rs.tr.what;
  if (w === "blocks") { if (!tryDepthOn()) return null; spec.blocks = true; }
  else if (w === "runs") { if (!tryDepthOn() || !rs.blocks) return null; spec.runs = [rs.blocks]; }
  else if (w === "widths") { const ws = tryWidthOnShape() ? tryWidths() : []; if (!ws.length) return null; spec.widths = ws; }
  else {
    const c = tryChoice();
    if (!c.blocks.length && c.width == null) return null;
    spec.choices = [c.width == null ? { blocks: c.blocks } : { blocks: c.blocks, width: c.width }];
  }
  return spec;
}
const tryNeedsMeasure = (spec) => !!spec && (spec.widths.length > 0 || spec.choices.some((c) => c.width != null)) && !(rs.meas && rs.meas.exists);

function fmtDur(s) {
  if (s == null) return "?";
  if (s < 90) return `about ${Math.max(1, Math.round(s))} seconds`;
  if (s < 5400) return `about ${Math.round(s / 60)} minutes`;
  return `about ${(s / 3600).toFixed(1)} hours`;
}
const pct = (x) => x == null ? "–" : (x >= 0 ? "+" : "−") + Math.abs(x * 100).toFixed(1) + " %";

// keeps the plan (how many cuts, how long) and the saved answers in step with the card; called after every render, asks the worker only when the request changed
function tryRefresh() {
  if (page !== "reshape" || !rs.state || !rs.options || rs.tr.running) return;
  const spec = trySpec(), tr = rs.tr, sig = spec ? JSON.stringify(spec) : "";
  if (sig !== tr.planFor) {
    tr.planFor = sig; tr.plan = null; tr.planErr = null;
    if (spec && !tryNeedsMeasure(spec)) {
      const token = ++tr.planTok;
      invoke("reshape_try_plan", { spec }).then((p) => { if (token === tr.planTok) { tr.plan = p; if (page === "reshape") render(); } })
        .catch((e) => { if (token === tr.planTok) { tr.planErr = String(e); if (page === "reshape") render(); } });
    }
  }
  const rsig = rs.model + "|" + trySamples() + "|" + rs.load;
  if (rsig !== tr.resFor && rs.state.dataset_version) {
    tr.resFor = rsig;
    invoke("reshape_try_results", { model: rs.model, samples: trySamples(), len: 512, load: rs.load }).then((r) => { if (rsig === tr.resFor && !tr.running) { tr.analysis = r && r.exists ? r.analysis : null; tr.sps = r && r.exists ? r.seconds_per_pass : null; if (page === "reshape") render(); } })
      .catch(() => {});
  }
}

// finds the answer for the person's choice in the analysis (null while it was not tried)
function tryLookup(a, ch) {
  if (!a || !ch) return null;
  const same = (x, y) => x.length === y.length && x.every((v, i) => v === y[i]);
  let r = null;
  if (!ch.blocks.length) r = ch.width != null ? (a.widths || []).find((w) => w.intermediate === ch.width) : null;
  else if (ch.width == null) {
    r = ch.blocks.length === 1 ? (a.blocks || []).find((b) => b.block === ch.blocks[0])
      : (((a.runs || {})[String(ch.blocks.length)] || {}).windows || []).find((w) => same(w.blocks, ch.blocks)) || (a.choices || []).find((c) => c.intermediate == null && same(c.blocks, ch.blocks));
  } else r = (a.choices || []).find((c) => c.intermediate === ch.width && same(c.blocks, ch.blocks));
  return r && r.tried ? r : null;
}

function bandCss(b) { return b ? `var(--band-${b})` : "var(--line)"; }

// one bar per block: the rise of the error when ONLY this block is skipped (colour = band); the program's influence score is the diamond; click a block to choose it
function tryBlockChart(a, ins) {
  const nb = ins.blocks, W = 560, H = 190, P = 34, rows = {};
  (a ? a.blocks : []).forEach((r) => { rows[r.block] = r; });
  const edges = a ? a.bands.edges : [0.03, 0.1];
  const rels = Object.values(rows).map((r) => r.rel).filter((x) => x != null), top = Math.max(0.12, ...rels) * 1.05;
  const infl = Object.values(rows).map((r) => r.influence).filter((x) => x != null), imax = Math.max(...infl, 1e-9);
  const bw = (W - P - 8) / nb, base = H - 26, ph = H - 26 - 22, Y = (v) => base - (Math.max(0, v) / top) * ph;
  const chosen = new Set(rs.tr.sel);
  const bars = Array.from({ length: nb }, (_, i) => {
    const x = P + i * bw + 1, w = Math.max(3, bw - 2), r = rows[i], prot = i < rs.pf || i >= nb - rs.pl;
    if (prot) return `<rect x="${x.toFixed(1)}" y="${base - 6}" width="${w.toFixed(1)}" height="6" fill="var(--line)"><title>block ${i}: protected, never removed</title></rect>`;
    const h = r && r.rel != null ? Math.max(2, base - Y(r.rel)) : 0, sel = chosen.has(i);
    const body = r && r.tried ? `<rect x="${x.toFixed(1)}" y="${(base - h).toFixed(1)}" width="${w.toFixed(1)}" height="${h.toFixed(1)}" fill="${bandCss(r.band)}"/>`
      : `<rect x="${x.toFixed(1)}" y="${base - 10}" width="${w.toFixed(1)}" height="10" fill="none" stroke="var(--dim)" stroke-dasharray="2 2"/>`;
    const mark = r && r.influence != null ? `<path d="M${(x + w / 2).toFixed(1)} ${(base - (r.influence / imax) * ph - 4).toFixed(1)} l4 4 l-4 4 l-4 -4 z" fill="var(--text)" opacity=".75"/>` : "";
    const tip = `block ${i} (layers ${rs.options.inspect.period * i}–${rs.options.inspect.period * i + rs.options.inspect.period - 1})` + (r && r.tried ? `: error ${pct(r.rel)}, perplexity ${r.ppl}` : ": not tried yet") + (r && r.influence != null ? `, influence ${r.influence.toFixed(3)}` : "") + (sel ? " · chosen" : "");
    return `<g data-trblock="${i}" style="cursor:pointer"><rect x="${x.toFixed(1)}" y="20" width="${w.toFixed(1)}" height="${base - 20}" fill="transparent"/>${body}${mark}${sel ? `<rect x="${(x - 1).toFixed(1)}" y="20" width="${(w + 2).toFixed(1)}" height="${base - 20}" fill="none" stroke="var(--text)" stroke-width="2.5"/>` : ""}<title>${esc(tip)}</title></g>`;
  }).join("");
  const lines = edges.map((e) => `<line x1="${P}" x2="${W - 8}" y1="${Y(e).toFixed(1)}" y2="${Y(e).toFixed(1)}" stroke="var(--dim)" stroke-dasharray="3 3" opacity=".6"/><text x="2" y="${(Y(e) + 3).toFixed(1)}" class="axis">+${Math.round(e * 100)} %</text>`).join("");
  return `<svg viewBox="0 0 ${W} ${H}" class="chart" role="img" aria-label="What skipping each block costs">${lines}${bars}
    <line x1="${P}" y1="${base}" x2="${W - 8}" y2="${base}" stroke="var(--line)"/><text x="${P}" y="${H - 8}" class="axis">block 0</text><text x="${W - 70}" y="${H - 8}" class="axis">block ${nb - 1}</text>
    <text x="${P + 4}" y="12" class="axis">bar = rise of the error with only this block skipped · ◆ = the program's influence score · outlined = chosen</text></svg>`;
}

function tryLegend(a) {
  const e = a ? a.bands.edges : [0.03, 0.1];
  return `<div class="dim small"><span class="tr-chip" style="background:var(--band-small)"></span> under +${Math.round(e[0] * 100)} % <span class="tr-chip" style="background:var(--band-noticeable)"></span> +${Math.round(e[0] * 100)} to +${Math.round(e[1] * 100)} % <span class="tr-chip" style="background:var(--band-large)"></span> over +${Math.round(e[1] * 100)} % — rules of thumb, not measurements.</div>`;
}

function tryRunsTable(a, k) {
  const r = a && a.runs && a.runs[String(k)];
  if (!r) return "";
  const star = r.best, def = r.default, rows = r.windows.map((w) => {
    const mark = (star && star.start === w.start ? "★ cheapest " : "") + (def && def.blocks && def.blocks[0] === w.start && def.blocks.length === k ? "◆ the program's choice" : "");
    const on = rs.tr.sel.length === w.blocks.length && w.blocks.every((b) => rs.tr.sel.includes(b));
    return `<tr data-trrun="${w.start}" style="cursor:pointer;${on ? "background:var(--sel)" : ""}"><td>${w.blocks.length === 1 ? w.blocks[0] : w.blocks[0] + "–" + w.blocks[w.blocks.length - 1]}</td><td>${esc(ranges(w.layers))}</td>
      <td>${w.tried ? `<span class="tr-chip" style="background:${bandCss(w.band)}"></span> ${pct(w.rel)}` : "–"}</td><td>${w.tried ? w.ppl : "–"}</td><td>${w.distance != null ? w.distance.toFixed(3) : "–"}</td><td class="small">${mark}</td></tr>`;
  }).join("");
  let line = "";
  if (def && def.tried && star) {
    line = star.blocks.join() === (def.blocks || []).join() ? `<p>The program's choice (blocks ${esc(ranges(def.blocks))}) is also the cheapest run: it costs <strong>${pct(def.rel)}</strong>.</p>`
      : `<p>The program's default choice (blocks ${esc(ranges(def.blocks))}) costs <strong>${pct(def.rel)}</strong>; the cheapest run (blocks ${esc(ranges(star.blocks))}) costs <strong>${pct(star.rel)}</strong>.</p>`;
  } else if (star) line = `<p>The cheapest run is blocks ${esc(ranges(star.blocks))}: <strong>${pct(star.rel)}</strong>.${def && !def.tried ? " The program's own choice has not been tried yet." : ""}</p>`;
  const unstable = r.stability && !r.stability.agree ? `<div class="warn">The two halves of your text disagree about which run is cheapest. Treat this ranking as uncertain; trying Full (more text) may help.</div>` : "";
  return `<h4>Runs of ${k} block${k === 1 ? "" : "s"}${r.complete ? "" : " (not all tried yet)"}</h4>${line}${unstable}<table class="tbl"><tr><th>Blocks</th><th>Layers</th><th>Error</th><th>Perplexity</th><th>Influence</th><th></th></tr>${rows}</table>`;
}

function tryWidthChart(a, ins) {
  const rows = (a ? a.widths : []).filter((w) => w.tried);
  if (!rows.length) return "";
  const full = rs.options.width.dims.intermediate, W = 560, H = 150, P = 34;
  const top = Math.max(0.12, ...rows.map((r) => r.rel)) * 1.05, X = (w) => P + ((full - w) / (full - Math.min(...rows.map((r) => r.intermediate)) + 1e-9)) * (W - P - 16), Y = (v) => H - 24 - (Math.max(0, v) / top) * (H - 44);
  const pts = rows.slice().sort((x, y) => y.intermediate - x.intermediate).map((r) => `${X(r.intermediate).toFixed(1)},${Y(r.rel).toFixed(1)}`).join(" ");
  const dots = rows.map((r) => `<circle data-trwidth="${r.intermediate}" cx="${X(r.intermediate).toFixed(1)}" cy="${Y(r.rel).toFixed(1)}" r="${rs.wi === r.intermediate ? 6 : 4}" fill="${bandCss(r.band)}" stroke="var(--text)" stroke-width="${rs.wi === r.intermediate ? 2 : 0.5}" style="cursor:pointer"><title>MLP ${thou(r.intermediate)} (−${Math.round(100 * (1 - r.intermediate / full))} %): error ${pct(r.rel)}, perplexity ${r.ppl}</title></circle>`).join("");
  const edges = a.bands.edges.map((e) => `<line x1="${P}" x2="${W - 8}" y1="${Y(e).toFixed(1)}" y2="${Y(e).toFixed(1)}" stroke="var(--dim)" stroke-dasharray="3 3" opacity=".6"/><text x="2" y="${(Y(e) + 3).toFixed(1)}" class="axis">+${Math.round(e * 100)} %</text>`).join("");
  return `<svg viewBox="0 0 ${W} ${H}" class="chart" role="img" aria-label="Quality against size for thinner layers">${edges}<polyline fill="none" stroke="var(--dim)" points="${pts}"/>${dots}
    <line x1="${P}" y1="${H - 24}" x2="${W - 8}" y2="${H - 24}" stroke="var(--line)"/><text x="${P}" y="${H - 8}" class="axis">MLP ${thou(full)} (the original)</text><text x="${W - 150}" y="${H - 8}" class="axis">thinner → (click to choose)</text>
    <text x="${P + 4}" y="12" class="axis">rise of the error with only the neurons below the cut switched off</text></svg>`;
}

function tryWidthTable(a) {
  const rows = (a ? a.widths : []).filter((w) => w.tried);
  if (!rows.length) return "";
  const full = rs.options.width.dims.intermediate;
  const unstable = a.widths_stability && !a.widths_stability.agree ? `<div class="warn">The two halves of your text disagree about the order of the widths. Treat this curve as uncertain.</div>` : "";
  return `${unstable}<details><summary>The numbers</summary><table class="tbl"><tr><th>MLP</th><th>Cut</th><th>Error</th><th>Perplexity</th></tr>${rows.slice().sort((x, y) => y.intermediate - x.intermediate).map((r) =>
    `<tr data-trwidth="${r.intermediate}" style="cursor:pointer;${rs.wi === r.intermediate ? "background:var(--sel)" : ""}"><td>${thou(r.intermediate)}</td><td>−${Math.round(100 * (1 - r.intermediate / full))} %</td><td><span class="tr-chip" style="background:${bandCss(r.band)}"></span> ${pct(r.rel)}</td><td>${r.ppl}</td></tr>`).join("")}</table></details>`;
}

// the plain verdict for one answer, with the honest part that never changes
function tryVerdict(a, r, what) {
  const [e1, e2] = a.bands.edges, band = r.band, base = a.base;
  const sentence = band === "small" ? "a small rise" : band === "noticeable" ? "a noticeable rise" : "a large rise";
  return `<div class="tr-verdict"><p><strong>${esc(what)}</strong>: the text-prediction error on your own text goes from ${base.loss.toFixed(3)} to ${r.loss.toFixed(3)} (<strong>${pct(r.rel)}</strong>; perplexity ${base.ppl} → ${r.ppl}). That is
    <span class="tr-chip" style="background:${bandCss(band)}"></span> <strong>${sentence}</strong>.</p></div>`;
}

const TRY_HONEST = (a) => `<p class="dim small">This is the error <strong>before</strong> healing: healing usually wins back part of it. A cheap cut now is not a guarantee that it heals well, and an expensive one can still heal into a usable model; nobody has measured how the two relate for your model yet. The colours are rules of thumb (under +${Math.round(a.bands.edges[0] * 100)} %, +${Math.round(a.bands.edges[0] * 100)} to +${Math.round(a.bands.edges[1] * 100)} %, over +${Math.round(a.bands.edges[1] * 100)} % of the original's error), not promises. Removing two blocks costs more than the two single costs added up; the "My choice" pass is the honest number. Small differences between two options are noise: the two halves of your text are compared below the chart.</p>`;

function tryCard() {
  const op = rs.options, ins = op && op.inspect, st = rs.state, tr = rs.tr;
  if (!ins || !ins.supported || rs.mode !== "prune" || !(op.options.length > 0 || tryWidthOnShape())) return "";
  if (!st.dataset_version) return `<div class="card wide"><h2>Try first</h2><p class="dim">Save a training-data version in Step 2 first: the cuts are tried on your own text.</p></div>`;
  const a = tr.analysis, spec = trySpec(), needM = tryNeedsMeasure(spec), run = tr.running;
  const what = TRY_WHAT.filter(([k]) => (k === "blocks" || k === "runs") ? tryDepthOn() : k === "widths" ? tryWidthOnShape() : true)
    .map(([k, l]) => `<label class="opt"><input type="radio" name="tr-what" value="${k}" ${tr.what === k ? "checked" : ""} ${run ? "disabled" : ""}> ${esc(l)}${k === "runs" && rs.blocks ? ` <span class="dim">(${rs.blocks} block${rs.blocks === 1 ? "" : "s"})</span>` : ""}</label>`).join("");
  const q = trySamples();
  const plan = tr.plan;
  const planLine = !spec ? `<span class="dim">${tr.what === "choice" ? "Click blocks in the chart (or pick a width) to choose what to try." : "Nothing to try with the settings above."}</span>`
    : needM ? `<span class="dim">The width needs the neuron measurement of this model first. <button class="link" id="tr-measure" ${rs.running ? "disabled" : ""}>Measure first</button></span>`
    : tr.planErr ? `<span class="err">${esc(tr.planErr)}</span>`
    : !plan ? `<span class="dim">Counting…</span>`
    : plan.passes_todo === 0 ? `<span class="dim">All ${plan.passes_total} answers are already kept; showing them costs nothing.</span>`
    : `<strong>${plan.passes_todo}</strong> pass${plan.passes_todo === 1 ? "" : "es"} over your text${plan.passes_cached ? ` (${plan.passes_cached} already kept)` : ""} × ${fmtDur(plan.seconds_per_pass).replace("about ", "")} each = <strong>${fmtDur(plan.seconds)}</strong>${plan.guess ? ` <span class="dim">(a guess from the model's size; the real time is measured on the first pass and kept)</span>` : ` <span class="dim">(from a pass that was timed)</span>`}`;
  const disabled = !spec || needM || run || rs.running || rs.testing || !!tr.planErr || (plan && plan.passes_todo === 0 && false);
  const progress = run ? `<div class="stepline">${esc(tr.text)}${tr.total ? ` · ${tr.done} of ${tr.total} done` : ""}</div>${tr.progress != null ? `<progress max="1" value="${tr.progress}"></progress>` : ""}` : "";
  let body = "";
  if (a && a.base && a.base.loss != null) {
    const parts = [];
    if (tryDepthOn() && a.blocks.some((r) => r.tried)) parts.push(`<h4>Skipping one block at a time</h4>${tryBlockChart(a, ins)}${tryLegend(a)}${a.blocks_stability && !a.blocks_stability.agree ? `<div class="warn">The two halves of your text disagree about which blocks are cheapest to skip. Treat the order as uncertain.</div>` : ""}`);
    else if (tryDepthOn() && (tr.what === "choice" || Object.keys(a.runs || {}).length)) parts.push(`<h4>Choose blocks</h4>${tryBlockChart(a, ins)}${tryLegend(a)}`);
    if (tryDepthOn() && rs.blocks && a.runs && a.runs[String(rs.blocks)]) parts.push(tryRunsTable(a, rs.blocks));
    if (tryWidthOnShape() && a.widths.some((w) => w.tried)) parts.push(`<h4>Thinner layers: quality against size</h4>${tryWidthChart(a, ins)}${tryLegend(a)}${tryWidthTable(a)}`);
    const ch = tryChoice(), hit = tryLookup(a, ch);
    const label = (ch.blocks.length ? `blocks ${ranges(ch.blocks)}` : "") + (ch.blocks.length && ch.width != null ? " and " : "") + (ch.width != null ? `MLP ${thou(ch.width)}` : "");
    const verdict = hit ? tryVerdict(a, hit, "Your choice (" + label + ")") : (ch.blocks.length || ch.width != null) ? `<p class="dim">Your choice (${esc(label)}) has not been tried yet: choose “My choice” above and press Try.</p>` : "";
    parts.push(`<h4>Verdict</h4>${verdict}<p>Original: error ${a.base.loss.toFixed(3)}, perplexity ${a.base.ppl}.</p>${TRY_HONEST(a)}
      <div class="row"><button class="primary" id="tr-use" ${hit && !run && !rs.running ? "" : "disabled"}>Use this choice</button><span class="dim small">${hit ? "fills the Make form below so the model is cut exactly as it was tried" : "try the choice first, then it can be used"}</span></div>`);
    body = parts.join("");
  }
  return `<div class="card wide"><h2>Try first <span class="tag">optional</span></h2>
    <p>Before a new model is written (tens of GB), find out what a cut costs. The program imitates the cut inside the loaded model and measures the same text-prediction error as the “before / after” numbers of a made model. <strong>Nothing is saved except a small results file.</strong></p>
    <div class="row">${what}</div>
    <p>Text: <label class="opt" style="display:inline"><input type="radio" name="tr-q" value="quick" ${q === 64 ? "checked" : ""} ${run ? "disabled" : ""}> Quick (64 samples)</label> <label class="opt" style="display:inline"><input type="radio" name="tr-q" value="full" ${q === 128 ? "checked" : ""} ${run ? "disabled" : ""}> Full (128 samples)</label>
      <span class="dim small">${ins.total_params / 1e9 >= 15 ? "Quick is the default for models of 15 B or more." : "Full is the default for smaller models."}</span></p>
    <p id="tr-plan">${planLine}</p>
    <div class="row"><button class="primary" id="tr-go" ${disabled ? "disabled" : ""}>${tr.what === "choice" ? "Preview my choice" : "Try"}</button><button class="secondary" id="tr-stop" ${run ? "" : "disabled"}>Stop</button><span class="dim small">Stopping keeps the cuts already tried.</span></div>
    ${progress}${body}</div>`;
}

function tryToggle(b) {
  const i = rs.tr.sel.indexOf(b);
  if (i >= 0) rs.tr.sel.splice(i, 1); else rs.tr.sel.push(b);
  rs.tr.sel.sort((x, y) => x - y);
  rs.tr.what = rs.tr.what === "runs" || rs.tr.what === "blocks" ? rs.tr.what : "choice";
}

// "Use this choice": fills the Make form with exactly what was tried
function tryUse() {
  const c = tryChoice(), ins = rs.options.inspect;
  if (c.blocks.length) {
    const layers = c.blocks.flatMap((b) => Array.from({ length: ins.period }, (_, i) => b * ins.period + i));
    rs.dropText = ranges(layers);
    if (rs.options.options.some((o) => o.blocks === c.blocks.length)) rs.blocks = c.blocks.length;
    if (c.width != null && rs.shape === "depth") rs.shape = "both";
  }
  if (c.width != null) rs.wi = c.width;
  rs.error = null;
  rs.ok = "The Make form now cuts exactly what was tried" + (c.blocks.length ? `: layers ${ranges(rs.dropText ? parseLayers(rs.dropText) : [])}` : "") + (c.width != null ? `, MLP ${thou(c.width)}` : "") + ".";
}

function onTryEvent(p) {
  const tr = rs.tr;
  if (p.kind === "progress") { tr.text = p.text || tr.text; tr.progress = p.value; }
  else if (p.kind === "log") rs.log.push(p.text);
  else if (p.kind === "try_pass") { tr.analysis = p.analysis; tr.done = p.done; tr.total = p.total; tr.sps = p.seconds_per_pass; }
  else if (p.kind === "error") { tr.running = false; tr.text = ""; tr.progress = null; rs.error = p.text; tr.planFor = null; tr.resFor = null; }
  else if (p.kind === "done") {
    tr.running = false; tr.text = ""; tr.progress = null; tr.planFor = null; tr.resFor = null;
    if (p.data && p.data.analysis) tr.analysis = p.data.analysis;
    rs.ok = p.data && p.data.stopped ? `Stopped. ${p.data.passes_done} of ${p.data.passes_total} cuts are kept; Try goes on from there.` : "Tried. The answers are kept; trying the same again costs nothing.";
  }
  if (page === "reshape") render();
}

function wireTry() {
  const on = (id, fn) => { const e = $(id); if (e) e.onclick = fn; };
  const tr = rs.tr;
  document.querySelectorAll("input[name=tr-what]").forEach((r) => r.onchange = () => { tr.what = r.value; render(); });
  document.querySelectorAll("input[name=tr-q]").forEach((r) => r.onchange = () => { tr.quick = r.value === "quick"; tr.analysis = null; render(); });
  document.querySelectorAll("[data-trblock]").forEach((g) => g.onclick = () => { tryToggle(+g.dataset.trblock); render(); });
  document.querySelectorAll("[data-trrun]").forEach((row) => row.onclick = () => {
    const w = (((tr.analysis || {}).runs || {})[String(rs.blocks)] || {}).windows || [], hit = w.find((x) => x.start === +row.dataset.trrun);
    if (hit) { tr.sel = [...hit.blocks]; render(); }
  });
  document.querySelectorAll("[data-trwidth]").forEach((e) => e.onclick = () => { if (rs.shape !== "depth") { rs.wi = +e.dataset.trwidth; render(); } });
  on("tr-go", async () => {
    const spec = trySpec(); if (!spec) return;
    tr.running = true; tr.text = "Starting…"; tr.progress = null; tr.done = 0; tr.total = 0; rs.error = null; rs.ok = null; rs.log = []; render();
    try { await invoke("reshape_try_start", { spec }); } catch (e) { tr.running = false; rs.error = String(e); render(); }
  });
  on("tr-stop", () => { tr.text = "Stopping…"; invoke("reshape_stop"); render(); });
  on("tr-use", () => { tryUse(); render(); });
  on("tr-measure", () => rsMeasure(false));
  tryRefresh();
}

function renderReshape() {
  const head = `<h1>🧬 Slim or grow a model</h1><p class="lead">Make a smaller or a bigger version of a model by taking whole blocks of layers out or copying some in. The help panel explains it.</p>`;
  if (!state.bundle) return head + `<p class="lead">Import your world data first (Step 1).</p>`;
  const st = rs.state;
  if (!st) return head + `<p class="lead">Loading…</p>`;
  const intro = `<div class="card wide"><h2>What this does</h2>
    <p><strong>Slim</strong> takes whole blocks of layers out where they change the text the least (measured on your own world's text). The model gets smaller, faster and needs less memory, but it answers <strong>worse until you train it again</strong>.</p>
    <p><strong>Grow</strong> copies a stretch of layers in. It starts out answering exactly like the original, and is then trained. <strong>Honest expectation:</strong> from one world's text the gain is usually small or none, and it needs about 40 % more memory, disk space and time. It pays off with much more training text than a world has.</p>
    <p><strong>Thinner layers</strong> (Qwen3.5 and 3.8 only) keep every layer but make each one narrower: the neurons that matter least, measured on your own text, are taken out of the MLP of every layer. This usually keeps more quality than removing whole layers for the same size, but it is slower to run, and it needs <strong>healing</strong>: the original model teaches the thinner one to answer like it did, which takes hours.</p>
    <p>Always test the new model, and usually train it in Step 3 afterwards.</p></div>`;
  const msg = (rs.error ? `<div class="err">${esc(rs.error)}</div>` : "") + (rs.ok ? `<div class="ok">${esc(rs.ok)}</div>` : "");
  const modelOpts = st.models.map((m) => `<option value="${esc(m.id)}" ${m.id === rs.model ? "selected" : ""}>${esc(m.name)} · ${(+m.params_b).toFixed(1)} B${m.reshape ? " · " + esc(m.reshape) : ""}</option>`).join("");
  const op = rs.options, ins = op && op.inspect;
  let info = "";
  if (rs.loadingOpts) info = `<p class="dim">Looking at the model… (the first time this can take a moment; nothing is downloaded)</p>`;
  else if (rs.optErr) info = `<div class="err">${esc(rs.optErr)}</div>`;
  else if (ins && !ins.supported) info = `<div class="err">${esc(ins.why || "This model cannot be slimmed or grown.")}</div>`;
  else if (ins) info = `<p>${ins.layers} layers: ${esc(patternText(ins))} · ${(ins.total_params / 1e9).toFixed(2)} B parameters · ${ins.has_vision ? "has a vision part (kept unchanged)" : "text only"}${ins.local ? "" : " · not on this computer yet: it is downloaded first (≈ " + (ins.source_bytes / 1e9).toFixed(0) + " GB)"}</p>`;
  const pick = `<div class="card wide"><h2>1 · Start from</h2><div class="row"><select id="rs-model" ${rs.running ? "disabled" : ""}>${modelOpts}</select></div>${info}</div>`;
  const ok = ins && ins.supported && !rs.optErr && (op.options.length > 0 || (op.width && op.width.supported && rs.mode === "prune"));
  const grow = rs.mode === "grow", o = rsOption();
  const idx = op ? Math.max(0, op.options.findIndex((x) => x.blocks === rs.blocks)) : 0;
  const warn = o ? o.warnings.map((w) => `<div class="${/little or no gain|worse|quarter|QLoRA|Even/.test(w) ? "warn" : "dim"}">${esc(w)}</div>`).join("") : "";
  const size = `<div class="card wide"><h2>2 · What to do</h2>
    <label class="opt"><input type="radio" name="rs-mode" value="prune" ${rs.mode === "prune" ? "checked" : ""} ${rs.running ? "disabled" : ""}> <strong>Slim it</strong> — take blocks of layers out</label>
    <label class="opt"><input type="radio" name="rs-mode" value="grow" ${grow ? "checked" : ""} ${rs.running ? "disabled" : ""}> <strong>Grow it</strong> — copy a stretch of layers in <span class="dim">(little or no gain from one world's text; about 40 % more memory)</span></label>
    ${ok && !grow && op.width ? rsHowUi(op) : ""}
    ${ok && rsWidthOn() ? rsWidthUi(op, st) : ""}
    ${ok && !rsWidthOn() ? `<div style="margin-top:12px"><div>${grow ? "Add" : "Remove"} <strong id="rs-blocks-n">${rs.blocks}</strong> block${rs.blocks === 1 ? "" : "s"} of ${ins.period} layers</div>
      <input type="range" id="rs-slider" min="0" max="${op.options.length - 1}" step="1" value="${idx}" style="width:100%;max-width:560px" ${rs.running ? "disabled" : ""}>
      <div id="rs-sizeinfo">${rsSizeInfo()}</div>${warn}</div>` : ""}
    ${ok ? "" : (ins && ins.supported && !rs.loadingOpts ? `<p class="dim">This model is too small to change in whole blocks.</p>` : "")}
    ${ok && !grow && rs.dropText && !state.settings.expert_mode ? `<div class="dim small">Your choice from “Try first” is used: layers ${esc(ranges(parseLayers(rs.dropText) || []))}. <button class="link" id="rs-cleardrop">Use the program's choice instead</button></div>` : ""}
    ${ok && grow ? `<p style="margin-bottom:4px">How should the new layers start?</p>
      <label class="opt"><input type="radio" name="rs-init" value="identity" ${rs.init === "identity" ? "checked" : ""}> <strong>As an exact copy of the original</strong> (recommended): the new layers are switched off, so nothing is lost before training</label>
      <label class="opt"><input type="radio" name="rs-init" value="copy" ${rs.init === "copy" ? "checked" : ""}> <strong>Copy the layers</strong> (SOLAR style): the copied layers work at once, which disturbs the model until it is trained</label>` : ""}
    ${ok && !grow && !rsWidthOn() ? `<p>Measure on: ${st.dataset_version ? `data version v${st.dataset_version} (world book + practice answers), <select id="rs-samples">${[64, 128, 256].map((n) => `<option ${rs.samples === n ? "selected" : ""}>${n}</option>`).join("")}</select> samples` : `<span class="err">Save a training-data version in Step 2 first: the layers are measured on your own text.</span>`}</p>
      <p class="dim small">The model is loaded ${op.load === "16bit" ? "in 16-bit" : op.load === "4bit" ? "in 4-bit (NF4) to fit the card; the ranking of the layers hardly changes" : "spread over the card and the computer's memory (slow)"} while measuring.</p>` : ""}
    ${ok && !grow && !rsWidthOn() && state.settings.expert_mode ? `<details open><summary>Expert settings</summary><div class="grid">
      <label>Which layers<select id="rs-strategy"><option value="contiguous" ${rs.strategy === "contiguous" ? "selected" : ""}>one run of blocks (recommended)</option><option value="separate" ${rs.strategy === "separate" ? "selected" : ""}>separate blocks wherever they matter least</option></select></label>
      <label>Never remove the first … blocks<input type="text" id="rs-pf" value="${rs.pf}"></label><label>… and the last … blocks<input type="text" id="rs-pl" value="${rs.pl}"></label>
      <label>Load for measuring<select id="rs-load">${[["auto", "automatic"], ["16bit", "16-bit"], ["4bit", "4-bit"], ["offload", "spread over card and memory"]].map(([k, l]) => `<option value="${k}" ${rs.load === k ? "selected" : ""}>${l}</option>`).join("")}</select></label>
      <label>Remove exactly these layers (skips the measuring)<input type="text" id="rs-drop" value="${esc(rs.dropText)}" placeholder="for example 36-51"></label></div></details>` : ""}</div>`;
  const dropList = parseLayers(rs.dropText);
  const wo = rsWidthOn() ? rsWidthOption() : null;
  const needGb = wo ? wo.disk_needed_gb : o ? o.disk_needed_gb : 0;
  const diskBad = st.free_disk_gb != null && (o || wo) && st.free_disk_gb < needGb;
  const noData = !grow && !st.dataset_version && (rs.shape !== "depth" || !(dropList && dropList.length));
  const why = !st.env_ready ? "The training software is not installed yet (Training setup)." : noData ? "Save a training-data version in Step 2 first." : diskBad ? `Not enough free disk space: ${needGb} GB needed, ${st.free_disk_gb} GB free.` : dropList === null ? "The layer list is not valid." : rsWidthOn() && !wo ? "Choose how thin the layers become." : "";
  const make = `<div class="card wide"><h2>3 · Make the model</h2>
    <div class="row"><button class="primary" id="rs-go" ${!ok || rs.running || rs.testing || rs.tr.running || why ? "disabled" : ""}>Make the model</button><button class="secondary" id="rs-stop" ${rs.running ? "" : "disabled"}>Stop</button></div>
    ${ok && why ? `<div class="err">${esc(why)}</div>` : ""}${ok && (wo || (!rsWidthOn() && o)) ? `<div class="dim small">Needs about ${needGb} GB of free disk space${st.free_disk_gb != null ? " (" + st.free_disk_gb + " GB free)" : ""}. The new model is saved inside the project, in <code>models/</code>.</div>` : ""}
    ${rs.running ? `<div class="stepline">${esc(rs.text)}</div>${rs.progress != null ? `<progress max="1" value="${rs.progress}"></progress>` : ""}` : ""}
    ${rs.log.length ? `<details><summary>Log</summary><pre class="log">${rs.log.slice(-120).map(esc).join("\n")}</pre></details>` : ""}</div>`;
  const made = st.made.filter((m) => m.status === "done" && m.summary);
  const shown = made.find((m) => m.name === rs.view) || made[0];
  const result = shown ? madeCard(shown) : "";
  const list = st.made.length ? `<div class="card wide"><h2>Models made so far</h2><table class="tbl"><tr><th>Name</th><th>Status</th><th>Layers</th><th>Size</th><th>Tested</th><th></th></tr>${st.made.map((m) =>
    `<tr><td>${esc(m.name)}</td><td>${esc(m.status)}${m.error ? `<div class="dim small">${esc(m.error)}</div>` : ""}</td><td>${m.summary ? m.summary.layers_before + " → " + m.summary.layers_after + (m.summary.width ? ", MLP " + thou(m.summary.width.before.intermediate_size) + " → " + thou(m.summary.width.after.intermediate_size) : "") : ""}</td><td>${m.size_gb ? m.size_gb + " GB" : ""}${m.healed ? ` <span class="tag">healed</span>` : ""}</td><td>${m.test ? esc(m.test.verdict.headline) : "–"}</td>
      <td>${m.summary ? `<button class="link" data-rsshow="${esc(m.name)}">Show</button> ` : ""}<button class="link" data-rsdelete="${esc(m.name)}" ${rs.running || rs.testing ? "disabled" : ""}>Delete</button></td></tr>`).join("")}</table></div>` : "";
  return head + msg + intro + pick + size + tryCard() + make + result + list;
}

async function rsRefresh(stale) {
  try {
    rs.state = await invoke("reshape_state");
    if (!rs.model && rs.state.models.length) rs.model = (rs.state.models.find((m) => m.id === "qwen3.5-9b") || rs.state.models[0]).id;
  } catch (e) { rs.error = String(e); }
  if (page === "reshape") render();
  if (rs.state && rs.model && !rs.options && !rs.loadingOpts) rsOptions();
  if (stale && rs.hl) rs.hl.stale = true;
  rsAfterState();
}

async function rsOptions() {
  const token = ++rs.token;
  rs.loadingOpts = true; rs.optErr = null; rs.options = null; render();
  try {
    const shape = rs.mode === "prune" ? rs.shape : "depth";
    const args = { model: rs.model, mode: rs.mode };
    if (shape !== "depth") {
      args.shape = shape;
      if (rs.lv) args.lin_v_heads = rsAxis("lin_v_heads");
      if (rs.hh) args.heads = rs.hh;
    }
    const o = await invoke("reshape_options", args);
    if (token !== rs.token) return;
    rs.options = o;
    if (o.width && !o.width.supported && rs.shape !== "depth") rs.shape = "depth";
    if (!o.options.some((x) => x.blocks === rs.blocks)) rs.blocks = o.options.length ? o.default_blocks : null;
    rsPickWidth();
  } catch (e) { if (token !== rs.token) return; rs.optErr = String(e); }
  rs.loadingOpts = false;
  if (page === "reshape") render();
  rsMeasInfo();
}

// the second step of a way with two value heads: 48 -> 32 (one step only)
function rsAxis(which) { const c = rs.options && rs.options.width && rs.options.width.choices; return c && c[which] && c[which][1] != null ? c[which][1] : null; }

// is there an earlier measurement of this model on the saved data with these settings? (for "measured earlier — reused")
async function rsMeasInfo() {
  if (!rs.model || !rs.state || !rs.state.dataset_version || rs.shape === "depth") return;
  try {
    const meas = await invoke("reshape_measurement", { model: rs.model, samples: rs.wsamples, len: 512, load: rs.load });
    rs.meas = meas || null;
  } catch (_) { rs.meas = null; }
  if (page === "reshape") render();
}

// the person's Heal card needs the options of the model that is shown
function rsAfterState() {
  const st = rs.state; if (!st) return;
  const made = st.made.filter((m) => m.status === "done" && m.summary);
  const shown = made.find((m) => m.name === rs.view) || made[0];
  if (shown && shown.summary.mode === "prune" && (!rs.hl || rs.hl.name !== shown.name || rs.hl.stale)) rsLoadHeal(shown.name);
}

async function rsLoadHeal(name) {
  rs.hl = { name, opts: null };
  try { rs.hl = { name, opts: await invoke("heal_options", { name }) }; } catch (e) { rs.hl = { name, opts: null, error: String(e) }; }
  if (page === "reshape") render();
  if (rs.hl && rs.hl.opts) rsHealEstimate();
}

function onHealEvent(p) {
  const run = rs.hlRun || (rs.hlRun = { running: true, step: "cache", text: "", progress: null, live: [] });
  if (p.kind === "progress") { run.step = p.heal_step || run.step; run.text = p.text || ""; run.progress = p.value; }
  else if (p.kind === "log") rs.log.push(p.text);
  else if (p.kind === "distill_metric" || p.kind === "distill_eval") { run.step = p.heal_step || run.step; run.live.push(p); }
  else if (p.kind === "error") { run.running = false; rs.error = p.text; rsRefresh(true); return; }
  else if (p.kind === "done") {
    run.running = false;
    if (p.data && p.data.stopped) rs.ok = "Stopped. Healing can be continued where it stopped.";
    else if (p.data) rs.ok = `Healed: ${p.data.name} now has a healed version, and it is in the model list of Step 3.`;
    setState_(); rsRefresh(true); return;
  }
  if (page === "reshape") render();
}

function onReshapeEvent(p) {
  if (p.phase === "heal") return onHealEvent(p);
  if (p.phase === "try") return onTryEvent(p);
  if (p.phase === "measure" && (p.kind === "done" || p.kind === "error")) {
    rs.running = false; rs.text = ""; rs.progress = null;
    if (p.kind === "error") rs.error = p.text; else rs.ok = "Measured. The result is kept for every size on this data.";
    rsMeasInfo(); return;
  }
  if (p.kind === "progress") { rs.text = p.text || p.step; rs.progress = p.value; }
  else if (p.kind === "log") rs.log.push(p.text);
  else if (p.kind === "error") { rs.running = false; rs.testing = null; rs.text = ""; rs.progress = null; rs.error = p.text; rsRefresh(); return; }
  else if (p.kind === "done") {
    const wasTest = rs.testing;
    rs.running = false; rs.testing = null; rs.text = ""; rs.progress = null;
    if (wasTest) rs.ok = "The test is finished. The result is below.";
    else if (p.data && p.data.stopped) rs.ok = "Stopped. The half-written model was removed.";
    else if (p.data) { rs.ok = `Done: ${p.data.name} is ready and is in the model list of Step 3.`; rs.view = p.data.name; }
    setState_(); rsRefresh(); return;
  }
  if (page === "reshape") render();
  const l = document.querySelector("details .log"); if (l) l.scrollTop = l.scrollHeight;
}

function rsSpec() {
  const drop = parseLayers(rs.dropText);
  const spec = { source_model_id: rs.model, mode: rs.mode, blocks: rs.blocks, strategy: rs.strategy, protect_first: rs.pf, protect_last: rs.pl, init: rs.init, calib_samples: rs.samples, calib_len: 512, load: rs.load,
    drop_layers: rs.mode === "prune" && drop && drop.length ? drop : null };
  if (rs.mode === "prune" && rs.shape !== "depth" && rs.wi != null) {          // thinner layers: width only has no blocks
    spec.blocks = rs.shape === "width" ? 0 : rs.blocks;
    spec.width = { intermediate: rs.wi, lin_v_heads: rs.lv ? rsAxis("lin_v_heads") : null, heads: rs.hh || null };
    spec.width_samples = rs.wsamples;
    if (rs.shape === "width") spec.drop_layers = null;
  }
  return spec;
}

// what the Heal card sends
function rsHealSpec() {
  const sp = rs.hlSpec, o = rs.hl && rs.hl.opts;
  return { tokens_m: sp.tokens_m || (o && o.default_tokens_m) || 20, general_share: sp.general ? +sp.share : 0, general: sp.general ? gcSource("heal") : null, alpha: +sp.alpha, lr: +sp.lr, mode: sp.mode || null,
    rank: sp.rank, seq: sp.seq, batch: sp.batch, per_update: sp.per_update, checkpointing: sp.checkpointing || "auto", method: sp.method || null, optimizer: sp.optimizer || null,
    kl_chunk: sp.kl_chunk, k: sp.k, teacher_load: sp.teacher_load || null, warmup: sp.warmup, kbit_upcast: sp.kbit_upcast === true ? true : null };
}

async function rsMeasure(force) {
  rs.running = true; rs.ok = null; rs.error = null; rs.log = []; rs.text = "Starting…"; rs.progress = null; render();
  try { await invoke("reshape_measure", { model: rs.model, samples: rs.wsamples, len: 512, load: rs.load, force }); } catch (e) { rs.running = false; rs.error = String(e); render(); }
}

function wireReshape() {
  if (page !== "reshape" || !rs.state) return;
  const on = (id, fn) => { const e = $(id); if (e) e.onclick = fn; };
  const m = $("rs-model"); if (m) m.onchange = () => { rs.model = m.value; rs.tr.sel = []; rs.tr.analysis = null; rs.tr.planFor = null; rs.tr.resFor = null; rs.dropText = ""; rs.blocks = null; rs.options = null; rs.ok = null; rs.error = null; rsOptions(); };
  document.querySelectorAll("input[name=rs-mode]").forEach((r) => r.onchange = () => { rs.mode = r.value; rs.blocks = null; rs.options = null; rs.ok = null; rsOptions(); });
  document.querySelectorAll("input[name=rs-init]").forEach((r) => r.onchange = () => { rs.init = r.value; });
  document.querySelectorAll("input[name=rs-shape]").forEach((r) => r.onchange = () => { rs.shape = r.value; rs.wi = null; rs.ok = null; rs.options = null; rsOptions(); });
  const sl = $("rs-slider");
  if (sl) {
    sl.oninput = () => {
      const o = rs.options.options[+sl.value]; rs.blocks = o.blocks; $("rs-blocks-n").textContent = o.blocks;
      if (rsWidthOn()) { rsPickWidth(); $("rs-sizeinfo").innerHTML = rsSizeInfo(); $("rs-wsizeinfo").innerHTML = rsWidthInfo(); } else $("rs-sizeinfo").innerHTML = rsSizeInfo();
    };
    sl.onchange = () => render();
  }
  const ws = $("rs-wslider");
  if (ws) {
    ws.oninput = () => { const g = rsWidthGroup(); rs.wi = g.options[+ws.value].intermediate; $("rs-w-n").textContent = thou(rs.wi); $("rs-wsizeinfo").innerHTML = rsWidthInfo(); };
    ws.onchange = () => render();
  }
  const lv = $("rs-lv"); if (lv) lv.onchange = () => { rs.lv = lv.checked; rs.options = null; rsOptions(); };
  const hh = $("rs-hh"); if (hh) hh.onchange = () => { rs.hh = hh.value ? +hh.value : null; rs.options = null; rsOptions(); };
  const wsm = $("rs-wsamples"); if (wsm) wsm.onchange = () => { rs.wsamples = +wsm.value; rsMeasInfo(); };
  const measure = rsMeasure;
  on("rs-measure-now", () => measure(false));
  on("rs-measure-again", () => measure(true));
  const sm = $("rs-samples"); if (sm) sm.onchange = () => { rs.samples = +sm.value; };
  const stg = $("rs-strategy"); if (stg) stg.onchange = () => { rs.strategy = stg.value; };
  const ld = $("rs-load"); if (ld) ld.onchange = () => { rs.load = ld.value; };
  const num = (id, key) => { const e = $(id); if (e) e.onchange = () => { const v = parseInt(e.value, 10); rs[key] = Number.isFinite(v) && v >= 0 && v <= 8 ? v : 1; render(); }; };
  num("rs-pf", "pf"); num("rs-pl", "pl");
  const dr = $("rs-drop"); if (dr) dr.onchange = () => { rs.dropText = dr.value; render(); };
  on("rs-go", async () => {
    rs.running = true; rs.ok = null; rs.error = null; rs.log = []; rs.text = "Starting…"; rs.progress = null; render();
    try { await invoke("reshape_start", { spec: rsSpec() }); } catch (e) { rs.running = false; rs.error = String(e); render(); }
  });
  on("rs-stop", () => { rs.text = "Stopping…"; invoke("reshape_stop"); render(); });
  document.querySelectorAll("[data-rsshow]").forEach((b) => b.onclick = () => { rs.view = b.dataset.rsshow; rsAfterState(); render(); });
  const runTest = (name, target) => async () => {
    rs.testing = name; rs.ok = null; rs.error = null; rs.text = "Starting…"; rs.progress = null; render();
    try { await invoke("reshape_test", { name, judge: false, target }); } catch (e) { rs.testing = null; rs.error = String(e); render(); }
  };
  document.querySelectorAll("[data-rstest]").forEach((b) => b.onclick = runTest(b.dataset.rstest, "slim"));
  document.querySelectorAll("[data-rstest-healed]").forEach((b) => b.onclick = runTest(b.dataset.rstestHealed, "healed"));
  // healing
  const hs = rs.hlSpec, hv = (id, fn) => { const e = $(id); if (e) e.onchange = () => { fn(e); render(); rsHealEstimate(); }; };
  hv("hl-general", (e) => { hs.general = e.checked; }); hv("hl-tokens", (e) => { hs.tokens_m = +e.value; }); 
  hv("hl-share", (e) => { const v = parseFloat(e.value); hs.share = Number.isFinite(v) && v >= 0 && v <= 0.9 ? v : 0.5; });
  hv("hl-alpha", (e) => { const v = parseFloat(e.value); hs.alpha = Number.isFinite(v) && v >= 0 && v <= 1 ? v : 1; });
  const hnum = (id, key, lo, hi) => hv(id, (e) => { const v = parseInt(e.value, 10); hs[key] = Number.isFinite(v) && v >= lo && v <= hi ? v : null; });
  hnum("hl-rank", "rank", 8, 128); hnum("hl-klchunk", "kl_chunk", 64, 512); hnum("hl-k", "k", 8, 128); hnum("hl-warmup", "warmup", 0, 1000); hnum("hl-perupdate", "per_update", 1, 256);
  hv("hl-seq", (e) => { hs.seq = +e.value; }); hv("hl-batch", (e) => { hs.batch = +e.value; }); hv("hl-ckpt", (e) => { hs.checkpointing = e.value; }); hv("hl-method", (e) => { hs.method = e.value || null; });
  hv("hl-optim", (e) => { hs.optimizer = e.value; }); hv("hl-tload", (e) => { hs.teacher_load = e.value === "auto" ? null : e.value; });
  hv("hl-upcast", (e) => { hs.kbit_upcast = e.checked ? true : null; });
  on("hl-reset", () => { Object.assign(hs, HEAL_AUTO); render(); rsHealEstimate(); });
  on("hl-safer", () => { const s = rs.hl.opts.safer; if (!s) return; hs.checkpointing = s.checkpointing; hs.batch = s.batch; hs.per_update = s.per_update; hs.kl_chunk = s.kl_chunk; hs.kbit_upcast = s.kbit_upcast === true ? true : null; render(); rsHealEstimate(); });
  hv("hl-lr", (e) => { const v = parseFloat(e.value); hs.lr = Number.isFinite(v) && v >= 1e-6 && v <= 1e-2 ? v : 1e-4; });
  hv("hl-mode", (e) => { hs.mode = e.value; });
  on("hl-go", async () => {
    rs.hlRun = { running: true, step: "cache", text: "", progress: null, live: [] }; rs.ok = null; rs.error = null; rs.log = []; render();
    try { await invoke("heal_start", { name: rs.hl.name, spec: rsHealSpec() }); } catch (e) { rs.hlRun.running = false; rs.error = String(e); render(); }
  });
  if (rs.hl && rs.hl.opts && hs.general) gcWire("heal", render);
  on("hl-usedefault", () => { if (gc.state) { gcScope("heal").sel = gc.state.default; gcScope("heal").check = null; gcPersist("heal"); render(); } });
  on("hl-stop", () => { if (rs.hlRun) rs.hlRun.text = "Stopping…"; invoke("heal_stop"); render(); });
  on("rs-cleardrop", () => { rs.dropText = ""; render(); });
  wireTry();
  document.querySelectorAll("[data-rstrain]").forEach((b) => b.onclick = () => { tr.model = b.dataset.rstrain; tr.over = {}; tr.plan = null; go("train"); });
  document.querySelectorAll("[data-rsdelete]").forEach((b) => b.onclick = async () => {
    if (!confirm("Delete this model and its files? A training run that used it must be deleted first.")) return;
    try { rs.state = await invoke("reshape_remove", { name: b.dataset.rsdelete }); rs.error = null; rs.ok = "Deleted."; if (rs.view === b.dataset.rsdelete) rs.view = null; } catch (e) { rs.error = String(e); }
    render();
  });
}

// ------------------------------------------------------------------------------------------------------------ guide
let gd = { html: null, report: null, copied: false };
function renderGuide() {
  if (gd.html == null) return `<h1>Guide</h1><p class="lead">Loading…</p>`;
  return `<div class="guide">${gd.html}</div>
    <div class="card wide"><h2>Something is wrong?</h2><p>The help panel on the right lists the usual problems. If it is not there, make a report and send it with your question. It contains no passwords or access keys.</p>
      <div class="row"><button class="primary" id="gd-report">Make a problem report</button>${gd.report ? `<button class="secondary" id="gd-copy">${gd.copied ? "Copied" : "Copy to the clipboard"}</button>` : ""}</div>
      ${gd.report ? `<textarea id="gd-text" rows="14" readonly>${esc(gd.report)}</textarea>` : ""}</div>`;
}
async function gdLoad() { gd.html = await invoke("help", { key: "guide" }); if (page === "guide") render(); }
function wireGuide() {
  if (page !== "guide") return;
  const on = (id, fn) => { const e = $(id); if (e) e.onclick = fn; };
  on("gd-report", async () => { gd.report = await invoke("problem_report"); gd.copied = false; render(); });
  on("gd-copy", async () => { try { await navigator.clipboard.writeText(gd.report); gd.copied = true; } catch (_) { $("gd-text").select(); } render(); });
}

function renderSoon(key) {
  const [t, lead] = TITLES[key];
  return `<h1>${t}</h1><p class="lead">${lead}</p><span class="soon">Coming in a later version. The help panel shows what this step will do.</span>`;
}

function render() {
  if (!state) return;
  renderSteps();
  const el = $("page");
  el.innerHTML = page === "welcome" ? renderWelcome() : page === "data" ? renderData() : page === "setup" ? renderSetup() : page === "dataset" ? renderDataset() : page === "train" ? renderTrain() : page === "reshape" ? renderReshape() : page === "use" ? renderUse() : page === "test" ? renderTest() : page === "guide" ? renderGuide() : renderSoon(page);
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
  wireTest();
  wireReshape();
  wireGuide();
  wireRetrain();
  wireUnsloth();
  on("recheck", refreshEnv);
  const nu = $("no-unsloth"); if (nu) nu.onchange = () => { env.noUnsloth = nu.checked; };
  on("install", () => startEnv("env_install", { unsloth: !env.noUnsloth }));
  on("selftest", () => startEnv("env_selftest"));
  on("cancelenv", () => invoke("env_cancel"));
  on("apply", applyFile);
  document.querySelectorAll(".rmworld").forEach((b) => b.onclick = async () => {
    if (!confirm("Remove this world from the project? Its files stay on disk; make the examples again in Step 2.")) return;
    const s = await call("remove_world", { slug: b.dataset.slug });
    if (s) { setState(s); message = { kind: "ok", text: "World removed. Make the examples again in Step 2." }; }
    render();
  });
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
  const sb = $("studio-book"); if (sb) sb.onchange = () => { ds.studioBook = sb.checked; };
  document.querySelectorAll(".studio-export").forEach((b) => b.onclick = async () => {
    const dest = await tauri.dialog.open({ directory: true, title: "Choose where to put the Unsloth Studio files" });
    if (!dest) return;
    ds.error = null; ds.ok = null;
    const r = await dsCall("studio_export", { version: Number(b.dataset.ver), dest, worldBook: ds.studioBook !== false });
    if (r) ds.ok = `Saved for Unsloth Studio: ${r.dir} (${r.rows} rows${r.eval ? ", " + r.eval + " held-out questions" : ""}). Open its README.md for the settings to choose.`;
    render();
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
  // the teacher
  const tcOpts = () => ({ kinds: [...document.querySelectorAll(".tk")].filter((c) => c.checked).map((c) => c.value), per_source: +$("t-per").value, general_percent: +$("t-gen").value, judge: $("t-judge").checked, drop_partly: $("t-partly").checked, limit: null });
  const tcSave = async () => { const r = await dsCall("teacher_set", { url: $("t-url").value, model: $("t-model").value, key: $("t-key").value, unloadAfter: $("t-unload").checked }); if (r) ds.state = r; return !!r; };
  const tcGo = async (cmd, args) => { ds.tc.opts = tcOpts(); ds.tc.open = $("t-more").open; if (!(await tcSave())) return render(); ds.error = null; ds.ok = null; ds.tc.running = true; ds.tc.text = "Starting…"; ds.tc.progress = null; ds.tc.report = null; render(); try { await invoke(cmd, args); } catch (e) { ds.tc.running = false; ds.error = String(e); render(); } };
  on("t-save", async () => { ds.tc.opts = tcOpts(); ds.ok = (await tcSave()) ? "Teacher saved." : null; render(); });
  on("t-check", () => { ds.tc.check = true; tcGo("teacher_check", {}); });
  on("t-run", () => { ds.tc.check = false; tcGo("teacher_start", { options: tcOpts() }); });
  on("t-trial", () => { ds.tc.check = false; tcGo("teacher_start", { options: { ...tcOpts(), limit: 5 } }); });
  on("t-stop", () => invoke("env_cancel"));
  on("t-clear", async () => { if (!confirm("Remove every example the teacher made? (Your other examples stay. The teacher's stored answers are kept, so making them again costs nothing.)")) return; await dsCall("teacher_clear", { kind: null, forgetCache: false }); dsRefresh(false); });
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
  if (key === "guide") gdLoad();
  if (key === "dataset" && state.bundle) { ds.ok = null; dsRefresh(false); }
  if (key === "train" && state.bundle) trRefresh();
  if (key === "reshape" && state.bundle) { rs.ok = null; rs.error = null; rsRefresh(); }
  if (key === "use" && state.bundle) usRefresh();
  if (key === "test" && state.bundle) { te.ok = null; teRefresh(); }
  if (key === "data" && state.project) { if (!us.state) usRefresh(); rtRefresh(); }
  $("help").innerHTML = await invoke("help", { key: key === "guide" ? "troubleshooting" : key });
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
  try { await tauri.event.listen("power-event", (e) => onPowerEvent(e.payload)); } catch (_) { /* previews */ }
  try { await tauri.event.listen("closing", (e) => { const d = document.createElement("div"); d.id = "closing"; d.textContent = (e.payload && e.payload.text) || "Saving, then closing…"; document.body.appendChild(d); }); } catch (_) { /* previews */ }
  try { await tauri.event.listen("bench-event", (e) => onBenchEvent(e.payload)); } catch (_) { /* previews */ }
  try { await tauri.event.listen("test-event", (e) => onTestEvent(e.payload)); } catch (_) { /* previews */ }
  try { await tauri.event.listen("teacher-event", (e) => onTeacherEvent(e.payload)); } catch (_) { /* previews */ }
  try { await tauri.event.listen("unsloth-event", (e) => onUnslothEvent(e.payload)); } catch (_) { /* previews */ }
  try { await tauri.event.listen("server-event", (e) => onServerEvent(e.payload)); } catch (_) { /* previews */ }
  try { await tauri.event.listen("export-event", (e) => onExportEvent(e.payload)); } catch (_) { /* previews */ }
  try { await tauri.event.listen("train-event", (e) => onTrainEvent(e.payload)); } catch (_) { /* previews */ }
  try { await tauri.event.listen("reshape-event", (e) => onReshapeEvent(e.payload)); } catch (_) { /* previews */ }
  try { await tauri.event.listen("env-event", (e) => onEnvEvent(e.payload)); } catch (_) { /* events are optional in previews */ }
  go("welcome");
}
init();
