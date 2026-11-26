// ND Lore Forge window. All logic lives in Rust (crates/ndforge-core); this file only draws the state it returns.
const tauri = window.__TAURI__;
const invoke = tauri.core.invoke;
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const TITLES = {
  welcome: ["Welcome to ND Lore Forge", "Teach a small AI model your nd-world world. Start by creating a project (one project = one world)."],
  test: ["Step 4 · Test", "Checks that the trained model knows your world better than the original."],
  imatrix: ["Importance matrices", "Measure which weights of a model matter most, keep the result, use it in exports and copy it elsewhere."],
  quantcheck: ["Compare quants", "Measure how far each smaller GGUF file is from the 16-bit model, as KL divergence, and see which size keeps the most."],
  tune: ["Tune this PC for a model", "Find the fastest settings for a model on this computer, test by test, and keep them as a preset you can select again."],
  storage: ["Storage", "See where the disk went and delete what you do not need. Deleting is permanent: the files are not moved to the Recycle Bin."],
  auto: ["Auto", "One click: measures, cuts, heals, checks and exports a model. It runs the same steps as the manual pages, one after the other."],
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

// Auto-cut shares the state of the Slim page (`rs`): what refreshes the Slim page refreshes the Auto page too.
const onAutoPage = () => page === "auto";
const onReshape = () => page === "reshape" || page === "auto";

function setState(s) { if (s) { state = s; } }

function renderSteps() {
  const items = [{ key: "welcome", label: "Welcome", unlocked: true, done: false }, ...state.steps];
  items.push({ key: "auto", label: "🚀 Auto", unlocked: !!state.bundle, done: false });
  items.push({ key: "reshape", label: "🧬 Slim or grow a model", unlocked: !!state.bundle, done: false });
  items.push({ key: "imatrix", label: "🧮 Importance matrices", unlocked: !!state.bundle, done: false });
  items.push({ key: "quantcheck", label: "📏 Compare quants", unlocked: !!state.bundle, done: false });
  items.push({ key: "tune", label: "🎛 Tune", unlocked: !!state.bundle, done: false });
  items.push({ key: "storage", label: "🧹 Storage", unlocked: !!state.project, done: false });
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
    ${recent ? `<h2>Recent projects</h2><ul class="recent">${recent}</ul>` : ""}${quCard()}
    ${state.project ? `<div class="row"><label class="opt" title="An Auto-cut or a healing that was cut off by closing the program (or a crash) is put in the queue and goes on where it stopped, when the project is opened."><input type="checkbox" id="auto-continue" ${state.settings.auto_continue_interrupted ? "checked" : ""}> Go on by itself with what was cut off when I open the project</label></div>
    <div class="row"><label class="opt" title="Also keeps the screen on while a long job runs. On a computer with Modern Standby this may be the only way to stop it from sleeping. Needs Keep this computer awake."><input type="checkbox" id="keep-display" ${state.settings.keep_display_on ? "checked" : ""}> Keep the screen on too while a long job runs</label></div>
    <div class="row"><label class="opt"><input type="checkbox" id="notify-on" ${state.settings.notifications !== false ? "checked" : ""}> Tell me when a job or the queue has finished (a message from the system)</label></div>
    <div class="row"><label>Web address that gets one line when a job ends (optional, for example an ntfy topic) <input type="text" id="webhook-url" value="${esc(state.settings.webhook_url || "")}" placeholder="https://ntfy.sh/my-topic" style="min-width:280px"></label><button class="secondary" id="webhook-save">Save</button></div>
    <p class="dim small">The line says only what kind of job it was and how it ended, never a name or a folder.</p>` : ""}`;
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
      ${tc.running || tc.text ? `<div class="stepline" aria-live="polite">${esc(tc.text)}</div>${tc.progress != null ? `<progress max="1" value="${tc.progress}"></progress>` : ""}` : ""}${rep}</div>`;
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
    ${b.running || b.text ? `<div class="stepline" aria-live="polite">${esc(b.text)}</div>${b.progress != null ? `<progress max="1" value="${b.progress}"></progress>` : ""}` : ""}${b.error ? `<div class="err">${esc(b.error)}</div>` : ""}
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
  const ctrl = `${pfBox("train", { server: tr.server || "" })}<div class="row"><button class="primary" id="tr-start" ${tr.running || !tr.model || (!tr.server && !st.env_ready) || (tr.server && !(srvStatus && srvStatus.gpu && !srvStatus.busy)) || pfBlocked("train") ? "disabled" : ""}>${tr.server && srv ? "Start training on " + esc(srv.name) : "Start training"}</button>
    <button class="secondary" id="tr-stop" title="Finishes the current step, saves a checkpoint and stops. The run can be continued later, even after closing the program." ${tr.running && !tr.stopping ? "" : "disabled"}>${tr.stopping ? "Stopping…" : "Stop and save"}</button>
    ${canResume ? `<button class="secondary" id="tr-resume" data-run="${esc(last.name)}">Continue the last run (${esc(last.model)}${last.step ? `, from step ${last.step}${last.total_steps ? " of " + last.total_steps : ""}` : ""})</button>` : ""}</div>
    <div class="row power"><label class="opt" title="Stops the computer from going to sleep while a training runs (the screen may still turn off). Applies from the next start."><input type="checkbox" id="tr-awake" ${state.settings.keep_awake ? "checked" : ""} ${tr.running ? "disabled" : ""}> Keep this computer awake during training</label>
      <label class="opt" title="When a job has finished normally and the queue has nothing more to do, the computer is switched off after a 60-second warning that you can cancel. Not after a stop or an error. Switches itself off again after use."><input type="checkbox" id="tr-shutdown" ${tr.shutdownAfter ? "checked" : ""}> Shut down this computer when the training and the queue have finished</label></div>`;
  const env = !st.env_ready ? `<div class="err">The training software is not installed yet. Open "⚙ Training setup" first.</div>` : "";
  const live = (tr.running || tr.metrics.length || tr.error || tr.notice) ? `<div class="card wide"><h2>${tr.running ? "Training…" : "Last run"}</h2>
    ${tr.stepText ? `<div class="stepline" aria-live="polite">${esc(tr.stepText)}</div>` : ""}${tr.progress != null ? `<progress max="1" value="${tr.progress}"></progress>` : ""}
    ${tr.error ? errDiv(tr.error) : ""}${tr.notice ? `<div class="ok">${esc(tr.notice)}</div>` : ""}
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
  bar.innerHTML = `<span>Everything has finished. This computer shuts down in <strong>${p.left}</strong> s.</span> <button id="pw-cancel">Cancel shutdown</button>`;
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
// ------------------------------------------------------------------------------------------- mixed precision (our own recipe): docs/MIXED_QUANTS.md
function usQuantBox(q) {
  return `<label class="opt" title="${esc(q.name)}"><input type="checkbox" class="us-q" value="${q.id}" ${us.quants.includes(q.id) ? "checked" : ""} ${us.what === "merged" ? "disabled" : ""}> ${esc(q.label || q.name.split(":")[0])}</label>`;
}

// the file whose tensor shapes the size estimate reads (any GGUF of the model: only its header is read): a file of this run's exports, else any, else a typed path
function usMixedFile(st) {
  const files = (st.mixed && st.mixed.files) || [];
  if (us.mx.file === "__typed") return us.mx.typed.trim();
  if (us.mx.file) return us.mx.file;
  const mine = files.find((f) => f.run === us.run) || files[0];
  return mine ? mine.path : "";
}

function mixedPlanHtml() {
  const pl = us.mx.plan;
  if (us.mx.busy) return `<p class="dim small">Reading the model's tensor shapes…</p>`;
  if (us.mx.error) return `<div class="warn small" id="us-mxerror">${esc(us.mx.error)}</div>`;
  if (!pl) return "";
  const rows = pl.presets.map((p) => `<tr><td>${esc(p.label)}</td><td>${p.estimate.gb} GB</td><td>${p.estimate.bpw != null ? p.estimate.bpw : "–"}</td><td>${esc(p.base)}</td></tr>`).join("");
  const plain = pl.plain_bpw || {};
  const rules = pl.presets.map((p) => `<details><summary>${esc(p.label)}: the rules llama-quantize gets</summary><pre class="log">${[`--output-tensor-type ${p.output}`, `--token-embedding-type ${p.embd}`].concat(p.rules.map((r) => `--tensor-type ${r}`)).map(esc).join("\n")}\n(base type ${esc(p.base)})</pre>${p.notes.map((n) => `<p class="dim small">${esc(n)}</p>`).join("")}</details>`).join("");
  const fit = pl.fit ? (pl.fit.ok ? `<div class="small" id="us-mxfit"><strong>${esc(pl.fit.label)}:</strong> about ${pl.fit.estimate.gb} GB (${pl.fit.estimate.bpw} bits per weight) for the ${pl.fit.target_bytes ? (pl.fit.target_bytes / 1e9).toFixed(2) : ""} GB asked${pl.fit.fits ? "" : ", <strong>which the lowest types cannot reach</strong>"}.
      ${(pl.fit.notes || []).map((n) => `<br>${esc(n)}`).join("")}<details><summary>The ${pl.fit.rules.length} rules this gives</summary><pre class="log">${pl.fit.rules.map((r) => esc("--tensor-type " + r)).join("\n")}</pre></details></div>`
    : `<div class="warn small" id="us-mxfiterror">${esc(pl.fit.error)}</div>`) : "";
  return `<div id="us-mxplan"><table class="qk-table"><thead><tr><th>${esc(pl.recipe)}</th><th>Estimated size</th><th>Bits per weight</th><th>Base type</th></tr></thead><tbody>${rows}</tbody></table>
    <p class="dim small">Estimated from the ${pl.params ? (pl.params / 1e9).toFixed(1) + " billion" : ""} weights and shapes in the file's header (architecture ${esc(pl.arch || "unknown")}, ${pl.n_layer != null ? pl.n_layer : "?"} layers). For comparison, llama.cpp's own table gives the plain sizes about ${plain.Q4_K_M} (Q4_K_M), ${plain.Q3_K_M} (Q3_K_M) and ${plain.Q2_K} (Q2_K) bits per weight on a typical model; a plain size's real figure depends on the model. The real size is shown after the export; <strong>Compare quants</strong> says how good each one is.</p>
    ${rules}${fit}</div>`;
}

function mixedBox(st) {
  if (us.what === "merged") return "";
  const mixed = st.mixed_quants || [];
  if (!mixed.length) return "";
  const any = mixed.some((q) => us.quants.includes(q.id));
  const files = (st.mixed && st.mixed.files) || [];
  const boxes = mixed.map(usQuantBox).join(" ");
  const fitOn = us.quants.includes("mixed_fit");
  const measuredOn = us.quants.includes("mixed_measured");
  const lib = (st.imatrix && st.imatrix.library) || [];
  const fileOpts = files.map((f) => `<option value="${esc(f.path)}" ${usMixedFile(st) === f.path ? "selected" : ""}>${esc(f.label)}</option>`).join("") + `<option value="__typed" ${us.mx.file === "__typed" ? "selected" : ""}>A GGUF file I choose…</option>`;
  const detail = any ? `${us.im ? "" : llamaNotice("all")}<p class="dim small" id="us-mxwhy">Made with llama.cpp: <code>llama-quantize</code> gives single tensors their own types. If Unsloth is the back end that would make the files, llama.cpp makes the <em>whole batch</em> instead (Unsloth cannot do per-tensor types) and the result says so. More bits go to the embedding and output tensors, the attention k / v / output, the linear-attention projections, ffn_down and the first and last layers; fewer to the ffn_up / gate of the middle layers. The importance matrix, when you use one, is applied to these sizes too. <strong>${esc((st.mixed || {}).recipe || "Mixed precision (our own recipe)")}</strong>: they are our own recipes, not copies of anyone else's, and nobody has measured yet whether they beat the plain sizes: that is what <strong>Compare quants</strong> is for.</p>
    <div class="row"><label>Estimate the size from <select id="us-mxfile">${fileOpts}</select></label>${us.mx.file === "__typed" ? `<input type="text" id="us-mxtyped" value="${esc(us.mx.typed)}" placeholder="C:\\models\\model-Q4_K_M.gguf" style="min-width:260px">` : ""}<button class="secondary" id="us-mxplanbtn" ${us.mx.busy || !usMixedFile(st) ? "disabled" : ""}>Show the estimated sizes</button></div>
    ${files.length || us.mx.file === "__typed" ? "" : `<p class="dim small">No GGUF of this model exists yet. Choose any GGUF file of it (any size, only its tensor shapes are read) to see the estimated sizes, or make the files and read the real ones afterwards.</p>`}
    ${fitOn ? `<div class="row" id="us-mxfitbox"><strong>Fit to a size (experimental)</strong> <label>Target size <input type="text" id="us-mxtarget" value="${esc(us.mx.target)}" placeholder="12.5" style="min-width:80px;flex:0"> GB</label>
      <label>Statistics from <select id="us-mxmatrix"><option value="">the matrix of this export</option>${lib.map((e) => `<option value="${esc(e.id)}" ${us.mx.matrix === e.id ? "selected" : ""}>${esc(e.label || e.id)}</option>`).join("")}</select></label><button class="secondary" id="us-mxfitbtn" ${us.mx.busy || !usMixedFile(st) || !(+us.mx.target > 0) ? "disabled" : ""}>Work out the rules</button></div>
      <p class="dim small">Experimental, and said so: every large tensor starts at the lowest type and the ones the matrix says matter most per extra byte get more bits until the size is used. It is a <strong>guess</strong> from the matrix's per-tensor statistics, not a measurement. It needs a saved matrix (without one, the closest recipe is used instead and the result says so). Press the button to see the rules before you export.</p>${!(+us.mx.target > 0) ? `<p class="warn small" id="us-mxtargetwhy">Give the size in GB that the fit should aim for.</p>` : ""}` : ""}
    ${measuredOn ? measuredFields(lib) : ""}
    ${mixedPlanHtml()}` : "";
  return `<div class="row mixedbox"><span class="quants"><strong>${esc((st.mixed || {}).recipe || "Mixed precision (our own recipe)")}:</strong> ${boxes}</span></div>${detail}${qfBox()}`;
}

// ------------------------------------------------------------------------------------------------ Mixed: measured fit to a size (docs/MIXED_QUANTS.md "Fit to a size (measured)")
const QF_LADDER = ["Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_0"];
const qf = { open: false, state: null, loading: false, file: "", typed: "", matrix: "", target: "", floor: "Q3_K", ceiling: "Q8_0", rows: 128, engine: "auto", est: null, estBusy: false, estError: "",
  running: false, step: "", progress: null, log: [], result: null, error: "", ok: "", stopped: false, make: false, compare: false, text: "data" };

function qfFile() {
  if (qf.file === "__typed") return qf.typed.trim();
  if (qf.file) return qf.file;
  const r = ((qf.state || {}).references || [])[0];
  return r ? r.path : "";
}

function qfSpec(extra) {
  return { gguf: qfFile(), matrix: qf.matrix || null, target_gb: Number(qf.target) || 0, floor: qf.floor, ceiling: qf.ceiling, sample_rows: Number(qf.rows) || 0, engine: qf.engine,
    make: qf.make, neighbours: qf.make, compare: qf.make && qf.compare ? { kind: qf.text === "general" ? "general" : "data", run: us.run || null } : null, ...(extra || {}) };
}

const qfGb = (n) => (n == null ? "?" : (+n).toFixed(2));

function qfResultHtml() {
  const r = qf.result;
  if (!r) return "";
  const rep = r.report || {};
  const rows = (rep.rows || []).map((x) => `<tr${x.id === "plan" ? ' class="qk-front"' : ""}><td>${esc(x.label)}</td><td>${qfGb(x.gb)} GB</td><td>${x.error}</td></tr>`).join("");
  const types = Object.entries(rep.types || {}).map(([t, v]) => `${esc(t)}: ${v.tensors} tensors`).join(", ");
  const roles = Object.entries(rep.bpw_by_role || {}).map(([k, v]) => `${esc(k)} ${v}`).join(" · ");
  const cmp = r.comparison || null;
  const prom = ((cmp && cmp.promotion) || []).map((x) => `<p class="${x.promoted ? "ok" : "dim"} small" id="qf-promotion">${esc(x.reason)}</p>`).join("");
  const verdict = cmp && cmp.verdict ? `<ul class="small" id="qf-verdict">${cmp.verdict.map((v) => `<li>${esc(v)}</li>`).join("")}</ul>` : "";
  return `<div id="qf-result"><p class="small"><strong>${r.fits ? "Plan" : "The size is below the smallest this model can be"}:</strong> ${qfGb((r.plan || {}).total / 1e9)} GB (measured with the ${esc(r.engine)} engine${r.fits ? "" : ", the smallest plan is shown"}).</p>
    <table class="qk-table"><thead><tr><th>${esc("Plan against the plain sizes")}</th><th>Size</th><th>Predicted weighted error</th></tr></thead><tbody>${rows}</tbody></table>
    <p class="dim small">Types chosen: ${types}.<br>Bits per weight by role: ${roles}.</p>
    <details><summary>The ${(r.rules || []).length} rules this gives</summary><pre class="log">${(r.flags || []).slice(0, 4).concat((r.rules || []).map((x) => "--tensor-type " + x)).map(esc).join("\n")}</pre></details>
    ${(r.notes || []).map((n) => `<p class="warn small">${esc(n)}</p>`).join("")}
    <p class="dim small" id="qf-caveat">${esc(rep.caveats || (qf.state || {}).caveats || "")}</p>${prom}${verdict}</div>`;
}

function qfBox() {
  const q = qf.state;
  const refs = (q && q.references) || [];
  const lib = ((us.state || {}).imatrix || {}).library || [];
  const opts = (list, cur) => list.map((x) => `<option value="${x}" ${cur === x ? "selected" : ""}>${x}</option>`).join("");
  const fileOpts = refs.map((f) => `<option value="${esc(f.path)}" ${qfFile() === f.path ? "selected" : ""}>${esc(f.label)}</option>`).join("") + `<option value="__typed" ${qf.file === "__typed" ? "selected" : ""}>A 16-bit GGUF I choose…</option>`;
  const e = qf.est;
  const plain = e && e.plain_estimates ? e.plain_estimates : {};
  const gb = Number(qf.target) || 0;
  const why = !qfFile() ? "Choose the 16-bit GGUF file." : !(gb > 0) ? "Give the size in GB." : "";
  const below = e && e.min_gb != null && gb > 0 && gb < e.min_gb ? `<p class="warn small" id="qf-below">${gb} GB is below the smallest this model can be with ${esc(qf.floor)} as the lowest type (${qfGb(e.min_gb)} GB). Ask for more, or lower the lowest type.</p>` : "";
  const estHtml = qf.estBusy ? `<p class="dim small">Reading the file's header…</p>` : qf.estError ? `<p class="warn small" id="qf-esterror">${esc(qf.estError)}</p>` : e ? `<div class="small" id="qf-est"><strong>${esc(e.minutes_note || "")}</strong>
      <br>Engine: ${esc(e.engine)}${e.llama_quantize ? ` (llama-quantize found${e.llama_version ? ", " + esc(e.llama_version) : ""}; <code>--tensor-type-file</code> ${e.tensor_type_file ? "yes" : "no"})` : " (llama-quantize not found: the numpy engine measures; making the file needs llama.cpp)"}.
      <br>Looks at ${(e.sampled_weights / 1e6).toFixed(0)} million weights (${(e.read_bytes / 1e9).toFixed(2)} GB read)${e.disk_need ? `, needs about ${(e.disk_need / 1e9).toFixed(1)} GB of free disk for its sample file` : ""}. Smallest ${qfGb(e.min_gb)} GB, largest ${qfGb(e.max_gb)} GB (the vision file is separate and not counted). Plain sizes: ${Object.entries(plain).map(([k, v]) => `${esc(k)} about ${v} GB`).join(", ")}.
      ${(e.notes || []).map((n) => `<br><span class="warn">${esc(n)}</span>`).join("")}</div>` : "";
  const prog = qf.running ? `<div class="progress"><div style="width:${Math.round((qf.progress || 0) * 100)}%"></div></div><p class="dim small" id="qf-step">${esc(qf.step)}</p>` : "";
  const busy = qf.running;
  return `<details id="qf-box" ${qf.open ? "open" : ""}><summary><strong>Fit to a size (measured)</strong> · each tensor's error is measured on your weights</summary>
    <p class="dim small">Starts from a <strong>16-bit GGUF</strong> (the <code>model-f16.gguf</code> an export makes) and an importance matrix. It measures what every type (Q2_K to Q8_0) does to every tensor on a sample of rows, weighted by the matrix, and gives each tensor the type that spends the bytes best for the size you ask. It is a <strong>measurement of the weights, not of the model</strong>: <strong>Compare quants</strong> (KL divergence) is the judge.</p>
    ${q ? "" : `<p class="dim small">Loading…</p>`}
    <div class="row"><label>16-bit file <select id="qf-file">${fileOpts}</select></label>${qf.file === "__typed" ? `<input type="text" id="qf-typed" value="${esc(qf.typed)}" placeholder="C:\\models\\model-f16.gguf" style="min-width:260px">` : ""}
      <label>Importance matrix <select id="qf-matrix"><option value="">none (every channel counts the same)</option>${lib.map((m) => `<option value="${esc(m.id)}" ${qf.matrix === m.id ? "selected" : ""}>${esc(m.label || m.id)}</option>`).join("")}</select></label></div>
    <p class="dim small">For a model you cut or healed, use the matrix adapted to it (Importance matrices page, Adapt); a matrix of another model is refused.</p>
    <div class="row"><label>Target size <input type="text" id="qf-target" value="${esc(qf.target)}" placeholder="10.6" style="min-width:80px;flex:0"> GB</label>${gb > 0 ? `<span class="dim small" id="qf-gib">= ${(gb * 1e9 / 1073741824).toFixed(2)} GiB</span>` : ""}
      <button class="secondary" id="qf-q4" ${plain.Q4_K_M ? "" : "disabled"} title="Needs the estimate">same size as Q4_K_M</button><button class="secondary" id="qf-q5" ${plain.Q5_K_M ? "" : "disabled"} title="Needs the estimate">same size as Q5_K_M</button></div>
    <details><summary>More settings</summary><div class="row"><label>Rows looked at per tensor <select id="qf-rows">${[64, 128, 256, 512].map((n) => `<option value="${n}" ${+qf.rows === n ? "selected" : ""}>${n}</option>`).join("")}</select></label>
      <label>Lowest type <select id="qf-floor">${opts(QF_LADDER, qf.floor)}</select></label><label>Highest type <select id="qf-ceiling">${opts(QF_LADDER, qf.ceiling)}</select></label>
      <label>Measured with <select id="qf-engine">${opts(["auto", "llama", "reference"], qf.engine)}</select></label></div></details>
    <div class="row"><button class="secondary" id="qf-estbtn" ${busy || qf.estBusy || !qfFile() ? "disabled" : ""}>Estimate</button>
      <button class="primary" id="qf-run" ${busy || why ? "disabled" : ""}>${qf.stopped ? "Resume" : "Measure and plan"}</button>${busy ? `<button class="secondary" id="qf-stop">Stop</button>` : ""}</div>
    <div class="row"><label class="opt"><input type="checkbox" id="qf-make" ${qf.make ? "checked" : ""}> Make the file (and the plain sizes next to it)</label>
      <label class="opt"><input type="checkbox" id="qf-compare" ${qf.compare ? "checked" : ""} ${qf.make ? "" : "disabled"}> and compare them</label>
      <label>on <select id="qf-text" ${qf.make && qf.compare ? "" : "disabled"}><option value="data" ${qf.text === "data" ? "selected" : ""}>the held-out part of my saved data</option><option value="general" ${qf.text === "general" ? "selected" : ""}>general text</option></select></label></div>
    ${why ? `<p class="dim small" id="qf-why">${esc(why)}</p>` : ""}${below}${estHtml}${prog}
    ${qf.error ? `<div class="warn small" id="qf-error">${esc(qf.error)}</div>` : ""}${qf.ok ? `<p class="ok small" id="qf-ok">${esc(qf.ok)}</p>` : ""}${qfResultHtml()}
    ${q && q.verdict && !qf.result ? `<p class="dim small" id="qf-lastverdict">Last verdict (${esc(String(q.verdict.target_gb))} GB): ${esc(((q.verdict.verdict || [])[0]) || "")}</p>` : ""}</details>`;
}

async function qfLoad() {
  if (qf.loading) return;
  qf.loading = true;
  try { qf.state = await invoke("quantfit_state"); } catch (e) { qf.error = String(e); }
  qf.loading = false;
  if (page === "use") render();
}

async function qfEstimate() {
  qf.estBusy = true; qf.estError = ""; qf.est = null; render();
  try { qf.est = await invoke("quantfit_estimate", { spec: qfSpec({ make: false, neighbours: false, compare: null }) }); } catch (e) { qf.estError = String(e); }
  qf.estBusy = false; render();
}

async function qfRun() {
  qf.running = true; qf.error = ""; qf.ok = ""; qf.result = null; qf.log = []; qf.step = "Starting…"; qf.progress = null; qf.stopped = false; render();
  try { await invoke("quantfit_start", { spec: qfSpec() }); } catch (e) { qf.running = false; qf.error = String(e); qf.step = ""; render(); }
}

function onQuantfitEvent(p) {
  if (p.kind === "log") qf.log.push(p.text);
  else if (p.kind === "progress") { if (p.text) qf.step = p.text; if (p.value != null) qf.progress = p.value; }
  else if (p.kind === "done") {
    qf.running = false; qf.progress = null; qf.step = ""; qf.stopped = false; qf.result = p.data || null; qf.error = "";
    qf.ok = qf.result && qf.result.files && qf.result.files.length ? "The files are made." : "The plan is ready.";
    qfLoad();
  } else if (p.kind === "error") {
    qf.running = false; qf.progress = null; qf.step = "";
    if (p.text === "Cancelled.") { qf.stopped = true; qf.error = "Stopped. What was measured is kept: press Resume to go on."; } else qf.error = p.text;
  }
  if (page === "use") render();
}

function wireQuantfit() {
  const on = (id, fn) => { const e = $(id); if (e) e.onclick = fn; };
  const sel = (id, key, redraw) => { const e = $(id); if (e) e.onchange = () => { qf[key] = e.value; qf.stopped = false; if (redraw) render(); }; };
  const d = $("qf-box"); if (d) d.ontoggle = () => { qf.open = !!d.open; if (qf.open && !qf.state) qfLoad(); };
  if (qf.open && !qf.state && !qf.loading) qfLoad();
  sel("qf-file", "file", true); sel("qf-typed", "typed", true); sel("qf-matrix", "matrix", false); sel("qf-rows", "rows", false); sel("qf-floor", "floor", true); sel("qf-ceiling", "ceiling", false); sel("qf-engine", "engine", false); sel("qf-text", "text", false);
  const tg = $("qf-target"); if (tg) tg.onchange = () => { qf.target = tg.value.trim(); qf.stopped = false; render(); };
  const mk = $("qf-make"); if (mk) mk.onchange = () => { qf.make = mk.checked; if (!qf.make) qf.compare = false; render(); };
  const cp = $("qf-compare"); if (cp) cp.onchange = () => { qf.compare = cp.checked; render(); };
  on("qf-q4", () => { if (qf.est && qf.est.plain_estimates.Q4_K_M) { qf.target = String(qf.est.plain_estimates.Q4_K_M); render(); } });
  on("qf-q5", () => { if (qf.est && qf.est.plain_estimates.Q5_K_M) { qf.target = String(qf.est.plain_estimates.Q5_K_M); render(); } });
  on("qf-estbtn", qfEstimate);
  on("qf-run", qfRun);
  on("qf-stop", () => invoke("quantfit_stop"));
}

// the measured size on the export page: the target and the matrix that weights it (the same fields as "Fit to a size")
function measuredFields(lib) {
  return `<div class="row" id="us-mmbox"><strong>Mixed: measured fit to a size</strong> <label>Target size <input type="text" id="us-mmtarget" value="${esc(us.mm.target)}" placeholder="10.6" style="min-width:80px;flex:0"> GB</label>
      <label>Importance matrix <select id="us-mmmatrix"><option value="">the matrix of this export</option>${lib.map((e) => `<option value="${esc(e.id)}" ${us.mm.matrix === e.id ? "selected" : ""}>${esc(e.label || e.id)}</option>`).join("")}</select></label></div>
      <p class="dim small">Measures what every type does to every tensor of this model after it is converted (a sample of the rows, weighted by the matrix) and gives each tensor the type that spends the bytes best. The measures are kept, so a second size is quick. If it cannot be done, the guessed fit is used and the result says so. Judge it with <strong>Compare quants</strong>.</p>${!(+us.mm.target > 0) ? `<p class="warn small" id="us-mmtargetwhy">Give the size in GB that the measured fit should aim for.</p>` : ""}`;
}

async function usMixedPlan(withFit) {
  const st = us.state, file = usMixedFile(st);
  us.mx.busy = true; us.mx.error = ""; us.mx.plan = null; render();
  try { us.mx.plan = await invoke("mixed_plan", { gguf: file, targetGb: withFit ? Number(us.mx.target) : null, matrix: withFit && us.mx.matrix ? us.mx.matrix : null }); } catch (e) { us.mx.error = String(e); }
  us.mx.busy = false; render();
}


let us = { mx: { file: "", typed: "", target: "", matrix: "", plan: null, busy: false, error: "", key: "" }, mm: { target: "", matrix: "" }, state: null, run: "", what: "gguf", quant: "q4_k_m", quants: ["q4_k_m"], numCtx: 8192, im: false, imChunks: 128, imReuse: null, imNextn: true, imSaved: "", imCheck: null, imAdd: false, imAddLabel: "", server: "", modelName: "", running: false, step: "", progress: null, log: [], error: null, ok: null, probe: null, check: null, adding: false };
const copyBtn = (text) => `<button class="link" data-copy="${esc(text)}">Copy</button>`;

function nextnHtml(im) {
  const n = nextnState(im);
  if (!n.head) return "";
  return `<div class="row"><label class="opt ${n.unsupported ? "dim" : ""}" title="This model has a speculative-decoding head (MTP). The matrix can measure that head too (llama-imatrix option --nextn)."><input type="checkbox" id="us-imnextn" ${n.on ? "checked" : ""} ${n.unsupported ? "disabled" : ""}> Include the speculative-decoding head (MTP) in the matrix</label></div>
    ${n.unsupported ? `<p class="dim small">Not available: ${esc(im.nextn_reason || "this llama.cpp has no --nextn option (an older build)")}. The matrix is made without the head.</p>` : `<p class="dim small">Used only if your llama-imatrix has <code>--nextn</code>; if it refuses, the matrix is made without the head and the page says why.</p>`}`;
}

// the importance-matrix box of the export page (GGUF exports only): off unless ticked; "Reuse the saved matrix" is on when a matrix of this run exists
// the speculative-decoding (MTP) head: the tick exists only for a model that has one (`im.mtp[run]` is true); greyed with the reason when llama-imatrix is known not to support it
function nextnState(im) {
  const head = ((im && im.mtp) || {})[us.run] === true;
  const unsupported = head && im.nextn_supported === false;
  return { head, unsupported, on: head && !unsupported && us.imNextn };
}

// "Use a saved matrix" (the matrix library, page "Importance matrices"): the export measures nothing and gives that file to llama-quantize / Unsloth as a fresh one
function savedPick(im) {
  const lib = im.library || [];
  const opts = lib.map((e) => `<option value="${esc(e.id)}" ${us.imSaved === e.id ? "selected" : ""}>${esc(e.label || e.id)} · ${esc(e.model || "other model")} · ${e.chunks != null ? e.chunks + " chunks" : "unknown amount"} · ${esc(String(e.created || "").slice(0, 10))}</option>`).join("");
  const add = us.imAdd ? `<div class="row"><label>Label <input type="text" id="us-imaddlabel" value="${esc(us.imAddLabel)}" maxlength="80" placeholder="for example: from my friend, Q4 test"></label><button class="secondary" id="us-imaddfile">Choose the matrix file…</button><button class="link" id="us-imaddcancel">Cancel</button></div>
    <p class="dim small">A .gguf matrix (the current format) or a legacy .dat file, for example one made on another computer. It is copied into the library of this project.</p>` : "";
  return `<div class="row"><label>Use a saved matrix <select id="us-imsaved"><option value="">Measure a new one now</option>${opts}</select></label>
      <button class="secondary" id="us-imadd">Add a matrix file from my computer…</button><button class="link" id="us-imlib">Open the matrix library</button></div>${add}
    ${lib.length ? "" : `<p class="dim small">The library is empty. Make a matrix on its own on the "Importance matrices" page, or add a file.</p>`}`;
}
function savedNote() {
  const c = us.imCheck && us.imCheck.id === us.imSaved && us.imCheck.run === us.run ? us.imCheck : null;
  const warn = c && c.warnings && c.warnings.length ? `<div class="warn small" id="us-imwarn"><strong>Check before you export.</strong> ${c.warnings.map(esc).join("<br>")}</div>` : c && c.error ? `<div class="warn small" id="us-imwarn">${esc(c.error)}</div>` : "";
  return `<p class="dim small">Nothing is measured: this matrix is copied into the export and given to <code>llama-quantize</code> (or Unsloth) as it is, for the sizes below Q8_0. It should have been made from the same model.</p>${warn}`;
}
function usCheckSaved() {
  if (!us.imSaved || !us.run) return;
  const key = us.imSaved + "|" + us.run;
  if (us.imCheckKey === key) return;
  us.imCheckKey = key;
  invoke("imatrix_check", { id: us.imSaved, run: us.run })
    .then((r) => { us.imCheck = { id: us.imSaved, run: us.run, ...(r || {}) }; if (page === "use") render(); })
    .catch((e) => { us.imCheck = { id: us.imSaved, run: us.run, error: String(e), warnings: [] }; if (page === "use") render(); });
}

function imatrixBox(st) {
  const im = st.imatrix;
  if (!im || us.what === "merged") return "";
  gcUse(im.general);
  const saved = (im.saved || {})[us.run];
  const reuse = us.imReuse === null ? !!saved : us.imReuse;
  const helps = us.quants.some((q) => !(im.no_matrix_for || []).includes(q));
  const est = ((im.estimates || {})[us.run] || {})[String(us.imChunks)];
  const names = { 64: "64 chunks (quick)", 128: "128 chunks (recommended)", 256: "256 chunks (thorough)" };
  const measure = `<div class="row"><label>Calibration text <select id="us-imchunks">${im.chunks.map((n) => `<option value="${n}" ${us.imChunks === n ? "selected" : ""}>${names[n] || n + " chunks"}</option>`).join("")}</select></label>
      <span class="dim">${est ? `${est.tokens.toLocaleString("en")} tokens, about ${est.gpu_minutes} min on a graphics card (up to ${est.cpu_minutes} min on the processor): a rough guess, the log shows the real progress` : ""}</span></div>
    ${nextnHtml(im)}${gcHtml("imatrix")}<p class="dim small">About ${Math.round((im.general_share || 0.2) * 100)} % of the calibration text is general text from the dataset above; the rest is your own data. A matrix is only reused for the same dataset.</p>
    ${saved ? `<label class="opt" title="${esc(saved.dir)}"><input type="checkbox" id="us-imreuse" ${reuse ? "checked" : ""}> Reuse the saved matrix (${saved.chunks} chunks, made ${esc(String(saved.created || "").slice(0, 10))}, if it still fits this model and data)</label>` : `<p class="dim">No saved matrix of this run yet: it will be made now and kept next to the files.</p>`}
    ${helps ? "" : `<p class="dim">None of the chosen sizes is below Q8_0, so there is nothing to improve: the matrix would not be made.</p>`}`;
  const detail = us.im ? savedPick(im) + (us.imSaved ? savedNote() : measure) : "";
  return `<div class="row imatrix"><label class="opt" title="llama.cpp reads about ${im.ctx * im.default_chunks} tokens of your data and some general text and learns which weights matter most; the small sizes then keep those more exactly"><input type="checkbox" id="us-im" ${us.im ? "checked" : ""}> Better small sizes (importance matrix)</label></div>
    <p class="dim">Makes Q6_K, Q5_K_M and Q4_K_M lose less quality (Q8_0 and F16 stay as they are). llama.cpp (<code>llama-imatrix</code>) measures the matrix; adds a few minutes. If the installed Unsloth can apply a matrix, Unsloth makes the files with it, otherwise llama.cpp does (convert, then <code>llama-quantize</code>); the list below says which. Without llama.cpp, or if anything about it fails, the files are made without the matrix and the page says so.</p>${us.im ? llamaNotice("all") : ""}${detail}`;
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
    <div class="row"><label>Run <select id="us-run">${runOpts}</select></label><span class="quants">File sizes: ${st.quants.map(usQuantBox).join(" ")}</span>
      <label>Context <select id="us-ctx">${[4096, 8192, 16384, 32768].map((n) => `<option ${us.numCtx === n ? "selected" : ""}>${n}</option>`).join("")}</select></label></div>
    <div class="row"><label>Make <select id="us-what"><option value="gguf" ${us.what === "gguf" ? "selected" : ""}>GGUF + Modelfile (for Ollama)</option><option value="gguf_merged" ${us.what === "gguf_merged" ? "selected" : ""}>GGUF + the merged model (safetensors)</option><option value="merged" ${us.what === "merged" ? "selected" : ""}>Only the merged model (safetensors, no GGUF)</option></select></label></div>
    ${mixedBox(st)}${imatrixBox(st)}
    <p class="dim">Joins the trained add-on into the model. ${us.what === "merged" ? "Makes the <strong>merged 16-bit model</strong> as ordinary safetensors files (10 to 20 GB for the bigger models), which loads anywhere without the add-on." : "Makes a <strong>GGUF</strong> file with a <strong>Modelfile</strong> (a few minutes; each file is 1 to 6 GB). Tick several sizes to make them all from one conversion; each becomes its own entry below." + (us.what === "gguf_merged" ? " Also keeps the merged 16-bit model (10 to 20 GB more)." : "")}</p>
    ${pfBox("export", { run: us.run, what: us.what, quants: us.what === "merged" ? [] : us.quants, imatrix: us.what !== "merged" && !!us.im })}
    <div class="row"><button class="primary" id="us-make" ${busy || !st.env_ready || pfBlocked("export") || (us.what !== "merged" && !us.quants.length) || (us.what !== "merged" && us.im && !us.imSaved && gcBlocked("imatrix")) || (us.what !== "merged" && us.quants.includes("mixed_fit") && !(+us.mx.target > 0)) || (us.what !== "merged" && us.quants.includes("mixed_measured") && !(+us.mm.target > 0)) ? "disabled" : ""}>Make the files</button>${quBtn("export", `${us.what !== "merged" && !us.quants.length ? "disabled" : ""}`)}<button class="secondary" id="us-lora" ${busy ? "disabled" : ""} title="Copies the small LoRA add-on, with a README on how to load it in Unsloth or PEFT or merge it yourself">Save the LoRA add-on…</button>${!st.env_ready ? ` <span class="err">Open "⚙ Training setup" first (not needed to copy the LoRA add-on).</span>` : ""}</div>${us.what !== "merged" && us.im && gcBlocked("imatrix") ? `<p class="warn small">${esc(gcBlocked("imatrix"))}</p>` : ""}</div>`;
  const ex = st.exports.length ? st.exports.map((e) => `<li><strong>${esc(e.name)}</strong> · ${e.gguf_file === null ? "merged model only" : e.size_mb != null ? e.size_mb + " MB" : "file missing"}${e.merged_exists ? ` · merged model kept` : ""}${e.has_vision ? ` · <span title="${esc(e.vision_note || "")}">vision file included</span> (pictures work in Unsloth Studio and llama.cpp; the Modelfile lists the vision file, so Ollama keeps pictures when its build loads this family's projector — see VISION.md in the folder)` : (e.modalities && e.modalities.length ? ` · <span class="dim" title="${esc(e.vision_note || "")}">no vision file</span>` : "")}${e.mixed ? ` · <span title="${esc((e.mixed.note || "") + " " + (e.mixed.notes || []).join(" "))}"><strong>${esc(e.mixed.recipe || "Mixed precision (our own recipe)")}</strong>: ${esc(e.mixed.label || "")}${e.mixed.estimate && e.mixed.estimate.gb ? `, estimated ${e.mixed.estimate.gb} GB` : ""}${e.mixed.fit ? ", fitted to " + (e.mixed.fit.target_bytes / 1e9).toFixed(2) + " GB (experimental)" : ""}${e.mixed.fell_back_to ? `, the fit fell back to ${esc(e.mixed.fell_back_to)}` : ""}</span>` : ""}${e.imatrix && e.imatrix.requested && e.via ? ` · <span class="dim">GGUF made by ${e.via === "unsloth" ? "Unsloth" : "llama.cpp"}</span>` : ""}${e.imatrix && e.imatrix.requested ? (e.imatrix.applied ? ` · <span title="${esc(e.imatrix.note || "")}">importance matrix used (${e.imatrix.chunks} chunks${e.imatrix.reused ? ", saved one" : ""}${e.imatrix.nextn ? ", with the speculative-decoding head" : ""})</span>` : ` · <span class="dim" title="${esc(e.imatrix.reason || e.imatrix.note || "")}">no importance matrix${e.imatrix.reason ? " (" + esc(e.imatrix.reason) + ")" : ""}</span>`) : ""}${e.imatrix && e.imatrix.note ? `<br><span class="dim">Importance matrix: ${esc(e.imatrix.note)}</span>` : ""}<br><span class="dim">${esc(e.dir)}</span> ${copyBtn(e.dir)}
      ${e.installed && e.installed.length ? `<br><span class="dim">Installed: ${e.installed.map((i) => esc(i.model) + " @ " + esc(i.url)).join(", ")}</span>` : ""} ${e.can_restore_vision ? `<button class="link" data-rsvision="${esc(e.name)}" title="The model has a vision part but this export has no projector file (its origin may have been deleted). Convert one from the model and add it, so pictures work again.">Restore vision</button>` : ""} <button class="link" data-stdel="export|${esc(e.name)}" title="Opens the Storage page with this export ticked; nothing is deleted before you confirm.">Delete…</button></li>`).join("") : "";
  const gg = st.exports.filter((e) => e.gguf_file !== null && e.run === us.run);
  const list = ex ? `<div class="card wide"><h2>Your files</h2><ul class="exlist">${ex}</ul>${gg.length > 1 ? `<div class="row"><button class="secondary" id="us-compare" title="Measures how far each of these files is from the 16-bit model (KL divergence)">Compare these sizes</button></div>` : ""}</div>` : "";
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
  const prog = (us.running || us.step) ? `<div class="stepline" aria-live="polite">${esc(us.step)}</div>${us.progress != null ? `<progress max="1" value="${us.progress}"></progress>` : ""}${us.running && us.pending === "make" ? ` <button class="secondary" id="us-cancel" title="Stops the export; while the importance matrix is made, llama.cpp is stopped at once">Cancel</button>` : ""}` : "";
  const msg = `${errBox(us.error, "all")}${us.ok ? `<div class="ok">${esc(us.ok)}</div>` : ""}`;
  const log = us.log.length ? `<details><summary>Log</summary><pre class="log">${us.log.slice(-100).map(esc).join("\n")}</pre></details>` : "";
  return `<h1>Step 5 · Use in nd-world</h1><p class="lead">Turns the trained model into a file nd-world can use, installs it into Ollama and shows what to set. The help panel explains each part.</p>
    ${files}${prog}${msg}${log}${list}${install}${how}`;
}

async function usRefresh() {
  try { us.state = await invoke("export_state"); } catch (e) { us.error = String(e); }
  if (us.imSaved && us.state && !((us.state.imatrix && us.state.imatrix.library) || []).some((e) => e.id === us.imSaved)) { us.imSaved = ""; us.imCheck = null; us.imCheckKey = null; }
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
    else if (d.restored) us.ok = `Vision restored for "${d.restored}": the projector file now sits next to the GGUF, so pictures work again (Unsloth Studio / llama.cpp; Ollama when its build loads this family's projector).`;
    else if (d.name && d.run) {
      const notes = [...new Set((d.exports || [d]).map((m) => m.imatrix && m.imatrix.note).filter(Boolean))];
      const mnotes = [...new Set((d.exports || [d]).map((m) => m.mixed && m.mixed.note).filter(Boolean))];
      us.ok = `Files made: ${d.name}.${d.merged_dir ? " Merged model: " + d.merged_dir : ""}${notes.length ? " Importance matrix: " + notes.join(" ") : ""}${mnotes.length ? " " + mnotes.join(" ") : ""}`;
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

// what "Make the files" sends to export_start (the queue keeps the same)
function usMakeArgs() {
  return { run: us.run, quant: us.quant, numCtx: us.numCtx, what: us.what, quants: us.what === "merged" ? [us.quants[0] || "q4_k_m"] : us.quants,
    imatrix: us.what !== "merged" && us.im ? { enabled: true, chunks: us.imChunks, reuse: us.imReuse === null ? !!((us.state.imatrix.saved || {})[us.run]) : us.imReuse, nextn: nextnState(us.state.imatrix).on, general: gcSource("imatrix"), ...(us.imSaved ? { use_saved: us.imSaved } : {}) } : null,
    ...(us.what !== "merged" && (us.quants.includes("mixed_fit") || us.quants.includes("mixed_measured"))
      ? { mixed: { ...(us.quants.includes("mixed_fit") ? { fit: { target_gb: Number(us.mx.target), matrix: us.mx.matrix || null } } : {}),
                   ...(us.quants.includes("mixed_measured") ? { measured: { target_gb: Number(us.mm.target), matrix: us.mm.matrix || null } } : {}) } } : {}) };
}

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
  const usr = $("us-run"); if (usr) usr.onchange = () => { us.run = usr.value; us.imCheck = null; us.imCheckKey = null; render(); };
  document.querySelectorAll(".us-q").forEach((c) => c.onchange = () => { us.quants = [...document.querySelectorAll(".us-q")].filter((x) => x.checked).map((x) => x.value); render(); }); const wh = $("us-what"); if (wh) wh.onchange = () => { us.what = wh.value; render(); }; bind("us-ctx", "numCtx", true); bind("us-exp", "exp"); bind("us-server", "server");
  const mxf = $("us-mxfile"); if (mxf) mxf.onchange = () => { us.mx.file = mxf.value; us.mx.plan = null; us.mx.error = ""; render(); };
  const mxt = $("us-mxtyped"); if (mxt) mxt.onchange = () => { us.mx.typed = mxt.value; us.mx.plan = null; };
  const mxg = $("us-mxtarget"); if (mxg) mxg.onchange = () => { us.mx.target = mxg.value.trim(); render(); };
  const mxm = $("us-mxmatrix"); if (mxm) mxm.onchange = () => { us.mx.matrix = mxm.value; };
  const mmt = $("us-mmtarget"); if (mmt) mmt.onchange = () => { us.mm.target = mmt.value.trim(); render(); };
  const mmm = $("us-mmmatrix"); if (mmm) mmm.onchange = () => { us.mm.matrix = mmm.value; };
  wireQuantfit();
  on("us-mxplanbtn", () => usMixedPlan(false));
  on("us-mxfitbtn", () => usMixedPlan(true));
  const imx = $("us-im"); if (imx) imx.onchange = () => { us.im = imx.checked; render(); };
  const imc = $("us-imchunks"); if (imc) imc.onchange = () => { us.imChunks = Number(imc.value); render(); };
  const imn = $("us-imnextn"); if (imn) imn.onchange = () => { us.imNextn = imn.checked; render(); };
  const imr = $("us-imreuse"); if (imr) imr.onchange = () => { us.imReuse = imr.checked; render(); };
  const ims = $("us-imsaved"); if (ims) ims.onchange = () => { us.imSaved = ims.value; us.imCheck = null; us.imCheckKey = null; render(); };
  on("us-imlib", () => go("imatrix"));
  on("us-compare", () => qkFromRun(us.run));
  on("us-imadd", () => { us.imAdd = true; render(); });
  on("us-imaddcancel", () => { us.imAdd = false; render(); });
  const ial = $("us-imaddlabel"); if (ial) ial.onchange = () => { us.imAddLabel = ial.value; };
  on("us-imaddfile", async () => {
    const f = await tauri.dialog.open({ multiple: false, title: "Choose an importance matrix file", filters: [{ name: "Importance matrix", extensions: ["gguf", "dat"] }, { name: "All files", extensions: ["*"] }] });
    if (!f) return;
    try {
      const r = await invoke("imatrix_import", { file: f, label: us.imAddLabel });
      us.imAdd = false; us.imAddLabel = ""; us.error = null; us.ok = `Added to the library${r.added && r.added.warning ? ": " + r.added.warning : "."}`;
      await usRefresh(); us.imSaved = r.added.id; us.imCheck = null; us.imCheckKey = null;
    } catch (e) { us.error = String(e); }
    render();
  });
  usCheckSaved();
  if (us.im && us.what !== "merged") gcWire("imatrix", render);
  on("us-cancel", () => invoke("env_cancel"));
  const nm = $("us-name"); if (nm) nm.onchange = () => { us.modelName = nm.value.trim(); render(); };
  document.querySelectorAll("[data-copy]").forEach((b) => b.onclick = async () => { try { await navigator.clipboard.writeText(b.dataset.copy); b.textContent = "Copied"; setTimeout(() => (b.textContent = "Copy"), 1500); } catch (_) { b.textContent = "Select and copy by hand"; } });
  document.querySelectorAll("[data-rsvision]").forEach((b) => b.onclick = () => usRun("restore", "export_restore_vision", { export: b.dataset.rsvision }));
  on("us-make", () => usRun("make", "export_start", usMakeArgs()));
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

// ------------------------------------------------------------------------------------------- importance matrices (the matrix library)
// A matrix made on its own (no GGUF export), kept in the project (imatrix/<id>/), reused by exports ("Use a saved matrix" on Step 5) and copied out. docs/IMATRIX.md.
let im = { state: null, src: "run", run: "", model: "", modelPath: "", gguf: "", tokenizer: "", cal: "data", textFile: "", share: 20, chunks: 128, nextn: true, dat: false, label: "",
  addOpen: false, addLabel: "", addExport: "", copyFor: "", copyDir: "", mergePicked: [], mergeFiles: [], mergeFileInput: "", mergePub: false, mergePubBase: "", mergePubInfo: null, mergePubBusy: false, mergePubError: "", mergeLabel: "", mergePub2: false, mergeWeights: {}, adSrc: "published", adId: "", adFile: "", adBase: "", adModel: "", adWeight: "1", adKept: "", adLabel: "", adPubInfo: null, adPubBusy: false, adPubError: "", adPreview: null, adPreviewKey: "", running: false, pending: null, step: "", progress: null, log: [], error: null, ok: null, result: null, saved: null };

const IM_INTRO = `<p>An <strong>importance matrix</strong> is a short measurement: the model reads some typical text, and llama.cpp notes how strongly each part of it is used. When the model is later shrunk to about 4 or 5 bits (Q4_K_M, Q5_K_M, Q6_K), the rounding then takes more care of the parts that matter, so the small file loses less quality. For Q8_0 and F16 it does nothing.</p>
  <p class="dim">You can make one here without making any GGUF file, keep it, use it in later exports (Step 5, "Use a saved matrix"), or save a copy to use with <code>llama-quantize</code> on another computer or to share. It needs llama.cpp (<code>llama-imatrix</code>): the program can install it for you (the button appears below when it is missing; <strong>Training setup</strong> has the same). A copy of your own in <code>LLAMA_CPP_DIR</code> always wins.</p>`;

function imEstimate(st) {
  const e = st.estimates || {};
  const key = im.src === "run" ? "run:" + im.run : im.src === "model" && im.model ? "model:" + im.model : "default";
  return (e[key] || e.default || {})[String(im.chunks)] || null;
}

function imEntryHtml(e) {
  const g = e.general_used === true ? "with general text" + (e.general_source && e.general_source !== "default" ? " (" + e.general_source + ")" : "") : e.general_used === false ? "your own text only" : "general text unknown";
  const head = e.nextn === true ? "MTP head included" : e.nextn_asked ? "MTP head left out" : "no MTP head";
  const copying = im.copyFor === e.id ? `<div class="row"><label>Save into the folder <input type="text" id="im-copydir" value="${esc(im.copyDir)}" placeholder="C:\\matrices" style="min-width:260px"></label><button class="secondary" id="im-copybrowse">Browse…</button><button class="primary" id="im-copygo">Save</button><button class="link" id="im-copycancel">Cancel</button></div>
    <p class="dim small">A new folder named <code>${esc(e.id)}</code> is made inside it (a number is added if that name is taken: nothing is overwritten). It holds imatrix.gguf${e.has_dat ? ", imatrix.dat" : ""} and imatrix-info.json.</p>` : "";
  const stats = e.statistics && e.statistics.lines && e.statistics.lines.length ? `<details><summary>What the matrix looks like</summary><ul class="small">${e.statistics.lines.map((l) => `<li>${esc(l)}</li>`).join("")}</ul></details>` : "";
  const notes = (e.notes || []).length ? `<p class="dim small">${e.notes.map(esc).join(" ")}</p>` : "";
  const off = im.running || e.in_use;
  const parents = (e.source_kind === "merged" || e.source_kind === "adapted") && (e.parents || []).length ? `<br><span class="dim small">Parents: ${e.parents.map((x) => esc(x.label || x.path) + (x.chunks != null ? ` (${x.chunks} chunks)` : "") + (x.kind === "published" ? ` · from ${esc(x.repo || "the Hub")}` : "")).join(" + ")}</span>` : "";
  const ad = e.source_kind === "adapted" && e.adapt ? `<br><span class="dim small">Adapted to ${esc(e.adapt.model)}: keeps ${e.adapt.layers_after} of ${e.adapt.layers_before} layers; ffn_down ${(e.adapt.ffn_down_before || 0).toLocaleString("en")}${e.adapt.ffn_down_after !== e.adapt.ffn_down_before ? " → " + (e.adapt.ffn_down_after || 0).toLocaleString("en") : ""} per layer${e.adapt.weight != null && e.adapt.weight !== 1 ? "; weight " + e.adapt.weight : ""}. It still describes the unmodified model's activations.</span>` : "";
  return `<li class="imentry"><label class="opt" title="Tick two or more matrices to merge them (see Merge matrices below)"><input type="checkbox" class="im-pick" value="${esc(e.id)}" ${im.mergePicked.includes(e.id) ? "checked" : ""} ${im.running ? "disabled" : ""}></label> <strong>${esc(e.label || e.id)}</strong> <span class="dim">${esc(e.id)}</span>${e.source_kind === "merged" ? ` <span class="tag">merged</span>` : ""}${e.source_kind === "adapted" ? ` <span class="tag">adapted</span>` : ""}<br>
    <span class="dim small">Model: ${esc(e.model || "not stated")} · ${e.size_mb} MB · ${e.chunks != null ? e.chunks + " chunks" : "unknown amount of text"} · ${esc(g)} · ${esc(head)} · ${esc(String(e.created || "").slice(0, 10))} · made by ${esc(e.via || "unknown")}${e.has_dat ? " · legacy .dat too" : ""}${e.has_gguf ? "" : " · legacy .dat only"}</span><br>
    <span class="dim small">Made from: ${esc(e.source || "unknown")}${e.calibration ? " · text: " + esc(e.calibration) : ""}</span>${parents}${ad}${e.in_use ? ` <span class="warn small">in use right now</span>` : ""}
    <div class="row"><button class="secondary im-copy" data-id="${esc(e.id)}" ${im.running ? "disabled" : ""}>Save a copy…</button><button class="secondary im-open" data-id="${esc(e.id)}">Open folder</button>
      ${e.has_gguf && !e.has_dat ? `<button class="secondary im-dat" data-id="${esc(e.id)}" ${off ? "disabled" : ""} title="llama-imatrix converts the matrix to the old format some tools still read">Make a .dat (legacy)</button>` : ""}
      <button class="secondary im-del" data-id="${esc(e.id)}" ${off ? "disabled" : ""} title="${e.in_use ? "A job uses this matrix right now" : "Deletes the matrix from the project"}">Delete</button></div>${copying}${stats}${notes}</li>`;
}

// the Hugging Face id of the model a new matrix is made from, when the Hub has one ("also merge Unsloth's published matrix" is offered only then)
function imBase(st) {
  if (im.src === "run") { const r = (st.runs || []).find((x) => x.name === im.run); return r && r.base ? r.base : ""; }
  if (im.src === "model") { if (im.model === "__typed") return /^[^/\\:.][^/\\:]*\/[^/\\:]+$/.test(im.modelPath.trim()) ? im.modelPath.trim() : ""; const m = (st.models || []).find((x) => x.id === im.model); return m && /^[^/\\:.][^/\\:]*\/[^/\\:]+$/.test(m.repo) ? m.repo : m && m.cut_origin ? m.cut_origin : ""; }
  return /^[^/\\:.][^/\\:]*\/[^/\\:]+$/.test(im.tokenizer.trim()) ? im.tokenizer.trim() : "";
}

// is the model of the Create form one made on the Slim page? (its published matrix is then ADAPTED to it, never merged raw)
function imCutSource(st) {
  if (im.src === "run") { const r = (st.runs || []).find((x) => x.name === im.run); return !!(r && r.cut); }
  if (im.src === "model") { const m = (st.models || []).find((x) => x.id === im.model); return !!(m && m.cut_origin); }
  return false;
}

function imFormHtml(st) {
  const runs = st.runs || [], models = st.models || [];
  if (!im.run && runs.length) im.run = runs[0].name;
  if (!im.model && models.length) im.model = models[0].id;
  const srcOpts = [["run", "A trained run (its add-on merged into its model)"], ["model", "A model (the model list, a folder or a Hugging Face id)"], ["gguf", "A GGUF file I already have"]]
    .map(([k, t]) => `<option value="${k}" ${im.src === k ? "selected" : ""}>${t}</option>`).join("");
  let src = "";
  if (im.src === "run") {
    src = runs.length ? `<div class="row"><label>Run <select id="im-run">${runs.map((r) => `<option value="${esc(r.name)}" ${im.run === r.name ? "selected" : ""}>${esc(r.name)} · ${esc(r.model)} · ${esc(r.status)}</option>`).join("")}</select></label></div>
      <p class="dim small">The run's add-on is merged into its base model, then the model is converted to a 16-bit GGUF file with llama.cpp (as the export does). If an earlier export of this run left such a file, it is used instead. Needs the training software.</p>`
      : `<p class="dim">There is no trained run yet (Step 3). Choose another source.</p>`;
  } else if (im.src === "model") {
    src = `<div class="row"><label>Model <select id="im-model">${models.map((m) => `<option value="${esc(m.id)}" ${im.model === m.id && !im.modelPath ? "selected" : ""}>${esc(m.name)} · ${(+m.params_b).toFixed(1)} B${m.reshape ? " · " + esc(m.reshape) : ""}</option>`).join("")}<option value="__typed" ${im.model === "__typed" ? "selected" : ""}>A folder or a Hugging Face id…</option></select></label></div>
      ${im.model === "__typed" ? `<div class="row"><label>Folder or id <input type="text" id="im-modelpath" value="${esc(im.modelPath)}" placeholder="C:\\models\\mymodel   or   owner/name" style="min-width:300px"></label><button class="secondary" id="im-modelbrowse">Browse…</button></div>` : ""}
      <p class="dim small">A full-precision model (the folder with config.json and the .safetensors files, or its Hugging Face id: it is downloaded if needed). It is converted to a 16-bit GGUF file with llama.cpp. Made, slimmed and healed models are in the list.</p>`;
  } else {
    src = `<div class="row"><label>GGUF file <input type="text" id="im-gguf" value="${esc(im.gguf)}" placeholder="C:\\models\\model-f16.gguf" style="min-width:300px"></label><button class="secondary" id="im-ggufbrowse">Browse…</button></div>
      ${im.cal === "data" ? `<div class="row"><label>Made from (folder or Hugging Face id) <input type="text" id="im-tok" value="${esc(im.tokenizer)}" placeholder="owner/name" style="min-width:300px"></label><button class="secondary" id="im-tokbrowse">Browse…</button></div>
      <p class="dim small">Needed only for your saved data: the chat template of that model writes the practice conversations the way the model sees them.</p>` : ""}
      <p class="dim small">A 16-bit (F16 or BF16) or Q8_0 file is best; any GGUF that <code>llama-imatrix</code> accepts works. <strong>A more precise file gives a better matrix.</strong></p>`;
  }
  const blocked = im.cal === "data" && im.share > 0 ? gcBlocked("imatrix") : "";
  const cal = `<div class="row"><label>Calibration text <select id="im-cal"><option value="data" ${im.cal === "data" ? "selected" : ""}>My saved training data${st.data_version != null ? " (version " + st.data_version + ")" : ""} and general text</option><option value="text" ${im.cal === "text" ? "selected" : ""}>A text file of my own</option></select></label></div>`
    + (im.cal === "data" ? (st.data_version == null ? `<p class="warn small">There is no saved version of the training data yet (Step 2). Save one, or choose a text file of your own.</p>` : "")
      + `<div class="row"><label>General text <select id="im-share">${[0, 10, 20, 30, 40].map((n) => `<option value="${n}" ${im.share === n ? "selected" : ""}>${n === 0 ? "none (only my data)" : "about " + n + " %"}${n === 20 ? " (recommended)" : ""}</option>`).join("")}</select></label></div>${im.share > 0 ? gcHtml("imatrix") : ""}
      <p class="dim small">Your data is written through the model's chat template (practice conversations, world-book texts as they are), about ${100 - im.share} % of the text; the rest is general text, so the matrix does not only know your world.</p>`
      : `<div class="row"><label>Text file <input type="text" id="im-text" value="${esc(im.textFile)}" placeholder="C:\\texts\\calibration.txt" style="min-width:300px"></label><button class="secondary" id="im-textbrowse">Browse…</button></div>
      <p class="dim small">Plain text, used as it is (a few hundred KB or more; text that looks like what the model will be asked). Nothing is added to it.</p>`);
  const est = imEstimate(st);
  const chunks = `<div class="row"><label>Amount of text <select id="im-chunks">${st.chunks.map((n) => `<option value="${n}" ${im.chunks === n ? "selected" : ""}>${{ 64: "64 chunks (quick)", 128: "128 chunks (recommended)", 256: "256 chunks (thorough)" }[n] || n + " chunks"}</option>`).join("")}</select></label>
      <span class="dim" id="im-est">${est ? `${est.tokens.toLocaleString("en")} tokens, about ${est.gpu_minutes} min on a graphics card (up to ${est.cpu_minutes} min on the processor)${im.src !== "gguf" ? ", plus the time to convert the model" : ""}: a rough guess, the log shows the real progress` : ""}</span></div>
    <div class="row"><label class="opt" title="If the model has a speculative-decoding head (MTP) and your llama-imatrix has --nextn, the head is measured too; otherwise it is left out and the result says so"><input type="checkbox" id="im-nextn" ${im.nextn ? "checked" : ""}> Include the speculative-decoding head (MTP) if the model has one</label>
      <label class="opt"><input type="checkbox" id="im-dat" ${im.dat ? "checked" : ""}> Also make a legacy .dat file</label></div>
    <div class="row"><label>Label <input type="text" id="im-label" value="${esc(im.label)}" maxlength="80" placeholder="for example: Q4 test with my own text"></label></div>
    ${imBase(st) ? `<div class="row"><label class="opt" title="After the matrix is made, a second entry is made by merging it with the matrix Unsloth published for ${esc(imBase(st))}"><input type="checkbox" id="im-mergepub2" ${im.mergePub2 ? "checked" : ""}> Also merge Unsloth's published matrix for the base model (${esc(imBase(st))})</label></div>
      ${im.mergePub2 ? `<p class="dim small">A second entry is made after yours (yours is kept). It downloads the published matrix from the Hub, adds its statistics to yours (a large published matrix can outweigh a small one of your own), and describes the unmodified base model. If the download or the merge fails, your matrix is kept and the reason is shown.</p>
      ${imCutSource(st) ? `<p class="warn small" id="im-mergepub2-cut"><strong>This model was cut on the Slim page.</strong> The published matrix describes the original model, so it is first adapted to your cut model (the layers it kept, the MLP cut the way yours was) and only then merged with yours: it is never merged as it is. It still describes the unmodified model's activations, so it is an approximation. Use the <em>Adapt to my cut model</em> card to choose a weight (the published matrix has far more text and otherwise outweighs yours).</p>` : ""}` : ""}` : ""}`;
  const incomplete = im.src === "run" ? !runs.length : im.src === "model" ? (im.model === "__typed" ? !im.modelPath.trim() : !im.model) : !im.gguf.trim();
  const calMissing = im.cal === "text" ? !im.textFile.trim() : st.data_version == null;
  const why = !st.env_ready && im.src !== "gguf" ? "The training software is not installed yet (Training setup)." : incomplete ? "Choose the model first." : calMissing ? (im.cal === "text" ? "Choose the text file." : "Save a version of the training data first.") : blocked;
  const prog = im.running && im.pending === "create" ? `<div class="stepline" aria-live="polite">${esc(im.step)}</div>${im.progress != null ? `<progress max="1" value="${im.progress}"></progress>` : ""} <button class="secondary" id="im-stop" title="llama.cpp is stopped at once; nothing is kept">Stop</button>` : "";
  return `<div class="card wide"><h2>Create a matrix</h2>
    <div class="row"><label>The model comes from <select id="im-src" ${im.running ? "disabled" : ""}>${srcOpts}</select></label></div>${src}${cal}${chunks}
    ${why ? `<p class="dim small" id="im-why">${esc(why)}</p>` : ""}
    ${pfBox("imatrix_make", {})}<div class="row"><button class="primary" id="im-create" ${im.running || why || pfBlocked("imatrix_make") ? "disabled" : ""}>Create the matrix</button></div>${prog}</div>`;
}

// "Merge matrices": the ticked matrices of the library, files, and/or Unsloth's published matrix for a base model, combined by llama-imatrix into a new entry. docs/IMATRIX.md.
function imMergeHtml(st) {
  const bases = st.published_bases || [];
  if (!im.mergePubBase && bases.length) im.mergePubBase = bases[0].repo;
  const info = im.mergePubInfo;
  const pubInfo = im.mergePubBusy ? `<p class="dim small">Looking it up on the Hub…</p>` : im.mergePubError ? `<div class="warn small" id="im-pub-error">${esc(im.mergePubError)}</div>`
    : info ? `<div class="small" id="im-pub-info"><strong>${esc(info.repo)}</strong> · file <code>${esc(info.file)}</code> · ${info.size != null ? (info.size / 1048576).toFixed(1) + " MB" : "size not stated"}${info.gated ? " · <strong>gated</strong>" : " · not gated"}<br>
      ${info.cached ? `Already downloaded: <code>${esc(info.cache_file)}</code> (nothing will be downloaded).` : `<strong>This merge downloads ${info.size != null ? (info.size / 1048576).toFixed(1) + " MB" : "the file"} from huggingface.co</strong> into this project's <code>imatrix_downloads</code> folder, once; nothing else is downloaded.`}
      Source: <code>${esc(info.page)}</code></div>` : "";
  const files = im.mergeFiles.map((f, i) => `<li><span>${esc(f)}</span> <button class="link im-unmergefile" data-i="${i}">Remove</button></li>`).join("");
  const n = im.mergePicked.length + im.mergeFiles.length + (im.mergePub ? 1 : 0);
  const badWeight = imMergeInputs().some((i) => !(imWeightOf(i.key) > 0));
  const cutNames = [...new Set(im.mergePicked.map((id) => ((st.entries || []).find((e) => e.id === id) || {}).cut_model).filter(Boolean).map((c) => c.name))];
  const cutInfo = im.mergePub && cutNames.length ? cutNames.map((nm) => { const c = (st.cut_models || []).find((x) => x.name === nm); return `<p class="ok small" id="im-merge-autoadapt">The published matrix is <strong>adapted first</strong> to your cut model <strong>${esc(nm)}</strong>${c ? ` (it keeps ${c.layers_after} of ${c.layers_before} layers${c.thinner ? `; ffn_down ${c.intermediate_before.toLocaleString("en")} → ${c.intermediate_after.toLocaleString("en")} per layer` : ""})` : ""}, because the matrix you ticked was measured on that model. It is never merged as it is: its layers would not be yours. See <em>Adapt to my cut model</em> below.</p>`; }).join("") : "";
  const wrows = imMergeInputs().map((i) => `<li>${esc(i.label)} <label class="opt">weight <input type="number" class="im-mw" data-key="${esc(i.key)}" min="0" step="any" value="${esc(String(im.mergeWeights[i.key] ?? "1"))}" style="width:90px"></label></li>`).join("");
  const why = !st.env_ready ? "The merge runs in the training software's worker, which is not set up yet (open Training setup)." : n < 2 ? "Tick two or more matrices above, or add a file, or use Unsloth's published matrix: a merge needs at least two things." : im.mergePub && !im.mergePubBase ? "Choose the base model." : badWeight ? "A weight must be a number above 0." : "";
  const prog = im.running && im.pending === "merge" ? `<div class="stepline" aria-live="polite">${esc(im.step)}</div>${im.progress != null ? `<progress max="1" value="${im.progress}"></progress>` : ""} <button class="secondary" id="im-stop" title="Stops at once; nothing is kept">Stop</button>` : "";
  return `<div class="card wide" id="im-merge"><h2>Merge matrices</h2>${llamaNotice("merge")}
    <p>Combine two or more matrices into a new one: matrices of your own (tick them in the list above), matrix files, and/or <strong>Unsloth's published matrix</strong> for the base model. llama.cpp (<code>llama-imatrix --in-file a --in-file b</code>) does it; the matrices you merge are not changed.</p>
    <div class="warn small"><strong>What merging does, honestly.</strong> Merging <em>adds</em> the statistics of the matrices. A matrix measured on more text therefore outweighs a smaller one: a large published matrix can drown a small one of your own (make yours with more chunks if you want it to count). And Unsloth's published matrix describes the <em>unmodified base model</em>, not your fine-tune: a reasonable start for a light LoRA, less so for a heavily changed model. Matrices are checked against each other from their headers; a mismatch is a warning, and if llama-imatrix refuses, its own message is shown.</div>
    <p class="dim small" id="im-merge-count">${im.mergePicked.length} ticked in the list${im.mergeFiles.length ? ` · ${im.mergeFiles.length} file(s)` : ""}${im.mergePub ? " · the published matrix" : ""}.</p>
    <ul class="exlist">${files}</ul>
    <div class="row"><label>Add a matrix file <input type="text" id="im-mergefile" value="${esc(im.mergeFileInput)}" placeholder="C:\\matrices\\imatrix.gguf" style="min-width:300px"></label><button class="secondary" id="im-mergefileadd">Add</button><button class="secondary" id="im-mergefilebrowse">Browse…</button></div>
    <div class="row"><label class="opt"><input type="checkbox" id="im-mergepub" ${im.mergePub ? "checked" : ""} ${bases.length ? "" : "disabled"}> Also use Unsloth's published matrix for the base model</label>
      ${im.mergePub && bases.length ? `<label>Base model <select id="im-mergepubbase">${bases.map((b) => `<option value="${esc(b.repo)}" ${im.mergePubBase === b.repo ? "selected" : ""}>${esc(b.label)}</option>`).join("")}</select></label><button class="secondary" id="im-pub-lookup" ${im.mergePubBusy ? "disabled" : ""}>Look it up</button>` : ""}</div>
    ${bases.length ? "" : `<p class="dim small">No base model on the Hugging Face Hub is known in this project yet (train a run on a model of the list first).</p>`}
    ${im.mergePub ? `<p class="dim small">The repository is named the way Unsloth's own code does it: <code>unsloth/&lt;model name&gt;-GGUF</code>, file <code>imatrix_unsloth.gguf</code> (or <code>.dat</code>). It is downloaded from the Hub, so this one step needs the internet; offline, not found and gated are explained in plain words.</p>${pubInfo}` : ""}
    ${cutInfo}
    ${wrows ? `<details id="im-merge-weights"${Object.keys(im.mergeWeights).length ? " open" : ""}><summary>Weights (optional)</summary><p class="dim small">Merging adds the sums and the counts, so a matrix measured on more text counts more. A weight multiplies the sums <em>and</em> the counts of one matrix before they are merged: its mean square per channel stays, it just counts that many times as strongly. Unsloth's published matrix counts about 1,149,614 rows of text; 128 chunks of your own at a context of 512 are about 65,000, so merged as they are the published one outweighs yours about 17 to 1. A weight of 0.05 on the published one makes them about even. The result lists the weights that were used.</p><ul class="small">${wrows}</ul></details>` : ""}
    <div class="row"><label>Label <input type="text" id="im-mergelabel" value="${esc(im.mergeLabel)}" maxlength="80" placeholder="for example: mine + Unsloth's"></label></div>
    ${why ? `<p class="dim small" id="im-merge-why">${esc(why)}</p>` : ""}
    <div class="row"><button class="primary" id="im-merge-go" ${im.running || why ? "disabled" : ""}>Merge</button></div>${prog}</div>`;
}

// what is in the merge, in the order it is sent, with the key its weight is kept under
function imMergeInputs() {
  const st = im.state || {};
  const out = im.mergePicked.map((id) => { const e = (st.entries || []).find((x) => x.id === id); return { key: "lib:" + id, label: (e && (e.label || e.id)) || id, spec: { kind: "library", id } }; });
  im.mergeFiles.forEach((file) => out.push({ key: "file:" + file, label: file, spec: { kind: "file", file } }));
  if (im.mergePub) out.push({ key: "pub", label: "Unsloth's published matrix", spec: { kind: "published", base: im.mergePubBase } });
  return out;
}
function imWeightOf(key) { const v = im.mergeWeights[key]; return v === undefined || String(v).trim() === "" ? 1 : Number(v); }

function imMergeSpec() {
  // a weight is sent only when it is not the default 1, so a merge without weights is the spec it always was
  const inputs = imMergeInputs().map((i) => { const w = imWeightOf(i.key); return w !== 1 ? { ...i.spec, weight: w } : i.spec; });
  return { inputs, label: im.mergeLabel.trim() };
}

// ---- "Adapt to my cut model": a matrix of the ORIGINAL model cut to a model made on the Slim page (layers kept and renumbered, ffn_down sliced). docs/IMATRIX.md.
function imAdaptSource(st) {
  const entries = (st.entries || []).filter((e) => e.has_gguf && e.source_kind !== "adapted");
  if (im.adSrc === "library") { if (!entries.some((e) => e.id === im.adId)) im.adId = entries.length ? entries[0].id : ""; return im.adId ? { kind: "library", id: im.adId } : null; }
  if (im.adSrc === "file") return im.adFile.trim() ? { kind: "file", file: im.adFile.trim() } : null;
  const cut = (st.cut_models || []).find((c) => c.name === im.adModel);
  const bases = st.published_bases || [];
  if (!im.adBase) im.adBase = (cut && cut.origin_repo && bases.some((b) => b.repo === cut.origin_repo) ? cut.origin_repo : bases.length ? bases[0].repo : "");
  return im.adBase ? { kind: "published", base: im.adBase } : null;
}

function imAdaptSpec(st) {
  const source = imAdaptSource(st);
  const kept = im.adKept.split(/[\s,;]+/).filter(Boolean).map(Number);
  const w = im.adWeight.trim() === "" ? 1 : Number(im.adWeight);
  const size = im.adSrc === "published" && im.adPubInfo && im.adPubInfo.size ? im.adPubInfo.size : null;
  return { source, model: im.adModel, weight: w === 1 ? null : w, label: im.adLabel.trim(), kept_layers: kept.length ? kept : null, source_size: size };
}

async function imAdaptPreview() {
  const st = im.state;
  if (!st || page !== "imatrix") return;
  const spec = imAdaptSpec(st);
  if (!spec.source || !spec.model || !(spec.weight === null || spec.weight > 0)) { im.adPreview = null; im.adPreviewKey = ""; return; }
  const key = JSON.stringify(spec);
  if (key === im.adPreviewKey) return;
  im.adPreviewKey = key; im.adPreview = null;
  try { im.adPreview = await invoke("imatrix_adapt_preview", { spec }); } catch (e) { im.adPreview = { ok: false, code: "error", text: String(e) }; }
  if (page === "imatrix" && im.adPreviewKey === key) render();
}

function imAdaptHtml(st) {
  imAdaptSource(st);                         // fills in the default list entry / base model before the form is drawn
  const cuts = st.cut_models || [];
  if (!im.adModel || !cuts.some((c) => c.name === im.adModel)) im.adModel = cuts.length ? cuts[0].name : "";
  const cut = cuts.find((c) => c.name === im.adModel);
  const entries = (st.entries || []).filter((e) => e.has_gguf && e.source_kind !== "adapted");
  const bases = st.published_bases || [];
  const srcOpts = [["published", "Unsloth's published matrix for the base model"], ["library", "A matrix of my list (of the original model)"], ["file", "A matrix file (of the original model)"]]
    .map(([k, t]) => `<option value="${k}" ${im.adSrc === k ? "selected" : ""}>${t}</option>`).join("");
  let src = "";
  if (im.adSrc === "library") src = entries.length ? `<div class="row"><label>Matrix <select id="im-ad-lib">${entries.map((e) => `<option value="${esc(e.id)}" ${im.adId === e.id ? "selected" : ""}>${esc(e.label || e.id)} · ${esc(e.id)}</option>`).join("")}</select></label></div>` : `<p class="dim small">There is no matrix of the original model in your list.</p>`;
  else if (im.adSrc === "file") src = `<div class="row"><label>Matrix file <input type="text" id="im-ad-file" value="${esc(im.adFile)}" placeholder="C:\\matrices\\imatrix.gguf" style="min-width:300px"></label><button class="secondary" id="im-ad-filebrowse">Browse…</button></div>`;
  else {
    const info = im.adPubInfo;
    src = bases.length ? `<div class="row"><label>Base model <select id="im-ad-base">${bases.map((b) => `<option value="${esc(b.repo)}" ${im.adBase === b.repo ? "selected" : ""}>${esc(b.label)}</option>`).join("")}</select></label><button class="secondary" id="im-ad-lookup" ${im.adPubBusy ? "disabled" : ""}>Look it up</button></div>`
      : `<p class="dim small">No base model on the Hugging Face Hub is known in this project yet.</p>`;
    src += im.adPubBusy ? `<p class="dim small">Looking it up on the Hub…</p>` : im.adPubError ? `<div class="warn small" id="im-ad-pub-error">${esc(im.adPubError)}</div>`
      : info ? `<div class="small" id="im-ad-pub-info"><strong>${esc(info.repo)}</strong> · ${info.size != null ? (info.size / 1048576).toFixed(1) + " MB" : "size not stated"}${info.gated ? " · <strong>gated</strong>" : ""}<br>${info.cached ? "Already downloaded; nothing will be downloaded." : `<strong>Adapting downloads ${info.size != null ? (info.size / 1048576).toFixed(1) + " MB" : "the file"} from huggingface.co</strong> into this project's <code>imatrix_downloads</code> folder, once.`}</div>` : "";
  }
  const model = cuts.length ? `<div class="row"><label>My cut model <select id="im-ad-model">${cuts.map((c) => `<option value="${esc(c.name)}" ${im.adModel === c.name ? "selected" : ""}>${esc(c.name)} · ${c.layers_after} of ${c.layers_before} layers${c.thinner ? ", MLP " + c.intermediate_before.toLocaleString("en") + " → " + c.intermediate_after.toLocaleString("en") : ""}${c.healed ? " · healed" : ""}</option>`).join("")}</select></label></div>
    ${cut && cut.refusal && !cut.needs_layer_map ? `<div class="warn small" id="im-ad-refusal">${esc(cut.refusal)}</div>` : ""}`
    : `<p class="dim small" id="im-ad-nomodel">There is no model made on the "Slim or grow a model" page in this project yet. Make one there (cut layers and/or thin the MLP), then come back.</p>`;
  const pv = im.adPreview;
  const needMap = (cut && cut.needs_layer_map) || (pv && pv.code === "needs_layer_map");
  const kept = needMap ? `<div class="row"><label>Layers of the original it kept <input type="text" id="im-ad-kept" value="${esc(im.adKept)}" placeholder="0, 1, 2, 3, 8, 9, 10, 11" style="min-width:300px"></label></div>
    <p class="dim small">This model was made by an older version of the program that did not write down which layers it kept. Type their numbers in the original model, counted from 0 (the program cannot guess them).</p>` : "";
  const preview = pv ? (pv.ok ? `<p class="ok small" id="im-ad-preview"><strong>${esc(pv.text)}</strong>${pv.exact ? "" : " (an estimate until the matrix file is at hand)"}.${(pv.notes || []).map((t) => " " + esc(t)).join("")}</p>` : `<div class="warn small" id="im-ad-preview-refusal">${esc(pv.text)}</div>`) : "";
  const wnum = im.adWeight.trim() === "" ? 1 : Number(im.adWeight);
  const why = !st.env_ready ? "The adaptation runs in the training software's worker, which is not set up yet (open Training setup)." : !cuts.length ? "Make a cut model first." : cut && cut.refusal && !cut.needs_layer_map ? cut.refusal
    : !imAdaptSource(st) ? "Choose the matrix to adapt." : !(wnum > 0) ? "The weight must be a number above 0." : pv && pv.ok === false ? pv.text : "";
  const prog = im.running && im.pending === "adapt" ? `<div class="stepline" aria-live="polite">${esc(im.step)}</div>${im.progress != null ? `<progress max="1" value="${im.progress}"></progress>` : ""} <button class="secondary" id="im-stop" title="Stops at once; nothing is kept">Stop</button>` : "";
  return `<div class="card wide" id="im-adapt"><h2>Adapt to my cut model</h2>
    <p>A model you cut on the <strong>Slim or grow a model</strong> page has fewer layers and/or thinner MLPs than the model a matrix was measured on. Unsloth publishes the matrix it used for the original model; this card <strong>adapts</strong> a matrix like that to your cut model <em>without measuring anything</em>: the layers your model dropped are dropped from the matrix and the others are renumbered, and the per-neuron statistics of <code>ffn_down</code> are cut to the neurons your model kept. Nothing else of the matrix changes (the other entries look at the hidden state, which Slim does not change).</p>
    <div class="warn small" id="im-ad-honest"><strong>What this is, honestly.</strong> The adapted matrix still describes the <em>unmodified</em> model's activations. For a model that was cut and healed it is an <em>approximation</em>; it is best merged with a matrix of your own, measured on the healed model (use the weight below, and the Merge card). It is <strong>not Unsloth's Dynamic recipe</strong>, which is unpublished and cannot be reproduced here. Whether it improves the small sizes of your cut model is measured with <em>Compare quants</em>; that has not been tried yet.</div>
    <div class="row"><label>The matrix comes from <select id="im-ad-src" ${im.running ? "disabled" : ""}>${srcOpts}</select></label></div>${src}${model}${kept}
    <div class="row"><label>Weight <input type="number" id="im-ad-weight" min="0" step="any" value="${esc(im.adWeight)}" style="width:100px"></label><span class="dim small">1 = as it is</span></div>
    <p class="dim small" id="im-ad-weight-text">Merging <em>adds</em> the statistics, so a matrix with more text outweighs one with less. The published matrix counts about 1,149,614 rows of text; a matrix of your own measured on 128 chunks of 512 about 65,000. Merged as they are, the published one outweighs yours about 17 to 1. The weight multiplies the sums <em>and</em> the counts of the adapted matrix (its mean square per channel stays), so it only changes how strongly it counts when merged: 0.05 makes the two about even, 0.1 about 2 to 1. Leave it at 1 if you only want the adapted matrix.</p>
    <div class="row"><label>Label <input type="text" id="im-ad-label" value="${esc(im.adLabel)}" maxlength="80" placeholder="for example: Unsloth's, cut to my 16B"></label></div>
    ${preview}${why ? `<p class="dim small" id="im-ad-why">${esc(why)}</p>` : ""}
    <div class="row"><button class="primary" id="im-adapt-go" ${im.running || why ? "disabled" : ""}>Adapt</button></div>${prog}</div>`;
}

function renderImatrix() {
  if (!state.bundle) return `<h1>Importance matrices</h1><p class="lead">Import your world data first (Step 1).</p>`;
  const st = im.state;
  if (!st) return `<h1>Importance matrices</h1><p class="lead">Loading…</p>`;
  gcUse(st.general);
  const entries = st.entries || [];
  const list = entries.length ? `<ul class="exlist">${entries.map(imEntryHtml).join("")}</ul>` : `<p class="dim">No matrix saved in this project yet. Create one below, or add a file.</p>`;
  const exp = (st.exports || []).filter((x) => !x.added);
  const add = `<div class="row"><button class="secondary" id="im-addopen" ${im.running ? "disabled" : ""}>Add a matrix file from my computer…</button>${exp.length ? `<label>or from an earlier export <select id="im-addexp"><option value="">choose…</option>${exp.map((x) => `<option value="${esc(x.name)}" ${im.addExport === x.name ? "selected" : ""}>${esc(x.name)}${x.chunks != null ? " · " + x.chunks + " chunks" : ""}</option>`).join("")}</select></label><button class="secondary" id="im-addexpgo" ${im.addExport ? "" : "disabled"}>Add to library</button>` : ""}</div>
    ${im.addOpen ? `<div class="row"><label>Label <input type="text" id="im-addlabel" value="${esc(im.addLabel)}" maxlength="80" placeholder="for example: from my friend"></label><button class="secondary" id="im-addfile">Choose the matrix file…</button><button class="link" id="im-addcancel">Cancel</button></div>
      <p class="dim small">A .gguf matrix (the current format) or a legacy .dat file. It is copied into the library of this project. Matrices that exports saved next to their GGUF files are not moved: "Add to library" copies one in.</p>` : ""}`;
  const r = im.result;
  const mergedPart = r && r.merged && r.merged.entry ? `<div class="ok" id="im-merged-result"><strong>Also merged with Unsloth's published matrix:</strong> ${esc(r.merged.entry.label || r.merged.entry.id)} (${r.merged.entry.size_mb} MB). Your own matrix is kept as well.${(r.merged.warnings || []).map((w) => `<div class="warn small">${esc(w)}</div>`).join("")}</div>` : r && r.merge_error ? `<div class="warn" id="im-merge-error">${esc(r.merge_error)}</div>` : "";
  const mergeResult = r && r.kind === "merge" && r.entry ? `<div class="ok" id="im-merge-result"><strong>Merged matrix ready:</strong> ${esc(r.entry.label || r.entry.id)} (${r.entry.size_mb} MB). It is in the list above and can be used in exports and in Compare quants like any other.</div>
      ${(r.warnings || []).map((w) => `<div class="warn small">${esc(w)}</div>`).join("")}
      ${(r.weights || []).length ? `<ul class="small" id="im-merge-weights-used">${r.weights.map((w) => `<li>${esc(w.label)}: ${w.tokens != null ? Math.round(w.tokens).toLocaleString("en") + " rows of text" : w.chunks != null ? w.chunks + " chunks" : "amount of text unknown"}${w.weight != null && w.weight !== 1 ? ` (weight ${w.weight} used)` : ""}${w.share != null ? ` · about ${Math.round(w.share * 100)} % of the merged statistics` : ""}</li>`).join("")}</ul>` : ""}` : "";
  const adaptResult = r && r.kind === "adapt" && r.entry ? `<div class="ok" id="im-adapt-result"><strong>Adapted matrix ready:</strong> ${esc(r.entry.label || r.entry.id)} (${r.entry.size_mb} MB)${r.numbers ? `: keeps ${r.numbers.layers_after} of ${r.numbers.layers_before} layers; ffn_down ${(r.numbers.ffn_down_before || 0).toLocaleString("en")} → ${(r.numbers.ffn_down_after || 0).toLocaleString("en")} per layer` : ""}${r.adapted && r.adapted.weight !== 1 ? `; weight ${r.adapted.weight}` : ""}. It is in the list above; merge it with a matrix of your own on the cut model (Merge card), or use it in exports and Compare quants like any other.</div>
      ${(r.notes || []).map((t) => `<div class="${t.startsWith("Warning") ? "warn " : "dim "}small">${esc(t)}</div>`).join("")}` : "";
  const result = adaptResult || mergeResult || (r && r.entry ? `<div class="ok"><strong>Matrix ready:</strong> ${esc(r.entry.label || r.entry.id)} (${r.entry.size_mb} MB).${(r.notes || []).length ? "<br>" + r.notes.map(esc).join("<br>") : ""}
      ${r.statistics && r.statistics.lines && r.statistics.lines.length ? `<ul class="small">${r.statistics.lines.map((l) => `<li>${esc(l)}</li>`).join("")}</ul>` : `<span class="dim small"> (This llama-imatrix gave no statistics.)</span>`}</div>${mergedPart}` : "");
  const msg = `${errBox(im.error, "imatrix")}${im.ok ? `<div class="ok">${esc(im.ok)}</div>` : ""}`;
  const log = im.log.length ? `<details><summary>Log</summary><pre class="log">${im.log.slice(-100).map(esc).join("\n")}</pre></details>` : "";
  const prog = im.running && im.pending === "dat" ? `<div class="stepline" aria-live="polite">${esc(im.step)}</div> <button class="secondary" id="im-stop">Stop</button>` : "";
  return `<h1>Importance matrices</h1><p class="lead">Measure which parts of a model matter most, keep the result, and use it for better small GGUF files.</p>
    <div class="card wide"><h2>What it is</h2>${IM_INTRO}${llamaNotice("imatrix")}<div class="row"><button class="secondary" id="im-tostep5">Go to Step 5 (make the files)</button></div></div>
    <div class="card wide"><h2>Your matrices</h2>${list}${add}${prog}</div>
    ${imMergeHtml(st)}${imAdaptHtml(st)}${imFormHtml(st)}${msg}${result}${log}`;
}

async function imRefresh() {
  try { im.state = await invoke("imatrix_list"); } catch (e) { im.error = String(e); }
  if (page === "imatrix") render();
}

function imSpec() {
  const model = im.model === "__typed" ? { path: im.modelPath.trim() } : { model: im.model };
  const source = im.src === "run" ? { kind: "run", run: im.run } : im.src === "model" ? { kind: "model", ...model } : { kind: "gguf", file: im.gguf.trim(), tokenizer: im.tokenizer.trim() || null };
  return { source, calibration: im.cal, text_file: im.cal === "text" ? im.textFile.trim() : null, general: im.cal === "data" && im.share > 0 ? gcSource("imatrix") : null,
    general_share: im.cal === "data" ? im.share / 100 : null, chunks: im.chunks, nextn: im.nextn, dat: im.dat, label: im.label.trim(), ...(im.mergePub2 && im.state && imBase(im.state) ? { merge_published: true } : {}) };
}

function onImatrixEvent(p) {
  if (p.kind === "log") im.log.push(p.text);
  else if (p.kind === "progress") { if (p.text) im.step = p.text; if (p.value != null) im.progress = p.value; }
  else if (p.kind === "done") {
    im.running = false; im.progress = null; im.step = "";
    const d = p.data || {};
    if (im.pending === "dat") im.ok = "The legacy .dat file was made.";
    else if (im.pending === "merge") { im.result = { ...d, kind: "merge" }; im.ok = null; im.mergePicked = []; im.mergeFiles = []; im.mergeLabel = ""; im.mergeWeights = {}; }
    else if (im.pending === "adapt") { im.result = { ...d, kind: "adapt" }; im.ok = null; im.adLabel = ""; }
    else { im.result = d; im.ok = null; im.label = ""; }
    imRefresh();
  } else if (p.kind === "error") {
    im.running = false; im.progress = null; im.step = "";
    im.error = p.text === "Cancelled." ? (im.pending === "create" || im.pending === "adapt" ? "Cancelled. Nothing was kept." : "Cancelled. Nothing was kept; the matrices you merge are unchanged.") : p.text;
    imRefresh();
  }
  if (page === "imatrix") render();
}

async function imRun(kind, cmd, args) {
  im.running = true; im.pending = kind; im.error = null; im.ok = null; im.result = null; im.log = []; im.step = "Starting…"; im.progress = null;
  render();
  try { await invoke(cmd, args); } catch (e) { im.running = false; im.error = String(e); im.step = ""; render(); }
}

function wireImatrix() {
  if (page !== "imatrix" || !im.state) return;
  const on = (id, fn) => { const e = $(id); if (e) e.onclick = fn; };
  const bind = (id, key, num, redraw) => { const e = $(id); if (e) e.onchange = () => { im[key] = num ? Number(e.value) : e.value; if (redraw) render(); }; };
  const pick = (id, key, dir, title, filters) => on(id, async () => {
    const f = await tauri.dialog.open({ multiple: false, directory: dir, title, filters });
    if (f) { im[key] = f; render(); }
  });
  bind("im-src", "src", false, true); bind("im-run", "run", false, true); bind("im-model", "model", false, true); bind("im-modelpath", "modelPath"); bind("im-gguf", "gguf"); bind("im-tok", "tokenizer");
  bind("im-cal", "cal", false, true); bind("im-share", "share", true, true); bind("im-text", "textFile"); bind("im-chunks", "chunks", true, true); bind("im-label", "label");
  const nx = $("im-nextn"); if (nx) nx.onchange = () => { im.nextn = nx.checked; };
  const dt = $("im-dat"); if (dt) dt.onchange = () => { im.dat = dt.checked; };
  pick("im-modelbrowse", "modelPath", true, "Choose the model's folder");
  pick("im-ggufbrowse", "gguf", false, "Choose a GGUF file", [{ name: "GGUF", extensions: ["gguf"] }, { name: "All files", extensions: ["*"] }]);
  pick("im-tokbrowse", "tokenizer", true, "Choose the folder of the model this file was made from");
  pick("im-textbrowse", "textFile", false, "Choose a text file", [{ name: "Text", extensions: ["txt", "text", "md"] }, { name: "All files", extensions: ["*"] }]);
  if (im.cal === "data" && im.share > 0) gcWire("imatrix", render);
  on("im-create", () => imRun("create", "imatrix_create", { spec: imSpec() }));
  const m2 = $("im-mergepub2"); if (m2) m2.onchange = () => { im.mergePub2 = m2.checked; render(); };
  document.querySelectorAll(".im-pick").forEach((c) => c.onchange = () => { im.mergePicked = [...document.querySelectorAll(".im-pick")].filter((x) => x.checked).map((x) => x.value); render(); });
  document.querySelectorAll(".im-unmergefile").forEach((b) => b.onclick = () => { im.mergeFiles.splice(Number(b.dataset.i), 1); render(); });
  const mf = $("im-mergefile"); if (mf) mf.onchange = () => { im.mergeFileInput = mf.value; };
  on("im-mergefileadd", () => { const v = ($("im-mergefile") ? $("im-mergefile").value : im.mergeFileInput).trim(); if (v && !im.mergeFiles.includes(v)) im.mergeFiles.push(v); im.mergeFileInput = ""; render(); });
  on("im-mergefilebrowse", async () => { const f = await tauri.dialog.open({ multiple: false, title: "Choose an importance matrix file", filters: [{ name: "Importance matrix", extensions: ["gguf", "dat"] }, { name: "All files", extensions: ["*"] }] }); if (f && !im.mergeFiles.includes(f)) { im.mergeFiles.push(f); render(); } });
  const mp = $("im-mergepub"); if (mp) mp.onchange = () => { im.mergePub = mp.checked; im.mergePubInfo = null; im.mergePubError = ""; render(); };
  const mb = $("im-mergepubbase"); if (mb) mb.onchange = () => { im.mergePubBase = mb.value; im.mergePubInfo = null; im.mergePubError = ""; render(); };
  const ml = $("im-mergelabel"); if (ml) ml.onchange = () => { im.mergeLabel = ml.value; };
  on("im-pub-lookup", async () => {
    im.mergePubBusy = true; im.mergePubError = ""; im.mergePubInfo = null; render();
    try { im.mergePubInfo = await invoke("imatrix_published", { base: im.mergePubBase }); } catch (e) { im.mergePubError = String(e); }
    im.mergePubBusy = false; render();
  });
  on("im-merge-go", () => imRun("merge", "imatrix_merge", { spec: imMergeSpec() }));
  document.querySelectorAll(".im-mw").forEach((i) => i.onchange = () => { im.mergeWeights[i.dataset.key] = i.value; render(); });
  const adBind = (id, key, redraw) => { const e = $(id); if (e) e.onchange = () => { im[key] = e.value; if (key === "adSrc" || key === "adBase") { im.adPubInfo = null; im.adPubError = ""; } if (redraw) render(); }; };
  adBind("im-ad-src", "adSrc", true); adBind("im-ad-lib", "adId", true); adBind("im-ad-file", "adFile", true); adBind("im-ad-base", "adBase", true); adBind("im-ad-model", "adModel", true);
  adBind("im-ad-kept", "adKept", true); adBind("im-ad-weight", "adWeight", true); adBind("im-ad-label", "adLabel", false);
  on("im-ad-filebrowse", async () => { const f = await tauri.dialog.open({ multiple: false, title: "Choose an importance matrix file", filters: [{ name: "Importance matrix", extensions: ["gguf"] }, { name: "All files", extensions: ["*"] }] }); if (f) { im.adFile = f; render(); } });
  on("im-ad-lookup", async () => {
    im.adPubBusy = true; im.adPubError = ""; im.adPubInfo = null; render();
    try { im.adPubInfo = await invoke("imatrix_published", { base: im.adBase }); } catch (e) { im.adPubError = String(e); }
    im.adPubBusy = false; render();
  });
  on("im-adapt-go", () => imRun("adapt", "imatrix_adapt", { spec: imAdaptSpec(im.state) }));
  imAdaptPreview();
  on("im-stop", () => invoke("imatrix_stop"));
  on("im-tostep5", () => go("use"));
  // the list
  document.querySelectorAll(".im-copy").forEach((b) => b.onclick = () => { im.copyFor = b.dataset.id; render(); });
  document.querySelectorAll(".im-open").forEach((b) => b.onclick = async () => { try { await invoke("imatrix_open_folder", { id: b.dataset.id }); } catch (e) { im.error = String(e); render(); } });
  document.querySelectorAll(".im-dat").forEach((b) => b.onclick = () => imRun("dat", "imatrix_make_dat", { id: b.dataset.id }));
  document.querySelectorAll(".im-del").forEach((b) => b.onclick = async () => {
    if (!confirm("Delete this matrix from the project? Copies you saved elsewhere stay.")) return;
    try { im.state = await invoke("imatrix_delete", { id: b.dataset.id }); im.ok = "Deleted."; im.error = null; } catch (e) { im.error = String(e); im.ok = null; }
    render();
  });
  const cd = $("im-copydir"); if (cd) cd.onchange = () => { im.copyDir = cd.value; };
  on("im-copybrowse", async () => { const d = await tauri.dialog.open({ directory: true, title: "Choose the folder to save the copy in" }); if (d) { im.copyDir = d; render(); } });
  on("im-copycancel", () => { im.copyFor = ""; render(); });
  on("im-copygo", async () => {
    const dest = ($("im-copydir") ? $("im-copydir").value : im.copyDir).trim();
    im.copyDir = dest;
    try { const to = await invoke("imatrix_save_copy", { id: im.copyFor, dest }); im.saved = to; im.ok = `Saved a copy in ${to}`; im.error = null; im.copyFor = ""; } catch (e) { im.error = String(e); im.ok = null; }
    render();
  });
  // adding
  on("im-addopen", () => { im.addOpen = true; render(); });
  on("im-addcancel", () => { im.addOpen = false; render(); });
  const al = $("im-addlabel"); if (al) al.onchange = () => { im.addLabel = al.value; };
  on("im-addfile", async () => {
    const f = await tauri.dialog.open({ multiple: false, title: "Choose an importance matrix file", filters: [{ name: "Importance matrix", extensions: ["gguf", "dat"] }, { name: "All files", extensions: ["*"] }] });
    if (!f) return;
    try { const r = await invoke("imatrix_import", { file: f, label: im.addLabel }); im.state = r.state; im.addOpen = false; im.addLabel = ""; im.ok = "Added to the library." + (r.added && r.added.warning ? " " + r.added.warning : ""); im.error = null; } catch (e) { im.error = String(e); im.ok = null; }
    render();
  });
  const ae = $("im-addexp"); if (ae) ae.onchange = () => { im.addExport = ae.value; render(); };
  on("im-addexpgo", async () => {
    try { const r = await invoke("imatrix_import", { export: im.addExport, label: im.addLabel }); im.state = r.state; im.addExport = ""; im.ok = "Added to the library."; im.error = null; } catch (e) { im.error = String(e); im.ok = null; }
    render();
  });
}

// ------------------------------------------------------------------------------------------- compare quants
// How far is each smaller GGUF from the 16-bit model? llama.cpp's llama-perplexity measures the KL divergence on a test text; the table, the chart and the words are drawn
// from results.json (written by the worker). docs/QUANT_COMPARE.md.
let qk = { state: null, refSel: "", refFile: "", makeSrc: "run", makeRun: "", makeModel: "", picked: null, typed: [], typedInput: "", text: "data", textFile: "", tokRun: "", tokTyped: "", tokMode: "run", chunks: 16,
  keep: false, plan: null, planRef: undefined, planBusy: false, running: false, pending: null, step: "", progress: null, log: [], error: null, ok: null, shown: null, resume: "" };

const QK_INTRO = `<p>A smaller GGUF file (Q4_K_M, a mixed-precision size ...) rounds the model's numbers, so its answers drift a little from the 16-bit original's. This page measures that drift: the 16-bit model reads a test text once and saves how likely it finds every next word; each smaller file then reads the same text and is compared with it. The figure is the <strong>KL divergence</strong> (KLD): 0 means the same answers, the larger, the further away. It also shows how often both files pick the same top word.</p>
  <p class="dim">KLD is a <strong>proxy</strong>: Unsloth reports it for its own quants too, but it does not replace trying the model on your own questions (Step 4 · Test). The test text is, by default, <strong>text the matrix was not measured on</strong> (the held-out part of your saved data), so a matrix does not get a head start. It needs llama.cpp with <code>llama-perplexity</code>: the program can install it for you (the button appears below when it is missing); a copy of your own in <code>LLAMA_CPP_DIR</code> always wins. The 16-bit model's answers are saved in one big temporary file: the page shows its size before anything runs.</p>`;

const qkGb = (b) => (b / 1e9).toFixed(2);
const qkNum = (v, d = 4) => (v == null || Number.isNaN(+v) ? "–" : (+v).toFixed(d));

function qkChartHtml(r) {
  const rows = (r.rows || []).filter((x) => x.status === "done" && x.mean_kld != null && x.size);
  if (rows.length < 1) return "";
  const W = 600, H = 340, L = 64, R = 20, T = 34, B = 48;
  const xs = rows.map((x) => x.size / 1e9), ys = rows.map((x) => Math.max(x.mean_kld, 1e-6));
  const x0 = Math.min(...xs) * 0.95, x1 = Math.max(...xs) * 1.05 || 1;
  const logY = Math.max(...ys) / Math.min(...ys) > 30;
  const f = logY ? (v) => Math.log10(v) : (v) => v;
  const y0 = logY ? f(Math.min(...ys)) - 0.2 : 0, y1 = logY ? f(Math.max(...ys)) + 0.2 : Math.max(...ys) * 1.1;
  const px = (v) => L + ((v - x0) / (x1 - x0 || 1)) * (W - L - R), py = (v) => H - B - ((f(Math.max(v, 1e-6)) - y0) / (y1 - y0 || 1)) * (H - T - B);
  const front = new Set(r.pareto || []);
  const line = rows.filter((x) => front.has(x.id)).sort((a, b) => a.size - b.size);
  const xt = [0, 1, 2, 3, 4].map((i) => x0 + ((x1 - x0) * i) / 4);
  const yt = [0, 1, 2, 3, 4].map((i) => (logY ? Math.pow(10, y0 + ((y1 - y0) * i) / 4) : y0 + ((y1 - y0) * i) / 4));
  const grid = `<style>.qk-g{stroke:var(--line);stroke-width:1}.qk-t{fill:var(--dim);font-size:11px}.qk-l{fill:var(--text);font-size:11px}</style>`
    + xt.map((v) => `<line class="qk-g" x1="${px(v)}" x2="${px(v)}" y1="${T}" y2="${H - B}"/><text class="qk-t" x="${px(v)}" y="${H - B + 16}" text-anchor="middle">${v.toFixed(1)}</text>`).join("")
    + yt.map((v) => `<line class="qk-g" x1="${L}" x2="${W - R}" y1="${py(v)}" y2="${py(v)}"/><text class="qk-t" x="${L - 6}" y="${py(v) + 4}" text-anchor="end">${v < 0.01 ? v.toExponential(0) : v.toFixed(3)}</text>`).join("");
  const poly = line.length > 1 ? `<polyline points="${line.map((x) => `${px(x.size / 1e9)},${py(x.mean_kld)}`).join(" ")}" style="fill:none;stroke:var(--accent);stroke-width:2"/>` : "";
  const dots = rows.map((x) => { const on = front.has(x.id); const cx = px(x.size / 1e9), cy = py(x.mean_kld);
    return `<g><title>${esc(x.label)}: ${qkGb(x.size)} GB, mean KLD ${qkNum(x.mean_kld)}</title><circle cx="${cx}" cy="${cy}" r="5" style="fill:${on ? "var(--accent)" : "var(--panel)"};stroke:var(--accent);stroke-width:2"/><text class="qk-l" x="${cx + 8}" y="${cy - 7}">${esc(String(x.label).slice(0, 26))}</text></g>`; }).join("");
  return `<svg id="qk-chart" role="img" aria-label="Mean KL divergence against file size; lower left is better" viewBox="0 0 ${W} ${H}" width="100%" style="max-width:${W}px;background:var(--panel);border:1px solid var(--line);border-radius:10px">
    <text class="qk-l" x="${L}" y="20" style="font-weight:600">KLD against size${logY ? " (log scale)" : ""}: lower left is better. The line joins the files nothing smaller beats.</text>${grid}${poly}${dots}
    <text class="qk-t" x="${(L + W - R) / 2}" y="${H - 8}" text-anchor="middle">file size (GB)</text><text class="qk-t" transform="translate(14 ${(T + H - B) / 2}) rotate(-90)" text-anchor="middle">mean KL divergence</text></svg>`;
}

function qkResultHtml(r) {
  if (!r) return "";
  const rows = [...(r.rows || [])].sort((a, b) => (a.size || 1e18) - (b.size || 1e18));
  const front = new Set(r.pareto || []);
  const body = rows.map((x) => `<tr class="${x.status === "done" ? "" : "dim"}"><td>${front.has(x.id) ? "● " : ""}${esc(x.label)}${x.partial ? ` <span class="dim" title="Only the running figures were printed, not the final statistics">(running figure)</span>` : ""}</td><td>${x.size ? qkGb(x.size) : "–"}</td><td>${x.bpw != null ? (+x.bpw).toFixed(2) : "–"}</td>
    <td>${x.status === "done" ? qkNum(x.mean_kld) : `<span class="err" title="${esc(x.error || "")}">${x.status === "pending" ? "not measured yet" : "failed"}</span>`}</td><td>${qkNum(x.median_kld)}</td><td>${qkNum(x.p99)}</td><td>${qkNum(x.max_kld, 3)}</td>
    <td>${x.same_top != null ? (+x.same_top).toFixed(1) + " %" : "–"}</td><td>${x.ppl != null ? (+x.ppl).toFixed(3) : "–"}</td></tr>${x.status === "failed" && x.error ? `<tr class="dim small"><td colspan="9">${esc(x.error)}</td></tr>` : ""}`).join("");
  const ref = r.reference || {};
  return `<div class="card wide" id="qk-result"><h2>Result: ${esc(r.id || "")}</h2>
    <p class="dim small">Reference: ${esc(ref.name || "")}${ref.params ? ` · ${(ref.params / 1e9).toFixed(1)} billion weights` : ""} · test text: ${esc((r.text || {}).source || "")} · ${(r.settings || {}).chunks} chunks of ${(r.settings || {}).ctx} tokens · ${esc((r.key || {}).llama_cpp || "llama.cpp version unknown")}${r.state && r.state !== "done" ? ` · <strong>${esc(r.state)}</strong>` : ""}</p>
    <div style="overflow-x:auto"><table class="qk-table"><thead><tr><th>File (● on the line)</th><th>Size (GB)</th><th>Bits per weight</th><th>Mean KLD</th><th>Median</th><th>99 %</th><th>Max</th><th>Same top word</th><th>PPL</th></tr></thead><tbody>${body}</tbody></table></div>
    <p class="dim small">Size in GB (10⁹ bytes); bits per weight = file size × 8 ÷ the weights in the file. Mean KLD is the figure to compare; the 99 % and maximum show the rare big misses. Lower is closer to the 16-bit model.</p>
    ${qkChartHtml(r)}
    <h3>In words</h3><ul>${(r.verdict || []).map((v) => `<li>${esc(v)}</li>`).join("")}</ul>
    ${(r.notes || []).length ? `<p class="dim small">${r.notes.map(esc).join("<br>")}</p>` : ""}</div>`;
}

function qkEntryHtml(e) {
  const states = { done: "finished", stopped: "stopped (the finished files are kept)", failed: "ended with an error", running: "was running" };
  const off = qk.running || e.running;
  return `<li class="imentry"><strong>${esc(e.id)}</strong> <span class="dim small">${esc(states[e.state] || e.state || "")}</span><br>
    <span class="dim small">Reference ${esc(e.reference || "?")} · ${e.rows_done} of ${e.rows} files measured · ${e.chunks} chunks · ${esc(String(e.created || "").slice(0, 10))}${e.base_kept ? ` · the 16-bit model's answers are kept (${e.base_mb} MB)` : ""}</span>
    <div class="row"><button class="secondary qk-show" data-id="${esc(e.id)}">Show</button>
      ${e.rows_done < e.rows || e.state === "stopped" ? `<button class="secondary qk-continue" data-id="${esc(e.id)}" ${off ? "disabled" : ""} title="Goes on with the files that are not measured yet, without reading the text again when the saved answers are still there">Continue</button>` : ""}
      ${e.base_kept ? `<button class="secondary qk-dellog" data-id="${esc(e.id)}" ${off ? "disabled" : ""} title="Frees ${e.base_mb} MB; the table stays">Delete the saved answers (${e.base_mb} MB)</button>` : ""}
      <button class="secondary qk-del" data-id="${esc(e.id)}" ${off ? "disabled" : ""}>Delete</button></div></li>`;
}

function qkRefPath(st) {
  if (qk.refSel === "__file") return qk.refFile.trim();
  if (qk.refSel === "__make") return "";
  return qk.refSel;
}

function qkPlanHtml() {
  const pl = qk.plan;
  if (!pl) return qk.planBusy ? `<p class="dim small">Checking the disk…</p>` : "";
  const rows = pl.options.map((o) => `<tr class="${o.fits ? "" : "dim"}"><td>${o.chunks}</td><td>${o.scored.toLocaleString("en")}</td><td>${o.gb} GB</td><td>${o.fits ? "fits" : "does not fit"}</td></tr>`).join("");
  return `<div id="qk-plan"><table class="qk-table"><thead><tr><th>Chunks</th><th>Words compared</th><th>Temporary file</th><th>Disk</th></tr></thead><tbody>${rows}</tbody></table>
    <p class="dim small">${pl.vocab_assumed ? "The reference's vocabulary is not known yet, so " + pl.vocab.toLocaleString("en") + " words are assumed. " : `The vocabulary has ${pl.vocab.toLocaleString("en")} words. `}The temporary file holds the 16-bit model's answers for every compared word (about 2 bytes per vocabulary word) and is deleted when the comparison ends${qk.keep ? " (you chose to keep it)" : ""}. Free on this disk: ${pl.free != null ? qkGb(pl.free) + " GB" : "unknown"}; a tenth is kept in reserve.</p>
    ${pl.refusal ? `<div class="err" id="qk-refusal">${esc(pl.refusal)}</div>` : ""}</div>`;
}

function qkFormHtml(st) {
  const c = st.candidates || { references: [], quants: [] };
  if (!qk.refSel) qk.refSel = c.references.length ? c.references[0].path : (st.runs || []).length ? "__make" : "__file";
  if (qk.picked === null) qk.picked = [];
  if (!qk.makeRun && (st.runs || []).length) qk.makeRun = st.runs[0].name;
  if (!qk.makeModel && (st.models || []).length) qk.makeModel = st.models[0].id;
  if (!qk.tokRun && (st.runs || []).length) qk.tokRun = st.runs[0].name;
  const refOpts = c.references.map((r) => `<option value="${esc(r.path)}" ${qk.refSel === r.path ? "selected" : ""}>${esc(r.label)} · ${r.size_mb} MB</option>`).join("")
    + `<option value="__file" ${qk.refSel === "__file" ? "selected" : ""}>A 16-bit GGUF file I choose…</option><option value="__make" ${qk.refSel === "__make" ? "selected" : ""}>Make a 16-bit file with llama.cpp…</option>`;
  let refExtra = "";
  if (qk.refSel === "__file") refExtra = `<div class="row"><label>File <input type="text" id="qk-reffile" value="${esc(qk.refFile)}" placeholder="C:\\models\\model-f16.gguf" style="min-width:300px"></label><button class="secondary" id="qk-refbrowse">Browse…</button></div>
    <p class="dim small">An F16 or BF16 file of the same model as the files below. A file that is already quantised can be chosen, but then the figures say how far the others are from <em>it</em>.</p>`;
  if (qk.refSel === "__make") refExtra = `<div class="row"><label>Made from <select id="qk-makesrc"><option value="run" ${qk.makeSrc === "run" ? "selected" : ""}>A trained run</option><option value="model" ${qk.makeSrc === "model" ? "selected" : ""}>A model of the list</option></select></label>
    ${qk.makeSrc === "run" ? `<label>Run <select id="qk-makerun">${(st.runs || []).map((r) => `<option value="${esc(r.name)}" ${qk.makeRun === r.name ? "selected" : ""}>${esc(r.name)} · ${esc(r.model)}</option>`).join("")}</select></label>`
      : `<label>Model <select id="qk-makemodel">${(st.models || []).map((m) => `<option value="${esc(m.id)}" ${qk.makeModel === m.id ? "selected" : ""}>${esc(m.name)}</option>`).join("")}</select></label>`}</div>
    <p class="dim small">The model is merged (a trained run) and converted to a 16-bit GGUF file with llama.cpp's converter; it needs the training software, about two bytes per weight of free disk (a 27 B model: 54 GB) and some minutes. The file is kept in the comparison's folder and deleted with it.</p>`;
  const quants = c.quants.length ? c.quants.map((q) => `<li><label class="opt"><input type="checkbox" class="qk-pick" value="${esc(q.path)}" ${qk.picked.includes(q.path) ? "checked" : ""}> ${esc(q.label)} <span class="dim small">· ${q.size_mb} MB · ${esc(q.export)}</span></label></li>`).join("")
    : `<li class="dim">No exported GGUF files in this project yet. Make some on Step 5, or add files by their path.</li>`;
  const typed = qk.typed.map((t, i) => `<li><span>${esc(t)}</span> <button class="link qk-untype" data-i="${i}">Remove</button></li>`).join("");
  const run = (st.runs || []).map((r) => `<option value="${esc(r.name)}" ${qk.tokRun === r.name && qk.tokMode === "run" ? "selected" : ""}>${esc(r.name)} · ${esc(r.model)}</option>`).join("");
  const dataInfo = st.data_version == null ? `<p class="warn small">There is no saved version of the training data yet (Step 2). Save one, or choose a text file of your own.</p>`
    : `<p class="dim small">${st.has_test_rows ? `The held-out questions of data version ${st.data_version}: conversations the training never saw, and (by design) not the text any matrix was measured on.` : `Version ${st.data_version} holds out no test questions, so a sample of its training rows is used (another sample than the matrix calibration's); files measured on text they were trained on look a little better than on new text, the ranking is still fair.`} They are written through the model's chat template with its special markers taken out (llama-perplexity cannot read them as markers).</p>`;
  const textBlock = `<div class="row"><label>Test text <select id="qk-text"><option value="data" ${qk.text === "data" ? "selected" : ""}>The held-out part of my saved data</option><option value="file" ${qk.text === "file" ? "selected" : ""}>A text file of my own</option></select></label></div>`
    + (qk.text === "data" ? dataInfo + `<div class="row"><label>The files were made from <select id="qk-tokrun">${run}<option value="__typed" ${qk.tokMode === "typed" || !(st.runs || []).length ? "selected" : ""}>A model folder or Hugging Face id…</option></select></label>${qk.tokMode === "typed" || !(st.runs || []).length ? `<label>Folder or id <input type="text" id="qk-toktyped" value="${esc(qk.tokTyped)}" placeholder="owner/name" style="min-width:260px"></label>` : ""}</div>`
      : `<div class="row"><label>Text file <input type="text" id="qk-textfile" value="${esc(qk.textFile)}" placeholder="C:\\texts\\test.txt" style="min-width:300px"></label><button class="secondary" id="qk-textbrowse">Browse…</button></div><p class="dim small">Plain text, used as it is (a few thousand words or more).</p>`);
  const pl = qk.plan;
  const names = (n) => { const o = pl && pl.options.find((x) => x.chunks === n); return `${n} chunks${o ? ` (${o.gb} GB temporary file${o.fits ? "" : ", too big for the disk"})` : ""}${n === 16 ? " · recommended" : ""}`; };
  const refPath = qkRefPath(st);
  const total = qk.picked.length + qk.typed.length;
  const typedModel = qk.tokMode === "typed" || !(st.runs || []).length;
  const noData = qk.text === "data" && (st.data_version == null || (typedModel && !qk.tokTyped.trim()));
  const chosen = pl && pl.options.find((o) => o.chunks === qk.chunks);
  const why = !st.env_ready ? "The training software is not installed yet (Training setup): it runs the comparison." : qk.refSel === "__file" && !qk.refFile.trim() ? "Choose the 16-bit file."
    : !total ? "Choose at least one file to compare." : qk.text === "file" && !qk.textFile.trim() ? "Choose the text file." : noData ? "Choose the test text (save a data version, or use a text file)."
    : pl && pl.refusal ? pl.refusal : chosen && !chosen.fits ? "That amount of text needs more free disk than there is. Choose fewer chunks." : "";
  const prog = qk.running ? `<div class="stepline" aria-live="polite">${esc(qk.step)}</div>${qk.progress != null ? `<progress max="1" value="${qk.progress}"></progress>` : ""} <button class="secondary" id="qk-stop" title="llama.cpp is stopped at once; the files that are finished stay measured">Stop</button>` : "";
  return `<div class="card wide"><h2>New comparison</h2>
    <h3>1 · The 16-bit reference</h3><div class="row"><label>Reference <select id="qk-ref" ${qk.running ? "disabled" : ""}>${refOpts}</select></label></div>${refExtra}
    ${c.references.length ? "" : `<p class="dim small">No 16-bit file was found in your exports (a size "F16" in Step 5 makes one). Choose a file, or let llama.cpp make one.</p>`}
    <h3>2 · The files to compare</h3><ul class="exlist" id="qk-quants">${quants}${typed}</ul>
    <div class="row"><label>Add a file by its path <input type="text" id="qk-typed" value="${esc(qk.typedInput)}" placeholder="C:\\models\\model-Q4_K_M.gguf" style="min-width:300px"></label><button class="secondary" id="qk-typeadd">Add</button><button class="secondary" id="qk-typebrowse">Browse…</button></div>
    <p class="dim small">Files made with and without an importance matrix, and the mixed-precision sizes, go in the same table: tick them all.</p>
    <h3>3 · The test text</h3>${textBlock}
    <h3>4 · How much text</h3><div class="row"><label>Amount <select id="qk-chunks">${[4, 8, 16, 32, 64, 128].map((n) => `<option value="${n}" ${qk.chunks === n ? "selected" : ""}>${names(n)}</option>`).join("")}</select></label>
      <label class="opt"><input type="checkbox" id="qk-keep" ${qk.keep ? "checked" : ""}> Keep the 16-bit model's answers afterwards</label></div>${qkPlanHtml()}
    <p class="dim small">More text gives steadier figures but a bigger temporary file and a longer run: the 16-bit model reads the text once, then every file reads it again.</p>
    ${why ? `<p class="dim small" id="qk-why">${esc(why)}</p>` : ""}
    ${pfBox("quantcheck", {})}<div class="row"><button class="primary" id="qk-start" ${qk.running || why || pfBlocked("quantcheck") ? "disabled" : ""}>Compare</button>${quBtn("quantcheck", why ? "disabled" : "")}</div>${prog}</div>`;
}

function renderQuantcheck() {
  if (!state.bundle) return `<h1>Compare quants</h1><p class="lead">Import your world data first (Step 1).</p>`;
  const st = qk.state;
  if (!st) return `<h1>Compare quants</h1><p class="lead">Loading…</p>`;
  const entries = st.entries || [];
  const list = entries.length ? `<ul class="exlist">${entries.map(qkEntryHtml).join("")}</ul>` : `<p class="dim">No comparison saved in this project yet. Make one below.</p>`;
  const msg = `${errBox(qk.error, "perplexity")}${qk.ok ? `<div class="ok">${esc(qk.ok)}</div>` : ""}`;
  const log = qk.log.length ? `<details><summary>Log</summary><pre class="log">${qk.log.slice(-100).map(esc).join("\n")}</pre></details>` : "";
  return `<h1>Compare quants</h1><p class="lead">See how much of the 16-bit model each smaller file keeps, and which size is worth it.</p>
    <div class="card wide"><h2>What it measures</h2>${QK_INTRO}${llamaNotice("perplexity")}<div class="row"><button class="secondary" id="qk-tostep5">Go to Step 5 (make the files)</button></div></div>
    <div class="card wide"><h2>Your comparisons</h2>${list}</div>${qkFormHtml(st)}${msg}${log}${qkResultHtml(qk.shown)}`;
}

async function qkRefresh() {
  try { qk.state = await invoke("quantcheck_state"); } catch (e) { qk.error = String(e); }
  if (page === "quantcheck") { render(); qkCheckPlan(); }
}

// the disk need of the 16-bit model's answers, for the chosen reference: asked once per reference
function qkCheckPlan() {
  if (!qk.state || page !== "quantcheck") return;
  const ref = qkRefPath(qk.state);
  if (qk.planRef === ref || qk.planBusy) return;
  qk.planRef = ref; qk.planBusy = true;
  invoke("quantcheck_plan", { reference: ref || null })
    .then((r) => { qk.plan = r; if (r && r.default_chunks && !r.options.find((o) => o.chunks === qk.chunks && o.fits)) qk.chunks = r.default_chunks; })
    .catch((e) => { qk.plan = null; qk.error = String(e); })
    .finally(() => { qk.planBusy = false; if (page === "quantcheck") render(); });
}

function qkSpec() {
  const st = qk.state, c = st.candidates;
  const picks = c.quants.filter((q) => qk.picked.includes(q.path)).map((q) => ({ path: q.path, label: q.label, group: q.group, matrix: q.matrix, mixed: q.mixed }))
    .concat(qk.typed.map((t) => ({ path: t, label: t.split(/[\\/]/).pop(), group: null, matrix: null, mixed: null })));
  const reference = qk.refSel === "__make" ? { kind: "make", source: qk.makeSrc === "run" ? { kind: "run", run: qk.makeRun } : { kind: "model", model: qk.makeModel } } : { kind: "file", file: qkRefPath(st) };
  const typedModel = qk.tokMode === "typed" || !(st.runs || []).length;
  const text = qk.text === "file" ? { kind: "file", file: qk.textFile.trim() } : { kind: "data", run: typedModel ? null : qk.tokRun, tokenizer: typedModel ? qk.tokTyped.trim() || null : null };
  return { reference, quants: picks, text, chunks: qk.chunks, keep_logits: qk.keep, resume: qk.resume || null };
}

function onQuantcheckEvent(p) {
  if (p.kind === "log") qk.log.push(p.text);
  else if (p.kind === "progress") { if (p.text) qk.step = p.text; if (p.value != null) qk.progress = p.value; }
  else if (p.kind === "done") {
    qk.running = false; qk.progress = null; qk.step = ""; qk.resume = "";
    qk.shown = p.data || null; qk.ok = "The comparison is ready."; qk.error = null;
    qkRefresh();
  } else if (p.kind === "error") {
    qk.running = false; qk.progress = null; qk.step = ""; qk.resume = "";
    qk.error = p.text === "Cancelled." ? "Stopped. The files that were already measured are kept (see Your comparisons: Continue)." : p.text;
    qkRefresh();
  }
  if (page === "quantcheck") render();
}

async function qkRun(spec) {
  qk.running = true; qk.error = null; qk.ok = null; qk.shown = null; qk.log = []; qk.step = "Starting…"; qk.progress = null;
  render();
  try { await invoke("quantcheck_start", { spec }); } catch (e) { qk.running = false; qk.error = String(e); qk.step = ""; render(); }
}

function wireQuantcheck() {
  if (page !== "quantcheck" || !qk.state) return;
  const on = (id, fn) => { const e = $(id); if (e) e.onclick = fn; };
  const bind = (id, key, num, redraw) => { const e = $(id); if (e) e.onchange = () => { qk[key] = num ? Number(e.value) : e.value; if (redraw) render(); }; };
  bind("qk-ref", "refSel", false, true); bind("qk-reffile", "refFile", false, false); bind("qk-makesrc", "makeSrc", false, true); bind("qk-makerun", "makeRun"); bind("qk-makemodel", "makeModel");
  bind("qk-text", "text", false, true); bind("qk-textfile", "textFile"); bind("qk-chunks", "chunks", true, true); bind("qk-toktyped", "tokTyped");
  const tr = $("qk-tokrun"); if (tr) tr.onchange = () => { if (tr.value === "__typed") { qk.tokMode = "typed"; } else { qk.tokRun = tr.value; qk.tokMode = "run"; } render(); };
  const kp = $("qk-keep"); if (kp) kp.onchange = () => { qk.keep = kp.checked; render(); };
  const ty = $("qk-typed"); if (ty) ty.onchange = () => { qk.typedInput = ty.value; };
  document.querySelectorAll(".qk-pick").forEach((c) => c.onchange = () => { qk.picked = [...document.querySelectorAll(".qk-pick")].filter((x) => x.checked).map((x) => x.value); render(); });
  document.querySelectorAll(".qk-untype").forEach((b) => b.onclick = () => { qk.typed.splice(Number(b.dataset.i), 1); render(); });
  on("qk-typeadd", () => { const v = ($("qk-typed") ? $("qk-typed").value : qk.typedInput).trim(); if (v && !qk.typed.includes(v)) qk.typed.push(v); qk.typedInput = ""; render(); });
  const pick = (id, apply, dir, title, filters) => on(id, async () => { const f = await tauri.dialog.open({ multiple: false, directory: dir, title, filters }); if (f) { apply(f); render(); } });
  const gg = [{ name: "GGUF", extensions: ["gguf"] }, { name: "All files", extensions: ["*"] }];
  pick("qk-refbrowse", (f) => { qk.refFile = f; }, false, "Choose the 16-bit GGUF file", gg);
  pick("qk-typebrowse", (f) => { if (!qk.typed.includes(f)) qk.typed.push(f); }, false, "Choose a GGUF file to compare", gg);
  pick("qk-textbrowse", (f) => { qk.textFile = f; }, false, "Choose a text file", [{ name: "Text", extensions: ["txt", "text", "md"] }, { name: "All files", extensions: ["*"] }]);
  on("qk-tostep5", () => go("use"));
  on("qk-start", () => qkRun(qkSpec()));
  on("qk-stop", () => invoke("quantcheck_stop"));
  document.querySelectorAll(".qk-show").forEach((b) => b.onclick = async () => { try { qk.shown = await invoke("quantcheck_get", { id: b.dataset.id }); qk.error = null; } catch (e) { qk.error = String(e); } render(); });
  document.querySelectorAll(".qk-continue").forEach((b) => b.onclick = async () => {
    // goes on with the files the comparison had, plus the ones ticked now; the saved answers and the saved test text are used when they are still there
    try {
      const r = await invoke("quantcheck_get", { id: b.dataset.id });
      const spec = qkSpec(); spec.resume = b.dataset.id; spec.chunks = (r.settings || {}).chunks || spec.chunks; spec.reference = { kind: "file", file: (r.reference || {}).path };
      const have = new Set(spec.quants.map((q) => q.path));
      (r.rows || []).forEach((x) => { if (!have.has(x.path)) spec.quants.push({ path: x.path, label: x.label, group: x.group, matrix: x.matrix, mixed: x.mixed }); });
      qk.resume = b.dataset.id; qkRun(spec);
    } catch (e) { qk.error = String(e); render(); }
  });
  document.querySelectorAll(".qk-dellog").forEach((b) => b.onclick = async () => {
    if (!confirm("Delete the saved answers of the 16-bit model? The table stays; continuing later has to read the text again.")) return;
    try { qk.state = await invoke("quantcheck_delete_logits", { id: b.dataset.id }); qk.ok = "The saved answers were deleted."; qk.error = null; } catch (e) { qk.error = String(e); qk.ok = null; }
    render();
  });
  document.querySelectorAll(".qk-del").forEach((b) => b.onclick = async () => {
    if (!confirm("Delete this comparison with its results? A 16-bit reference that the program made for it is deleted too.")) return;
    try { qk.state = await invoke("quantcheck_delete", { id: b.dataset.id }); if (qk.shown && qk.shown.id === b.dataset.id) qk.shown = null; qk.ok = "Deleted."; qk.error = null; } catch (e) { qk.error = String(e); qk.ok = null; }
    render();
  });
  qkCheckPlan();
}

// the "Compare these sizes" button of the export results: the files of one run, ticked
async function qkFromRun(run) {
  await qkRefresh();
  const c = (qk.state && qk.state.candidates) || { quants: [] };
  qk.picked = c.quants.filter((q) => q.run === run).map((q) => q.path);
  qk.tokRun = run; qk.tokMode = "run"; qk.shown = null;
  go("quantcheck");
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
    ${un.busy ? `<div class="stepline" aria-live="polite">Asking Unsloth…</div>` : ""}${un.result ? `<div class="ok">${esc(un.result)}</div>` : ""}${un.error ? `<div class="err">${esc(un.error)}</div>` : ""}</div>`;
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

// ------------------------------------------------------------------------------------------------------------ the job queue (crates/ndforge-core/src/queue.rs)
// Jobs the person wants done one after the other. "Add to queue" sits next to Start; the list is on the Home page. The program starts the next job by itself when one ends
// (after its checklist: an entry that fails it is skipped with the reason) and waits after a restart or a Stop until "Run the queue" is pressed.
const qu = { view: null, loaded: false, note: "" };
async function quLoad() {
  try { qu.view = await invoke("queue_list"); } catch (_) { qu.view = null; }
  qu.loaded = true;
  if (page === "welcome") render();
}
async function quDo(cmd, args) {
  try { qu.view = await invoke(cmd, args || {}); qu.note = ""; } catch (e) { qu.note = String(e); }
  render();
}
const quBtn = (kind, extra) => `<button class="secondary" data-quadd="${kind}" ${extra || ""} title="Runs after what is running now; the program starts it by itself when the computer is free">Add to queue</button>`;
const QU_ARGS = {
  auto: () => ({ spec: acSpec(), resume: null }),
  export: () => usMakeArgs(),
  export_model: () => rsxMakeArgs(),
  heal: () => ({ name: rs.hl.name, spec: rsHealSpec() }),
  quantcheck: () => ({ spec: qkSpec() }),
  think_check: (b) => { const m = (rs.state.made || []).find((x) => x.name === rs.tk.name); return { spec: { name: rs.tk.name, target: b.dataset.target, quick: m ? tkQuick(m) : true, gguf: b.dataset.target === "gguf" ? rs.tk.gguf : null, ...tkSpecExtra() } }; },
};
async function quAdd(kind, btn) {
  const make = QU_ARGS[kind];
  if (!make) return;
  try { qu.view = await invoke("queue_add", { kind, args: make(btn) }); qu.note = "Added to the queue (see the Home page)."; } catch (e) { qu.note = String(e); }
  render();
}
const QU_STATUS = { queued: "waiting", running: "running", done: "done", failed: "failed", stopped: "stopped", skipped: "skipped" };
function quCard() {
  const v = qu.view;
  if (!v || !v.items || !v.items.length) return qu.note ? `<p class="dim small">${esc(qu.note)}</p>` : "";
  const rows = v.items.map((i, n) => `<li class="qu-${esc(i.status)}"><strong>${esc(i.label || i.kind)}</strong> · ${esc(QU_STATUS[i.status] || i.status)}${i.kind === "auto" && i.status === "running" && ac.prog ? ` <span class="dim" data-quprog="auto">${Math.round(ac.prog.pct)} %${ac.prog.state !== "estimating" && ac.prog.left_s ? ` · ${esc(acProgLeft(ac.prog).replace(/ \(done around.*$/, ""))}` : ""}</span>` : ""}${i.error ? ` <span class="dim">${esc(fixActions(i.error).text)}</span>` : ""}
      ${i.status === "queued" ? `<button class="secondary" data-qumove="${esc(i.id)}" data-to="${n - 1}" aria-label="Move up">↑</button><button class="secondary" data-qumove="${esc(i.id)}" data-to="${n + 1}" aria-label="Move down">↓</button>` : ""}
      ${i.status !== "running" ? `<button class="secondary" data-quremove="${esc(i.id)}">Remove</button>` : ""}</li>`).join("");
  const waiting = v.queued || 0;
  return `<div class="card wide" id="qu-card"><h2>Job queue</h2><ul class="qu-list">${rows}</ul>
    <div class="row">${v.paused ? `<button class="primary" id="qu-run" ${waiting ? "" : "disabled"}>Run the queue</button><span class="dim small">The queue is waiting${waiting ? ": nothing starts until you press Run" : ""}.</span>` : `<button class="secondary" id="qu-pause">Pause the queue</button>`}
      <button class="secondary" id="qu-clear">Clear the finished entries</button></div>${qu.note ? `<p class="dim small">${esc(qu.note)}</p>` : ""}</div>`;
}
function wireQueue() {
  document.querySelectorAll("[data-quadd]").forEach((b) => b.onclick = () => quAdd(b.dataset.quadd, b));
  document.querySelectorAll("[data-quremove]").forEach((b) => b.onclick = () => quDo("queue_remove", { id: b.dataset.quremove }));
  document.querySelectorAll("[data-qumove]").forEach((b) => b.onclick = () => quDo("queue_move", { id: b.dataset.qumove, to: Math.max(0, +b.dataset.to) }));
  const on = (id, fn) => { const e = $(id); if (e) e.onclick = fn; };
  on("qu-run", () => quDo("queue_pause", { on: false }));
  on("qu-pause", () => quDo("queue_pause", { on: true }));
  on("qu-clear", () => quDo("queue_clear_finished"));
}

// ------------------------------------------------------------------------------------------------------------ the preflight checklist (crates/ndforge-core/src/preflight.rs)
// One list above a Start button: what the job needs and whether it is there (green / yellow / red). A red light disables Start. The program is asked again when the page's
// facts change, after 15 seconds, or when something that fixes a light has finished (pfReset).
const pf = {};
function pfReset() { for (const k of Object.keys(pf)) delete pf[k]; }
function pfAsk(kind, spec) {
  const key = JSON.stringify(spec || {});
  const e = pf[kind] || (pf[kind] = { key: null, asked: null, at: 0, checks: null, can_start: true, busy: false });
  if (e.busy || (e.asked === key && Date.now() - e.at < 15000)) return e;
  e.asked = key;
  e.busy = true;
  Promise.resolve(invoke("preflight", { kind, spec: spec || {} }))
    .then((r) => { e.checks = r && Array.isArray(r.checks) ? r.checks : null; e.can_start = !(r && r.can_start === false); })
    .catch(() => { e.checks = null; e.can_start = true; })
    .finally(() => { e.key = key; e.at = Date.now(); e.busy = false; try { render(); } catch (_) { /* a preview without a window */ } });
  return e;
}
function pfBox(kind, spec) {
  const e = pfAsk(kind, spec);
  if (!e.checks || !e.checks.length) return "";
  const bad = e.checks.some((c) => c.light === "fail");
  const mark = { ok: "✓", warn: "!", fail: "✕" };
  const rows = e.checks.map((c) => `<li class="pf-${esc(c.light)}"><span class="pf-dot" aria-hidden="true">${mark[c.light] || "·"}</span> <strong>${esc(c.label)}</strong> ${esc(c.detail)}${c.fix && c.light !== "ok" ? ` <span class="dim">${esc(c.fix)}</span>` : ""}${c.action === "keep_awake_on" && c.light !== "ok" ? ` <button class="secondary" data-pf-action="keep_awake_on">Keep the computer awake</button>` : ""}</li>`).join("");
  return `<div class="pf ${bad ? "pf-bad" : ""}" role="status" aria-live="polite" data-pf="${esc(kind)}"><div class="small dim">Before you start</div><ul>${rows}</ul></div>`;
}
// a button next to a light (the program names it in the check's `action`): "keep_awake_on" switches the setting on, then the list is asked again
async function pfAction(act) {
  if (act !== "keep_awake_on") return;
  try { setState(await invoke("set_setting", { key: "keep_awake", value: true })); } catch (_) { /* the list stays as it was */ }
  pfReset();
  try { render(); } catch (_) { /* a preview without a window */ }
}
try { document.addEventListener("click", (ev) => { const t = ev.target && ev.target.closest && ev.target.closest("[data-pf-action]"); if (t) pfAction(t.dataset.pfAction); }); } catch (_) { /* a preview without a window */ }
const pfBlocked = (kind) => { const e = pf[kind]; return !!(e && e.checks && e.can_start === false); };

// ------------------------------------------------------------------------------------------------------------ llama.cpp: the installer (docs/LLAMA_CPP_INSTALL.md)
// One state for every place that needs llama.cpp (Training setup, Step 5, Importance matrices, Compare quants): the status the workers will find, the plan to confirm, the progress.
let lc = { status: null, planning: false, confirm: null, running: false, step: "", progress: null, bytes: null, log: [], error: null, result: null, update: null, updating: false, backend: "auto", pendingAuto: false };
const LLAMA_MARK = "[llama_cpp_missing]";                // the program ends the text of a worker error with this when llama.cpp (or a part of it) was not found
const llamaMarked = (t) => typeof t === "string" && t.includes(LLAMA_MARK);
const stripLlamaMark = (t) => (typeof t === "string" ? t.split(LLAMA_MARK).join("").trim() : t);

// ------------------------------------------------------------------------------------------------------------ worker errors -> fix-it buttons
// The program ends a worker error with ` [fix:<code>]` for some codes (envmgr/jobs.rs FIX_CODES, and missing_module for a missing Python library). fixActions() takes the marks out of
// the text for display and gives the buttons for them; fixDo() does what a button says. Codes without a button here (distill_unstable, reshape_refused, bad_job) only lose the mark.
const FIX_RE = /\s*\[fix:([a-z_]+)\]/g;
const FIX_BUTTONS = { disk_full: [["open", "Open the project folder"]], distill_oom: [["safer", "Use safer settings"]], missing_module: [["repair", "Repair the training software"]] };
function fixActions(text) {
  if (typeof text !== "string") return { text, html: "" };
  const codes = [];
  const shown = text.replace(FIX_RE, (_, c) => { codes.push(c); return ""; }).trim();
  const btns = [...new Set(codes)].flatMap((c) => FIX_BUTTONS[c] || []).map(([act, label]) => `<button class="secondary" data-fix="${act}">${esc(label)}</button>`).join(" ");
  return { text: shown, html: btns ? `<div class="row fix-row">${btns}</div>` : "" };
}
// `<div class="err">…</div>` for a worker error, with its buttons (tag: "div" or "p")
function errDiv(text, tag) {
  const f = fixActions(text);
  return `<${tag || "div"} class="err">${esc(f.text)}</${tag || "div"}>${f.html}`;
}
function healUseSafer() {
  const s = rs.hl && rs.hl.opts && rs.hl.opts.safer, hs = rs.hlSpec;
  if (!s || !hs) return false;
  hs.checkpointing = s.checkpointing; hs.batch = s.batch; hs.per_update = s.per_update; hs.kl_chunk = s.kl_chunk; hs.kbit_upcast = s.kbit_upcast === true ? true : null; hs.head_checkpoint = s.head_checkpoint === "off" ? "off" : null;
  return true;
}
async function fixDo(act) {
  try {
    if (act === "open") await invoke("imatrix_open_folder", { path: (state && state.project && state.project.folder) || "" });
    else if (act === "safer") {
      if (rs.hl && rs.hl.name) await rsLoadHeal(rs.hl.name);
      if (healUseSafer()) { render(); rsHealEstimate(); } else render();
    } else if (act === "repair") { await go("setup"); startEnv("env_install", { unsloth: !env.noUnsloth }); }
  } catch (e) { alert(String(e)); }
}
const LC_WHY = { all: "It is needed to make GGUF files with llama.cpp (the importance matrix, the mixed sizes).", imatrix: "It is needed to measure an importance matrix.", perplexity: "Compare quants needs its llama-perplexity program.", merge: "Merging matrices needs its llama-imatrix program." };

// is a part that the page needs missing? (null status = not known yet: say nothing)
function lcGap(need) {
  const s = lc.status;
  if (!s) return false;
  const t = s.tools || {};
  if (need === "perplexity") return !t.perplexity;
  if (need === "imatrix") return !t.imatrix || (!s.converter_found && typeof im !== "undefined" && im.src !== "gguf");
  if (need === "merge") return !t.imatrix;
  return !s.found;
}

async function lcRefresh() {
  const had = !!(lc.status && lc.status.found);
  try { lc.status = await invoke("llamacpp_status"); } catch (_) { lc.status = null; }
  if (had !== !!(lc.status && lc.status.found)) pfReset();            // llama.cpp appeared or went: the lists must be asked again
  if (["setup", "use", "imatrix", "quantcheck"].includes(page)) render();
}

const lcMb = (n) => (n >= 1e9 ? (n / 1e9).toFixed(1) + " GB" : Math.round(n / 1e6) + " MB");

// the plan is shown first (what, from where, how big); nothing is downloaded until the person agrees (one click). `auto`: the person already agreed (the CPU build offer).
async function lcPlan(backend, auto) {
  lc.planning = true; lc.error = null; lc.result = null; lc.confirm = null; lc.pendingAuto = !!auto;
  render();
  try {
    const plan = await invoke("llamacpp_plan", { backend: backend || lc.backend || "auto", tag: null });
    lc.planning = false;
    if (auto) { lc.confirm = plan; await lcInstall(); return; }
    lc.confirm = plan;
  } catch (e) { lc.planning = false; lc.error = String(e); }
  render();
}

async function lcInstall() {
  const plan = lc.confirm;
  if (!plan) return;
  lc.confirm = null; lc.running = true; lc.log = []; lc.step = "Starting…"; lc.progress = 0; lc.bytes = null; lc.error = null; lc.result = null;
  render();
  try { await invoke("llamacpp_install", { backend: plan.backend, tag: plan.tag }); } catch (e) { lc.running = false; lc.error = String(e); lc.step = ""; render(); }
}

// the events of the install arrive on the environment channel; the page that started it knows it by `lc.running`
function onLlamaEvent(p) {
  if (p.kind === "log") lc.log.push(p.text);
  else if (p.kind === "step") { lc.step = `Step ${p.index} of ${p.total}: ${p.text}`; }
  else if (p.kind === "progress") { if (p.text) lc.step = p.text; if (p.value != null) lc.progress = p.value; if (p.bytes != null) lc.bytes = [p.bytes, p.total]; }
  else if (p.kind === "done") { lc.running = false; lc.progress = null; lc.step = ""; lc.result = p.data || {}; lc.error = null; lcAfter(); return; }
  else if (p.kind === "error") {
    lc.running = false; lc.progress = null; lc.step = "";
    lc.error = p.text === "Cancelled." ? "Cancelled. Nothing was changed." : p.text;
    lcAfter(); return;
  }
  if (["setup", "use", "imatrix", "quantcheck"].includes(page)) { render(); const l = $("lc-log"); if (l) l.scrollTop = l.scrollHeight; }
}

// after an install (or a failure): the status, then every page that waited for llama.cpp is read again so the person can simply retry
async function lcAfter() {
  await lcRefresh();
  if (lc.result) {
    if (llamaMarked(im.error)) im.error = null;
    if (llamaMarked(qk.error)) qk.error = null;
    if (llamaMarked(us.error)) us.error = null;
    if (page === "use" && state.bundle) usRefresh();
    if (page === "imatrix" && state.bundle) imRefresh();
    if (page === "quantcheck" && state.bundle) qkRefresh();
  }
}

function lcBackendName(plan) {
  return ({ cuda: `NVIDIA (CUDA ${plan.cuda_version || ""})`.replace(" )", ")"), vulkan: "Vulkan (AMD, Intel, NVIDIA)", cpu: "processor only", metal: "Apple (Metal)", build: "built here with CUDA" })[plan.backend] || plan.backend;
}

function lcConfirmHtml(plan) {
  const files = (plan.assets || []).map((a) => `<code>${esc(a.name)}</code> (${lcMb(a.size)})`).join(", ");
  const notes = (plan.notes || []).map((n) => `<li>${esc(n)}</li>`).join("");
  const build = plan.backend === "build";
  return `<div class="warn lc-confirm"><strong>Install llama.cpp ${esc(plan.tag)}: ${esc(lcBackendName(plan))}?</strong>
    <p class="small">${esc(plan.reason || "")}${plan.gpu ? ` (Your card: ${esc(plan.gpu.name)}.)` : ""}</p>
    ${notes ? `<ul class="small">${notes}</ul>` : ""}
    <p class="small">${build ? "The source of llama.cpp is downloaded (about " + lcMb(plan.total_bytes) + ") and built on this computer, which takes 20 minutes or more. You can cancel at any time." : `This downloads about <strong>${lcMb(plan.total_bytes)}</strong> from github.com: ${files || "the program files"}, and the source files of the same release for the converter (about 30 MB).`}
    It is installed into this program's own data folder; nothing needs administrator rights. The programs are not signed: Windows SmartScreen or an antivirus may warn about them. A folder you name in <code>LLAMA_CPP_DIR</code> still wins.</p>
    <div class="row"><button class="primary lc-go">Download and install (${lcMb(plan.total_bytes)})</button><button class="secondary lc-notnow">Not now</button></div></div>`;
}

function lcProgressHtml() {
  return `<div class="lc-progress"><div class="stepline" aria-live="polite">${esc(lc.step)}</div>${lc.progress != null ? `<progress max="1" value="${lc.progress}"></progress>` : ""}${lc.bytes ? `<span class="dim small"> ${lcMb(lc.bytes[0])} of about ${lcMb(lc.bytes[1])}</span>` : ""}
    <div class="row"><button class="secondary lc-cancel">Cancel</button></div>${lc.log.length ? `<pre class="log" id="lc-log">${lc.log.slice(-60).map(esc).join("\n")}</pre>` : ""}</div>`;
}

function lcResultHtml() {
  const r = lc.result;
  if (!r) return "";
  const v = Object.entries(r.versions || {}).map(([k, x]) => `${esc(k)}: ${esc(x)}`).join(" · ");
  return `<div class="ok lc-result"><strong>llama.cpp ${esc(r.tag || "")} is installed</strong> (${esc(r.backend || "")}${r.cuda_version ? " " + esc(r.cuda_version) : ""}). ${v ? esc(v) + ". " : ""}${r.converter_check ? esc(r.converter_check) + " " : ""}You can go on now; press the button you pressed before once more.</div>`;
}

function lcErrorHtml() {
  if (!lc.error) return "";
  const f = lc.status && lc.status.last_failure;
  const offer = f && f.offer_cpu && lc.error !== "Cancelled. Nothing was changed." ? `<div class="row"><button class="primary lc-cpu">Install the CPU build instead</button><span class="dim small">It needs no graphics card or driver and is slower for the measurements, but it works.</span></div>` : "";
  return `<div class="err lc-error">${esc(lc.error)}</div>${offer}`;
}

// the line shown wherever llama.cpp is needed: a note and "Install llama.cpp now" while it is missing, the plan / the progress / the result while the install goes
function llamaNotice(need, force) {
  if (lc.running) return `<div class="warn llama-notice"><strong>Installing llama.cpp…</strong>${lcProgressHtml()}</div>`;
  if (lc.planning) return `<div class="warn llama-notice"><strong>Looking up the llama.cpp release…</strong></div>`;
  if (lc.confirm) return lcConfirmHtml(lc.confirm);
  if (lc.result && (lcGap(need) === false)) return lcResultHtml();
  if (!force && !lcGap(need)) return "";
  return `<div class="warn llama-notice" data-need="${esc(need)}"><strong>llama.cpp is not installed (or not found).</strong> ${esc(LC_WHY[need] || LC_WHY.all)}
    <div class="row"><button class="primary lc-start">Install llama.cpp now</button></div>
    <p class="dim small">The program downloads it from GitHub into its own folder (you see the size first). To use your own copy instead, set the environment variable <code>LLAMA_CPP_DIR</code> to its folder: that always wins. The program looks in LLAMA_CPP_DIR, then in the folder it installed llama.cpp into, then in a folder called llama.cpp in your home folder, then next to the program; <code>llama-imatrix</code> and <code>llama-perplexity</code> are also looked for on the PATH (the converter script is not).</p>${lcErrorHtml()}</div>`;
}

// an error box; a worker error that ends with the mark gets the install button under it
function errBox(text, need) {
  if (!text) return "";
  const body = errDiv(stripLlamaMark(text));
  return llamaMarked(text) ? body + llamaNotice(need || "all", true) : body;
}

// the section of the Training setup page
function renderLlamaSection() {
  const s = lc.status;
  const expert = state.settings && state.settings.expert_mode;
  const busy = lc.running || lc.planning || env.running;
  let line = "<p class=\"dim\">Looking…</p>";
  if (s) {
    const inst = s.installed;
    const f = s.found;
    const how = f ? f.how_label : "";
    line = f
      ? `<p><span class="lamp">🟢</span> <strong>Found</strong> in <code>${esc(f.folder)}</code> (${esc(how)}).${inst && f.how === "program_install" ? ` Release <strong>${esc(inst.tag)}</strong>, ${esc(inst.backend)}${inst.cuda_version ? " " + esc(inst.cuda_version) : ""}, installed ${esc(String(inst.installed_at || "").slice(0, 10))}.` : ""}</p>
        <ul class="small dim"><li>llama-imatrix: ${s.tools && s.tools.imatrix ? esc(s.tools.imatrix.how_label) : "not found"}</li><li>llama-perplexity: ${s.tools && s.tools.perplexity ? esc(s.tools.perplexity.how_label) : "not found"}</li>${inst && f.how === "program_install" && inst.converter_check ? `<li>Converter: ${esc(inst.converter_check)}</li>` : ""}${inst && inst.versions ? Object.entries(inst.versions).map(([k, v]) => `<li>${esc(k)}: ${esc(v)}</li>`).join("") : ""}</ul>`
      : `<p><span class="lamp">🟡</span> <strong>Not installed.</strong> The program can install it for you.${s.folder_without_record ? ` (The folder <code>${esc(s.install_dir)}</code> exists but was not made by this program or is incomplete.)` : ""}</p>`;
    if (s.install_error) line += `<div class="warn small">The record of the installed copy is damaged (${esc(s.install_error)}). Press Install to make a new copy.</div>`;
  }
  const upd = lc.update && lc.update.newer ? `<div class="warn small lc-update">A newer release is available: <strong>${esc(lc.update.latest)}</strong> (installed: ${esc(lc.update.installed)}). <button class="secondary lc-start" data-backend="${esc((s && s.installed && s.installed.backend) || "auto")}">Update</button></div>` : lc.update ? `<p class="dim small">${lc.update.installed ? "This is the newest release (" + esc(lc.update.latest) + ")." : "The newest release is " + esc(lc.update.latest) + "."}</p>` : "";
  const ours = s && s.installed;
  const picker = expert ? `<label>Build <select id="lc-backend" ${busy ? "disabled" : ""}>${[["auto", "Automatic (the best for this computer)"], ["cuda", "NVIDIA (CUDA)"], ["vulkan", "Vulkan (AMD, Intel, NVIDIA)"], ["cpu", "Processor only"], ["build", "Build with CUDA from the source (Linux, long)"]].map(([v, t]) => `<option value="${v}" ${lc.backend === v ? "selected" : ""}>${t}</option>`).join("")}</select></label>` : "";
  return `<div class="card wide" id="lc-section"><h2>llama.cpp</h2>
    <p class="lead small">Makes GGUF files, importance matrices, quant comparisons and the mixed sizes. The program can install it for you; you do not have to build anything.</p>
    ${line}${upd}
    <div class="row">${picker}<button class="primary lc-start" ${busy ? "disabled" : ""}>${ours ? "Install again" : "Install llama.cpp"}</button>
      ${expert || (lc.status && lc.status.last_failure && lc.status.last_failure.offer_cpu) ? `<button class="secondary lc-cpu" ${busy ? "disabled" : ""}>Install the CPU build</button>` : ""}
      <button class="secondary lc-check" ${busy ? "disabled" : ""}>Check for a newer release</button>
      ${ours ? `<button class="secondary lc-remove" ${busy ? "disabled" : ""}>Remove</button>` : ""}</div>
    ${lc.planning ? `<p class="dim">Looking up the release…</p>` : ""}${lc.confirm ? lcConfirmHtml(lc.confirm) : ""}${lc.running ? lcProgressHtml() : ""}${lcErrorHtml()}${lcResultHtml()}
    <p class="dim small">Honestly: this downloads from GitHub (about 30 MB for a processor-only build, several hundred MB for a CUDA build; you see the size before anything starts). The programs are not signed, so Windows SmartScreen or an antivirus may warn about them. Nothing needs administrator rights. Your own copy named in <code>LLAMA_CPP_DIR</code> always wins. "Remove" deletes only the folder the program made.</p></div>`;
}

function wireLlama() {
  const all = (sel) => document.querySelectorAll(sel);
  all(".lc-start").forEach((b) => b.onclick = () => { const bk = b.dataset.backend; if (bk) lc.backend = bk; return lcPlan(lc.backend, false); });
  all(".lc-go").forEach((b) => b.onclick = () => lcInstall());
  all(".lc-notnow").forEach((b) => b.onclick = () => { lc.confirm = null; render(); });
  all(".lc-cancel").forEach((b) => b.onclick = () => invoke("llamacpp_cancel"));
  all(".lc-cpu").forEach((b) => b.onclick = () => lcPlan("cpu", true));
  all(".lc-check").forEach((b) => b.onclick = async () => {
    lc.updating = true; lc.error = null; render();
    try { lc.update = await invoke("llamacpp_check_update"); } catch (e) { lc.error = String(e); }
    lc.updating = false; render();
  });
  all(".lc-remove").forEach((b) => b.onclick = async () => {
    if (!confirm("Remove the llama.cpp this program installed? Only its own folder is deleted; a folder of yours (LLAMA_CPP_DIR) is never touched.")) return;
    try { lc.status = await invoke("llamacpp_remove"); lc.result = null; lc.update = null; lc.error = null; } catch (e) { lc.error = String(e); }
    render();
  });
  const sel = $("lc-backend"); if (sel) sel.onchange = () => { lc.backend = sel.value; };
}

const LIGHT = { ok: "🟢", warn: "🟡", fail: "🔴" };

// "Where the time went": the timing lines of the latest job logs, summed (crates/ndforge-core/src/timing.rs). Loaded on a click: it reads the saved logs.
const tm = { data: null, loading: false, err: null };
async function tmLoad() {
  tm.loading = true; tm.err = null; render();
  try { tm.data = await invoke("timing_report"); } catch (e) { tm.data = null; tm.err = String(e); }
  tm.loading = false; render();
}
function tmSecs(s) { return s >= 3600 ? (s / 3600).toFixed(1) + " h" : s >= 60 ? Math.round(s / 60) + " min" : Math.round(s) + " s"; }
function renderTimeCard() {
  const d = tm.data, sum = d && d.summary;
  const stages = sum && sum.stages ? Object.entries(sum.stages).sort((a, b) => b[1].seconds - a[1].seconds).slice(0, 12) : [];
  const body = tm.err ? `<p class="err">${esc(tm.err)}</p>` : !d ? `<p class="dim">The program writes down how long every part of a job took (loading a model, measuring, the teacher's answers, healing, saving, converting). Press the button to add up the latest ${30} jobs.</p>`
    : !stages.length ? `<p class="dim">No timing lines in the latest ${sum.logs} job logs yet: they come with the next job.</p>`
    : `<p class="dim small">${sum.logs} job logs, ${tmSecs(sum.job_seconds)} in all.</p>
      <table class="tbl" id="tm-table"><tr><th>Stage</th><th>Time</th><th>Times</th><th>Share</th></tr>${stages.map(([n, v]) => `<tr><td>${esc(n)}</td><td>${tmSecs(v.seconds)}</td><td>${v.count}</td><td>${Math.round(v.share * 100)} %</td></tr>`).join("")}</table>
      ${(d.advice || []).map((a) => `<p class="${a.worth ? "warn" : "dim"} small" data-tm-advice="${esc(a.id)}">${a.worth ? "<strong>Worth doing:</strong> " : ""}${esc(a.text)}</p>`).join("")}`;
  return `<div class="card wide" id="tm-card"><h2>Where the time went</h2>${body}
    <div class="row"><button class="secondary" id="tm-load" ${tm.loading ? "disabled" : ""}>${d ? "Look again" : "Show where the time went"}</button></div></div>`;
}

// ------------------------------------------------------------------------------------------------------- the vLLM engine (EXPERIMENTAL, off by default; docs/VLLM.md)
// Built from a research report and not yet run on this PC. Nothing is installed by the program; the status says what was found and what to install yourself.
const vl = { status: null, busy: false, err: null, saved: null, report: undefined, run: null, model: "", confirm: false, adapter: "" };
const VL_KINDS = [["auto", "Find it myself"], ["docker", "Docker Desktop (docker.exe)"], ["docker_wsl", "Docker inside WSL"], ["wslc", "WSLc (WSL 3.x containers, templates)"], ["wsl", "vLLM installed in WSL"], ["url", "A server that already runs"]];
function vllmSettings() { return (typeof state !== "undefined" && state && state.settings && state.settings.vllm) || { enabled: false, kind: "auto", image: "vllm/vllm-openai:v0.34.0", distro: "", venv: "~/.ndforge/vllm", url: "", model_volume: "ndforge-vllm-hf", model_dir: "", teacher_repo: "cyankiwi/Qwen3.8-27B-AWQ-INT4", adapter_rounds: false, kv_cache_dtype: "auto", kv_cache_dtype_teacher: false, llama_cache_type: "f16", wslc: {}, teacher: VL_TEACHER_DEFAULT }; }
const VL_TEACHER_DEFAULT = { enabled: false, route: "auto", pairs: { "Qwen/Qwen3.8-27B": "cyankiwi/Qwen3.8-27B-AWQ-INT4" }, confirmed: {}, required: false };
const vllmTeacher = () => ({ ...VL_TEACHER_DEFAULT, ...(vllmSettings().teacher || {}) });
const vlPairsText = (m) => Object.entries(m || {}).map(([a, b]) => `${a} = ${b}`).join("\n");
const vlPairsParse = (t) => Object.fromEntries(String(t || "").split("\n").map((l) => l.split("=").map((x) => x.trim())).filter((p) => p.length === 2 && p[0] && p[1]));
async function vllmDetect() {
  vl.busy = true; vl.err = null; render();
  try { vl.status = await invoke("vllm_status"); } catch (e) { vl.err = String(e); }
  vl.busy = false; render();
}
function renderVllm() {
  const v = vllmSettings(), st = vl.status;
  if (vl.report === undefined) { vl.report = null; setTimeout(vllmLoadReport, 0); }
  const lamp = { ok: "●", warn: "!", fail: "✕" };
  const lines = st ? `<ul class="checks">${st.lines.map((l) => `<li class="check lt-${esc(l.light)}"><span class="lamp">${lamp[l.light] || "·"}</span><div><strong>${esc(l.label)}</strong><div class="detail">${esc(l.detail)}</div></div></li>`).join("")}</ul>${st.guide.length ? `<ul class="small dim">${st.guide.map((g) => `<li>${esc(g)}</li>`).join("")}</ul>` : ""}` : `<p class="dim small">Press “Look for the engine” to see what this computer has (it only asks small programs; nothing is started or installed).</p>`;
  const f = (id, label, val, ph, w) => `<label>${label} <input type="text" id="${id}" value="${esc(val || "")}" placeholder="${esc(ph || "")}" style="min-width:${w || 220}px"></label>`;
  return `<div class="card wide" id="vl-card"><h2>vLLM engine (experimental)</h2>
    <p class="warn small">Built from a research report, not yet run on this PC. It is off unless you switch it on, and nothing here installs Docker, WSL or a driver for you. Run the vLLM self-test first; every speed in the app for it is an estimate.</p>
    <div class="row"><label class="opt"><input type="checkbox" id="vl-enabled" ${v.enabled ? "checked" : ""}> Offer the vLLM engine in the thinking check</label></div>
    <div class="row"><label>Way to run it <select id="vl-kind">${VL_KINDS.map(([k, n]) => `<option value="${k}" ${v.kind === k ? "selected" : ""}>${esc(n)}</option>`).join("")}</select></label>
      ${f("vl-image", "Docker image", v.image, "vllm/vllm-openai:v0.34.0", 280)}</div>
    <div class="row">${f("vl-distro", "WSL distribution", v.distro, "(the default one)")}${f("vl-venv", "vLLM folder in WSL", v.venv, "~/.ndforge/vllm")}${f("vl-url", "Server address (way: url)", v.url, "http://127.0.0.1:8000")}</div>
    <div class="row">${f("vl-volume", "Docker volume for models", v.model_volume, "ndforge-vllm-hf")}${f("vl-dir", "Model folder (optional, slow under /mnt/)", v.model_dir, "")}${f("vl-teacher", "Pre-quantized original (27 B)", v.teacher_repo, "", 300)}</div>
    <div class="row"><label class="opt" title="Untested: LoRA on the linear-attention layers is reported garbled in some versions. The standard engine stays the exact way until an agreement run."><input type="checkbox" id="vl-adapter" ${v.adapter_rounds ? "checked" : ""}> Check the rounds of a heal (model + LoRA) with vLLM (untested, off)</label></div>
    ${vllmKvCard(v)}
    ${vllmTeacherCard()}
    <details><summary>WSLc command templates (advanced, JSON)</summary><textarea id="vl-wslc" rows="4" style="width:100%">${esc(JSON.stringify(v.wslc || {}, null, 1))}</textarea><p class="dim small">Keys: run, stop, remove, logs, list, pull, gputest, gpu_flag, shm, publish, env, test_image. See docs/VLLM.md.</p></details>
    <div class="row"><button class="secondary" id="vl-save">Save</button><button class="secondary" id="vl-detect" ${vl.busy ? "disabled" : ""}>Look for the engine</button></div>
    ${vl.saved ? `<p class="ok">${esc(vl.saved)}</p>` : ""}${vl.err ? `<div class="err">${esc(vl.err)}${memHint(vl.err)}</div>` : ""}${lines}${renderVllmTest(v)}</div>`;
}
// the optional 8-bit KV cache (docs/VLLM.md): off by default; the teacher's answers need a second, explicit confirmation
function vllmKvCard(v) {
  const sel = (id, cur, opts) => `<select id="${id}">${opts.map(([k, n]) => `<option value="${k}" ${cur === k ? "selected" : ""}>${esc(n)}</option>`).join("")}</select>`;
  return `<fieldset class="soft" id="vl-kv-card"><legend>8-bit KV cache (experimental, off)</legend>
    <p class="dim small">Stores the attention cache in 8 bits: a tiny numeric change. Measured on a 5090 with Qwen3.8-27B AWQ: 71,141 cached tokens with auto, 85,196 with fp8 (only about 1.2 times, not 2 times) and no speed gain (564 and 548 tokens per second on long answers): Qwen3.8 is a hybrid, only its full-attention layers have a KV cache and the page size is dominated by the DeltaNet state. A plain transformer would gain more; this is not measured. Thinking checks and trace generation may use it (the result says "kv cache fp8" and has its own key); the teacher's answers for healing do not, unless you confirm below.</p>
    <div class="row"><label>vLLM KV cache ${sel("vl-kv", v.kv_cache_dtype || "auto", [["auto", "auto (the default)"], ["fp8", "fp8 (8 bit)"]])}</label>
      <label class="opt"><input type="checkbox" id="vl-kv-teacher" ${v.kv_cache_dtype_teacher ? "checked" : ""}> Also use it for the original's answers (the training target; cached answers carry it in their identity and are never mixed with exact ones)</label></div>
    <div class="row"><label>llama.cpp KV cache (GGUF thinking check) ${sel("vl-llama-kv", v.llama_cache_type || "f16", [["f16", "f16 (the default)"], ["q8_0", "q8_0 (8 bit, needs flash attention)"]])}</label></div></fieldset>`;
}
// the original's answers through vLLM (docs/VLLM.md, "Teacher cache"): the teacher must be YOUR original model, so a pre-quantized copy is used only for the original it was made from
function vllmTeacherCard() {
  const t = vllmTeacher();
  const route = (k, n) => `<option value="${k}" ${t.route === k ? "selected" : ""}>${esc(n)}</option>`;
  return `<fieldset class="soft" id="vl-teacher-card"><legend>The original's answers through vLLM (experimental)</legend>
    <p class="dim small">Off unless you tick it. The heal's teacher must be your ORIGINAL model. A pre-quantized checkpoint is a quantization of one repository, so it is used only for that original (the list below) or one you confirm; every other original is loaded from its own folder and quantized to 4 bits while loading, which is the standard teacher's number format (how vLLM does that is unverified and its speed unknown).</p>
    <div class="row"><label class="opt"><input type="checkbox" id="vl-t-on" ${t.enabled ? "checked" : ""}> Offer "make the original's answers with vLLM" on the Heal card</label></div>
    <div class="row"><label>Route <select id="vl-t-route">${route("auto", "Automatic: a confirmed copy, else 4-bit while loading")}${route("bnb", "Always 4-bit while loading, from the original's folder")}${route("prequant", "Only a confirmed pre-quantized copy (refuses otherwise)")}</select></label>
      <label class="opt"><input type="checkbox" id="vl-t-required" ${t.required ? "checked" : ""}> A heal stops when vLLM does not work (else it goes on with the standard teacher and says so)</label></div>
    <label>Confirmed pairs: original = its quantized copy, one per line (the default list holds the one the research report names)<textarea id="vl-t-pairs" rows="2" style="width:100%">${esc(vlPairsText(t.pairs))}</textarea></label>
    <label>Copies you confirm: "this repository is a quantized copy of my original model", one per line: original = copy<textarea id="vl-t-confirmed" rows="2" style="width:100%" placeholder="your/original = your/original-AWQ">${esc(vlPairsText(t.confirmed))}</textarea></label>
  </fieldset>`;
}
// the self-test: the report's 1-hour experiment, run for you (docs/VLLM.md). Everything stoppable; the engine is always stopped again.
function renderVllmTest(v) {
  const r = vl.report, busy = !!vl.run || rs.running || env.running;
  const word = { pass: "Pass", partial: "Partly", fail: "Fail", stopped: "Stopped" };
  const rep = r ? `<div class="${r.verdict === "pass" ? "ok" : r.verdict === "partial" ? "warn" : "err"}" id="vl-verdict"><strong>${esc(word[r.verdict] || r.verdict)}</strong>: ${esc(r.verdict_text)}
      <ul class="small">${(r.criteria || []).map((c) => `<li>${c.ok ? "✓" : c.ok === false ? "✕" : "?"} ${esc(c.label)}: ${esc(typeof c.value === "object" ? JSON.stringify(c.value) : c.value == null ? "not measured" : String(c.value))}</li>`).join("")}</ul>
      <p class="dim small">(also in the problem-report bundle). ${r.answer_tokens ? `Answers: median ${r.answer_tokens.median} tokens. ` : ""}${r.long_tokens_per_s != null ? `Long answers: ${r.long_tokens_per_s} tokens/s. ` : ""}${r.tokens_per_s != null ? `Measured: ${r.tokens_per_s} tokens/s in ${r.wall_seconds} s; engine ready after ${r.startup_seconds} s.` : ""}</p></div>` : "";
  return `<h3>vLLM self-test</h3><p class="dim small">Runs the research report's experiment for you: checks the driver and that a container can use the graphics card, starts the engine on a model, waits for it, asks the Quick thinking suite with 40 questions at once, measures start-up time and tokens per second, compares with your saved standard-engine numbers, stops the engine and writes a report with a plain verdict. A first run downloads the engine's image and the model.</p>
    <div class="row"><label>Model folder or Hugging Face id <input type="text" id="vl-tmodel" value="${esc(vl.model)}" placeholder="${esc(v.teacher_repo || "")}" style="min-width:320px"></label>
      <span class="dim small">Empty = the default, ${esc(v.teacher_repo || "the configured pre-quantized original")} (about 21 GB, 4-bit). A 16-bit model larger than the card is refused before anything starts.</span>
      <label>LoRA folder (optional: agreement check) <input type="text" id="vl-tadapter" value="${esc(vl.adapter)}" style="min-width:220px"></label></div>
    <div class="row"><label class="opt"><input type="checkbox" id="vl-tkv" ${vl.kv ? "checked" : ""}> Also test the 8-bit KV cache (a second engine start: reports the cache size and speed for both)</label></div>
    <div class="row"><label class="opt"><input type="checkbox" id="vl-tconfirm" ${vl.confirm ? "checked" : ""}> I agree to the download: the default model is about 21 GB (plus the engine's image, several GB)</label></div>
    <div class="row"><button class="primary" id="vl-test" ${busy || !v.enabled ? "disabled" : ""} title="${v.enabled ? "" : "Switch the engine on above first"}">Run the vLLM self-test</button><button class="secondary" id="vl-inflight" ${busy || !v.enabled ? "disabled" : ""} title="Starts the engine on the model above (a folder: your original) with the in-flight 4-bit route alone, no Hugging Face process, and says whether it worked or which lines name the cause. Minutes, not hours.">Test the in-flight 4-bit route on my original</button>${vl.run ? `<button class="secondary" id="vl-teststop">Stop</button>` : ""}</div>
    ${vl.run ? `<div class="stepline" aria-live="polite">${esc(vl.run.text)}</div>${vl.run.progress != null ? `<progress max="1" value="${vl.run.progress}"></progress>` : ""}` : ""}${rep}${vllmFilesBox(vl.files, "vl-files", "The report is saved in:")}`;
}
// the report's folder as text, with Copy path / Open the folder / Copy the whole report, and the text inline (also after a failure): the owner must be able to paste it into an issue
function vllmFilesBox(files, id, title) {
  if (!files || !(files.exists || files.dir)) return "";
  const dir = files.dir || "", text = files.markdown || files.text || "";
  return `<div class="card-in" id="${id}"><p class="small">${esc(title)} <code class="vl-path" data-path>${esc(dir)}</code>
    <button class="secondary" data-vlcopy="${esc(dir)}">Copy path</button><button class="secondary" data-vlopen="${esc(dir)}">Open the folder</button>${text ? `<button class="secondary" data-vlcopy-text="${id}">Copy the whole report</button>` : ""}</p>
    ${text ? `<details ${files.open === false ? "" : "open"}><summary class="small">The report (text)</summary><pre class="vl-report small" id="${id}-text" style="white-space:pre-wrap;max-height:28em;overflow:auto">${esc(text)}</pre></details>` : `<p class="dim small">No report file yet.</p>`}</div>`;
}
function wireVllmFiles() {
  document.querySelectorAll("[data-vlcopy]").forEach((b) => b.onclick = async () => { try { await navigator.clipboard.writeText(b.dataset.vlcopy); b.textContent = "Copied"; } catch (_) { b.textContent = "Select the path and copy by hand"; } });
  document.querySelectorAll("[data-vlcopy-text]").forEach((b) => b.onclick = async () => { const e = $(b.dataset.vlcopyText + "-text"); try { await navigator.clipboard.writeText(e ? e.textContent : ""); b.textContent = "Copied"; } catch (_) { b.textContent = "Select the text and copy by hand"; } });
  document.querySelectorAll("[data-vlopen]").forEach((b) => b.onclick = async () => { try { await invoke("imatrix_open_folder", { path: b.dataset.vlopen }); } catch (e) { b.textContent = String(e).slice(0, 120); } });
}
async function vllmLoadReport() {
  vl.report = null;
  try { vl.report = await invoke("vllm_selftest_report"); } catch (_) { vl.report = null; }
  try { vl.files = await invoke("vllm_selftest_files"); } catch (_) { vl.files = null; }
  try { render(); } catch (_) { /* a preview without a window */ }
}
function onVllmEvent(p) {
  if (p.kind === "progress") { if (vl.run) { vl.run.text = p.text || p.step; vl.run.progress = p.value; } }
  else if (p.kind === "error") { vl.run = null; vl.err = p.text; }
  else if (p.kind === "done") { vl.run = null; vllmLoadReport(); return; }
  if (page === "setup") render();
}
function wireVllm() {
  const val = (id) => (($(id) || {}).value || "").trim();
  const sv = $("vl-save"); if (sv) sv.onclick = async () => {
    let wslc = {};
    try { wslc = JSON.parse(val("vl-wslc") || "{}"); } catch (_) { vl.err = "The WSLc templates are not valid JSON."; render(); return; }
    const v = { enabled: $("vl-enabled").checked, kind: val("vl-kind") || "auto", image: val("vl-image"), distro: val("vl-distro"), venv: val("vl-venv"), url: val("vl-url"), model_volume: val("vl-volume"), model_dir: val("vl-dir"),
      teacher_repo: val("vl-teacher"), adapter_rounds: $("vl-adapter").checked, wslc, kv_cache_dtype: val("vl-kv") || "auto", kv_cache_dtype_teacher: !!($("vl-kv-teacher") || {}).checked, llama_cache_type: val("vl-llama-kv") || "f16",
      teacher: { enabled: $("vl-t-on").checked, route: val("vl-t-route") || "auto", required: $("vl-t-required").checked, pairs: vlPairsParse(($("vl-t-pairs") || {}).value), confirmed: vlPairsParse(($("vl-t-confirmed") || {}).value) } };
    try { setState(await invoke("vllm_set", { vllm: v })); vl.err = null; vl.saved = "Saved."; } catch (e) { vl.err = String(e); vl.saved = null; }
    render();
  };
  wireVllmFiles();
  const dt = $("vl-detect"); if (dt) dt.onclick = vllmDetect;
  const tm = $("vl-tmodel"); if (tm) tm.onchange = () => { vl.model = tm.value.trim(); };
  const ta = $("vl-tadapter"); if (ta) ta.onchange = () => { vl.adapter = ta.value.trim(); };
  const tc = $("vl-tconfirm"); if (tc) tc.onchange = () => { vl.confirm = tc.checked; };
  const tk8 = $("vl-tkv"); if (tk8) tk8.onchange = () => { vl.kv = tk8.checked; };
  const tt = $("vl-test"); if (tt) tt.onclick = () => {
    vl.model = (($("vl-tmodel") || {}).value || "").trim(); vl.adapter = (($("vl-tadapter") || {}).value || "").trim(); vl.confirm = !!($("vl-tconfirm") || {}).checked;
    vl.run = { text: "Starting…", progress: null }; vl.err = null; render();
    invoke("vllm_selftest_start", { spec: { model: vl.model, confirm_download: vl.confirm, adapter: vl.adapter || null, ...(vl.kv ? { kv_fp8_test: true } : {}) } }).catch((e) => { vl.run = null; vl.err = String(e); render(); });
  };
  const tf = $("vl-inflight"); if (tf) tf.onclick = () => {
    vl.model = (($("vl-tmodel") || {}).value || "").trim(); vl.confirm = !!($("vl-tconfirm") || {}).checked;
    vl.run = { text: "Starting…", progress: null }; vl.err = null; render();
    invoke("vllm_selftest_start", { spec: { model: vl.model, confirm_download: vl.confirm, adapter: null, in_flight: true } }).catch((e) => { vl.run = null; vl.err = String(e); render(); });
  };
  const ts = $("vl-teststop"); if (ts) ts.onclick = () => { if (vl.run) vl.run.text = "Stopping…"; invoke("reshape_stop"); render(); };
}

// ------------------------------------------------------------------------------------------------ the memory panel (docs/MEMORY.md): footer numbers, a popover, "Free GPU" / "Free RAM"
const mem = { st: null, open: false, plan: null, stop: false, run: false, result: null, err: null, timer: null };
const memGb = (mb) => (mb / 1024).toFixed(1);
function memLine() {
  const s = mem.st; if (!s) return "Memory";
  const g = s.gpu ? `GPU ${memGb(s.gpu.free_mb)} of ${memGb(s.gpu.total_mb)} GB free` : "GPU not found";
  const r = s.ram ? `RAM ${memGb(s.ram.free_mb)} of ${memGb(s.ram.total_mb)} GB free` : "";
  return [g, r, s.ram && s.ram.app_mb ? `this app ${memGb(s.ram.app_mb)} GB` : ""].filter(Boolean).join(" · ");
}
function memBar(used, total) { return `<div class="mem-bar" aria-hidden="true"><i style="width:${total ? Math.min(100, Math.round((100 * used) / total)) : 0}%"></i></div>`; }
function memPlanHtml(p) {
  const need = (p.items || []).some((i) => i.needs_stop && i.refused);
  const items = (p.items || []).map((i) => `<li><strong>${esc(i.label)}</strong> <span class="dim">${esc(i.detail)}</span>${i.refused ? ` <span class="warn">Not now: ${esc(i.refused)}.</span>` : ""}</li>`).join("");
  const others = (p.others || []).map((o) => `<li>${esc(o.name)} (pid ${esc(String(o.pid))})${o.used_mb != null ? `, ${esc(memGb(o.used_mb))} GB` : ""} <span class="dim">${esc(o.note || "")}</span></li>`).join("");
  const wsl = p.kind === "wsl";
  return `<div id="mem-confirm" role="alertdialog" aria-label="Confirm"><h3>${wsl ? "Release the WSL memory?" : p.kind === "gpu" ? "Free the graphics memory this program holds?" : "Free the RAM this program holds?"}</h3>
    ${wsl ? `<p class="warn"><strong>Warning:</strong> this ends every WSL distribution on this computer, also the ones you use for other things.</p>` : ""}
    ${items ? `<p class="small">This will end or release exactly:</p><ul class="small">${items}</ul>` : `<p>This program holds nothing that can be freed right now.</p>`}
    ${others ? `<p class="small">Other programs hold memory too. They are only listed: close them yourself, this program never ends them.</p><ul class="small">${others}</ul>` : ""}
    ${(p.notes || []).map((n) => `<p class="dim small">${esc(n)}</p>`).join("")}
    ${need ? `<label class="opt"><input type="checkbox" id="mem-stop" ${mem.stop ? "checked" : ""}> Stop the running job too (it is stopped the normal way, its state is saved and Continue works)</label>` : ""}
    <div class="row"><button class="primary" id="mem-go" ${mem.run || !(p.items || []).length ? "disabled" : ""}>${wsl ? "Release and end WSL" : "Free it"}</button><button class="secondary" id="mem-no">Cancel</button></div></div>`;
}
function memResultHtml(r) {
  const n = (x) => (x ? `${memGb(x.free_mb)} of ${memGb(x.total_mb)} GB free` : "not known");
  const row = (l, b, a) => `<tr><td>${l}</td><td>${n(b)}</td><td>${n(a)}</td></tr>`;
  return `<div id="mem-result"><h3>Done</h3><table><tr><th></th><th>Before</th><th>After</th></tr>${row("GPU", r.before.gpu, r.after.gpu)}${row("RAM", r.before.ram, r.after.ram)}</table>
    ${(r.done || []).map((d) => `<div class="small ok">✓ ${esc(d.label)}${d.freed_mb ? ` (${esc(String(d.freed_mb))} MB)` : ""}</div>`).join("")}
    ${(r.skipped || []).map((d) => `<div class="small warn">Left alone: ${esc(d.label)} — ${esc(d.why)}</div>`).join("")}
    ${(r.failed || []).map((d) => `<div class="small err">Could not: ${esc(d.label)} — ${esc(d.why)}</div>`).join("")}
    ${(r.others || []).length ? `<p class="dim small">Still using memory and not ours: ${(r.others || []).map((o) => esc(o.name) + (o.used_mb != null ? " " + memGb(o.used_mb) + " GB" : "")).join(", ")}. Close them yourself.</p>` : ""}
    <p class="dim small">The free RAM includes Windows' file cache; there is no RAM cleaner here.</p></div>`;
}
function memPopHtml() {
  const s = mem.st, g = s && s.gpu, r = s && s.ram;
  const numbers = `<table>
    <tr><th>Graphics memory</th><td>${g ? `${esc(memGb(g.used_mb))} used, ${esc(memGb(g.free_mb))} free of ${esc(memGb(g.total_mb))} GB${g.app_mb != null ? `; this app ${esc(memGb(g.app_mb))} GB` : ""}` : "no NVIDIA card found"}</td></tr>
    <tr><td colspan="2">${g ? memBar(g.used_mb, g.total_mb) : ""}</td></tr>
    <tr><th>RAM</th><td>${r ? `${esc(memGb(r.used_mb))} used, ${esc(memGb(r.free_mb))} free of ${esc(memGb(r.total_mb))} GB; this app ${esc(memGb(r.app_mb))} GB` : "not known"}</td></tr>
    <tr><td colspan="2">${r ? memBar(r.used_mb, r.total_mb) : ""}</td></tr></table>
    ${g && !g.per_process ? `<p class="dim small" id="mem-noproc">The driver does not say which program uses how much graphics memory (usual on Windows): only the totals are shown.</p>` : ""}
    ${s && (s.containers || []).length ? `<p class="small">Engine containers: ${(s.containers || []).map((c) => esc(c.name) + (c.running ? " (running)" : " (stopped)")).join(", ")}</p>` : ""}
    ${g && (g.others || []).length ? `<p class="small">Other programs on the card: ${(g.others || []).map((o) => esc(o.name) + (o.used_mb != null ? " " + esc(memGb(o.used_mb)) + " GB" : "")).join(", ")} <span class="dim">(listed only)</span></p>` : ""}`;
  const body = mem.err ? `<div class="err">${esc(mem.err)}</div>` : "";
  return `<div class="row"><strong>Memory</strong><span class="spacer"></span><button class="link" id="mem-close">Close</button></div>${numbers}
    <div class="row"><button class="secondary" id="mem-gpu" ${mem.run ? "disabled" : ""}>Free GPU…</button><button class="secondary" id="mem-ram" ${mem.run ? "disabled" : ""}>Free RAM…</button>${s && s.wsl ? `<button class="link" id="mem-wsl" ${mem.run ? "disabled" : ""}>Release WSL memory…</button>` : ""}</div>
    <p class="dim small">Free GPU and Free RAM only end what this program started (its engine containers and its leftover workers). Nothing happens before you confirm.</p>
    ${body}${mem.plan ? memPlanHtml(mem.plan) : ""}${mem.result ? memResultHtml(mem.result) : ""}`;
}
function renderMemory() {
  const b = $("mem-btn"); if (b) { b.textContent = memLine(); try { b.setAttribute("aria-expanded", mem.open ? "true" : "false"); } catch (_) { /* a preview */ } }
  const pop = $("mem-pop"); if (!pop) return;
  pop.hidden = !mem.open;
  if (mem.open) { pop.innerHTML = memPopHtml(); wireMemory(); }
}
async function memPoll() {
  try { mem.st = await invoke("memory_status"); } catch (_) { /* no numbers: the button just says Memory */ }
  renderMemory();
  clearTimeout(mem.timer);
  const t = setTimeout(memPoll, mem.st && mem.st.busy ? 2000 : 5000);          // every ~2 s while a job runs, else every few seconds
  mem.timer = t; if (t && t.unref) t.unref();
}
async function memAsk(kind) {
  mem.open = true; mem.err = null; mem.result = null; mem.plan = null; mem.stop = false; renderMemory();
  try { const p = await invoke("memory_free_plan", { kind }); mem.plan = { ...p, kind }; } catch (e) { mem.err = String(e); }
  renderMemory();
}
async function memGo() {
  const p = mem.plan; if (!p || mem.run) return;
  mem.run = true; renderMemory();
  try { mem.result = await invoke("memory_free", { kind: p.kind, planToken: p.token, stopRunning: !!mem.stop }); mem.plan = null; mem.err = null; }
  catch (e) { mem.err = String(e); }
  mem.run = false;
  try { mem.st = await invoke("memory_status"); } catch (_) { /* keep the old numbers */ }
  renderMemory();
}
function wireMemory() {
  const on = (id, fn) => { const e = $(id); if (e) e.onclick = fn; };
  on("mem-close", () => { mem.open = false; renderMemory(); });
  on("mem-gpu", () => memAsk("gpu")); on("mem-ram", () => memAsk("ram")); on("mem-wsl", () => memAsk("wsl"));
  on("mem-go", memGo); on("mem-no", () => { mem.plan = null; renderMemory(); });
  const st = $("mem-stop"); if (st) st.onchange = () => { mem.stop = st.checked; };
}
function memHint(text) { return /free memory|out of memory|in use by other programs|desired GPU memory|graphics (card|memory)/i.test(String(text || "")) ? ` <button class="link" data-memfree="gpu">Free GPU first</button>` : ""; }
try { document.addEventListener("click", (ev) => { const t = ev.target && ev.target.closest && ev.target.closest("[data-memfree]"); if (t) memAsk(t.dataset.memfree); }); } catch (_) { /* a preview without a window */ }
try { const mb = document.getElementById("mem-btn"); if (mb) mb.onclick = () => { mem.open = !mem.open; if (mem.open && !mem.st) memPoll(); renderMemory(); }; } catch (_) { /* a preview */ }

// ------------------------------------------------------------------------------------------------ Tune: tested settings kept as presets you can see and select (docs/TUNE.md)
const tn = { model: "", view: null, sel: "", err: null, ok: null, ask: null, preview: null, busy: false, loadingFor: null };
const TN_KINDS = { exact: "tested for this model", same_shape: "same shape", lineage: "from the original", nearest: "nearest size", other_pc: "another PC" };
function jobWhy() {
  const parts = [];
  if (rs.running) parts.push("a Slim, heal or check job");
  if (rs.testing) parts.push("a model test");
  if (typeof hsp !== "undefined" && hsp.run) parts.push("Measure heal speed");
  if (typeof hb !== "undefined" && rs.hb && rs.hb.running) parts.push("Find the fastest loss settings");
  if (vch.run) parts.push("a vLLM check");
  if (ac && ac.running) parts.push("an Auto-cut run");
  if (mem.st && mem.st.busy && !parts.length) parts.push("a job");
  return parts.length ? `${parts.join(" and ")} is running` : "";
}
const tnBusy = () => !!jobWhy();
function tuneCard() {
  const sp = rs.hlSpec || {}, kn = (rs.hlEst && rs.hlEst.est && rs.hlEst.est.knobs) || {};
  const pick = (v, a) => (v !== undefined && v !== null ? v : a !== undefined ? a : null);
  return { seq: pick(sp.seq, kn.seq), batch: pick(sp.batch, kn.batch), per_update: pick(sp.per_update, kn.batch && kn.grad_accum ? kn.batch * kn.grad_accum : null), checkpointing: sp.checkpointing || "auto",
    rank: pick(sp.rank, kn.rank), method: sp.method || (kn.qlora === true ? "qlora" : kn.qlora === false ? "lora" : null), optimizer: sp.optimizer || null, kl_chunk: pick(sp.kl_chunk, kn.kl_chunk), k: pick(sp.k, null),
    teacher_load: sp.teacher_load || null, warmup: pick(sp.warmup, null), head_checkpoint: sp.head_checkpoint === "off" ? "off" : "on", student_batch_tokens: +(sp.student_batch_tokens || 0) };
}
function tuneApplyToCard(changes) {
  const hs = rs.hlSpec; if (!hs) return;
  for (const c of changes) {
    const k = c.setting, v = c.to;
    if (k === "head_checkpoint") hs.head_checkpoint = v === "off" ? "off" : null;
    else if (k === "student_batch_tokens") hs.student_batch_tokens = +v >= 512 ? +v : null;
    else if (k === "checkpointing") hs.checkpointing = v;
    else hs[k] = v;
  }
}
async function tuneLoad(model) {
  if (!model) return;
  tn.loadingFor = model; tn.err = null;
  try { const v = await invoke("tune_presets", { model }); if (tn.loadingFor === model) { tn.view = v; tn.model = model; if (!v.presets.some((x) => x.id === tn.sel)) tn.sel = ""; } }
  catch (e) { tn.view = null; tn.model = model; tn.err = String(e); }
  try { render(); } catch (_) { /* a preview without a window */ }
}
function tuneEnsure(model) { if (model && tn.model !== model && tn.loadingFor !== model) tuneLoad(model); }
const tnSelected = () => (tn.view && tn.view.presets.find((x) => x.id === tn.sel)) || null;
function tuneGainText(p) { const g = (p.gains || []).filter((x) => x.gain > 1.0).map((x) => `${esc(x.function.replace(/^.*\./, ""))} ×${(+x.gain).toFixed(2)}`); return g.join(", "); }
function tuneBar(where, model) {
  tuneEnsure(model);
  const v = tn.view, off = tnBusy(), why = jobWhy();
  const list = v && v.presets || [];
  const opts = [`<option value="">${list.length ? "Choose a preset…" : "No preset for this model yet"}</option>`].concat(list.map((p) => `<option value="${esc(p.id)}" ${p.id === tn.sel ? "selected" : ""} ${p.kind === "other_pc" ? "disabled" : ""}>${esc(p.name)} · ${esc(TN_KINDS[p.kind] || p.kind)}${p.stale.length ? " · stale, re-test?" : ""}${p.scope === "library" ? " · library" : ""}</option>`)).join("");
  const sel = tnSelected();
  const b = (id, label, extra = "", tip = "") => `<button class="secondary" id="tn-${id}-${where}" data-tn="${id}" ${extra}${tip ? ` title="${esc(tip)}"` : ""}>${label}</button>`;
  const noSel = sel ? "" : "disabled";
  const ask = tn.ask && tn.ask.where === where ? tuneAskHtml(where) : "";
  return `<div class="tn-bar" id="tn-bar-${where}"><div class="row"><strong>Preset</strong> <select id="tn-sel-${where}" data-tn-sel="${where}">${opts}</select>
    ${b("use", "Use…", noSel + (off ? " disabled" : ""), off ? `Off because ${why}` : "Shows what would change on the card, and what it is worth, before anything is applied")}
    ${b("save", "Save current as preset…", off ? "disabled" : "", off ? `Off because ${why}` : "Keeps the settings of this card as a preset you can select again")}
    ${where !== "tune" ? `<button class="primary" id="tn-auto-${where}" data-tn="auto" ${off ? "disabled" : ""} title="${off ? esc(`Off because ${why}`) : "Tests every setting of this card for this model on this PC, one after the other, then shows what to change and saves a preset"}">Auto: test and choose the best settings</button>` : ""}${b("rename", "Rename", noSel)}${b("dup", "Duplicate", noSel)}${b("del", "Delete", noSel)}${b("export", "Export", noSel)}${b("import", "Import", "")}</div>
    ${sel ? `<p class="small dim" id="tn-note-${where}">${esc(sel.note || "")}${tuneGainText(sel) ? ` · measured: ${tuneGainText(sel)}` : ""}${sel.stale.length ? ` · <span class="warn">stale: ${esc(sel.stale.map((x) => x.reasons.join("; ")).join("; "))} (re-test)</span>` : ""}</p>` : ""}
    ${off ? `<p class="dim small" id="tn-why-${where}">Some buttons are off because ${esc(why)}. Wait for it or press its Stop.</p>` : ""}
    ${tn.err && where !== "tune" ? `<p class="err small">${esc(tn.err)}</p>` : ""}${tn.ok ? `<p class="ok small">${esc(tn.ok)}</p>` : ""}${ask}${tn.preview && tn.preview.where === where ? tuneStageHtml(where) : ""}</div>`;
}
function tuneAskHtml(where) {
  const a = tn.ask, sel = tnSelected();
  const label = { save: "Name of the new preset", rename: "New name", dup: "Name of the copy", del: `Type the preset's exact name (${sel ? esc(sel.name) : ""}) to delete it`, import: "" }[a.kind] || "";
  return `<div class="row" id="tn-ask-${where}"><label>${label}<br><input type="text" id="tn-ask-input-${where}" value="${esc(a.value || "")}" style="min-width:320px"></label>
    <button class="primary" id="tn-ask-ok-${where}">${a.kind === "del" ? "Delete permanently" : "OK"}</button><button class="secondary" id="tn-ask-no-${where}">Cancel</button></div>`;
}
function tuneStageHtml(where) {
  const pv = tn.preview.data;
  const rows = pv.changes.map((c) => `<tr><td>${esc(c.label)}</td><td>${esc(String(c.from ?? "automatic"))}</td><td><strong>${esc(String(c.to))}</strong></td><td>${c.gain && c.gain > 1.0 ? "×" + (+c.gain).toFixed(2) : ""}</td><td class="dim small">${esc(c.class === "quality" ? "quality knob: based on a short trial, not a full heal" : c.class === "tol" ? "same results within tolerance" : c.class === "exact" ? "same results" : c.class || "")}</td></tr>`).join("");
  return `<div class="card" id="tn-preview-${where}"><h3>Use “${esc(pv.name)}”?</h3>
    <p class="small ${pv.whole ? "dim" : "warn"}">${esc(pv.note)}${pv.whole ? "" : " Only the settings that carry over to another model are offered."}</p>
    ${pv.stale.length ? `<p class="warn small">Stale: ${esc(pv.stale.map((x) => x.reasons.join("; ")).join("; "))}. You can still use it; re-testing takes about a quarter of an hour.</p>` : ""}
    ${rows ? `<table class="tbl"><tr><th>Setting</th><th>Now</th><th>Becomes</th><th>Measured gain</th><th></th></tr>${rows}</table>` : `<p>The card already has these settings. Nothing would change.</p>`}
    <div class="row"><button class="primary" id="tn-apply-${where}" ${rows ? "" : "disabled"}>Apply all</button><button class="secondary" id="tn-apply-no-${where}">Cancel</button></div></div>`;
}
function tuneName(sel) { return (sel && sel.name) || ""; }
// the one-click Auto for a model, from any page (the Heal card, the Auto page, the Tune page): the same plan, the same confirmation, the same run
async function tuneAutoFor(model) {
  if (!model) return;
  tn.pick = model; tu.plan = null; tu.state = null; tu.preview = null; tu.spv = {}; tu.applied = null; tn.view = null; tn.model = "";
  await go("tune"); await tuLoadPlan(); await tuGo();
}
const acTuneModel = () => { const made = ((rs.state && rs.state.made) || []).filter((m) => m.status === "done"); return (made.find((m) => m.name === tn.pick) || made[0] || {}).name || ""; };
async function tuneAction(where, kind) {
  const sel = tnSelected(), model = where === "tune" ? tn.pick : (rs.hl && rs.hl.name);
  tn.err = null; tn.ok = null;
  if (kind === "auto") { await tuneAutoFor(model); return; }
  if (kind === "use") {
    if (!sel) return;
    try { tn.preview = { where, data: await invoke("tune_apply_preview", { presetId: sel.id, scope: sel.scope, model, target: "heal_card", current: tuneCard() }) }; } catch (e) { tn.err = String(e); }
  } else if (kind === "save") tn.ask = { where, kind, value: "" };
  else if (kind === "rename" || kind === "dup") tn.ask = { where, kind, value: kind === "rename" ? tuneName(sel) : tuneName(sel) + " copy" };
  else if (kind === "del") tn.ask = { where, kind, value: "" };
  else if (kind === "export" && sel) {
    try { const dir = await tauri.dialog.open({ directory: true, title: "Choose the folder for the preset file" }); if (dir) { const f = String(dir).replace(/[\\/]+$/, "") + "/" + tuneName(sel).replace(/[^A-Za-z0-9._ -]+/g, "_") + ".ndtune.json"; await invoke("tune_preset_export", { id: sel.id, scope: sel.scope, path: f }); tn.ok = `Exported to ${f}`; } } catch (e) { tn.err = String(e); }
  } else if (kind === "import") {
    try { const f = await tauri.dialog.open({ multiple: false, title: "Choose a preset file", filters: [{ name: "Preset", extensions: ["json"] }] }); if (f) { tn.lastImport = String(f); await invoke("tune_preset_import", { path: String(f), overwrite: false }); tn.ok = "Imported into the library."; await tuneLoad(model); } }
    catch (e) { const t = String(e); if (/already in the library/.test(t) && confirm(t + "\n\nReplace it?")) { try { await invoke("tune_preset_import", { path: tn.lastImport, overwrite: true }); } catch (e2) { tn.err = String(e2); } } else tn.err = t; }
  }
  render();
}
async function tuneAskOk(where) {
  const a = tn.ask, sel = tnSelected(), model = where === "tune" ? tn.pick : (rs.hl && rs.hl.name);
  const input = $(`tn-ask-input-${where}`), val = (input ? input.value : a.value || "").trim();
  tn.err = null; tn.ok = null;
  try {
    if (a.kind === "save") { const r = await invoke("tune_preset_save", { model, name: val, scope: "project", card: tuneCard() }); tn.sel = r.id; tn.ok = `Saved “${r.name}”.`; }
    else if (a.kind === "rename" && sel) { await invoke("tune_preset_rename", { id: sel.id, scope: sel.scope, name: val }); tn.ok = "Renamed."; }
    else if (a.kind === "dup" && sel) { const r = await invoke("tune_preset_duplicate", { id: sel.id, scope: sel.scope, name: val }); tn.sel = r.id; tn.ok = `Copied as “${r.name}”.`; }
    else if (a.kind === "del" && sel) { await invoke("tune_preset_delete", { id: sel.id, scope: sel.scope, confirm: val }); tn.sel = ""; tn.ok = "Deleted."; }
    tn.ask = null; await tuneLoad(model);
  } catch (e) { tn.err = String(e); render(); }
}
function wireTune() {
  const tm = $("tn-model"); if (tm) tm.onchange = () => { tn.pick = tm.value; tn.sel = ""; tn.preview = null; tn.view = null; tn.model = ""; tu.plan = null; tu.state = null; tu.preview = null; tu.applied = null; tuneLoad(tn.pick); render(); };
  wireTuneAuto();
  document.querySelectorAll("[data-tn]").forEach((b) => { b.onclick = () => tuneAction(b.id.split("-").pop(), b.dataset.tn); });
  document.querySelectorAll("[data-tn-sel]").forEach((e) => { e.onchange = () => { tn.sel = e.value; tn.preview = null; tn.ok = null; render(); }; });
  for (const where of ["heal", "tune", "auto"]) {
    const ok = $(`tn-ask-ok-${where}`); if (ok) ok.onclick = () => tuneAskOk(where);
    const no = $(`tn-ask-no-${where}`); if (no) no.onclick = () => { tn.ask = null; render(); };
    const ap = $(`tn-apply-${where}`); if (ap) ap.onclick = () => { if (where === "tune") { tn.ok = "Open the Heal card of the model to use a preset."; } else tuneApplyToCard(tn.preview.data.changes); tn.preview = null; tn.ok = tn.ok || "Applied to the card. Nothing is saved until you start a heal."; render(); try { rsHealEstimate(); } catch (_) { /* a preview */ } };
    const an = $(`tn-apply-no-${where}`); if (an) an.onclick = () => { tn.preview = null; render(); };
  }
}
// ---- "Auto: test and choose the best settings": the runner of the Tune page (docs/TUNE.md)
const tu = { spv: {}, plan: null, ticked: null, maxH: "", trialK: 400, state: null, running: false, text: "", err: null, rows: {}, preview: null, runId: null, applied: null };
const TU_CLASS = { exact: "same results", tol: "same results within tolerance", stat: "same on average", quality: "quality knob: based on a short trial, not a full heal" };
const TU_TIER = { 1: "Speed and memory: the same results, tested in minutes", 2: "Quality settings: short trial heals on your rows (a short trial, not a full heal)", 3: "Other functions" };
function tuSpec() {
  const model = tn.pick;
  const ticked = tu.ticked || {};
  return { model, tests: Object.keys(ticked).filter((k) => ticked[k]), card: rs.hl && rs.hl.name === model ? tuneCard() : {}, max_hours: +tu.maxH > 0 ? +tu.maxH : null, trial_tokens_k: +tu.trialK || 400 };
}
async function tuLoadPlan() {
  if (!tn.pick) return;
  try {
    if (!tu.ticked) { tu.ticked = {}; }
    const first = !tu.plan;
    const plan = await invoke("tune_plan", { spec: tuSpec() });
    if (first || !tu.plan) for (const t of plan.tests) if (!(t.id in tu.ticked)) tu.ticked[t.id] = !!t.default_on && t.built && !t.unavailable;
    tu.plan = plan; tu.err = null;
  } catch (e) { tu.err = String(e); tu.plan = null; }
  try { render(); } catch (_) { /* a preview */ }
}
const tuMin = (m) => (m[1] >= 90 ? `${(m[0] / 60).toFixed(1)} to ${(m[1] / 60).toFixed(1)} h` : `${m[0]} to ${m[1]} min`);
function tuTierHtml(tier, locked) {
  const tests = tu.plan.tests.filter((t) => t.tier === tier);
  if (!tests.length) return "";
  const sum = tests.filter((t) => tu.ticked[t.id] && t.built && !t.unavailable).reduce((a, t) => [a[0] + t.minutes[0], a[1] + t.minutes[1]], [0, 0]);
  const rows = tests.map((t) => `<li class="${t.unavailable ? "dim" : ""}"><label class="opt"><input type="checkbox" data-tu-test="${esc(t.id)}" ${tu.ticked[t.id] ? "checked" : ""} ${t.unavailable || locked ? "disabled" : ""}> <strong>${esc(t.label)}</strong> <span class="tag">${esc(TU_CLASS[t.class] || t.class)}</span> <span class="dim small">${t.unavailable ? "not available: " + esc(t.unavailable) : tuMin(t.minutes)}</span></label><div class="dim small">${esc(t.what)}</div></li>`).join("");
  return `<h3>${esc(TU_TIER[tier])}${sum[1] ? ` <span class="dim small">(${tuMin(sum)})</span>` : ""}</h3><ul class="plain">${rows}</ul>`;
}
function tuLiveHtml() {
  const st = tu.state;
  if (!st) return tu.running ? `<div class="stepline" aria-live="polite">${esc(tu.text || "Starting…")}</div>` : "";
  const rows = st.tests.map((t) => `<tr class="${t.status === "done" ? "" : "dim"}"><td>${esc((tu.plan && (tu.plan.tests.find((x) => x.id === t.id) || {}).label) || t.id)}</td><td>${esc(t.status.replace("_", " "))}</td><td>${esc((t.decision && t.decision.why) || t.note || "")}</td><td>${t.result && t.result.gain > 1 ? "×" + (+t.result.gain).toFixed(2) : ""}</td><td class="dim small">${esc(t.result ? TU_CLASS[t.result.class] || t.result.class : "")}</td></tr>`).join("");
  const cand = (r) => (r.ngl !== undefined ? `${r.ngl} layers on the card` : r.write_mb_s !== undefined ? "Disk" : String(r.batch_tokens ?? r.label ?? r.id ?? ""));
  const result = (r) => (r.tokens_per_s ? Math.round(r.tokens_per_s).toLocaleString("en-US") + " tokens/s" : r.held_after != null ? `held-out loss ${(+r.held_after).toFixed(4)} (short trial)` : r.write_mb_s !== undefined ? `writes ${r.write_mb_s} MB/s, reads ${r.read_mb_s} MB/s` : String(r.error || ""));
  const live = Object.values(tu.rows).flat().slice(-12).map((r) => `<tr><td>${esc(cand(r))}</td><td>${esc(result(r))}</td><td>${r.peak_vram_mb ? (r.peak_vram_mb / 1024).toFixed(1) + " GB" : ""}</td></tr>`).join("");
  return `${tu.running ? `<div class="stepline" aria-live="polite" id="tu-text">${esc(tu.text || "Running…")}</div>` : ""}${st.note ? `<p class="warn" id="tu-wait">${esc(st.note)}</p>` : ""}
    <table class="tbl" id="tu-table"><tr><th>Test</th><th>State</th><th>Decision</th><th>Gain</th><th>Kind</th></tr>${rows}</table>${live ? `<table class="tbl small" id="tu-live"><tr><th>Candidate</th><th>Speed</th><th>Peak memory</th></tr>${live}</table>` : ""}`;
}
const TU_SETTINGS_TARGETS = ["settings_teacher", "settings_vllm", "settings_llama", "settings_forward"];
const tuSettingsRows = () => TU_SETTINGS_TARGETS.flatMap((t) => (tu.spv && tu.spv[t] ? tu.spv[t].changes : []));
function tuApplyHtml() {
  const pv = tu.preview; if (!pv) return "";
  const row = (c, where) => `<tr><td>${esc(c.label)}</td><td class="dim small">${where}</td><td>${esc(String(c.from ?? "automatic"))}</td><td><strong>${esc(String(c.to))}</strong></td><td>${c.gain && c.gain > 1 ? "×" + (+c.gain).toFixed(2) : ""}</td><td class="dim small">${esc(TU_CLASS[c.class] || c.class || "")}</td></tr>`;
  const sets = tuSettingsRows();
  const rows = pv.changes.map((c) => row(c, "Heal card")).join("") + sets.map((c) => row(c, "Settings")).join("");
  return `<div class="card" id="tu-apply"><h3>Apply and save preset</h3><p class="dim small">The preset “${esc(pv.name)}” is already saved (you can select it later on the Heal card, the Auto page and here). Applying fills the Heal card's settings${sets.length ? " and saves the listed program settings (the original's batch, the vLLM engine, llama.cpp's layers)" : ""}; nothing else changes until you start something.</p>
    ${rows ? `<table class="tbl" id="tu-apply-table"><tr><th>Setting</th><th>Where</th><th>Now</th><th>Becomes</th><th>Measured gain</th><th></th></tr>${rows}</table>` : `<p>Your settings are already the best that was found. Nothing would change.</p>`}
    <div class="row"><button class="primary" id="tu-apply-all" ${rows ? "" : "disabled"}>Apply all</button><button class="secondary" id="tu-apply-no">Not now</button></div></div>`;
}
function tuAutoHtml() {
  if (!tn.pick) return "";
  if (!tu.plan && !tu.err) { tuLoadPlan(); return `<div class="card wide"><p class="dim">Planning…</p></div>`; }
  const off = tnBusy() && !tu.running;
  const why = off ? jobWhy() : "";
  const p = tu.plan;
  const locked = tu.running;
  const total = p ? tuMin(p.total_minutes) : "";
  const done = p && p.done_between ? `${new Date(p.done_between[0] * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })} to ${new Date(p.done_between[1] * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}` : "";
  const resume = tu.state && ["stopped", "waiting_gpu"].includes(tu.state.status) && !tu.running;
  return `<div class="card wide" id="tu-auto"><h2>Auto: test and choose the best settings</h2>
    <p class="dim small">One click runs every test below, one after the other, for this model on this PC, and at the end shows every change with its measured gain. It is saved as a preset; applying it is one confirmation. Everything is ticked by default; the most valuable tests run first, so a run that is stopped still gives a useful preset.</p>
    ${tu.err ? `<p class="err">${esc(tu.err)}</p>` : ""}
    ${p ? [1, 2, 3].map((t) => tuTierHtml(t, locked)).join("") : ""}
    <div class="row"><label>I have at most <input type="text" id="tu-hours" value="${esc(tu.maxH)}" placeholder="no limit" style="width:70px;flex:0" ${locked ? "disabled" : ""}> hours</label>
      <label>Trial budget of a quality test <select id="tu-trial" ${locked ? "disabled" : ""}>${[300, 400, 500].map((k) => `<option value="${k}" ${+tu.trialK === k ? "selected" : ""}>${k} thousand tokens each</option>`).join("")}</select></label></div>
    ${p ? `<p id="tu-total"><strong>About ${esc(total)}</strong>${done ? `, done around ${esc(done)}` : ""} for the ticked tests.</p>` : ""}
    <div class="row"><button class="primary big" id="tu-go" ${off || tu.running || !p || !p.total_minutes[1] ? "disabled" : ""} title="${off ? esc(`Off because ${why}`) : "Runs the ticked tests, then shows what to change"}">${resume ? "Continue the tests" : "Auto: test and choose the best settings"}</button>${tu.running ? `<button class="secondary" id="tu-stop">Stop</button>` : ""}</div>
    ${off ? `<p class="dim small" id="tu-why">The button is off because ${esc(why)}.</p>` : ""}
    ${tuLiveHtml()}${tuApplyHtml()}</div>`;
}
async function tuGo() {
  const resume = tu.state && ["stopped", "waiting_gpu"].includes(tu.state.status);
  if (!resume) {
    const p = tu.plan; if (!p) return;
    if (!confirm(`Test this PC for this model?\n\nAbout ${tuMin(p.total_minutes)} for the ticked tests. You can stop at any time; a stopped run still saves what it found.`)) return;
  }
  tu.err = null; tu.preview = null; tu.running = true; tu.text = "Starting…"; tu.rows = {}; render();
  try {
    if (resume) { await invoke("tune_continue", { runId: tu.state.id }); tu.runId = tu.state.id; }
    else { const r = await invoke("tune_start", { spec: tuSpec() }); tu.runId = r.run; tu.state = null; }
  } catch (e) { tu.running = false; tu.err = String(e); }
  render();
}
async function tuRefreshState() {
  if (!tu.runId) return;
  try { tu.state = await invoke("tune_state", { runId: tu.runId }); } catch (_) { /* the run is not saved yet */ }
}
async function onTuneEvent(p) {
  if (p.kind === "tune_test") { tu.text = `Test ${p.index + 1} of ${p.total}: ${p.label || p.id} (${p.status})`; await tuRefreshState(); }
  else if (p.kind === "tune_row") { if (p.row) (tu.rows[p.id] || (tu.rows[p.id] = [])).push(p.row); }
  else if (p.kind === "tune_progress") { const e = p.event || {}; if (e.text || e.step) tu.text = e.text || e.step; }
  else if (p.kind === "tune_wait") { tu.running = false; await tuRefreshState(); }
  else if (p.kind === "error") { tu.running = false; tu.err = p.text; }
  else if (p.kind === "done") {
    tu.running = false; await tuRefreshState();
    if (p.data && p.data.status === "stopped") { /* a stopped run: Continue is offered */ }
    else if (tu.state && tu.state.preset_id) {
      try { tu.preview = await invoke("tune_apply_preview", { presetId: tu.state.preset_id, scope: "project", model: tn.pick, target: "heal_card", current: rs.hl && rs.hl.name === tn.pick ? tuneCard() : {} }); } catch (e) { tu.err = String(e); }
      // the saved settings a test may have decided (the original's batch, the vLLM engine, llama.cpp's layers): the same confirmation lists them, each applied with the token of its own fresh preview
      tu.spv = {};
      for (const target of TU_SETTINGS_TARGETS) { try { const pv = await invoke("tune_apply_preview", { presetId: tu.state.preset_id, scope: "project", model: tn.pick, target, current: {} }); if (pv && pv.changes && pv.changes.length) tu.spv[target] = pv; } catch (_) { /* nothing for this target */ } }
      tn.model = ""; tuneLoad(tn.pick);
    }
  }
  if (page === "tune") render();
}
function wireTuneAuto() {
  document.querySelectorAll("[data-tu-test]").forEach((c) => { c.onchange = () => { tu.ticked[c.dataset.tuTest] = c.checked; tu.plan = null; tuLoadPlan(); }; });
  const h = $("tu-hours"); if (h) h.onchange = () => { tu.maxH = h.value.trim(); tu.plan = null; tuLoadPlan(); };
  const t = $("tu-trial"); if (t) t.onchange = () => { tu.trialK = +t.value; tu.plan = null; tuLoadPlan(); };
  const g = $("tu-go"); if (g) g.onclick = tuGo;
  const st = $("tu-stop"); if (st) st.onclick = () => { tu.text = "Stopping…"; invoke("tune_stop"); render(); };
  const all = $("tu-apply-all"); if (all) all.onclick = async () => {
    const card = tu.preview.changes, sets = tuSettingsRows();
    tuneApplyToCard(card);
    let saved = 0;
    for (const target of TU_SETTINGS_TARGETS) { const pv = tu.spv && tu.spv[target]; if (!pv) continue; try { await invoke("tune_apply", { presetId: tu.state.preset_id, scope: "project", model: tn.pick, target, token: pv.token }); saved += pv.changes.length; } catch (e) { tu.err = String(e); } }
    tu.applied = `Applied ${card.length} change${card.length === 1 ? "" : "s"} to the Heal card${sets.length ? ` and saved ${saved} program setting${saved === 1 ? "" : "s"}` : ""}.`; tu.preview = null; tu.spv = {};
    try { rsHealEstimate(); } catch (_) { /* a preview */ } render();
  };
  const no = $("tu-apply-no"); if (no) no.onclick = () => { tu.preview = null; render(); };
}

function renderTune() {
  const head = `<h1>🎛 ${TITLES.tune[0]}</h1><p class="lead">${TITLES.tune[1]}</p>`;
  const made = ((rs.state && rs.state.made) || []).filter((m) => m.status === "done");
  if (!tn.pick && made.length) tn.pick = made[0].name;
  const picker = made.length ? `<div class="card wide"><h2>Model</h2><div class="row"><select id="tn-model">${made.map((m) => `<option value="${esc(m.name)}" ${m.name === tn.pick ? "selected" : ""}>${esc(m.name)}${m.size_gb ? ` · ${esc(String(m.size_gb))} GB` : ""}</option>`).join("")}</select></div>
      ${tn.view && tn.model === tn.pick ? `<p class="dim small">${esc(String(tn.view.model.layers))} layers, hidden size ${esc(String(tn.view.model.hidden))}, ${(tn.view.model.params / 1e9).toFixed(1)} B parameters on ${esc(tn.view.machine.gpu)} (${Math.round(tn.view.machine.vram_mb / 1024)} GB).</p>` : ""}</div>`
    : `<div class="card wide"><p>There is no finished model to tune yet: make one on the Slim page, or run Auto.</p></div>`;
  const bar = tn.pick ? `<div class="card wide"><h2>Tested settings for this model</h2>${tuneBar("tune", tn.pick)}${tuneListHtml()}</div>` : "";
  return head + picker + tuAutoHtml() + (tu.applied ? `<p class="ok" id="tu-applied">${esc(tu.applied)}</p>` : "") + bar;
}
function tuneListHtml() {
  const v = tn.view; if (!v || tn.model !== tn.pick) return `<p class="dim">Looking…</p>`;
  if (v.unreadable && v.unreadable.length) var bad = `<p class="warn small">${v.unreadable.map((u) => esc(u.file) + (u.kept_as ? ` (kept as ${esc(u.kept_as)})` : "") + ": " + esc(u.why)).join("; ")}</p>`;
  const rows = v.presets.map((p) => `<tr class="${p.id === tn.sel ? "best" : ""}"><td>${esc(p.name)}</td><td>${esc(TN_KINDS[p.kind] || p.kind)}${p.scope === "library" ? " (library)" : ""}</td><td>${esc(String(p.created).slice(0, 10))}</td><td>${tuneGainText(p)}</td><td>${p.stale.length ? `<span class="warn">stale</span>` : ""}</td></tr>`).join("");
  return `${rows ? `<table class="tbl" id="tn-list"><tr><th>Preset</th><th>Match</th><th>Date</th><th>Measured</th><th></th></tr>${rows}</table>` : `<p class="dim" id="tn-empty">No preset yet. Save the settings of the Heal card as one, or let Auto test this PC.</p>`}${bad || ""}`;
}

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
      ${st && st.env_installed && (st.checks || []).some((c) => (c.id === "kernels" || c.id === "triton") && c.light !== "ok") ? `<button class="secondary" id="kernels" ${busy ? "disabled" : ""} title="Installs flash-linear-attention (and causal-conv1d where a ready-made package exists) so that Qwen3.5's linear-attention layers run fast. Nothing else is changed.">Install the fast kernels</button>` : ""}
      <button class="secondary" id="cancelenv" ${busy ? "" : "disabled"}>Cancel</button>
    </div>
    ${env.step ? `<div class="stepline" aria-live="polite">${esc(env.step)}</div>` : ""}
    ${env.progress != null ? `<progress max="1" value="${env.progress}"></progress>` : ""}
    ${env.error ? `<div class="err">${esc(env.error)}</div>` : ""}${result}
    ${env.log.length ? `<pre class="log" id="log">${log}</pre>` : ""}
    ${renderTimeCard()}
    ${renderLlamaSection()}
    ${renderVllm()}
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
  if (p.task === "llamacpp" || (p.data && p.data.task === "llamacpp") || (lc.running && (p.kind === "done" || p.kind === "error" || p.kind === "log" || p.kind === "step" || p.kind === "progress"))) { onLlamaEvent(p); return; }
  if (p.kind === "log") env.log.push(p.text);
  else if (p.kind === "step") { env.step = `Step ${p.index} of ${p.total}: ${p.text}`; env.progress = (p.index - 1) / p.total; }
  else if (p.kind === "progress") { env.step = p.text || p.step; env.progress = p.value; }
  else if (p.kind === "done") { env.running = false; env.progress = 1; env.step = "Finished."; if (p.data && p.data.device_name) env.result = p.data; pfReset(); refreshEnv(); }
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
      ${te.running || te.text ? `<div class="stepline" aria-live="polite">${esc(te.text)}</div>${te.progress != null ? `<progress max="1" value="${te.progress}"></progress>` : ""}` : ""}</div>`;
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
      ${te.asking ? `<div class="stepline" aria-live="polite">${esc(te.text)}</div>` : ""}
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
const HEAL_AUTO = { rank: null, seq: null, batch: null, per_update: null, checkpointing: "auto", method: null, optimizer: null, kl_chunk: null, k: null, teacher_load: null, warmup: null, kbit_upcast: null, head_checkpoint: null, student_batch_tokens: null };
let rs = { state: null, model: "", mode: "prune", options: null, optErr: null, loadingOpts: false, token: 0, blocks: null, init: "identity", samples: 128, strategy: "contiguous", pf: 1, pl: 1, load: "auto",
  dropText: "", running: false, testing: null, text: "", progress: null, log: [], error: null, ok: null, view: null,
  // thinner layers (width): the way ("depth" | "width" | "both"), the chosen MLP width, the Expert axes, measuring, and healing
  shape: "depth", wi: null, lv: false, hh: null, wsamples: 256, scoring: "activation", recon: false, meas: null, measModel: null,
  // "Try first": what to try, the clicked blocks, the plan (how many cuts, how long), the kept answers and a run in progress
  tr: { what: "blocks", scoring: false, recon: false, se: { target: "b", b: null, gb: "", pct: 75, budget: 20, keep: null, axes: "both", think: true, confirm: true, many: false, beyond: false, thorough: false, plan: null, planErr: null, planFor: "", planTok: 0, running: false, text: "", progress: null, done: 0, total: 0, round: null, left: null, best: null, rows: [], result: null, err: null, partial: null, id: null }, quick: null, sel: [], plan: null, planErr: null, planFor: null, planTok: 0, resFor: null, analysis: null, sps: null, running: false, text: "", progress: null, done: 0, total: 0 },
  hs: { cand: null, studentPick: "", studentFolder: "", studentHub: "", teacherPick: "", teacherFolder: "", teacherHub: "", checking: false, check: null, checkKey: null, err: null },
  hl: null, hlRun: null, hlSpec: { general: true, tokens_m: null, dataset: "", share: 0.5, shareSet: false, recipe: null, alpha: 1, lr: 0.0001, mode: "", rounds: false, roundM: 5, firstCheck: 10, roundCheck: "quick", stopRule: "pass", ...HEAL_AUTO }, hlEst: null, hlSeed: {},
  // the thinking check of the shown model: its saved numbers, the choice of quick or full, the GGUF to check, and a check in progress
  tk: { name: null, data: null, quick: null, gguf: "", run: null, err: null, ok: null } };

// the "Save a model as GGUF (no training needed)" card of this page (docs/RESHAPE.md): the model to save (a made one, one of the list, or a folder on this computer),
// the same sizes and options as the export of a training run (Step 5), and the run in progress. Events come on "reshape-event" tagged {"phase": "export"}.
let rsx = { state: null, src: "made", pick: "", folder: "", quants: ["q4_k_m"], numCtx: 8192, thinking: false,
  im: false, imChunks: 128, imReuse: null, imNextn: true, imSaved: "", mx: { target: "", matrix: "" }, mm: { target: "", matrix: "" },
  running: false, pending: null, text: "", progress: null, log: [], error: null, ok: null };

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

// Where the parameters are: "text layers 19.00 B · embeddings 1.27 B · ... · file total 19.81 B" (the same words as `ParamsBreakdown::line` in the program).
function breakdownLine(b) {
  if (!b || b.total == null) return "";
  const x = (n) => ((+n || 0) / 1e9).toFixed(2) + " B";
  const parts = [`text layers ${x(b.text_layers)}`, `embeddings ${x(b.embeddings)}`];
  if (b.output > 0) parts.push(`output ${x(b.output)}`);
  if (b.vision > 0) parts.push(`vision ${x(b.vision)}`);
  if (b.mtp > 0) parts.push(`multi-token head ${x(b.mtp)}`);
  if (b.mtp_removed > 0) parts.push(`multi-token head removed ${x(b.mtp_removed)}`);
  parts.push(`other ${x(b.other)}`);
  return `${parts.join(" · ")} · file total ${x(b.total)}`;
}

function rsSizeInfo() {
  const o = rsOption(), op = rs.options, v = op && op.vram_gb;
  if (!o) return `<span class="dim">Nothing can be changed in whole blocks for this model.</span>`;
  const tick = (fits) => fits === true ? "✓" : fits === false ? "✗" : "?";
  return `<strong>${o.layers} layers · ${o.params_b} B parameters</strong> · ${o.disk_gb} GB on disk · training: LoRA ≈ ${o.vram_lora_gb} GB ${tick(o.fits_lora)}, QLoRA ≈ ${o.vram_qlora_gb} GB ${tick(o.fits_qlora)}${v ? ` on this card (${v} GB)` : ""}${o.breakdown ? `<br><span class="dim small" id="rs-breakdown">${esc(breakdownLine(o.breakdown))}</span>` : ""}`;
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
  return `MLP ${thou(o.intermediate)}${full ? " of " + thou(full) + " (−" + cut + " %)" : ""} · <strong>${o.layers} layers · ${o.params_b} B parameters${o.exact ? " (exact)" : ""}</strong> · ${o.disk_gb} GB on disk · training: LoRA ≈ ${o.vram_lora_gb} GB ${tick(o.fits_lora)}, QLoRA ≈ ${o.vram_qlora_gb} GB ${tick(o.fits_qlora)}${v ? ` on this card (${v} GB)` : ""}${o.breakdown ? `<br><span class="dim small" id="rs-breakdown">${esc(breakdownLine(o.breakdown))}</span>` : ""}`;
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
    <p class="dim small" id="rs-measinfo">${meas}</p>${expert ? rsScoringHtml() : ""}${axes}`;
}

// Expert: how the neurons that stay are chosen. The default is how much they fire; "output" (experimental) also counts how much each one writes into the layer's output.
function rsReconNumbers() {
  const a = rs.tr && rs.tr.analysis, rows = ((a && a.variants) || []).filter((r) => r.tried && r.recon && r.intermediate === rs.wi);
  if (!rows.length) return `<span class="dim small">Try first has not measured the refit for this width.</span>`;
  const pc = (x) => x == null ? "–" : Math.round(100 * x) + " %";
  return rows.map((r) => `<span class="dim small">Try first, ${esc(r.label)}: error ${(+r.loss).toFixed(3)} (${r.better_both_halves === true ? "better" : r.better_both_halves === false ? "not better" : "not compared"} than the plain cut in both halves); it gives back ${pc(r.recon.median_removed_share)} of what the cut took (median layer) on rows it was not fitted on.</span>`).join(" ");
}

function rsScoringHtml() {
  return `<p id="rs-recon-row"><label class="opt"><input type="checkbox" id="rs-recon" ${rs.recon ? "checked" : ""} ${rs.running ? "disabled" : ""}> Refit the kept neurons without training (experimental, off)</label>
    <span class="dim small">Changes the output weights of the kept neurons by least squares to make up for part of what the removed ones did. Needs about 10 to 20 minutes more (a formula, not a measurement) and about 6 GB of disk next to the measurement. Nobody has measured here that it keeps more quality; compare it in Try first, and a gain before healing may not survive healing.</span> ${rsReconNumbers()}</p>
  <p id="rs-scoring-row">Rank the neurons by <select id="rs-scoring" ${rs.running ? "disabled" : ""}><option value="activation" ${rs.scoring !== "output" ? "selected" : ""}>how much they fire (the default)</option><option value="output" ${rs.scoring === "output" ? "selected" : ""}>their expected output (experimental)</option></select>
    <span class="dim small">The second counts the mean square of what a neuron fires times the squared size of what it writes into the layer. Nobody has measured here that it keeps more quality; the “Try first” card below compares the two on your text and says whether both halves agree. A gain before healing may not survive healing.</span></p>`;
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
    const est = rs.hl.pair ? await invoke("heal_pair_estimate", { student: rs.hl.pair.student, teacher: rs.hl.pair.teacher, spec: rsHealSpec() }) : await invoke("heal_estimate", { name, spec: rsHealSpec() });
    if (est && rs.hl && rs.hl.name === name) { rs.hlEst = { name, key, est }; render(); }
  } catch (e) { /* the page keeps the estimate it has */ }
}

// the measured numbers of a healing run (saved ones, then the live ones of this session), as chart points
function healSeries() {
  const ev = ((rs.hl && rs.hl.opts && rs.hl.opts.metrics) || []).concat((rs.hlRun && rs.hlRun.live) || []);
  const seen = new Set(), kd = [], cal = [], hold = [];
  ev.forEach((e) => { const k = e.kind + ":" + e.step + ":" + (e.kd ?? e.calib_loss); if (seen.has(k)) return; seen.add(k); if (e.kind === "distill_metric") kd.push([e.step, e.kd]); else if (e.kind === "distill_eval") { cal.push([e.step, e.calib_loss]); if (e.holdout_loss != null) hold.push([e.step, e.holdout_loss]); } });
  return { kd, cal, hold };
}

// the Expert "Healing settings" section of the Heal card; null in a field of the spec = the automatic value, shown here from `kn`
// "Find the fastest loss settings": short real steps of the healing loss with several settings on this computer (the loss is the same in all of them), one row per setting.
function hbHtml(busy) {
  const hb = rs.hb || {}, res = hb.result || hb.last, rows = hb.running ? (hb.rows || []) : (res && res.rows) || [];
  const gb = (mb) => mb == null ? "–" : (mb / 1024).toFixed(1) + " GB";
  const body = rows.map((r) => `<tr${res && res.best === r.id && !hb.running ? ` class="best"` : ""}><td>${esc(r.label)}</td><td>${r.ok ? Math.round(r.tokens_per_s).toLocaleString("en-US") : "–"}</td><td>${r.ok && r.vs_current != null ? "×" + r.vs_current.toFixed(2) : "–"}</td><td>${r.ok ? gb(r.peak_vram_mb) : "–"}</td><td>${r.ok ? (res && res.best === r.id && !hb.running ? "fastest" : "ok") : r.oom ? "ran out of memory" : "failed"}</td></tr>`).join("");
  const best = res && res.best_settings && !hb.running ? res.best_settings : null;
  return `<div id="hb-box"><h4>Find the fastest loss settings</h4>
    <p class="dim small">Runs ${hb.steps || 12} real steps of the healing loss with the KL chunk 64 to 512 and, when it fits the card, with the output layer's logits kept (about a quarter of an hour in all). The loss is the same in every row; only speed and memory differ. Nothing is assumed: these numbers are this computer's. A real heal is a little slower (checkpoints, checks).</p>
    <div class="row"><button class="secondary" id="hb-go" ${busy || hb.running || !rs.hl || !rs.hl.opts || rs.running ? "disabled" : ""}>Find the fastest</button> <button class="secondary" id="hb-stop" ${hb.running ? "" : "disabled"}>Stop</button>
      <button class="secondary" id="hb-use" ${best && !busy ? "" : "disabled"}>Use the fastest${best ? ` (KL chunk ${best.kl_chunk}${best.head_checkpoint === "off" ? ", logits kept" : ""})` : ""}</button></div>
    ${hb.running ? `<div class="stepline" aria-live="polite">${esc(hb.text || "Starting…")}</div>` : ""}${hb.err ? `<p class="err">${esc(hb.err)}</p>` : ""}
    ${rows.length ? `<table class="tbl" id="hb-table"><tr><th>Setting</th><th>Tokens per second</th><th>Against yours</th><th>Peak memory</th><th></th></tr>${body}</table>` : ""}</div>`;
}

function onHealBenchEvent(p) {
  const hb = rs.hb || (rs.hb = {});
  if (p.kind === "heal_bench") { hb.text = `Setting ${p.index + 1} of ${p.total}: ${p.name}`; if (p.row) (hb.rows || (hb.rows = [])).push(p.row); }
  else if (p.kind === "error") { hb.running = false; hb.err = p.text; }
  else if (p.kind === "done") { hb.running = false; hb.err = null; if (p.data) { hb.result = p.data; hb.last = p.data; } }
  if (onReshape()) render();
}

function healSettings(sp, kn, plan, busy, upcastGb, headOff) {
  const d = busy ? "disabled" : "", v = (x, auto) => (x ?? auto);
  const sel = (id, cur, opts) => `<select id="${id}" ${d}>${opts.map(([val, label]) => `<option value="${val}" ${String(cur) === String(val) ? "selected" : ""}>${label}</option>`).join("")}</select>`;
  const pu = v(sp.per_update, (kn.batch || 1) * (kn.grad_accum || 1)), bt = v(sp.batch, kn.batch || 1);
  return `<details id="hl-settings" open><summary>Healing settings</summary>
    ${tuneBar("heal", rs.hl && rs.hl.name)}
    <p class="dim small">The safe settings for this model and card are already chosen. If healing runs out of memory, the program tries again by itself with safer ones.</p>
    <div class="grid">
    <label>Sequence length<br>${sel("hl-seq", v(sp.seq, kn.seq), [[512, "512"], [1024, "1024"], [2048, "2048"], [4096, "4096"], [8192, "8192 (long reasoning rows; needs a lot of memory)"]])}</label>
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
    <div class="row"><label>Tokens per forward pass (Expert; 0 = one row at a time, as always)<br><input type="text" id="hl-sbt" value="${v(sp.student_batch_tokens, 0)}" ${d} title="Rows of similar length are put together in one forward pass, up to this many tokens, padded. The same learning: every row keeps its weight and the update still holds the same rows. Measure heal speed recommends a value; 512 to 65,536."></label></div>
    <label class="opt"><input type="checkbox" id="hl-upcast" ${sp.kbit_upcast === true ? "checked" : ""} ${busy || kn.qlora === false ? "disabled" : ""}> Keep the embeddings and output layer in 32-bit (the old behaviour: about <span id="hl-upcast-gb">${upcastGb != null ? upcastGb : "4 to 5"}</span> GB more memory)${kn.qlora === false ? " - only for QLoRA" : ""}</label>
    <p class="dim small">By default the two big word tables stay in 16-bit while healing: they are not changed by the learning, so a 32-bit copy only costs memory. Changing this keeps the saved answers and the learning done so far. If healing runs out of memory with this on, the program turns it off first.</p>
    ${(() => { const on = sp.head_checkpoint === "off", gb = headOff && headOff.head_off_gb != null ? headOff.head_off_gb : null, fits = headOff ? headOff.head_off_fits : null; const off = busy || (!on && fits === false);
      return `<label class="opt"><input type="checkbox" id="hl-headkeep" ${on ? "checked" : ""} ${off ? "disabled" : ""}> Keep the output layer's logits for the backward pass (faster, the same learning; about <span id="hl-headkeep-gb">${gb != null ? gb : "several"}</span> GB more memory)${!on && fits === false ? " - does not fit this card" : ""}</label>
    <p class="dim small">Normally the logits of every chunk are made again in the backward pass to save memory. Keeping them stores about tokens × 250,000 × 10 bytes (about 20 GB for 8,192 tokens), so it is offered only when the estimate says it fits. It changes nothing about what is learned and keeps the saved answers and the learning done so far; how much faster it is is measured by "Find the fastest loss settings", not assumed. If healing runs out of memory with it on, the program turns it off first.</p>`; })()}
    ${hbHtml(busy)}
    <p class="dim small">A different sequence length, LoRA size or top-k means the original answers again or the learning starts again; the other settings keep what was saved.</p>
    <div class="row"><button class="secondary" id="hl-reset" ${d}>Reset to safe defaults</button></div></details>`;
}

// ---- healing in rounds (experimental, docs/HEAL_ROUNDS.md): the settings, the estimate line, the live table of rounds
const HL_ROUND_SIZES = [1, 2, 5, 10, 20];
const HL_MIN_STEPS = 8;
const HL_QUICK_NOISE = "A Quick check asks 16 questions: one or two answers decide pass or fail. That is why the merged model is checked again at the end; if it fails there, healing goes on and then needs two passed checks in a row.";
const HL_ADAPTER_NOTE = "The checks during healing look at the model as it is trained (4-bit with the learning on top); the merged model can score a little differently. The merged check decides.";
const HL_EXPERIMENTAL = "Experimental: whether healing in rounds ends as good as one long heal is not known yet.";
// the optimizer steps of a round, estimated the way the program does: a step reads about half of seq x sequences per update tokens
function hlRoundSteps(roundM, kn) { const seq = kn.seq || 2048, pu = (kn.batch || 1) * (kn.grad_accum || 1) || 32; return roundM * 1e6 / (0.5 * seq * pu); }
// the rounds part of what the Heal card sends (null when it is off); `cap` is the most tokens in million
function hlRoundsSpec(cap, extra) {
  const sp = rs.hlSpec; if (!sp.rounds) return null;
  const round = Math.min(sp.roundM, cap), first = Math.max(round, Math.min(sp.firstCheck, cap));
  return { round_m: round, max_tokens_m: cap, first_check_m: first, check: sp.roundCheck, stop: sp.stopRule, ...(extra || {}) };
}
function hlHours(m) { return m ? `${(m[0] / 60).toFixed(0)}-${(m[1] / 60).toFixed(0)}` : ""; }
// the rows of the table: the saved ones, and the decisions the live events brought after them
function hlRoundRows(h, run) {
  const saved = ((h && h.heal && h.heal.rounds && h.heal.rounds.rows) || []).map((r) => ({ n: r.n, tokens_m: r.tokens_m, calib_loss: r.calib_loss, check: r.check, decision: r.decision, seconds: (r.cache_s || 0) + (r.train_s || 0) + (r.check_s || 0) }));
  ((run && run.rounds) || []).filter((e) => e.decision && !["baseline", "final", "merged_fail"].includes(e.decision)).forEach((e) => {
    const i = saved.findIndex((r) => r.n === e.round), row = { n: e.round, tokens_m: e.tokens_m, calib_loss: e.calib_loss, check: e.check, decision: e.decision, seconds: null };
    if (i >= 0) saved[i] = { ...saved[i], ...row, seconds: saved[i].seconds }; else saved.push(row);
  });
  return saved.sort((a, b) => a.n - b.n);
}
function hlRoundsTable(h, run) {
  const rows = hlRoundRows(h, run), pct = (x) => (x == null ? "-" : Math.round(x * 100) + " %");
  if (!rows.length) return "";
  const dec = { continue: "goes on", stop_pass: "passed: stop", stop_plateau: "no progress: stop", stop_cap: "most tokens: stop", stop_time: "time limit: stop", stop_never: "most tokens: stop" };
  const body = rows.map((r) => { const m = (r.check && r.check.main) || null;
    return `<tr><td>${r.n}</td><td>${r.tokens_m}</td><td>${m ? pct(m.closed_rate) : "-"}</td><td>${m ? pct(m.accuracy) : "-"}</td><td>${m ? pct(m.loop_rate) : "-"}</td><td>${r.check && r.check.score != null ? (+r.check.score).toFixed(2) : "-"}</td><td>${r.calib_loss != null ? (+r.calib_loss).toFixed(3) : "-"}</td><td>${esc(dec[r.decision] || r.decision || "running")}</td><td>${r.seconds ? Math.round(r.seconds / 60) + " min" : "-"}</td></tr>`; }).join("");
  const merged = ((h && h.heal && h.heal.rounds && h.heal.rounds.merged_checks) || []).map((c) => `<p class="warn small">The merged model after round ${c.after_round} did not pass (${esc((c.failed || []).join(", "))}); healing went on.</p>`).join("");
  const fin = h && h.heal && h.heal.rounds && h.heal.rounds.final ? `<p class="small">The merged model's check (${esc(h.heal.rounds.final.mode || "")}): <strong>${esc(h.heal.rounds.final.verdict)}</strong>.</p>` : "";
  return `<table id="hl-rounds-table" class="small"><thead><tr><th>Round</th><th>Tokens so far (M)</th><th>Finishes reasoning</th><th>Right</th><th>Loops</th><th>Score</th><th>Loss on your text</th><th>Decision</th><th>Time</th></tr></thead><tbody>${body}</tbody></table>${merged}${fin}`;
}
// the controls of the card
// "Make the original's answers with vLLM (experimental)": only offered when Settings switch it on; the numbers are the person's own where they exist (docs/VLLM.md)
// "Measure heal speed" (docs/HEAL_SPEED.md): where the time of a heal goes on this computer; nothing runs unless the button is pressed
const hsp = { run: null, err: null, files: undefined, name: null, teacher: false };
function hlSpeedBox(busy) {
  const name = rs.hl && rs.hl.name;
  if (name && hsp.name !== name) { hsp.name = name; hsp.files = undefined; }
  if (name && hsp.files === undefined) { hsp.files = null; Promise.resolve(invoke("heal_speed_report", { name })).then((f) => { hsp.files = f || null; try { render(); } catch (_) { /* a preview without a window */ } }).catch(() => {}); }
  const f = hsp.files, hint = f && (f.windows_hints || []).find((h) => /: off|A display is attached|never changes/.test(h));
  const running = hsp.run !== null;
  return `<details id="hl-speed" ${f && f.exists ? "open" : ""}><summary class="small"><strong>Measure heal speed</strong> (where does the time go on this computer?)</summary>
    <p class="dim small">Profiles a few real steps of the healing step (forward, loss, backward, optimizer, the GatedDeltaNet and matrix-multiplication kernels), times one GatedDeltaNet forward + backward, and writes a report you can read or paste. Nothing runs unless you press the button; it takes a few minutes. The research report's speed claims are hypotheses until this measurement confirms them.</p>
    <div class="row"><label class="opt"><input type="checkbox" id="hl-speed-teacher" ${hsp.teacher ? "checked" : ""}> Also time the original's answers per rows-per-forward (loads the original: slow)</label>
      <button class="secondary" id="hl-speed-go" ${busy || running ? "disabled" : ""}>Measure heal speed</button>${running ? `<button class="secondary" id="hl-speed-stop">Stop</button>` : ""}</div>
    ${running ? `<div class="stepline" aria-live="polite">${esc(hsp.run.text)}</div>` : ""}${hsp.err ? `<div class="err">${esc(hsp.err)}</div>` : ""}${hint ? `<p class="warn small" id="hl-speed-hint">${esc(hint)}</p>` : ""}
    ${(() => { const r = f && f.recommended; if (!r) return ""; const cur = +((rs.hlSpec || {}).student_batch_tokens || 0);
      return r.value ? `<p id="hl-speed-rec" class="ok small"><strong>Recommended: ${esc(String(r.value))} tokens per forward pass.</strong> ${esc(r.why || "")} <button class="secondary" id="hl-speed-use" ${busy || cur === +r.value ? "disabled" : ""}>Use the recommended batch size</button> <span class="dim">(never automatic)</span></p>`
        : `<p id="hl-speed-rec" class="dim small">${esc(r.why || "No token budget is clearly faster.")}</p>`; })()}
    ${f ? vllmFilesBox({ ...f, open: false }, "hl-speed-files", "The report is saved in:") : ""}</details>`;
}
function onHealSpeedEvent(p) {
  if (p.kind === "heal_speed" || p.kind === "progress") { if (hsp.run) hsp.run.text = p.text || p.step || hsp.run.text; }
  else if (p.kind === "error") { hsp.run = null; hsp.err = p.text; hsp.files = undefined; }
  else if (p.kind === "done") { hsp.run = null; hsp.err = p.data && p.data.error ? p.data.error : null; hsp.files = undefined; }
  if (onReshape()) render();
}
const hlVllmAvailable = () => !!(vllmSettings().enabled && vllmTeacher().enabled);
const vch = { run: null, err: null, probe: null, probing: false };
function hlVllmBox(sp, est, busy) {
  if (!hlVllmAvailable()) return "";
  const v = est && est.vllm_teacher, ck = v && v.check && v.check.verdict ? v.check : null;
  const fh = (m) => (m[0] / 60).toFixed(1) + " to " + (m[1] / 60).toFixed(1);
  const verdict = ck ? `<p id="hl-vllm-verdict" class="${ck.verdict === "different" ? "err" : ck.verdict === "same" ? "ok" : "warn"}"><strong>Check: ${esc(ck.verdict)}</strong>${ck.current === false ? " (made for other settings: check again)" : ""}. ${esc(ck.text || "")}${ck.vllm ? ` <span class="dim">Start-up ${ck.vllm.startup_seconds != null ? Math.round(ck.vllm.startup_seconds) + " s" : "not timed"}, scoring ${ck.vllm.tokens_per_s != null ? Math.round(ck.vllm.tokens_per_s) + " tokens/s" : "not timed"}.</span>` : ""}</p>` : "";
  let html = `<label class="opt"><input type="checkbox" id="hl-vllm" ${sp.vllm ? "checked" : ""} ${busy ? "disabled" : ""}> <strong>Make the original's answers with vLLM</strong> — experimental</label>`;
  if (!sp.vllm) return html;
  html += `<div id="hl-vllm-box" class="card-in">`;
  if (v) {
    html += `<p class="${v.route.route === "refused" ? "err" : "dim"} small" id="hl-vllm-route">${esc(v.route.reason)}</p><p id="hl-vllm-est">${esc(v.text)}</p>`;
    html += (v.warnings || []).map((w) => `<p class="warn small">${esc(w)}</p>`).join("");
    html += `<details><summary class="small">Where these numbers come from</summary><ul class="small dim">${(v.basis || []).map((b) => `<li>${esc(b)}</li>`).join("")}</ul></details>`;
  }
  html += vprobeBox();
  html += verdict + vllmFilesBox(v && v.files && v.files.exists ? v.files : null, "hl-vllm-files", "The check is saved in:");
  const running = vch.run !== null;
  const offWhy = busy || running || tnBusy() ? (jobWhy() || "a job is running") : "";
  const offTip = offWhy ? `Off because ${offWhy}. ` : "";
  html += `<div class="row"><button class="secondary" id="hl-vllm-check" ${offWhy ? "disabled" : ""} title="${esc(offTip)}">Check the vLLM teacher</button><button class="secondary" id="hl-vllm-checkcmp" ${offWhy ? "disabled" : ""} title="${esc(offTip)}Scores the same 100 rows twice with vLLM, once with the exact KV cache and once with the 8-bit one, against the standard teacher's answers (made once), and prints both and the difference">Check auto and 8-bit cache</button><button class="secondary" id="hl-vllm-checkkv" ${offWhy ? "disabled" : ""} title="${esc(offTip)}Measures how much the 8-bit KV cache changes the teacher's answers (mean KL, top-1 agreement against the exact reference)">Check with the 8-bit KV cache</button>${running ? `<button class="secondary" id="hl-vllm-checkstop">Stop</button>` : ""}
    <span class="dim small">Compares its answers with the standard teacher's on 100 of your rows (uses an existing saved answer set as the reference when there is one).</span></div>
    ${offWhy ? `<p class="warn small" id="hl-vllm-why">These buttons are off because ${esc(offWhy)}. Wait for it to end, or press its Stop.</p>` : ""}
    ${running ? `<div class="stepline" aria-live="polite">${esc(vch.run.text)}</div>${vch.run.progress != null ? `<progress max="1" value="${vch.run.progress}"></progress>` : ""}` : ""}${vch.err ? `<div class="err">${esc(vch.err)}${memHint(vch.err)}</div>` : ""}</div>`;
  return html;
}
// "Test the container copy": which way of moving a 1 KB file into and out of the vLLM container works on this PC
function onVprobeEvent(p) {
  if (p.kind === "progress") vch.probeText = p.text || p.step;
  else if (p.kind === "error") { vch.probing = false; vch.probe = { ok: false, text: p.text }; }
  else if (p.kind === "done") { vch.probing = false; vch.probe = p.data || { ok: false, text: "No answer." }; }
  if (onReshape()) render();
}
function vprobeBox() {
  const r = vch.probe;
  const res = vch.probing ? `<div class="stepline" aria-live="polite">${esc(vch.probeText || "Testing…")}</div>` : r ? `<div id="hl-vllm-probe-result" class="${r.ok ? "ok" : "err"}"><strong>${r.ok ? "The container copy works" : "The container copy does not work"}</strong> — ${esc(r.text || "")}${r.tried ? `<ul class="small dim">${r.tried.map((t) => `<li>${t.ok ? "worked" : "failed"}: ${esc(t.route)} — ${esc(t.detail || "")}</li>`).join("")}</ul>` : ""}</div>` : "";
  return `<div class="row"><button class="secondary" id="hl-vllm-probe" ${vch.probing || vch.run ? "disabled" : ""} title="Starts the idle container, writes 1 KB in and reads it back by every way (a mounted folder, cp, stdin), removes the container. A minute at most.">Test the container copy</button><span class="dim small">Finds out in a minute how files get into the vLLM container on this PC, before a long cache job.</span></div>${res}`;
}
function onVcheckEvent(p) {
  if (p.kind === "progress") { if (vch.run) { vch.run.text = p.text || p.step; vch.run.progress = p.value; } }
  else if (p.kind === "error") { vch.run = null; vch.err = p.text; }
  else if (p.kind === "done") { vch.run = null; vch.err = null; try { rsHealEstimate(); } catch (_) { /* a preview without a window */ } }
  if (onReshape()) render();
}
function hlRoundsBox(h, sp, kn, est, busy, thinks, cap) {
  const d = busy ? "disabled" : "", rt = est && est.rounds ? est.rounds : null, tm = rt && rt.time;
  const label = thinks ? "Heal in rounds and check after each (stop early when it is good enough) — experimental" : "Heal in rounds and stop when the error stops falling — experimental";
  let html = `<label class="opt"><input type="checkbox" id="hl-rounds" ${sp.rounds ? "checked" : ""} ${d}> <strong>${label}</strong></label>`;
  if (!sp.rounds) return html;
  const steps = (m) => hlRoundSteps(m, kn);
  const opts = HL_ROUND_SIZES.map((m) => { const few = steps(m) < HL_MIN_STEPS;
    return `<option value="${m}" ${m === sp.roundM ? "selected" : ""} ${few || m > cap ? "disabled" : ""}>${m} million${m === 5 ? " (recommended)" : ""}${few ? ` — only about ${Math.floor(steps(m))} steps, at least ${HL_MIN_STEPS} are needed` : ""}</option>`; }).join("");
  const share = tm ? ` loading the models${thinks ? " and the check" : ""} add about ${Math.round(tm.per_round[0] - 0)}-${Math.round(tm.per_round[1])} min of work per round, ${Math.round(tm.overhead_share[0] * 100)}-${Math.round(tm.overhead_share[1] * 100)} % of it overhead` : "";
  const warnOver = rt && rt.warn_overhead ? ` <span class="warn">More than a fifth of every round is overhead: a longer round is cheaper.</span>` : "";
  html += `<div class="grid" id="hl-rounds-box">
    <label>Train before each check<br><select id="hl-round" ${d}>${opts}</select></label>
    ${thinks ? `<label>No check before<br><select id="hl-firstcheck" ${d}>${[5, 10, 20].map((m) => `<option value="${m}" ${m === sp.firstCheck ? "selected" : ""}>${m} million tokens</option>`).join("")}</select></label>
    <label>Check<br><select id="hl-roundcheck" ${d}><option value="quick" ${sp.roundCheck === "quick" ? "selected" : ""}>Quick (16 questions, about 10-15 min)</option><option value="full" ${sp.roundCheck === "full" ? "selected" : ""}>Full (32 questions, about 15-25 min)</option></select></label>` : ""}
    <label>Stop rule<br><select id="hl-stoprule" ${d}><option value="pass" ${sp.stopRule === "pass" ? "selected" : ""}>${thinks ? "Stop at the first passed check" : "Stop when the error stops falling"}</option><option value="never" ${sp.stopRule === "never" ? "selected" : ""}>Never stop early (record the checks; to compare with a normal heal)</option></select></label></div>
    <p class="dim small" id="hl-round-line">About ${Math.floor(steps(sp.roundM))} training steps per round;${share}.${warnOver} The original's answers are always saved first, round by round.</p>`;
  if (tm) {
    html += `<p class="small" id="hl-rounds-est">${thinks ? `If the first check passes: about ${hlHours(tm.best)} h. ` : ""}If it runs to the most tokens: about ${hlHours([tm.worst[0], tm.worst[1]])} h${rt.normal_heal ? ` (a normal heal: ${hlHours(rt.normal_heal)} h)` : ""}. <span class="dim">Formulas, not measurements.</span></p>`;
  }
  if (thinks) html += `<p class="warn small" id="hl-quick-noise">${esc(HL_QUICK_NOISE)}</p><p class="warn small" id="hl-adapter-note">${esc(HL_ADAPTER_NOTE)}</p>`;
  html += `<p class="warn small" id="hl-experimental">${esc(HL_EXPERIMENTAL)}</p>`;
  return html;
}
// after the most tokens without a pass: more rounds on the same run
function hlMoreBox(h, busy) {
  const r = h && h.heal && h.heal.rounds;
  if (!r || r.state !== "ended" || !r.final || ["pass", "warn"].includes(r.final.verdict)) return "";
  const cap = Math.round(r.cap_m || 0);
  return `<p class="warn small">The merged model did not pass its thinking check after ${cap} million tokens.</p><div class="row"><button class="secondary" id="hl-more" ${busy ? "disabled" : ""}>Heal 10 M more</button></div>`;
}

// "Heal it": the original teaches the thinner model to answer like it did
// `pair`: a model from elsewhere (or another original for a made one), chosen in the Heal section: the same card, bound to the pair
// What the rows of a heal cover (`coverage` of the worker's teacher pass): how many conversations are whole, cut or left out, the same for reasoning rows, what the kept tokens are, and the lengths.
function coverageHtml(c) {
  if (!c || !c.rows) return "";
  const r = c.rows, ch = c.chat_rows || {}, th = c.reasoning_rows || {}, sh = c.token_share || {};
  const pc = (x) => Math.round(100 * (+x || 0)) + "%";
  const rowsLine = `${r.project_text || 0} texts and ${r.project_chat || 0} conversations from your world` + (r.general_text || r.general_chat ? `, ${r.general_text || 0} texts and ${r.general_chat || 0} conversations of general text` : "");
  const chats = `<span>Conversations: <strong>${ch.complete || 0}</strong> whole, <strong>${ch.cut || 0}</strong> cut at the sequence length, <strong>${ch.skipped || 0}</strong> left out as too long.</span>`;
  const reasoning = th.known && (th.total || 0) > 0 ? `<span>Reasoning rows: <strong>${th.complete || 0}</strong> whole, <strong>${th.cut_answer || 0}</strong> lost part of the answer, <strong>${th.cut_reasoning || 0}</strong> cut inside the reasoning, <strong>${th.skipped || 0}</strong> left out.</span>` : "";
  const shares = `<span>The kept tokens are ${pc(sh.prompt)} question, ${pc(sh.reasoning)} reasoning, ${pc(sh.answer)} answer, ${pc(sh.plain)} plain text.</span>`;
  const lb = c.length_buckets || {};
  const lens = Object.keys(lb).length ? `<span>Lengths before cutting: ${Object.keys(lb).map((k) => `${esc(k)} ${lb[k]}`).join(" · ")}.</span>` : "";
  const notes = (c.notes || []).map((n) => `<p class="warn small">${esc(n)}</p>`).join("");
  return `<div class="dim small" id="hl-coverage"><span>What healing learns from: ${esc(rowsLine)}.</span> ${chats} ${reasoning} ${shares} ${lens}</div>${notes}`;
}

function healCard(m, pair) {
  const sm = m.summary;
  if (!sm || sm.mode !== "prune") return "";
  if (!pair && rs.hl && rs.hl.pair) return `<div class="card-in"><h3>Heal it</h3><p class="dim">${rs.hl.name === m.name ? "Another original is chosen in the <strong>Heal a model</strong> section above: the settings are there." : "The Heal settings are in the <strong>Heal a model</strong> section above (another model is chosen there). Choose this model there to heal it."}</p></div>`;
  const h = rs.hl && rs.hl.name === m.name ? rs.hl.opts : null, run = rs.hlRun;
  const wide = sm.shape === "width" || sm.shape === "both";
  const head = pair ? `<h3>Heal this model</h3>` : `<h3>Heal it ${wide ? "(recommended)" : "(optional)"}</h3>`;
  const thin = pair ? "the model" : "the thinner model";
  if (m.healed && pair) return `<div class="card-in">${head}<p>Healed: <strong>${esc((rs.state.models.find((x) => x.id === m.healed_model_id) || {}).name || m.healed_model_id || "")}</strong> is in the model list of Step 3.The original's saved answers stay until you delete this heal.</p>
    <div class="row">${m.healed_model_id ? `<button class="primary" data-rstrain="${esc(m.healed_model_id)}">Train the healed model on your world (Step 3)</button>` : ""}</div></div>`;
  if (m.healed) return `<div class="card-in">${head}<p>Healed: <strong>${esc((rs.state.models.find((x) => x.id === m.healed_model_id) || {}).name || m.healed_model_id || "")}</strong> is in the model list of Step 3 (${m.healed_size_gb} GB). ${m.cache_gb ? `The original's saved answers (${m.cache_gb} GB) stay until you delete this model.` : ""}</p>${rs.hl && rs.hl.name === m.name ? hlRoundsTable(rs.hl.opts, run) + hlMoreBox(rs.hl.opts, rs.running || (run && run.running)) : ""}</div>`;
  if (!h) return `<div class="card-in">${head}<p class="dim">Loading…</p></div>`;
  const sp = rs.hlSpec, st = (h.heal && h.heal.status) || null;
  const recipeOn = hlRecipeOn(h), tk = sp.tokens_m || (recipeOn ? h.recipe.tokens_m : h.default_tokens_m);
  const est = rs.hlEst && rs.hlEst.name === m.name ? rs.hlEst.est : null, plan = (est && est.plan) || h.plan || {}, kn = (est && est.knobs) || h.defaults || plan.knobs || {};
  const t = (h.tokens || []).find((x) => x.tokens_m === tk) || {};
  const expert = state.settings.expert_mode;
  gcUse(h.general);
  if (h.recipe && sp.recipe == null && gc.state && !rs.hlSeed[m.name]) {            // the recipe is on by default: its reasoning text is chosen once, unless the person already chose a text for healing
    rs.hlSeed[m.name] = true;
    if (!(gc.state.choice || {}).heal) { const sc = gcScope("heal"); sc.sel = h.recipe.general; sc.project = state && state.project ? state.project.folder : ""; sc.check = null; }
  }
  const blocked = sp.general ? gcBlocked("heal") : "";
  const verdictP = plan.verdict_text ? `<p class="${plan.verdict === "over" ? "err" : plan.verdict === "tight" ? "warn" : "dim"} small" id="hl-verdict">${esc(plan.verdict_text)}${plan.breakdown ? ` <span class="dim">Weights ${plan.breakdown.weights_gb} GB, LoRA and optimizer ${plan.breakdown.lora_gb} GB, activations ${plan.breakdown.activations_gb} GB, loss ${plan.breakdown.kl_gb} GB, the rest ${(plan.breakdown.overhead_gb + (plan.breakdown.total_gb - plan.breakdown.weights_gb - plan.breakdown.lora_gb - plan.breakdown.activations_gb - plan.breakdown.kl_gb - plan.breakdown.overhead_gb)).toFixed(1)} GB.</span>` : ""}</p>` : "";
  const mem = plan.fits === false ? `<div class="err">${esc(plan.why || "Healing does not fit this card.")}</div>${verdictP}`
    : `<p class="dim small">${plan.mode === "online" ? `The original and ${thin} run at the same time` : `The original answers first and saves its answers to disk, then ${thin} learns (they take turns)`}: about ${plan.est_gb} GB of memory${h.vram_gb ? " (this card has " + h.vram_gb + " GB)" : ""}. ${pair ? "The model" : "The thinner model"} is trained as ${plan.student_method === "qlora" ? "QLoRA" : "LoRA"}.</p>${verdictP}`;
  const busy = rs.running || rs.testing || (run && run.running);
  const prog = run && run.running ? `<div class="stepline" aria-live="polite">${esc(HEAL_STEPS[run.step] || "Starting…")}${run.text ? " · " + esc(run.text) : ""}</div>${run.progress != null ? `<progress max="1" value="${run.progress}"></progress>` : ""}` : "";
  const series = healSeries();
  const charts = lineChart(series.kd, `How far ${thin}'s answers are from the original's (KL)`.replace(/^How far the model's/, "How far this model's"), true) + lineChart(series.cal, "Text-prediction error on your data", true) + lineChart(series.hold, "Error on general text the healing never learned from (held out)", true);
  const loss = h.loss && h.loss.original != null && h.loss.thinned != null ? `<p class="dim small">Before healing: error ${(+h.loss.thinned).toFixed(2)} on your data; the original has ${(+h.loss.original).toFixed(2)}. Healing aims to bring the first close to the second.</p>` : pair ? `<p class="dim small">Nothing is recorded about how well this model answered before. The error on your data is measured at the start of healing and shown in the chart.</p>` : "";
  const usedG = h.heal && h.heal.general_used && h.heal.general_used.source ? `<p class="dim small">General text: ${h.heal.general_used.skipped ? "skipped" : "used"} <strong>${esc(h.heal.general_used.source)}</strong>${h.heal.general_used.rows ? ` (${h.heal.general_used.rows} rows${h.heal.general_used.detected ? ", read as " + esc(h.heal.general_used.detected.kind) : ""})` : ""}.</p>` : "";
  const hres = (h.heal && h.heal.result) || {};
  const holdP = hres.holdout_loss_after != null ? `<p class="dim small" id="hl-holdout">Held out (${hres.holdout_rows || "some"} general rows the healing never learned from): error ${hres.holdout_loss_before != null ? (+hres.holdout_loss_before).toFixed(2) + " → " : ""}${(+hres.holdout_loss_after).toFixed(2)}${hres.teacher_holdout_loss != null ? `; the original has ${(+hres.teacher_holdout_loss).toFixed(2)}` : ""}. The error on your own data above is measured on text the healing also learns from, so it reads better than it is.</p>` : "";
  const used = usedG + coverageHtml(h.heal && h.heal.coverage) + holdP;
  const other = sp.general && gc.state && gcSel("heal") !== gc.state.default;
  const note = h.heal && h.heal.note ? `<p class="warn">${esc(h.heal.note)}${other ? ` <button class="secondary" id="hl-usedefault">Use ${esc(gc.state.default)} instead</button>` : ""}</p>${used}` : used;
  const resume = st === "stopped" || st === "failed";
  const oomFailed = st === "failed" && h.heal && h.heal.oom === true;
  const tried = oomFailed && (h.heal.retries || []).length ? `<p class="dim small">Tried automatically: ${esc(h.heal.retries.map((r) => r.change).join(", then "))}.</p>` : "";
  const oomBlock = oomFailed ? `<div class="card-in" id="hl-oom">${tried}${h.safer
    ? `<p>The next safer setting that keeps what the model learns: <strong>${esc(h.safer.label)}</strong>. <button class="primary" id="hl-safer">Try again with safer settings</button></p>`
    : `<p class="warn">The automatic steps are used up. Next, in this order: a shorter sequence length, QLoRA, a lower LoRA size (Expert: Healing settings), a smaller model.</p>`}</div>` : "";
  return `<div class="card-in">${head}
    ${pair ? `<p>The original model teaches the model you chose to answer like it does, on your world's text and on general text. This brings back what cutting the model down took away. Your world's facts are still taught afterwards, in Step 3. The healed copy is a new folder inside this project; the model you chose is only read, never changed.</p>${h.name == null && rs.hs.check && rs.hs.check.download_gb ? `<p class="warn small">Starting downloads about ${rs.hs.check.download_gb} GB from Hugging Face first (into the Hugging Face cache on this computer).</p>` : ""}`
      : `<p>The original model teaches the thinner one to answer like it did, on your world's text and on general text. This is what brings back most of what thinning took away. Your world's facts are still taught afterwards, in Step 3.</p>`}
    ${h.recipe ? `<label class="opt"><input type="checkbox" id="hl-recipe" ${recipeOn ? "checked" : ""} ${busy ? "disabled" : ""}> <strong>Keep thinking</strong> (recommended for reasoning models). ${recipeOn ? `The healing text is ${Math.round(h.recipe.general_share * 100)} % reasoning traces that Qwen3.8-27B itself wrote (${esc(h.recipe.general)}, downloaded from Hugging Face) and the rest your world's text; texts up to ${h.recipe.seq} tokens long, LoRA size ${h.recipe.rank}, ${h.recipe.tokens_m} million tokens${h.recipe.tokens_m > 40 ? " (more, because more than 30 % was cut)" : ""}. It takes longer and needs more memory than the plain recipe. Without this the model learns to answer without thinking, and the exported file's chat template then answers without thinking too.` : `Off: the model is healed on answers without thinking, and the exported file answers without thinking unless asked.`}</label>` : ""}
    <label class="opt"><input type="checkbox" id="hl-general" ${sp.general ? "checked" : ""} ${busy ? "disabled" : ""}> Also use general text (about half of the healing text). <strong>It is downloaded from Hugging Face</strong> (or read from a file you choose) while the original answers; you need an internet connection for that. Without it, or with this box unticked, healing uses your world's text only (general ability comes back less).</label>
    ${sp.general ? gcHtml("heal") : ""}
    <div class="row">${sp.rounds ? "At most" : "Healing text:"} <select id="hl-tokens" ${busy ? "disabled" : ""}>${(h.tokens || []).map((x) => `<option value="${x.tokens_m}" ${x.tokens_m === tk ? "selected" : ""}>${x.tokens_m} million tokens</option>`).join("")}</select>
      <span class="dim small">${t.hours ? `about ${t.hours[0]}–${t.hours[1]} hours (an estimate until this card has been measured) · ` : ""}${t.cache_gb != null && plan.mode === "cache" ? `${t.cache_gb} GB of disk for the original's saved answers` : ""}</span></div>
    ${hlRoundsBox(h, sp, kn, est, busy, recipeOn, tk)}${sp.rounds ? hlRoundsTable(h, run) : ""}
    ${hlVllmBox(sp, est, busy)}
    ${hlSpeedBox(busy)}
    ${mem}${loss}${note}
    ${expert ? `<details><summary>Expert settings</summary><div class="grid">
      <label>General share<input type="text" id="hl-share" value="${sp.share}"></label>
      <label>Weight of copying the original (1 = only copy it)<input type="text" id="hl-alpha" value="${sp.alpha}"></label>
      <label>Learning rate<input type="text" id="hl-lr" value="${sp.lr}"></label>
      <label>Way<select id="hl-mode"><option value="" ${sp.mode ? "" : "selected"}>what fits (${esc(plan.mode)})</option><option value="cache" ${sp.mode === "cache" ? "selected" : ""}>the original answers first</option><option value="online" ${sp.mode === "online" ? "selected" : ""}>both at once</option></select></label></div></details>` : ""}
    ${expert ? healSettings(sp, kn, plan, busy, est && est.upcast_gb != null ? est.upcast_gb : h.upcast_gb, est && est.head_off_gb != null ? est : h) : ""}${oomBlock}
    ${pfBox("heal", { ...(hlVllmAvailable() && sp.vllm ? { vllm_teacher: true } : {}), tokens_m: sp.tokens_m, mode: sp.mode || plan.mode, verdict: plan.verdict, verdict_text: plan.verdict_text, thinks: !pair && !!(rs.options && rs.options.inspect && rs.options.inspect.model_type === "qwen3_5"), name: m.name })}
    <div class="row"><button class="primary" id="hl-go" ${busy || plan.fits === false || !h.has_data || !rs.state.env_ready || blocked || (pair && !hsPassed()) || pfBlocked("heal") ? "disabled" : ""}>${oomFailed ? "Continue with the settings above" : resume ? "Continue healing" : "Start healing"}</button>${!pair ? quBtn("heal") : ""}<button class="secondary" id="hl-stop" ${run && run.running ? "" : "disabled"}>Stop</button></div>
    ${blocked ? `<p class="warn small">${esc(blocked)}</p>` : ""}${pair && !hsPassed() ? `<p class="warn small">The two models have not passed the check (or the choice changed since): press Check above.</p>` : ""}
    ${resume && h.heal ? `<p class="dim small">${oomFailed ? "Healing stopped: " + esc(h.heal.error || "") : st === "failed" ? "Healing stopped with an error: " + esc(h.heal.error || "") : "Healing was stopped"}${!oomFailed && h.heal.step ? " during step " + esc(h.heal.step) : ""}. ${oomFailed ? "The saved answers and the training so far are kept. A changed sequence length, LoRA size or top-k makes the original answer again or starts the learning again; the other settings go on from the last checkpoint." : "Continue goes on where it stopped."}</p>` : ""}
    ${prog}${charts}</div>`;
}


// ------------------------------------------------------------------------------------------ Heal a model (any model, docs/HEAL_ANY_MODEL.md)
// Two pickers (the model to heal and its original), a Check in plain words, then the same Heal card as above, bound to the pair. A made model with its recorded original
// keeps the card of its result (same code, same state `rs.hl`); every other pair is healed from here.
function hsMade() { return rs.state ? rs.state.made.filter((m) => m.status === "done" && m.summary && m.summary.mode === "prune") : []; }
function hsPick() { const m = hsMade()[0]; return rs.hs.studentPick || (m ? "made:" + m.name : ""); }
function hsEntry(pick) {
  const c = rs.hs.cand; if (!c) return null;
  const id = pick.slice(pick.indexOf(":") + 1), list = pick.startsWith("mine:") ? c.models.mine : c.models.builtin;
  return (list || []).find((e) => e.id === id) || null;
}
function hsRun(pick) { const c = rs.hs.cand; return pick.startsWith("run:") && c ? (c.runs || []).find((r) => r.name === pick.slice(4)) || null : null; }
function hsStudent() {
  const h = rs.hs, pick = hsPick();
  if (pick.startsWith("made:")) return { kind: "made", name: pick.slice(5) };
  if (pick.startsWith("mine:") || pick.startsWith("built:")) { const e = hsEntry(pick); return e ? e.source : null; }
  if (pick.startsWith("run:")) { const r = hsRun(pick); return r ? r.student : null; }
  if (pick === "folder") return h.studentFolder.trim() ? { kind: "folder", path: h.studentFolder.trim() } : null;
  if (pick === "hub") return h.studentHub.trim() ? { kind: "hub", id: h.studentHub.trim() } : null;
  return null;
}
// null = the recorded original of a made model
function hsTeacher() {
  const h = rs.hs, pick = h.teacherPick, sp = hsPick();
  if (sp.startsWith("run:")) { const r = hsRun(sp); return r ? r.teacher : null; }
  if (pick === "guess") return h.guess ? { kind: "hub", id: h.guess.repo } : null;
  if (pick.startsWith("mine:") || pick.startsWith("built:")) { const e = hsEntry(pick); return e ? e.source : null; }
  if (pick === "folder") return h.teacherFolder.trim() ? { kind: "folder", path: h.teacherFolder.trim() } : null;
  if (pick === "hub") return h.teacherHub.trim() ? { kind: "hub", id: h.teacherHub.trim() } : null;
  return null;
}
const hsKey = () => JSON.stringify([hsStudent(), hsTeacher()]);
// a made model with its own recorded original: healed from the result card, as before
function hsCardMode() { const s = hsStudent(); return !!s && s.kind === "made" && !hsTeacher(); }
function hsPassed() { const h = rs.hs; return !!(h.check && h.check.ok && h.checkKey === hsKey()); }

async function hsLoadCand() {
  try { rs.hs.cand = await invoke("heal_candidates"); } catch (_) { rs.hs.cand = null; }
  if (onReshape()) render();
}
// the choice changed: the Check no longer belongs to it, and the card that was bound to the old pair goes
function hsChanged() {
  const h = rs.hs;
  h.check = null; h.checkKey = null; h.err = null;
  if (rs.hl && rs.hl.pair) rs.hl = null;
  const s = hsStudent(); if (s && s.kind === "made") rs.view = s.name;
  rsAfterState(); render();
}
async function hsCheck(refresh) {
  const h = rs.hs, st = hsStudent(), te = hsTeacher();
  if (!st) { h.err = "Choose the model to heal first."; render(); return; }
  const key = hsKey();
  h.checking = true; h.err = null; render();
  try {
    if (hsCardMode()) { h.check = await invoke("heal_check", { student: st, teacher: null, refresh }); h.checkKey = key; }
    else {
      const r = await invoke("heal_pair", { student: st, teacher: te, refresh });
      if (key !== hsKey()) { h.checking = false; return; }
      h.check = r ? r.check : null; h.checkKey = key;
      rs.hl = r && r.options ? { name: r.options.name || (st.kind === "made" ? st.name : "@pair"), opts: r.options, pair: { student: st, teacher: te } } : null;
      if (rs.hl) rsHealEstimate();
    }
    if (h.check && h.check.teacher_guess) h.guess = h.check.teacher_guess;
  } catch (e) { h.err = String(e); h.check = null; }
  h.checking = false; render();
}
// after a heal: the folder, the healed model and the charts come from the core again
async function hsReloadPair() {
  const pr = rs.hl && rs.hl.pair; if (!pr) return;
  try {
    const r = await invoke("heal_pair", { student: pr.student, teacher: pr.teacher, refresh: false });
    if (r && r.options && rs.hl && rs.hl.pair) { rs.hl = { name: r.options.name || "@pair", opts: r.options, pair: pr }; rsHealEstimate(); }
  } catch (_) { if (rs.hl) rs.hl.stale = false; }
  hsLoadCand();
}

function hsChecks(c) {
  const icon = { ok: "✓", warn: "⚠", refuse: "✗", note: "•" }, cls = { ok: "hs-ok", warn: "hs-warn", refuse: "hs-no", note: "hs-note" };
  return `<ul class="hs-check" id="hs-check-list">${c.items.map((i) => `<li class="${cls[i.level] || "hs-note"}"><span class="hs-ic">${icon[i.level] || "•"}</span> ${esc(i.text)}</li>`).join("")}</ul>
    <p class="${c.ok ? "ok" : "err"}" id="hs-verdict">${c.ok ? "These two models can be used for healing." : "These two models cannot be used for healing as chosen."}</p>`;
}

function healSection() {
  const h = rs.hs, st = rs.state, c = h.cand, made = hsMade();
  const busy = !!(rs.running || rs.testing || (rs.hlRun && rs.hlRun.running));
  const pick = hsPick(), opt = (v, t, cur) => `<option value="${esc(v)}" ${v === cur ? "selected" : ""}>${esc(t)}</option>`;
  const grp = (label, items) => items.length ? `<optgroup label="${esc(label)}">${items.join("")}</optgroup>` : "";
  const mine = (c && c.models.mine) || [], built = (c && c.models.builtin) || [], runs = (c && c.runs) || [];
  const dis = busy || h.checking ? "disabled" : "";
  const sOpts = (pick ? "" : opt("", "Choose a model…", "")) + grp("Made by this program", made.map((m) => opt("made:" + m.name, m.name + (m.healed ? " (healed)" : ""), pick)))
    + grp("My models", mine.map((e) => opt("mine:" + e.id, e.name, pick))) + grp("Earlier heals of other models", runs.map((r) => opt("run:" + r.name, `${r.student_name} (taught by ${r.teacher_name})${r.healed ? " - healed" : ""}`, pick)))
    + grp("Models that come with the program", built.map((e) => opt("built:" + e.id, e.name, pick)))
    + `<optgroup label="Somewhere else">${opt("folder", "A folder on this computer…", pick)}${opt("hub", "A Hugging Face model…", pick)}</optgroup>`;
  const st0 = hsStudent(), isRun = pick.startsWith("run:");
  const sMore = pick === "folder" ? `<div class="row"><input type="text" id="hs-sfolder" value="${esc(h.studentFolder)}" placeholder="The model's own folder: it holds config.json and the .safetensors files" style="flex:1" ${dis}> <button class="secondary" id="hs-sbrowse" ${dis}>Browse…</button></div>`
    : pick === "hub" ? `<div class="row"><input type="text" id="hs-shub" value="${esc(h.studentHub)}" placeholder="owner/model, for example Qwen/Qwen3-8B" style="flex:1" ${dis}></div>` : "";
  const builtinWarn = pick.startsWith("built:") ? `<p class="hs-warn small">This model ships with the program and has not been changed. Healing it is pointless unless it was cut down or changed first.</p>` : "";
  const madeStudent = st0 && st0.kind === "made";
  const guess = h.guess && !isRun ? opt("guess", `${h.guess.repo} (a guess: ${h.guess.from})`, h.teacherPick) : "";
  const tOpts = opt("", madeStudent ? "The model it was made from (recorded)" : "Choose the original…", h.teacherPick) + guess
    + grp("My models", mine.map((e) => opt("mine:" + e.id, e.name, h.teacherPick))) + grp("Models that come with the program", built.map((e) => opt("built:" + e.id, e.name, h.teacherPick)))
    + `<optgroup label="Somewhere else">${opt("folder", "A folder on this computer…", h.teacherPick)}${opt("hub", "A Hugging Face model…", h.teacherPick)}</optgroup>`;
  const tMore = h.teacherPick === "folder" ? `<div class="row"><input type="text" id="hs-tfolder" value="${esc(h.teacherFolder)}" placeholder="The original's folder" style="flex:1" ${dis}> <button class="secondary" id="hs-tbrowse" ${dis}>Browse…</button></div>`
    : h.teacherPick === "hub" ? `<div class="row"><input type="text" id="hs-thub" value="${esc(h.teacherHub)}" placeholder="owner/model" style="flex:1" ${dis}></div>` : "";
  const guessNote = h.guess && !isRun && !h.teacherPick && !madeStudent ? `<p class="dim small">The model card of the model to heal names <strong>${esc(h.guess.repo)}</strong> as its base (${esc(h.guess.from)}). That is a guess: pick it in the list only if it is the model it was really made from.</p>` : "";
  const tHtml = isRun ? `<p class="dim small">The original of this heal: <strong>${esc((hsRun(pick) || {}).teacher_name || "")}</strong>.</p>`
    : `<label>Original model (the teacher) <select id="hs-teacher" ${dis}>${tOpts}</select></label>${tMore}${guessNote}`;
  const checking = h.checking ? `<p class="dim" id="hs-checking">Checking the two models (their files are read; nothing is loaded or downloaded)…</p>` : "";
  const list = h.check && h.checkKey === hsKey() ? hsChecks(h.check) : "";
  const err = h.err ? errDiv(h.err) : "";
  let body = "";
  if (hsCardMode() && st0 && made.some((m) => m.name === st0.name)) body = `<p class="dim small" id="hs-cardnote">The settings, the estimate and the Start button for this model are in its result card below (<strong>Heal it</strong>). Pick another original above to teach it from a different model.</p>`;
  else if (rs.hl && rs.hl.pair && rs.hl.opts) {
    const o = rs.hl.opts;
    body = healCard({ name: rs.hl.name, healed: !!o.healed, healed_model_id: (o.heal && o.heal.result && o.heal.result.model_id) || null, healed_size_gb: null, cache_gb: null, summary: { mode: "prune", shape: "width" } }, true);
  }
  const del = isRun ? `<div class="row"><button class="link" id="hs-delrun" data-name="${esc(pick.slice(4))}" ${busy ? "disabled" : ""}>Delete this heal and its files</button></div>` : "";
  const noEnv = !st.env_ready ? `<p class="warn small">The training software is not installed yet (Training setup): the check needs it.</p>` : "";
  return `<div class="card wide" id="hs-section"><h2>Heal a model</h2>
    <p>Teach a smaller or cut-down model to answer like the original it came from. It works for the models made on this page <strong>and for any other</strong>: a pruned model downloaded from Hugging Face, or one made with another tool. Nothing is written until you press Start.</p>
    <label>Model to heal <select id="hs-student" ${dis}>${sOpts}</select></label>${sMore}${builtinWarn}${del}
    ${tHtml}
    <div class="row"><button class="primary" id="hs-check" ${!st0 || dis || !st.env_ready ? "disabled" : ""}>Check</button><span class="dim small">Reads the two models' files: size, layers, tokenizer, licence. A Hugging Face model that is not on this computer is not downloaded by this.</span></div>
    ${noEnv}${checking}${err}${list}${body}
    <p class="dim small">The healed copy is a derivative of both models. Check that their licences allow what you plan to do.</p></div>`;
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

// what became of the vision part and the multi-token head (the worker's `width.mtp` says what a width cut did to the head; a depth-only cut leaves it as it was)
function mtpText(wd) {
  const m = wd && wd.mtp ? String(wd.mtp) : "";
  if (m.startsWith("removed")) return "The vision part was copied unchanged. The multi-token head (speculative decoding) was removed: it no longer fits the thinner layers.";
  if (m.includes("sliced")) return "The vision part was copied unchanged. The multi-token head was cut to fit but was not measured or healed; do not use it for speculative decoding.";
  return "The vision part and the multi-token head were copied unchanged.";
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
    ${sm.params_breakdown ? `<p class="dim small" id="rs-made-breakdown">${esc(breakdownLine(sm.params_breakdown))}</p>` : ""}
    ${loss ? `<p>${esc(loss)}</p>` : ""}${stab}
    ${slim && sm.stable === false && !wide ? `<p class="warn">The measurement was not stable: two halves of your text chose different layers. Try 256 samples.</p>` : ""}
    ${(sm.notes || []).map((n) => `<p class="dim small">${esc(n)}</p>`).join("")}
    <p class="dim small">${mtpText(wd)} ${sm.load ? "Measured with the model loaded as " + esc(sm.load) + "." : ""} It is in the model list of Step 3 as “${esc((rs.state.models.find((x) => x.id === m.model_id) || {}).name || m.model_id || "")}”.</p>
    <div class="row"><button class="primary" data-rstest="${esc(m.name)}" ${busy || !rs.state.env_ready || !rs.state.dataset_version ? "disabled" : ""} title="${rs.state.dataset_version ? "" : "Save a training-data version in Step 2 first: the questions come from it."}">Test against the original</button>
      ${m.healed ? `<button class="primary" data-rstest-healed="${esc(m.name)}" ${busy || !rs.state.env_ready || !rs.state.dataset_version ? "disabled" : ""}>Test the healed model</button>` : ""}
      <button class="secondary" data-rstrain="${esc(m.healed ? m.healed_model_id || "" : m.model_id || "")}" ${(m.healed ? m.healed_model_id : m.model_id) ? "" : "disabled"}>Train ${m.healed ? "the healed model" : "it"} on your world (Step 3)</button>
      <button class="secondary" data-stdel="model_entry|${esc(m.name)}" data-rsdelete="${esc(m.name)}" ${busy ? "disabled" : ""}>Delete…</button></div>
    ${rs.testing === m.name ? `<div class="stepline" aria-live="polite">${esc(rs.text)}</div>${rs.progress != null ? `<progress max="1" value="${rs.progress}"></progress>` : ""}` : ""}${test}${sideBySide(m)}${healCard(m)}${thinkCard(m)}${traceCard(m)}</div>`;
}


// ------------------------------------------------------------------------------------------------------------ Auto-cut (one click)
// The steps of this page as one run: measure, try the cuts, cut, heal, check, export, optionally put the files into Unsloth Studio (docs/AUTO_CUT.md). The plan comes from the
// program (auto_plan, no job runs); the run's state is a file that the stepper reads (auto_status), so a stop or a closed program loses nothing.
let ac = { target: null, thinking: null, gate: true, gateN: 2, stages: 2, quick: false,
  sp: { search: null, stage_a: null, last: null, tq: null, check_cut: true, retry_heal: true, back_off: true, share: false, imatrix: true, hr: null, vl: null }, spInit: false, advOpen: false, maxOn: false, maxH: "", keep: false, many: false, notifyStages: true, publish: false, gbOn: false, gb: "", plan: null, planFor: "", planErr: null, planTok: 0,
  prog: null, pin: null, run: null, running: false, text: "", progress: null, err: null, report: null, loaded: false, share: null, shareOpen: false, shareNote: null };
// The speed settings of Auto-cut: the same table as crates/ndforge-core/src/reshape/auto_speed.json (ui/smoke/autocut.test.mjs compares them).
const AC_SPEED_PRESETS = [
  {"id": "thorough", "label": "Thorough (recommended recipe)", "blurb": "The recipe that was tested: two stages, the first on 40 M tokens and the last on the recipe's 40 or 80 M, full thinking checks, a check of every cut, a retry and a back-off when a thinking check fails.", "risk": "No shortcut. The only risk is the cut itself.", "set": {"search_minutes": 30, "stages": 2, "stage_a_tokens_m": null, "last_tokens_m": null, "share_teacher_cache": false, "quick": false, "think_quick": null, "think_gate": true, "think_gate_n": 2, "check_cut": true, "retry_heal": true, "back_off": true, "imatrix": true, "heal_rounds": null}},
  {"id": "balanced", "label": "Balanced", "blurb": "Two stages, a lighter first stage (20 M tokens), the same sequence length in both stages so the original writes its answers only once, quick thinking checks (16 questions) and a quick Try first. The last stage keeps the recipe's tokens.", "risk": "The first stage learns half as much and the quick checks are noisier. Runs that healed below the recipe have failed the thinking checks before; the checks will tell, and a failed check costs a retry.", "set": {"search_minutes": 20, "stages": 2, "stage_a_tokens_m": 20, "last_tokens_m": null, "share_teacher_cache": true, "quick": true, "think_quick": true, "think_gate": true, "think_gate_n": 2, "check_cut": true, "retry_heal": true, "back_off": true, "imatrix": true, "heal_rounds": null}},
  {"id": "fast", "label": "Fast (quick look)", "blurb": "One stage on 20 M tokens, quick checks, no check of the unhealed cut, no retry, no back-off and no importance matrix.", "risk": "Far below the recipe: a model that thinks will very likely lose its reasoning. Use it to see roughly how a cut behaves, not to keep the result.", "set": {"search_minutes": 10, "stages": 1, "stage_a_tokens_m": null, "last_tokens_m": 20, "share_teacher_cache": false, "quick": true, "think_quick": true, "think_gate": true, "think_gate_n": 1, "check_cut": false, "retry_heal": false, "back_off": false, "imatrix": false, "heal_rounds": null}},
  {"id": "early", "label": "Stop early when good (periodic checks, experimental)", "blurb": "Balanced, but every stage heals in rounds of 5 M tokens with a quick thinking check after each round (not before 10 M) and stops at the first round that passes, then checks the merged model. The cap stays the recipe's tokens. Experimental: nobody has measured yet whether healing in rounds ends as good as one long heal.", "risk": "A quick check asks 16 questions, so one or two answers decide pass or fail; the merged model is checked again at the end and decides. Every round costs about 25 minutes of loading and checking, so a run that never passes takes a few hours longer than Balanced. Whether rounds reach the quality of one long heal is not known yet.", "set": {"search_minutes": 20, "stages": 2, "stage_a_tokens_m": null, "last_tokens_m": null, "share_teacher_cache": true, "quick": true, "think_quick": true, "think_gate": true, "think_gate_n": 2, "check_cut": true, "retry_heal": true, "back_off": true, "imatrix": true, "heal_rounds": {"round_m": 5, "check": "quick"}}},
  {"id": "custom", "label": "Custom", "blurb": "Your own choice in the advanced speed settings below.", "risk": "Whatever you set below the recipe is listed in the plan and the report.", "set": {}},
];
const AC_CHIP = { pending: "waiting", running: "running", done: "done", skipped: "skipped", failed: "failed", stopped: "stopped" };
const AC_ICON = { pending: "○", running: "▶", done: "✓", skipped: "–", failed: "✗", stopped: "■" };
const AC_STATUS = { running: "Running", stopped: "Stopped", failed: "Stopped by an error", done: "Finished", gates_failed: "Stopped: the thinking checks failed" };

function acIns() { return rs.options && rs.options.inspect && rs.options.inspect.supported ? rs.options.inspect : null; }
function acDefaults() {
  const ins = acIns(); if (!ins) return;
  const total = ins.total_params / 1e9;
  if (ac.target == null) ac.target = Math.round(total * 0.76 * 2) / 2;
  if (ac.thinking == null) ac.thinking = ins.model_type === "qwen3_5";
  if (!ac.spInit) { ac.spInit = true; acApplyPreset(total >= 15 ? "balanced" : "thorough"); }      // a big model takes days: Balanced is chosen for it, the recommended recipe is one click away
  if (!ac.share) { const sh = state.settings && state.settings.studio_share; ac.share = sh ? { ...sh } : { host_path: "", container_path: "", register_scan_folder: true, finetune_handoff: false }; }
}
// the speed knobs as the program names them (the same fields as AC_SPEED_PRESETS[].set)
function acKnobs() {
  const sp = ac.sp;
  return { stages: ac.stages, stage_a_tokens_m: sp.stage_a, last_tokens_m: sp.last, share_teacher_cache: !!sp.share, quick: !!ac.quick, think_quick: sp.tq == null ? !!ac.quick : !!sp.tq, think_gate: !!ac.gate, think_gate_n: ac.gateN,
    check_cut: !!sp.check_cut, retry_heal: !!sp.retry_heal, back_off: !!sp.back_off, imatrix: !!sp.imatrix, heal_rounds: sp.hr || null };
}
const AC_THINK_KNOBS = ["think_quick", "think_gate", "think_gate_n", "check_cut", "retry_heal"];
// the preset whose values the knobs have; "custom" when they are a mix (the thinking knobs do not count for a model that does not think)
function acPresetNow() {
  if (ac.sp.vl && (ac.sp.vl.teacher_cache || ac.sp.vl.checks || ac.sp.vl.automatic)) return "custom";
  const kn = acKnobs();
  const hit = AC_SPEED_PRESETS.find((p) => p.id !== "custom" && Object.entries(p.set).every(([k, v]) => {
    if (k === "search_minutes" || (!ac.thinking && AC_THINK_KNOBS.includes(k))) return true;      // how long the search looks is not part of what makes a recipe
    if (k === "heal_rounds") return JSON.stringify(kn[k]) === JSON.stringify(v);        // an object: compared by its content
    return kn[k] === (k === "think_quick" && v === null ? p.set.quick : v);
  }));
  return hit ? hit.id : "custom";
}
function acApplyPreset(id) {
  const p = AC_SPEED_PRESETS.find((x) => x.id === id);
  if (!p) return;
  acApplySet(p.set);
  if (id !== "custom") ac.sp.vl = null;          // a preset is a recipe without the experimental engine (the program does the same)
}
// a choice of a knob (or a preset) sets these spec fields
function acApplySet(set) {
  for (const [k, v] of Object.entries(set)) {
    if (k === "stages") ac.stages = v; else if (k === "quick") ac.quick = !!v; else if (k === "think_gate") ac.gate = !!v; else if (k === "think_gate_n") ac.gateN = v;
    else if (k === "stage_a_tokens_m") ac.sp.stage_a = v; else if (k === "last_tokens_m") ac.sp.last = v; else if (k === "think_quick") ac.sp.tq = v;
    else if (k === "search_minutes") ac.sp.search = v;
    else if (k === "heal_rounds") ac.sp.hr = v ? { ...v } : null;
    else if (k === "vllm") ac.sp.vl = v ? { ...v } : null;
    else if (k === "share_teacher_cache") ac.sp.share = !!v; else if (k in ac.sp) ac.sp[k] = !!v;
  }
}
const AC_SPEED_KEYS = ["student_batch_tokens", "kl_chunk", "head_checkpoint", "checkpointing", "teacher_batch_tokens"];
const AC_QUALITY_KEYS = ["rank", "optimizer", "warmup", "lr"];
const AC_PROGRAM_KEYS = ["max_num_seqs", "vllm_batch_tokens", "gpu_util_pct", "llama_ngl", "try_calib_batch", "think_batch"];
// what a snapshot drives in an Auto-cut run, what it holds but an Auto run never takes (the quality knobs of a short trial are for the Heal card), and the program settings that are saved from the Tune page
function acTunedSplit(sn) {
  const v = (sn && sn.values) || {}, chosen = (sn && sn.chosen_keys) || [];
  const set = (k) => v[k] !== undefined && v[k] !== null && v[k] !== 0 && v[k] !== "on" && v[k] !== "auto";
  const drives = AC_SPEED_KEYS.filter((k) => set(k) && (k !== "teacher_batch_tokens" || chosen.includes(k)));
  const quality = AC_QUALITY_KEYS.filter((k) => chosen.includes(k) && v[k] !== undefined);
  const program = AC_PROGRAM_KEYS.filter((k) => chosen.includes(k));
  return { drives, quality, program, v };
}
function acTunedHtml(lock) {
  if (ac.tunedList === undefined) { ac.tunedList = null; Promise.resolve(invoke("tune_presets_here")).then((l) => { ac.tunedList = l || []; try { render(); } catch (_) { /* a preview */ } }).catch(() => { ac.tunedList = []; }); }
  const list = ac.tunedList || [];
  const sn = ac.tunedSnap;
  const sp = acTunedSplit(sn), name = (k) => k.replace(/_/g, " ");
  const gains = (p) => (p.gains || []).filter((g) => g.gain > 1.0).map((g) => `${g.function.replace(/^.*\./, "")} ×${(+g.gain).toFixed(2)}`).join(", ");
  const model = acTuneModel(), off = tnBusy() || !model, why = tnBusy() ? jobWhy() : "there is no finished model to test yet (run Auto once or make a cut)";
  return `<div class="row"><button class="primary" id="ac-tune-auto" ${off ? "disabled" : ""} title="${off ? esc(`Off because ${why}`) : `Tests this PC for ${esc(model)} (the Tune page's one click), then shows what to change and saves a preset`}">Auto: test and choose the best settings${model ? ` for ${esc(model)}` : ""}</button>
      <button class="link" id="ac-tune-go" title="Open the Tune page: the checklist, the results and every preset of a model">Open the Tune page…</button></div>
    ${off ? `<p class="dim small" id="ac-tune-why">The test button is off because ${esc(why)}.</p>` : ""}
    <label class="opt" id="ac-tuned-row">Use tested settings for this PC <select id="ac-tuned" ${lock}><option value="">none (the heals as they always were)</option>${list.map((p) => `<option value="${esc(p.id)}|${esc(p.scope)}" ${sn && sn.preset_id === p.id ? "selected" : ""}>${esc(p.name)} · ${esc(p.model)}${gains(p) ? ` · ${esc(gains(p))}` : ""}</option>`).join("")}</select></label>
    ${sn ? `<div class="small dim" id="ac-tuned-note"><p>The heals of this run use: ${esc(sp.drives.map((k) => `${name(k)} ${sp.v[k]}`).join(", ") || "your own settings (nothing differs)")}${sn.tokens_per_s ? `; the time estimate uses the measured ${Math.round(sn.tokens_per_s)} tokens/s of the student` : ""}. A snapshot: later edits of the preset do not change this run.</p>
      ${sp.quality.length ? `<p id="ac-tuned-quality">Not taken by Auto-cut: ${esc(sp.quality.map((k) => `${name(k)} ${sp.v[k]}`).join(", "))}. These came from short trial heals, not a full heal; use them from the Heal card if you want them.</p>` : ""}
      ${sp.program.length ? `<p id="ac-tuned-program">Program settings in this preset (${esc(sp.program.map(name).join(", "))}) are saved from the Tune page's Apply, not from here.</p>` : ""}</div>` : ""}`;
}
function acSpec() {
  const sp = ac.sp;
  return { model_id: rs.model, target_b: +ac.target, keep_thinking: !!ac.thinking, think_gate: !!ac.thinking && !!ac.gate, think_gate_n: ac.gateN, stages: ac.stages, quick: ac.quick, keep_intermediate: ac.keep, allow_many_blocks: ac.many, quants: ["q4_k_m"], studio_publish: ac.publish,
    ...(ac.gbOn && +ac.gb > 0 ? { target_gb: +ac.gb } : {}), preset: acPresetNow(), stage_a_tokens_m: sp.stage_a, last_tokens_m: sp.last, think_quick: sp.tq, check_cut: !!sp.check_cut, retry_heal: !!sp.retry_heal, back_off: !!sp.back_off,
    share_teacher_cache: !!sp.share, imatrix: !!sp.imatrix, ...(ac.tunedSnap ? { tuned: ac.tunedSnap } : {}), ...(sp.sbt ? { student_batch_tokens: +sp.sbt } : {}), ...(sp.hr ? { heal_rounds: sp.hr } : {}), ...(vllmSettings().enabled && sp.vl && (sp.vl.teacher_cache || sp.vl.checks || sp.vl.automatic) ? { vllm: { teacher_cache: !!sp.vl.teacher_cache, checks: !!sp.vl.checks, automatic: !!sp.vl.automatic, ...(sp.vl.kv_fp8 ? { kv_fp8: true } : {}) } } : {}), notify_stages: ac.notifyStages !== false, ...(sp.search ? { search_minutes: +sp.search } : {}), ...(ac.pin ? { cut: ac.pin } : {}), ...(ac.maxOn && +ac.maxH >= 1 ? { max_hours: +ac.maxH } : {}) };
}
async function acLoad() {
  try { const r = (await invoke("auto_status")) || {}; ac.run = r.run || null; ac.running = !!r.running; ac.loaded = true; } catch (e) { ac.err = String(e); }
  if (onReshape()) render();
}
// keeps the plan in step with the choices; asks the program only when the request changed
function acRefresh() {
  if (!onAutoPage() || !rs.state) return;
  if (!ac.loaded) { ac.loaded = true; acLoad(); }
  const ins = acIns(); acDefaults();
  if (!ins || ac.target == null) return;
  const sig = JSON.stringify([acSpec(), rs.state.dataset_version]);
  if (sig === ac.planFor) return;
  ac.planFor = sig; ac.plan = null; ac.planErr = null;
  const token = ++ac.planTok;
  invoke("auto_plan", { spec: acSpec(), inspect: ins }).then((r) => { if (token === ac.planTok) { ac.plan = r.plan; acRememberSpeed(r.plan); if (onReshape()) render(); } })
    .catch((e) => { if (token === ac.planTok) { ac.planErr = String(e); if (onReshape()) render(); } });
}

// a change of the total in hours: "−6 h to −11 h" (minutes in)
function acDelta(d) {
  if (!d) return "";
  const [lo, hi] = [Math.min(d[0], d[1]), Math.max(d[0], d[1])];
  if (Math.abs(lo) < 3 && Math.abs(hi) < 3) return "no change";
  const sign = hi < 0 ? "−" : lo > 0 ? "+" : "±", a = acDuration(Math.abs(hi < 0 ? hi : lo) / 60), b = acDuration(Math.max(Math.abs(lo), Math.abs(hi)) / 60);
  return a === b ? `${sign}${a}` : `${sign}${a} to ${sign}${b}`;
}
const acTotal = (t) => t ? `${acStepMin(t[0] * 60, t[1] * 60)}` : "";
// the speed settings: the presets with their totals, the biggest levers with the change each choice makes, the rest under "More speed settings", and the time limit
// The table of speed settings of the last plan that came, kept while the next one is on its way (`acRefresh` drops `ac.plan` at every change). The radios show the LOCAL state
// (`acOptionOn`) and a click uses the `set` of this table, so neither depends on a plan being there at that moment.
function acRememberSpeed(pl) {
  if (pl && pl.speed && pl.speed.knobs && pl.speed.knobs.length) ac.speedCache = { model: rs.model, speed: pl.speed };
}
function acSpeedTable(pl) {
  if (pl && pl.speed) return { sp: pl.speed, stale: false };
  const c = ac.speedCache;
  return c && c.model === rs.model ? { sp: c.speed, stale: true } : { sp: null, stale: false };
}
// does this choice of a knob match what the local state has now? (the same comparison as `acPresetNow`)
function acOptionOn(o) {
  const kn = acKnobs();
  return Object.entries(o.set || {}).every(([k, v]) => {
    if (k === "search_minutes") return (ac.sp.search || null) === (v || null);
    if (k === "think_quick" && v === null) return ac.sp.tq == null;
    if (k === "heal_rounds") return JSON.stringify(kn[k]) === JSON.stringify(v);
    return k in kn ? kn[k] === v : true;
  });
}
function acSpeedBlock(pl, live) {
  const tbl = acSpeedTable(pl), lock = live ? "disabled" : "", sp = tbl.sp, now = acPresetNow(), think = !!ac.thinking;
  const row = (p) => {
    const t = sp && sp.presets.find((x) => x.id === p.id), tt = p.id === "custom" ? sp && sp.total_h : t && t.total_h;
    const risk = p.id === "fast" && think ? `<div class="err" data-acspeed-warn="fast">${esc(p.risk)} A model that thinks will very likely lose its reasoning with these settings.</div>` : `<div class="dim small">${esc(p.risk)}</div>`;
    return `<label class="opt ac-preset" data-acpreset="${esc(p.id)}"><input type="radio" name="ac-preset" value="${esc(p.id)}" ${now === p.id ? "checked" : ""} ${lock}> <strong>${esc(p.label)}</strong>${tt ? ` <span class="ac-ptotal">about ${esc(acTotal(tt))}</span>` : ""}
      <div class="dim small">${esc(p.blurb)}</div>${now === p.id || p.id === "fast" ? risk : ""}${t && t.retry_h && p.id !== "custom" ? `<div class="dim small">If a thinking check fails, a retry adds ${esc(acTotal(t.retry_h))} (not counted).</div>` : ""}</label>`;
  };
  const optRow = (k) => `<tr data-acknob="${esc(k.knob)}"><td>${esc(k.label)}<div class="dim small">${esc(k.risk)}</div></td><td>${k.options.map((o, i) => `<label class="opt ac-choice"><input type="radio" name="ac-k-${esc(k.knob)}" data-acknob="${esc(k.knob)}" data-acopt="${i}" ${acOptionOn(o) ? "checked" : ""} ${lock || (o.total_h ? "" : "disabled")}> ${esc(o.label)}
      <span class="dim small ac-delta">${acOptionOn(o) ? "now" : esc(acDelta(o.delta_min))}</span>${o.note ? `<div class="dim small">${esc(o.note)}</div>` : ""}</label>`).join("")}</td></tr>`;
  const knobs = sp ? sp.knobs.filter((k) => !["stages", "quick", "vllm"].includes(k.knob)) : [];
  const big = knobs.filter((k) => ["last_tokens_m", "stage_a_tokens_m", "share_teacher_cache"].includes(k.knob)), rest = knobs.filter((k) => !big.includes(k));
  const bud = pl && pl.budget;
  const budList = bud ? (bud.dropped.length ? `<div id="ac-budget-list"><p class="small">To fit ${esc(String(bud.max_hours))} hours these settings are given up (the estimate goes from ${esc(acTotal(bud.asked_h))} to ${esc(acTotal(bud.total_h))}):</p>
      <ul>${bud.dropped.map((d) => `<li data-acdropped="${esc(d.knob)}"><strong>${esc(d.label)}</strong>: ${esc(d.to)} instead of ${esc(d.from)} (saves ${esc(acTotal(d.saves_h))}). <span class="dim small">${esc(d.risk)}</span></li>`).join("")}</ul></div>` : `<p class="dim small" id="ac-budget-ok">Your settings already fit.</p>`) : "";
  return `<div class="ac-speed" id="ac-speed"><h3>How careful, how fast${tbl.stale ? ` <span class="dim small" id="ac-updating">updating…</span>` : ""}</h3>
    <p class="dim small">Healing is most of the time. The recommended recipe is the safest; the others trade quality risk for hours, and every value that is not the recipe's is listed in the plan and in the report.</p>
    ${AC_SPEED_PRESETS.map(row).join("")}
    ${sp ? `<div class="ac-levers" id="ac-levers"><h4>What takes the time</h4><table class="tbl">${big.map(optRow).join("")}</table></div>` : ""}
    ${rest.length ? `<details id="ac-adv" ${ac.advOpen ? "open" : ""}><summary>More speed settings</summary><table class="tbl">${rest.map(optRow).join("")}</table></details>` : ""}
    ${ac.pin ? "" : `<p class="small" id="ac-searchrow">Find the best cut for <select id="ac-search" ${lock}>${[["", "no search (the classic Try first on a few cuts)"], [10, "10 minutes"], [20, "20 minutes"], [30, "30 minutes"], [60, "60 minutes"]].map(([v, l]) => `<option value="${v}" ${String(ac.sp.search == null ? "" : ac.sp.search) === String(v) ? "selected" : ""}>${l}</option>`).join("")}</select>
      <span class="dim small">It measures many cuts of the size you asked for in one go, ranks them on your own text and passes the best few on to the thinking check before healing. A low error is not proof of thinking.</span></p>`}
    <label class="opt"><input type="checkbox" id="ac-maxon" ${ac.maxOn ? "checked" : ""} ${lock}> I have at most <input type="text" id="ac-maxh" value="${esc(ac.maxH)}" placeholder="48" style="width:60px;flex:0" ${lock}> hours <span class="dim small">the least harmful settings are given up until it fits, or you are told it cannot</span></label>
    ${ac.maxOn && !(+ac.maxH >= 1) ? `<p class="warn" id="ac-maxwhy">Give the hours (1 or more).</p>` : ""}${ac.maxOn ? budList : ""}
    ${acTunedHtml(lock)}
    <label class="opt">Tokens per forward pass of the student <input type="text" id="ac-sbt" value="${esc(ac.sp.sbt || "")}" placeholder="0" style="width:70px;flex:0" ${lock}> <span class="dim small">off by default (0 = one row at a time). Rows of similar length share a forward pass, up to this many tokens: the same learning, usually much faster. Measure heal speed on the Heal card recommends a value (512 to 65,536).</span></label></div>`;
}

// the change one choice of a knob makes, as a small note after its label (nothing while there is no plan or for the choice that is on)
function acKnobDelta(pl, knob, labelStart) {
  const k = pl && pl.speed && pl.speed.knobs.find((x) => x.knob === knob), o = k && k.options.find((x) => x.label.startsWith(labelStart));
  return o && !o.current && o.delta_min ? ` <span class="dim small ac-delta">${esc(acDelta(o.delta_min))}</span>` : "";
}

// vLLM (experimental) in Auto-cut: two switches and "automatic", each with the change in time from YOUR measured numbers (the plan's own words), the prerequisites, and the decision of every step
function acVllmBox(pl, lock) {
  if (!vllmSettings().enabled) return "";
  const v = ac.sp.vl || {}, rep = vl.report, mi = rep && rep.model_info;
  const word = { pass: "Pass", partial: "Partly", fail: "Fail", stopped: "Stopped" };
  const td = (label, on) => `<li class="${on ? "ok" : "warn"}">${on ? "✓" : "!"} ${label}</li>`;
  const pre = [
    td(rep ? `Self-test run: ${esc(word[rep.verdict] || rep.verdict)}${mi ? ` — measured on ${esc(mi.model)}${mi.params_b != null ? ` (${mi.params_b} B)` : ""}, ${esc(mi.quant || "")}; the plan scales its numbers by model size or does not use them when the size is more than 2 times different` : " (an older report: it does not say which model, so its numbers are not used)"}` : "The vLLM self-test has not been run: every vLLM time below is the research report's estimate", !!rep && !!mi),
    td(rep && rep.long_tokens_per_s != null ? `Speed on long answers measured: ${rep.long_tokens_per_s} tokens per second` : "Speed on long (thinking-length) answers not measured: the Quick suite's short answers overstate it and are not used", !!rep && rep.long_tokens_per_s != null),
    td("The check of the vLLM teacher (Heal card, \"Check the vLLM teacher\") is run for each made model: a teacher the check calls \"different\" is refused and the heal falls back to the standard teacher", false),
  ].join("");
  const dec = pl && pl.time && pl.time.vllm ? pl.time.vllm : null;
  const reasons = dec ? `<ul class="small" id="ac-vllm-decisions">${dec.decisions.filter((d, i, a) => a.findIndex((x) => x.kind === d.kind && x.reason === d.reason) === i && d.reason).map((d) => `<li>${esc(d.reason)}</li>`).join("")}</ul><ul class="small dim">${dec.notes.map((n) => `<li>${esc(n)}</li>`).join("")}</ul>` : "";
  const delta = (l) => acKnobDelta(pl, "vllm", l);
  return `<fieldset class="soft" id="ac-vllm"><legend>vLLM (experimental)</legend>
    <p class="warn small">Built from a research report. It starts a container for every use: it pays for hours of the original's answers, and it does not pay for a short thinking check. A step that cannot use it goes on with the standard engine and the report says so.</p>
    <label class="opt"><input type="checkbox" id="ac-vl-t" ${v.teacher_cache && !v.automatic ? "checked" : ""} ${lock}> The original's answers with vLLM${delta("The original's answers")}</label>
    <label class="opt"><input type="checkbox" id="ac-vl-c" ${v.checks && !v.automatic ? "checked" : ""} ${lock}> The thinking checks with vLLM${delta("The thinking checks")}</label>
    <label class="opt"><input type="checkbox" id="ac-vl-a" ${v.automatic ? "checked" : ""} ${lock}> Automatic: vLLM only for a step where its estimate is lower than the standard engine's (the plan says why)${delta("Automatic")}</label>
    <label class="opt"><input type="checkbox" id="ac-vl-kv" ${v.kv_fp8 ? "checked" : ""} ${lock}> 8-bit KV cache for the thinking checks (experimental; never for the original's answers): more answers at once, usually faster, a tiny numeric change; the checks keep their own results</label>
    <ul class="small" id="ac-vllm-pre">${pre}</ul>${reasons}</fieldset>`;
}
// "Healing in 5 M rounds: about A-B h if the first check passes, up to C-D h." for every stage, and the warnings once (the plan's own words)
function acRoundsPlan(pl) {
  if (!pl || !pl.time || !ac.sp.hr) return "";
  const rows = (pl.time.steps || []).filter((x) => x.rounds && /\.heal$/.test(x.id));
  if (!rows.length) return "";
  const h = (m) => (m / 60).toFixed(0);
  const lines = rows.map((x) => { const r = x.rounds; return `<li data-acrounds="${esc(x.id)}">${rows.length > 1 ? `Stage ${esc(x.id.slice(1, x.id.indexOf(".")))}: ` : ""}Healing in ${r.round_m} M rounds: about ${h(r.best[0])}-${h(r.best[1])} h if the first check passes, up to ${h(r.worst[0])}-${h(r.worst[1])} h.</li>`; }).join("");
  const warn = (id, t) => `<div class="warn small" id="${id}">${esc(t)}</div>`;
  return `<div id="ac-rounds-plan"><ul class="small">${lines}</ul>${ac.thinking ? warn("ac-quick-noise", HL_QUICK_NOISE) + warn("ac-adapter-note", HL_ADAPTER_NOTE) : ""}${warn("ac-rounds-experimental", HL_EXPERIMENTAL)}</div>`;
}
// the chips of the rounds of the heal that runs: done, in work, to come
function acRoundChips(st) {
  const r = st && st.rounds; if (!r || !r.max) return "";
  const chips = Array.from({ length: r.max }, (_, i) => { const state = r.stopped || i < r.n - 1 ? "done" : i === r.n - 1 ? "now" : "next"; return `<span class="tag ac-rchip ac-rchip-${state}" title="Round ${i + 1}">${i + 1}</span>`; }).join(" ");
  return `<div class="ac-rchips small" id="ac-rchips">${chips}</div>`;
}

// "Change settings and continue" for a stopped or failed run: the same knobs as the start screen, locked ones greyed with the reason, the diff, the time left, a confirmation for what would be redone
const acc = { open: false, patch: {}, pv: null, busy: false, err: null, confirmed: false, id: null };
const ACC_ROWS = [
  ["last_tokens_m", "Heal tokens of the last stage (million)", "num", [null, 10, 20, 40, 80]], ["stage_a_tokens_m", "Heal tokens of the first stage (million)", "num", [null, 10, 20, 40, 80]],
  ["think_quick", "Quick thinking checks (16 questions)", "bool"], ["check_cut", "Check the unhealed cut", "bool"], ["retry_heal", "Heal again when a check fails", "bool"], ["back_off", "Cut one step less after a second failed check", "bool"],
  ["heal_rounds", "Heal in rounds of 5 M with a quick check after each (experimental)", "rounds"], ["share_teacher_cache", "Same sequence length in both stages", "bool"], ["imatrix", "Importance matrix for the export", "bool"],
  ["vllm.teacher_cache", "vLLM for the original's answers (experimental)", "vl"], ["vllm.checks", "vLLM for the thinking checks (experimental)", "vl"], ["vllm.kv_fp8", "8-bit KV cache for the checks (experimental)", "vl"],
];
const accCur = (k) => { const c = (acc.pv && acc.pv.current) || {}, [a, b] = k.split("."); return b ? ((c[a] || {})[b] ?? false) : c[a]; };
const accVal = (k) => { const [a, b] = k.split("."); if (b) { const cur = acc.patch[a] !== undefined ? acc.patch[a] : ((acc.pv && acc.pv.current) || {})[a]; return !!(cur || {})[b]; } return acc.patch[a] !== undefined ? acc.patch[a] : accCur(k); };
async function accPreview() {
  acc.busy = true; render();
  try { acc.pv = await invoke("auto_change_preview", { id: acc.id, patch: acc.patch }); acc.err = null; } catch (e) { acc.err = String(e); }
  acc.busy = false; render();
}
function accSet(k, v) {
  const [a, b] = k.split(".");
  if (b) { const cur = { ...(acc.patch[a] !== undefined ? acc.patch[a] : (((acc.pv || {}).current || {})[a] || {})) }; cur[b] = v; if (cur.kv_fp8 === false) delete cur.kv_fp8; acc.patch[a] = cur.teacher_cache || cur.checks || cur.kv_fp8 || cur.automatic ? { teacher_cache: !!cur.teacher_cache, checks: !!cur.checks, automatic: !!cur.automatic, ...(cur.kv_fp8 ? { kv_fp8: true } : {}) } : null; }
  else acc.patch[a] = v;
  acc.confirmed = false; accPreview();
}
function accPanel() {
  const pv = acc.pv; if (!acc.open) return "";
  const lock = (k) => ((pv && pv.locks) || []).find((l) => l.key === k.split(".")[0]) || { state: "free", reason: "" };
  const rows = ACC_ROWS.map(([k, label, kind, opts]) => {
    const l = lock(k), off = l.state === "locked" || acc.busy, v = accVal(k);
    const input = kind === "num" ? `<select data-acc="${k}" ${off ? "disabled" : ""}>${opts.map((o) => `<option value="${o ?? ""}" ${(v ?? null) === o ? "selected" : ""}>${o == null ? "the recipe's" : o}</option>`).join("")}</select>`
      : kind === "rounds" ? `<input type="checkbox" data-acc="${k}" ${v ? "checked" : ""} ${off ? "disabled" : ""}>` : `<input type="checkbox" data-acc="${k}" ${v ? "checked" : ""} ${off ? "disabled" : ""}>`;
    return `<li class="${l.state === "locked" ? "dim" : ""}" data-acc-row="${esc(k)}"><label>${input} ${esc(label)}</label> ${l.state === "locked" ? `<span class="small">🔒 ${esc(l.reason)}</span>` : l.state === "confirm" ? `<span class="small warn">needs your confirmation: ${esc(l.redo || l.reason)}</span>` : ""}</li>`;
  }).join("");
  const sug = pv && pv.suggestions && pv.suggestions.length ? `<p class="small"><strong>Suggestions for the step that stopped${pv.failed_step ? ` (${esc(pv.failed_step)})` : ""}:</strong></p><ul class="small" id="acc-suggest">${pv.suggestions.map((x, i) => `<li>${esc(x.text)} <button class="secondary" data-acc-sug="${i}">Use</button></li>`).join("")}</ul>` : "";
  const diff = pv && pv.diff && pv.diff.length ? `<ul id="acc-diff">${pv.diff.map((d) => `<li>${esc(d.text)}</li>`).join("")}</ul>` : `<p class="dim small" id="acc-diff">Nothing changed yet.</p>`;
  const rem = pv && pv.remaining_h && pv.was_remaining_h ? `<p id="acc-remaining">Time left: about ${pv.remaining_h[0]}-${pv.remaining_h[1]} h (was ${pv.was_remaining_h[0]}-${pv.was_remaining_h[1]} h).</p>` : "";
  const refused = pv && pv.refused && pv.refused.length ? pv.refused.map((r) => `<div class="err">${esc(r.key)}: ${esc(r.reason)}</div>`).join("") : "";
  const conf = pv && pv.confirm && pv.confirm.length ? `<div class="warn" id="acc-confirm"><label class="opt"><input type="checkbox" id="acc-confirmed" ${acc.confirmed ? "checked" : ""}> I understand that this redoes: ${pv.confirm.map((c) => esc(c.redo || c.reason)).join("; ")}.</label></div>` : "";
  const can = pv && pv.ok && !acc.busy && Object.keys(acc.patch).length && (!(pv.confirm || []).length || acc.confirmed);
  return `<div class="card-in" id="acc-panel"><h3>Change settings and continue</h3>
    <p class="dim small">Only settings that have not taken effect yet can change. The steps that are done are not run again, and the changes are kept in the run's report.</p>
    ${sug}<ul class="plain">${rows}</ul>${diff}${rem}${refused}${conf}${acc.err ? `<div class="err">${esc(acc.err)}</div>` : ""}
    <div class="row"><button class="primary" id="acc-go" ${can ? "" : "disabled"}>Continue with these settings</button><button class="secondary" id="acc-close">Cancel</button></div></div>`;
}
function wireAcc() {
  const op = $("ac-change"); if (op) op.onclick = () => { acc.open = !acc.open; acc.id = ac.run && ac.run.id; acc.patch = {}; acc.pv = null; acc.confirmed = false; if (acc.open) accPreview(); else render(); };
  document.querySelectorAll("[data-acc]").forEach((e) => e.onchange = () => { const k = e.dataset.acc; if (e.tagName === "SELECT") accSet(k, e.value === "" ? null : +e.value); else if (k === "heal_rounds") accSet(k, e.checked ? { round_m: 5, check: "quick" } : null); else accSet(k, e.checked); });
  document.querySelectorAll("[data-acc-sug]").forEach((b) => b.onclick = () => { const x = acc.pv.suggestions[+b.dataset.accSug]; Object.assign(acc.patch, x.patch); acc.confirmed = false; accPreview(); });
  const cf = $("acc-confirmed"); if (cf) cf.onchange = () => { acc.confirmed = cf.checked; render(); };
  const cl = $("acc-close"); if (cl) cl.onclick = () => { acc.open = false; render(); };
  const go = $("acc-go"); if (go) go.onclick = async () => {
    ac.err = null; ac.report = null; ac.running = true; ac.text = "Continuing with the changed settings…"; acc.open = false; render();
    try { await invoke("auto_start", { spec: null, resume: acc.id, changes: acc.patch, confirmed: acc.confirmed }); } catch (e) { ac.running = false; ac.err = String(e); render(); }
  };
}
function autoCard() {
  const ins = acIns();
  if (!rs.state) return "";
  acDefaults();
  const run = ac.run, live = ac.running || (run && run.status === "running");
  const total = ins ? ins.total_params / 1e9 : 0;
  const pl = ac.plan;
  const canThink = !!ins && ins.model_type === "qwen3_5";
  const lock = live ? "disabled" : "";
  const sizes = pl && pl.stages && pl.stages.length ? `<table class="tbl"><tr><th>Stage</th><th>Model</th><th>Cut</th><th>Heal</th><th>Card</th></tr>${pl.stages.map((g) => `<tr><td>${g.index} of ${g.of}</td><td>${g.params_b.toFixed(1)} B${g.blocks_cut ? `, ${g.blocks_cut} block${g.blocks_cut === 1 ? "" : "s"} out` : ""}${g.width ? `, MLP ${thou(g.width)}` : ""}</td><td>${g.cut_pct} % of the original</td><td>${g.tokens_m} M tokens, about ${g.hours[0]}–${g.hours[1]} h</td><td>${g.heal_gb} GB (${esc(g.verdict)})</td></tr>`).join("")}</table>
    ${pl.disk ? `<p class="dim small">Disk: up to ${Math.round(pl.disk.peak_gb)} GB at "${esc(pl.disk.peak_step)}", ${Math.round(pl.disk.final_gb)} GB left at the end${pl.disk.free_gb != null ? ` (${Math.round(pl.disk.free_gb)} GB free)` : ""}.</p>` : ""}` : "";
  const notes = pl ? pl.blockers.map((b) => `<div class="err">${esc(b)}</div>`).join("") + pl.warnings.map((w) => `<div class="warn">${esc(w)}</div>`).join("") + acRoundsPlan(pl) : "";
  const steps = run && run.steps ? `<ol class="stepper" id="ac-steps">${run.steps.map((x) => `<li class="ac-${esc(x.status)}" data-acstep="${esc(x.id)}"><span>${AC_ICON[x.status] || "○"}</span> ${esc(x.label)} <span class="tag ac-chip ac-chip-${esc(x.status)}">${esc(AC_CHIP[x.status] || x.status)}</span>${acStepFig(x) ? ` <span class="dim small ac-fig">${esc(acStepFig(x))}</span>` : ""}${x.note ? ` <span class="dim small">${esc(x.note)}</span>` : ""}</li>`).join("")}</ol>` : "";
  const status = run ? `<p><strong id="ac-status">${esc(AC_STATUS[run.status] || run.status)}</strong>${run.error ? `: ${esc(run.error)}` : ""} <span class="dim small">${esc(run.id)}</span></p>` : "";
  const ended = run && ["done", "gates_failed", "failed"].includes(run.status);
  const sh = ac.share || {};
  const studio = `<label class="opt"><input type="checkbox" id="ac-publish" ${ac.publish ? "checked" : ""} ${lock}> Put the GGUF files into Unsloth Studio's share when done</label>
    ${ac.publish || ac.shareOpen ? `<div class="grid" id="ac-share">
      <label>Studio share (the folder as this computer sees it)<input type="text" id="ac-sh-host" value="${esc(sh.host_path || "")}" placeholder="Y:\\unsloth\\studio"></label>
      <label>Path inside the container<input type="text" id="ac-sh-cont" value="${esc(sh.container_path || "")}" placeholder="/workspace (empty: found out from Studio)"></label>
      <label><input type="checkbox" id="ac-sh-reg" ${sh.register_scan_folder ? "checked" : ""}> Tell Studio about the exports folder</label>
      <label><input type="checkbox" id="ac-sh-hf" ${sh.finetune_handoff ? "checked" : ""}> Also copy the healed model for fine-tuning in Studio (large)</label>
      <div class="row"><button class="secondary" id="ac-sh-save">Save</button><button class="secondary" id="ac-sh-check">Check</button></div>
      ${ac.shareNote ? `<p class="dim small" id="ac-sh-note">${esc(ac.shareNote)}</p>` : ""}</div>` : ""}`;
  const rep = ac.report ? `<details open id="ac-report"><summary>Report</summary><pre class="log">${esc(ac.report)}</pre></details>` : "";
  return `<div class="card wide" id="ac-card"><h2>Auto-cut (one click)</h2>
    <p class="dim small">Measures the model, tries a few cuts, cuts in stages, heals each stage against the original, checks that it still thinks, and exports the GGUF files. It takes many hours; you can stop it and continue later. Nothing of the original is touched.</p>
    ${!ins ? `<p class="dim">Choose a model that can be slimmed in “Start from” above.</p>` : `
    <div class="row"><label>Aim for <strong id="ac-target-n">${(+ac.target).toFixed(1)}</strong> B parameters (the model has ${total.toFixed(1)} B)</label></div>
    <input type="range" id="ac-target" min="${Math.max(3, Math.ceil(total * 0.3))}" max="${Math.floor(total * 0.95 * 2) / 2}" step="0.5" value="${ac.target}" style="width:100%;max-width:560px" ${lock}>
    ${ac.pin ? `<div class="ac-pin" id="ac-pin"><strong>Chosen cut:</strong> ${ac.pin.blocks.length ? `blocks ${esc(ranges(ac.pin.blocks))} out` : "no blocks out"}${ac.pin.intermediate ? `, MLP ${thou(ac.pin.intermediate)}` : ""}${ac.pin.loss != null ? `, text-prediction error ${(+ac.pin.loss).toFixed(3)}` : ""} <span class="dim small">found by the search on the Try first card${ac.pin.think ? " (its thinking was checked)" : ""}; Auto-cut does not search or try again${ac.pin.think ? " and skips the thinking check before healing" : ""}.</span> <button class="secondary" id="ac-pin-remove" ${lock}>Remove</button></div>` : ""}
    <label class="opt"><input type="checkbox" id="ac-think" ${ac.thinking ? "checked" : ""} ${canThink && !live ? "" : "disabled"}> Keep it able to think (recommended): measure and heal on reasoning text, check the answers, keep the thinking template${canThink ? "" : " <span class=\"dim small\">(only Qwen3.5 / 3.8)</span>"}</label>
    ${ac.thinking ? `<label class="opt"><input type="checkbox" id="ac-gate" ${ac.gate ? "checked" : ""} ${live ? "disabled" : ""}> Check the thinking of the best 2 cuts before healing${pl && pl.think_gate_minutes ? ` (about ${Math.round(pl.think_gate_minutes)} min more)` : ""} <span class="dim small">a cut with a low error can still have lost its thinking; this chooses between the cuts by how they think</span></label>` : ""}
    ${acSpeedBlock(pl, live)}
    <div class="row"><label><input type="radio" name="ac-stages" value="2" ${ac.stages === 2 ? "checked" : ""} ${lock}> Two stages (recommended)${acKnobDelta(pl, "stages", "Two")}</label> <label><input type="radio" name="ac-stages" value="1" ${ac.stages === 1 ? "checked" : ""} ${lock}> One stage${acKnobDelta(pl, "stages", "One")}</label>
      <label><input type="radio" name="ac-quick" value="full" ${ac.quick ? "" : "checked"} ${lock}> Full${acKnobDelta(pl, "quick", "Full")}</label> <label><input type="radio" name="ac-quick" value="quick" ${ac.quick ? "checked" : ""} ${lock}> Quick${acKnobDelta(pl, "quick", "Quick")} <span class="dim small">(Try first)</span></label></div>
    <label class="opt"><input type="checkbox" id="ac-notify" ${ac.notifyStages !== false ? "checked" : ""} ${lock}> Tell me when a stage has been healed <span class="dim small">a message from the system with the percentage and the time left (needs “Tell me when a job has finished” on the Home page)</span></label>
    <label class="opt"><input type="checkbox" id="ac-keep" ${ac.keep ? "checked" : ""} ${lock}> Keep the intermediate files (needs much more disk)</label>
    ${pl && pl.candidates && pl.candidates.some((c) => c.needs_confirmation) ? `<label class="opt"><input type="checkbox" id="ac-many" ${ac.many ? "checked" : ""} ${lock}> I confirm that more than 4 whole blocks may be removed (reasoning suffers first)</label>` : ""}
    <label class="opt"><input type="checkbox" id="ac-gbon" ${ac.gbOn ? "checked" : ""} ${lock}> Also make a file of about <input type="text" id="ac-gb" value="${esc(ac.gb)}" placeholder="10.6" style="width:70px;flex:0" ${lock}> GB (measured mixed precision) and compare it with the plain sizes</label>
    ${ac.gbOn ? `<p class="dim small" id="ac-gbnote">Each tensor gets the type that spends the bytes best (measured on the healed model's weights, weighted by its importance matrix), then <strong>Compare quants</strong> (KL divergence) checks the file against the plain sizes next to it. It is published only when it is clearly better than the plain file of its size.${pl && pl.measured_minutes ? ` About ${Math.round(pl.measured_minutes[0])} min to measure and ${Math.round(pl.measured_minutes[1])} min to compare (guesses). The 16-bit file is kept until the comparison is done, so this needs more disk.` : ""}${!(+ac.gb > 0) ? ` <span class="warn" id="ac-gbwhy">Give the size in GB.</span>` : ""}</p>` : ""}
    ${studio}
    ${ac.planErr ? `<div class="err">${esc(ac.planErr)}</div>` : !pl ? `<p class="dim">Planning…</p>` : ""}${notes}${acVllmBox(pl, lock)}${sizes}${live ? "" : acTimeBlock(pl)}`}
    ${pfBox("auto", { studio_publish: !!acSpec().studio_publish, need_gb: pl && pl.disk ? pl.disk.peak_gb : null })}
    <div class="row"><button class="primary" id="ac-go" ${!pl || !pl.ok || live || rs.running || rs.testing || (rs.hlRun && rs.hlRun.running) || !rs.state.env_ready || pfBlocked("auto") ? "disabled" : ""}>Start Auto-cut</button>${quBtn("auto", `${!pl || !pl.ok ? "disabled" : ""}`)}
      ${live ? `<button class="secondary" id="ac-stop">Stop</button>` : ""}
      ${run && !live && (run.status === "stopped" || run.status === "failed") ? `<button class="primary" id="ac-continue">Continue</button><button class="secondary" id="ac-change">Change settings and continue</button>` : ""}
      ${ended ? `<button class="secondary" id="ac-report-btn">Show the report</button>` : ""}${ac.run ? `<button class="link" id="ac-free" title="Opens the Storage page: the models, the teacher's answers and the files this run made.">Free the disk space of this run…</button>` : ""}</div>
    ${!rs.state.env_ready ? `<div class="err">The training software is not installed yet (Training setup).</div>` : ""}
    ${accPanel()}
    ${acProgressBlock(run, live)}
    ${live && !acProg(run) && acEtaText(run, ac.eta) ? `<p class="dim small" id="ac-eta" aria-live="polite">${esc(acEtaText(run, ac.eta))} (${run.plan && run.plan.time ? "an estimate, step by step" : "an estimate; the checks and the export are counted at a fixed time"})</p>` : ""}
    ${live ? `<div class="stepline" aria-live="polite" id="ac-text">${esc(ac.text)}</div>${ac.progress != null ? `<progress max="1" value="${ac.progress}"></progress>` : ""}` : ""}
    ${ac.err ? errDiv(ac.err) : ""}${status}${steps}${rep}</div>`;
}

// "done around HH:MM": what is left of the plan (the running heal's own estimate when the training reports one, else the plan's hours) plus fixed allowances for the checks and the export
const AC_CHECK_H = 0.25, AC_EXPORT_H = 0.5;
function acEtaRange(run, eta, nowMs) {
  if (!run || !run.steps || run.status === "done") return null;
  const stages = (run.plan && run.plan.stages) || [];
  const time = run.plan && run.plan.time && run.plan.time.steps ? run.plan.time : null;
  const now = nowMs || Date.now();
  let lo = 0, hi = 0;
  for (const x of run.steps) {
    if (x.status === "done" || x.status === "skipped") continue;
    // the estimate made before the run: every step that is not done is left with its own range (a running step has the time it has used taken off)
    const t = time ? time.steps.find((y) => y.id === x.id) : null;
    if (t) {
      let a = t.minutes[0] / 60, b = t.minutes[1] / 60;
      if (x.status === "running") {
        const left = eta && eta.step === x.id ? Math.max(0, eta.s - (now - eta.at) / 1000) / 3600 : null;
        if (left != null) {
          const merge = (t.parts || []).filter((q) => q.id === "merge").reduce((m, q) => [m[0] + q.minutes[0] / 60, m[1] + q.minutes[1] / 60], [0, 0]);   // the merge comes after the training
          a = left + merge[0]; b = left + merge[1];
        } else if (x.started && !Number.isNaN(Date.parse(x.started))) {
          const used = Math.max(0, now - Date.parse(x.started)) / 3600e3;
          a = Math.max(0.02, a - used); b = Math.max(a, b - used);
        }
      }
      lo += a; hi += b;
      continue;
    }
    const m = /^s(\d+)\.(heal|check_cut|check_heal)$/.exec(x.id);
    if (x.id === "export") { lo += AC_EXPORT_H; hi += AC_EXPORT_H; }
    else if (x.id === "check_gguf" || (m && m[2] !== "heal")) { lo += AC_CHECK_H; hi += AC_CHECK_H; }
    else if (m) {
      const left = eta && eta.step === x.id && x.status === "running" ? Math.max(0, eta.s - (now - eta.at) / 1000) / 3600 : null;
      const g = stages[+m[1] - 1];
      if (left != null) { lo += left; hi += left; } else if (g && g.hours) { lo += g.hours[0]; hi += g.hours[1]; }
    }
  }
  return lo || hi ? { lo, hi } : null;
}
const acHM = (d) => String(d.getHours()).padStart(2, "0") + ":" + String(d.getMinutes()).padStart(2, "0");
// the clock time of `ms`; a time on another calendar day (local date, not 24 hours) says which: "tomorrow 14:49", or "Thu 14:49" two or more days on
const acDay = (ms, nowMs) => {
  const d = new Date(ms), n = new Date(nowMs);
  const days = Math.round((Date.UTC(d.getFullYear(), d.getMonth(), d.getDate()) - Date.UTC(n.getFullYear(), n.getMonth(), n.getDate())) / 864e5);
  return days <= 0 ? "" : days === 1 ? "tomorrow" : ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"][d.getDay()];
};
const acClock = (ms, nowMs) => { const day = nowMs == null ? "" : acDay(ms, nowMs); return (day ? day + " " : "") + acHM(new Date(ms)); };
// a duration in words without false precision: to the nearest 5 minutes below 10 hours, to the nearest 30 minutes from 10 hours up ("45 min", "1 h", "2 h 05 min", "20 h")
function acDuration(hours) {
  const step = hours >= 10 ? 30 : 5;
  const m = Math.max(5, Math.round(hours * 60 / step) * step), h = Math.floor(m / 60), r = m % 60;
  return h === 0 ? `${r} min` : r === 0 ? `${h} h` : `${h} h ${String(r).padStart(2, "0")} min`;
}
// a range of hours in words and the clock times it ends at: { left: "9 h 30 min to 14 h", when: "tomorrow 02:15–07:45" }
function acSpan(r, now) {
  const hiH = Math.max(r.lo, r.hi), lo = now + r.lo * 3600e3, hi = now + hiH * 3600e3;
  const a = acClock(lo, now), b = acClock(hi, now), da = acDay(lo, now), db = acDay(hi, now);
  const dl = acDuration(r.lo), dh = acDuration(hiH);
  return { left: dl === dh ? dl : `${dl} to ${dh}`, when: a === b ? a : da === db ? `${a}–${acHM(new Date(hi))}` : `${a}–${b}` };      // one day word for both ends when they share it
}
function acEtaText(run, eta, nowMs) {
  const r = acEtaRange(run, eta, nowMs);
  if (!r) return "";
  const s = acSpan(r, nowMs || Date.now());
  return `about ${s.left} left (done around ${s.when})`;
}

// "How long this will take": the estimate of the whole run (plan.time, made by the program from your own earlier jobs where it has them, else from formulas)
function acStepMin(lo, hi) {
  const f = (m) => (m < 1 ? "<1" : String(Math.round(m)));
  if (hi < 60) return lo < 1 && hi < 1 ? "under 1 min" : `${f(lo)}–${f(hi)} min`;
  const a = acDuration(lo / 60), b = acDuration(hi / 60);
  return a === b ? a : `${a} to ${b}`;
}
function acTimeBlock(pl, nowMs) {
  const t = pl && pl.time;
  if (!t || !t.steps || !t.steps.length) return "";
  const now = nowMs || Date.now(), r = { lo: t.total_h[0], hi: t.total_h[1] }, sp = acSpan(r, now);
  const tag = (b) => b === "your_history" ? `<span class="tag ac-hist">from your history</span>` : `<span class="tag ac-formula">formula</span>`;
  const partsOf = (x) => {
    const q = (x.parts || []).filter((y) => ["cache", "train", "merge"].includes(y.id));
    return q.length ? `<div class="dim small">${q.map((y) => `${esc(y.label)}: ${acStepMin(y.minutes[0], y.minutes[1])}`).join("; ")}</div>` : "";
  };
  const rows = t.steps.map((x) => `<tr data-actime="${esc(x.id)}"><td>${esc(x.label)}${partsOf(x)}${x.note ? `<div class="dim small">${esc(x.note)}</div>` : ""}</td><td>${acStepMin(x.minutes[0], x.minutes[1])}</td><td>${tag(x.basis)}</td></tr>`).join("");
  const lever = t.lever_h && t.lever_h[1] > 0.5 * r.hi ? `<p class="dim small" id="ac-time-lever">Most of the time is the teacher's answers and the heal: ${acStepMin(t.lever_h[0] * 60, t.lever_h[1] * 60)}. Fewer tokens or one stage shortens it.</p>` : "";
  const extra = (t.extra || []).map((e) => `<p class="dim small" data-actime-extra="${esc(e.id)}">Possible extra, not counted above: ${esc(e.label)}: ${acStepMin(e.minutes[0], e.minutes[1])}.</p>`).join("");
  return `<div class="ac-time" id="ac-time"><h3>How long this will take</h3>
    <p class="ac-total" id="ac-time-total">All of it together takes about <strong>${esc(sp.left)}</strong></p>
    <p id="ac-time-done">If you press Start now it should be done around ${esc(sp.when)}.</p>
    <details id="ac-time-steps"><summary>Every step</summary><table class="tbl"><tr><th>Step</th><th>Estimate</th><th>Where from</th></tr>${rows}</table></details>
    <p class="dim small" id="ac-time-basis">${esc(t.basis)}</p>
    <p class="dim small" id="ac-time-caveat">${esc((t.notes && t.notes[0]) || "This is an estimate, not a promise.")}</p>
    ${lever}${extra}${(t.notes || []).slice(1).map((n) => `<p class="dim small">${esc(n)}</p>`).join("")}</div>`;
}

// How far the run is (the `auto_progress` event of the program: the share of the whole run weighted by the time estimate, the time spent, the time left from the live speed).
// A run that sent none (an older one) keeps the older hours estimate above.
function acProg(run) {
  const p = ac.prog && run && ac.prog.run === run.id ? ac.prog : run && run.progress ? run.progress : null;
  return p && p.kind === "auto_progress" ? p : null;
}
function acProgLeft(pr, nowMs) {
  if (!pr || pr.state === "estimating" || !pr.left_s) return "estimating the time left…";
  const s = acSpan({ lo: pr.left_s[0] / 3600, hi: pr.left_s[1] / 3600 }, nowMs || Date.now());
  return `about ${s.left} left (done around ${s.when})`;
}
function acProgressBlock(run, live, nowMs) {
  const pr = acProg(run);
  if (!pr) return "";
  const pct = Math.round(pr.pct), st = pr.step && pr.step.id ? pr.step : null;
  const head = live ? `${pct} % · ${acDuration((pr.elapsed_s || 0) / 3600)} so far · ${acProgLeft(pr, nowMs)}` : `${run.status === "done" ? "Finished" : "Stopped at " + pct + " %"} · ${acDuration((pr.elapsed_s || 0) / 3600)} spent`;
  const chip = pr.stage && pr.stage.of > 1 ? `<span class="tag ac-stage" id="ac-stagechip">Stage ${pr.stage.n} of ${pr.stage.of}</span> ` : "";
  const sub = live && st ? `<div class="ac-substep">${st.sub ? `<div class="stepline" aria-live="polite" id="ac-sub">${esc(st.sub)}</div>` : ""}<progress max="100" value="${st.pct}" id="ac-stepbar"></progress> <span class="dim small">${esc(st.label)} (step ${st.index} of ${st.count})</span>${acRoundChips(st)}</div>` : "";
  const back = pr.retry_added_min ? `<p class="dim small" id="ac-retrynote">The bar stepped back once: a thinking check failed and the stage is healed again (about ${acDuration(pr.retry_added_min / 60)} more).</p>` : "";
  const cal = live && pr.calibration && Math.abs(pr.calibration - 1) >= 0.15 ? `<p class="dim small" id="ac-calnote">This computer is running ${pr.calibration > 1 ? "slower" : "faster"} than the plan so far (${pr.calibration.toFixed(2)} times); the steps still to come are counted that way.</p>` : "";
  return `<div class="ac-progress" id="ac-progress"><div class="ac-head">${chip}<strong id="ac-overall-text">${esc(head)}</strong></div>
    <progress max="100" value="${pr.pct}" id="ac-overall" aria-label="Auto-cut overall"></progress>${sub}${back}${cal}</div>`;
}
function acStepFig(x) {
  const bits = [];
  if (x.est) bits.push(`est. ${acStepMin(x.est[0], x.est[1])}`);
  if (x.status === "done" && x.started && x.finished) { const m = (Date.parse(x.finished) - Date.parse(x.started)) / 60000; if (m > 0) bits.push(`took ${acDuration(m / 60)}`); }
  return bits.join(" · ");
}

function onAutoEvent(p) {
  if (p.kind === "auto_progress") { ac.prog = p; if (onReshape() || page === "welcome") render(); return; }
  if (p.eta_s != null && p.auto_step) ac.eta = { s: +p.eta_s, step: p.auto_step, at: Date.now() };
  if (p.kind === "auto") { ac.text = p.note || ""; acLoad(); return; }
  if (p.kind === "progress") { ac.text = p.text || p.step || ""; ac.progress = p.value == null ? null : p.value; }
  else if (p.kind === "error") { ac.running = false; ac.err = p.text; ac.progress = null; acLoad(); return; }
  else if (p.kind === "done") { ac.running = false; ac.progress = null; ac.text = ""; if (p.data) ac.run = p.data; acLoad(); return; }
  if (onReshape()) render();
}

function wireAuto() {
  const set = (id, f) => { const e = $(id); if (e) e.onchange = () => { f(e); ac.err = null; render(); }; };
  set("ac-target", (e) => { ac.target = +e.value; });
  const t = $("ac-target"); if (t) t.oninput = () => { const n = $("ac-target-n"); if (n) n.textContent = (+t.value).toFixed(1); };
  set("ac-think", (e) => { ac.thinking = e.checked; }); set("ac-gate", (e) => { ac.gate = e.checked; }); set("ac-keep", (e) => { ac.keep = e.checked; }); set("ac-notify", (e) => { ac.notifyStages = e.checked; }); set("ac-many", (e) => { ac.many = e.checked; });
  set("ac-publish", (e) => { ac.publish = e.checked; });
  set("ac-gbon", (e) => { ac.gbOn = e.checked; }); set("ac-gb", (e) => { ac.gb = e.value.trim(); });
  document.querySelectorAll("input[name=ac-preset]").forEach((r) => r.onchange = () => { if (r.value === "custom") ac.advOpen = true; else acApplyPreset(r.value); render(); });
  document.querySelectorAll("input[data-acknob]").forEach((r) => r.onchange = () => {
    const sp = acSpeedTable(ac.plan).sp, k = sp && sp.knobs.find((x) => x.knob === r.dataset.acknob), o = k && k.options[+r.dataset.acopt];
    if (o) acApplySet(o.set);                          // never dropped: the table of the last plan is kept while a new one is on its way
    const y = typeof window !== "undefined" && window ? window.scrollY : 0;
    render();
    if (y && typeof window.scrollTo === "function") window.scrollTo(0, y);
  });
  const pr = $("ac-pin-remove"); if (pr) pr.onclick = () => { ac.pin = null; ac.err = null; render(); };
  const vlset = (k) => (e) => { const cur = ac.sp.vl || { teacher_cache: false, checks: false, automatic: false }; if (k === "automatic") { cur.automatic = e.checked; if (e.checked) { cur.teacher_cache = false; cur.checks = false; } } else { cur[k] = e.checked; if (e.checked) cur.automatic = false; } ac.sp.vl = cur.teacher_cache || cur.checks || cur.automatic ? cur : null; };
  set("ac-vl-kv", (e) => { const cur = ac.sp.vl || { teacher_cache: false, checks: false, automatic: false }; cur.kv_fp8 = e.checked; ac.sp.vl = cur.teacher_cache || cur.checks || cur.automatic || cur.kv_fp8 ? cur : null; });
  set("ac-vl-t", vlset("teacher_cache")); set("ac-vl-c", vlset("checks")); set("ac-vl-a", vlset("automatic"));
  set("ac-search", (e) => { ac.sp.search = e.value === "" ? null : +e.value; });
  const adv = $("ac-adv"); if (adv) adv.ontoggle = () => { ac.advOpen = adv.open; };
  const tg = $("ac-tune-go"); if (tg) tg.onclick = () => go("tune");
  const ta = $("ac-tune-auto"); if (ta) ta.onclick = () => tuneAutoFor(acTuneModel());
  const tuned = $("ac-tuned"); if (tuned) tuned.onchange = async () => { if (!tuned.value) { ac.tunedSnap = null; render(); return; } const [id, scope] = tuned.value.split("|"); try { ac.tunedSnap = await invoke("tune_snapshot", { presetId: id, scope }); } catch (e) { ac.tunedSnap = null; rs.error = String(e); } render(); };
  set("ac-sbt", (e) => { const n = Math.round(+e.value || 0); ac.sp.sbt = n >= 512 && n <= 65536 ? n : null; });
  set("ac-maxon", (e) => { ac.maxOn = e.checked; }); set("ac-maxh", (e) => { ac.maxH = e.value.trim(); });
  document.querySelectorAll("input[name=ac-stages]").forEach((r) => r.onchange = () => { ac.stages = +r.value; render(); });
  document.querySelectorAll("input[name=ac-quick]").forEach((r) => r.onchange = () => { ac.quick = r.value === "quick"; render(); });
  const share = () => ({ host_path: ($("ac-sh-host") || {}).value || "", container_path: ($("ac-sh-cont") || {}).value || "", register_scan_folder: !!($("ac-sh-reg") || {}).checked, finetune_handoff: !!($("ac-sh-hf") || {}).checked });
  const sv = $("ac-sh-save"); if (sv) sv.onclick = async () => { ac.share = share(); try { setState(await invoke("studio_share_set", { share: ac.share })); ac.shareNote = "Saved."; } catch (e) { ac.shareNote = String(e); } render(); };
  const ck = $("ac-sh-check"); if (ck) ck.onclick = async () => { ac.share = share(); ac.shareNote = "Checking…"; render(); try { const r = await invoke("studio_share_check", { share: ac.share }); ac.shareNote = r.note; } catch (e) { ac.shareNote = String(e); } render(); };
  const go = $("ac-go");
  if (go) go.onclick = async () => {
    ac.err = null; ac.report = null; ac.running = true; ac.text = "Starting…"; ac.progress = null; render();
    try { await invoke("auto_start", { spec: acSpec(), resume: null }); } catch (e) { ac.running = false; ac.err = String(e); render(); }
  };
  wireAcc();
  const co = $("ac-continue");
  if (co) co.onclick = async () => {
    ac.err = null; ac.report = null; ac.running = true; ac.text = "Continuing…"; render();
    try { await invoke("auto_start", { spec: null, resume: ac.run.id }); } catch (e) { ac.running = false; ac.err = String(e); render(); }
  };
  const st = $("ac-stop"); if (st) st.onclick = () => { ac.text = "Stopping…"; invoke("auto_stop"); render(); };
  const fr = $("ac-free"); if (fr) fr.onclick = () => go("storage");
  const rp = $("ac-report-btn");
  if (rp) rp.onclick = async () => { try { const r = await invoke("auto_report", { run: ac.run ? ac.run.id : null }); ac.report = r.markdown; } catch (e) { ac.err = String(e); } render(); };
  acRefresh();
}

// ------------------------------------------------------------------------------------------------------------ the thinking check
// Does the model still think before it answers, after it was cut and healed? A small suite is asked with thinking on (worker job think_check, docs/THINKING_CHECK.md);
// the table sets the numbers of the original, the cut, the healed model and a GGUF file side by side, and the gates say pass, warn or fail for each.
const TK_TARGETS = [["original", "Original"], ["cut", "Cut model"], ["healed", "Healed"], ["gguf", "GGUF file"]];
const TK_NUMBERS = [["think_rate", "Writes a reasoning trace", "rate"], ["closed_rate", "Finishes its reasoning and answers", "rate"], ["accuracy", "Right answers", "rate"], ["loop_rate", "Loops (lower is better)", "rate"],
  ["garbage_rate", "Garbled text (lower is better)", "rate"], ["truncated_rate", "Cut off by the length limit", "rate"], ["median_think_tokens", "Typical length of the reasoning (tokens)", "n"]];
const TK_GATE_WORD = { closed_rate: "finishing its reasoning", accuracy: "right answers", loop_rate: "loops", garbage_rate: "garbled text", median_think_tokens: "length of the reasoning" };

const tkBackend = () => (rs.tk.backend === "vllm" && vllmSettings().enabled ? "vllm" : rs.tk.backend === "llama" ? "llama" : "hf");
const tkSpecBackend = () => (tkBackend() === "vllm" ? "vllm" : null);
const tkSpecExtra = () => (tkBackend() === "vllm" ? { backend: "vllm" } : {});          // nothing added for the standard way: its spec is as it always was
function tkBackendRow() {
  const be = tkBackend(), on = !!vllmSettings().enabled;
  return `<div class="row"><label>Engine <select id="tk-backend"><option value="hf" ${be === "hf" ? "selected" : ""}>Transformers (default)</option><option value="llama" ${be === "llama" ? "selected" : ""}>llama-server (GGUF files)</option>`
    + `<option value="vllm" ${be === "vllm" ? "selected" : ""} ${on ? "" : "disabled"}>vLLM (experimental${on ? "" : ": switched off in Training setup"})</option></select></label>`
    + (be === "llama" ? ` <span class="dim small">llama-server runs GGUF files: choose one below.</span>` : "") + `</div>`;
}

function tkQuick(m) {
  if (rs.tk.quick != null) return rs.tk.quick;
  const b = m && m.summary ? (m.summary.params_after || 0) / 1e9 : 0;
  return b >= 15;
}
const tkCell = (v, kind) => v == null ? "–" : kind === "rate" ? Math.round(v * 100) + " %" : thou(Math.round(v));

function thinkCard(m) {
  const sm = m.summary;
  if (!sm || m.status !== "done") return "";
  const tk = rs.tk, d = tk.name === m.name ? tk.data : null;
  const quick = tkQuick(m), be = tkBackend(), set = (d && be === "vllm" ? d.vllm_modes : d && d.modes) || null, mode = set ? set[quick ? "quick" : "full"] : null, rows = mode ? mode.rows : {}, gates = mode ? mode.gates : {};
  const busy = rs.running || rs.testing || (rs.hlRun && rs.hlRun.running) || tk.run;
  const can = (t) => d && d.targets && (t === "gguf" ? !!tk.gguf : be !== "llama" && d.targets[t]);
  const btn = (t, label) => `<button class="${rows[t] ? "secondary" : "primary"}" data-tkgo="${t}" ${busy || !rs.state.env_ready || !can(t) || pfBlocked("think_check") ? "disabled" : ""}>${rows[t] ? "Check again: " : "Check "}${label}</button>${quBtn("think_check", `data-target="${t}" ${!rs.state.env_ready || !can(t) ? "disabled" : ""}`)}`;
  const cols = TK_TARGETS.filter(([t]) => rows[t]);
  const table = cols.length ? `<table class="tbl"><tr><th>${quick ? "16" : "32"} questions, thinking on</th>${cols.map(([, n]) => `<th>${n}</th>`).join("")}</tr>
      ${TK_NUMBERS.map(([k, name, kind]) => `<tr><td>${name}</td>${cols.map(([t]) => `<td>${tkCell(rows[t].main[k], kind)}</td>`).join("")}</tr>`).join("")}
      <tr><td>Thinking off (control): right answers</td>${cols.map(([t]) => `<td>${tkCell(rows[t].control && rows[t].control.n ? rows[t].control.accuracy : null, "rate")}</td>`).join("")}</tr>
      <tr><td><strong>Compared with the original</strong></td>${cols.map(([t]) => { const g = gates[t]; return `<td>${t === "original" ? "–" : g ? `<strong class="${g.verdict === "pass" ? "ok" : g.verdict === "warn" ? "warn" : "err"}" data-tkverdict="${t}">${g.verdict === "pass" ? "Pass" : g.verdict === "warn" ? "Close to the limit" : "Fail"}</strong>${g.failed.length ? `<br><span class="dim small">too weak: ${g.failed.map((x) => TK_GATE_WORD[x]).join(", ")}</span>` : g.warned.length ? `<br><span class="dim small">near the limit: ${g.warned.map((x) => TK_GATE_WORD[x]).join(", ")}</span>` : ""}` : "<span class=\"dim small\">check the original first</span>"}</td>`; }).join("")}</tr></table>
      <p class="dim small">${cols.map(([t]) => `${TK_TARGETS.find((x) => x[0] === t)[1]}: ${esc(rows[t].created || "")}${rows[t].backend === "llama" ? " (llama.cpp)" : rows[t].backend === "vllm" ? " (vLLM, experimental)" : ""}`).join(" · ")}</p>` : "";
  const gg = d && d.ggufs && d.ggufs.length ? `<select id="tk-gguf-pick"><option value="">Choose an exported GGUF…</option>${d.ggufs.map((g) => `<option value="${esc(g.path)}" ${g.path === tk.gguf ? "selected" : ""}>${esc(g.name)}</option>`).join("")}</select>` : "";
  const note = mode && !rows.original ? `<p class="dim small">Check the original first: the other numbers are judged against it (it is checked once and kept).</p>` : "";
  // a cut that goes beyond what Auto-cut does by itself (the program names how): the thinking numbers of the cut when it was checked, else a pointer to the button
  const guard = d && d.guardrails && d.guardrails.length ? (() => {
    const cm = rows.cut ? rows.cut.main : null;
    const nums = cm ? `Checked before healing: it finished its reasoning in ${Math.round((cm.closed_rate || 0) * 100)} % of the answers and looped in ${Math.round((cm.loop_rate || 0) * 100)} %${rows.original ? ` (the original: ${Math.round((rows.original.main.closed_rate || 0) * 100)} %)` : ""}.`
      : `It has not been checked for thinking yet: press “Check the cut model” below.`;
    return `<p class="warn" id="tk-guard">${d.guardrails.map(esc).join(" ")} ${esc(nums)}</p>`;
  })() : "";
  return `<div class="card-in" id="tk-card"><h3>Thinking check</h3>${guard}
    <p class="dim small">Does the model still think before it answers? It is asked ${quick ? 16 : 32} short questions (sums, logic, what a program prints, counting) with thinking switched on, and the table says how often it writes a reasoning trace, finishes it, finds the right answer and avoids loops. ${quick ? "Quick: about 10 minutes for a large model." : "Full: about 15 minutes for a large model."}</p>
    <div class="row"><label><input type="radio" name="tk-mode" value="quick" ${quick ? "checked" : ""}> Quick (16 questions)</label> <label><input type="radio" name="tk-mode" value="full" ${quick ? "" : "checked"}> Full (32 questions)</label></div>
    ${tkBackendRow()}${be === "vllm" ? `<p class="warn small" id="tk-vllm-note">vLLM is experimental: built from a research report and not yet run on this PC (run the vLLM self-test in Training setup first). Its numbers are kept apart from the standard engine's: a model is compared with the original checked the same way. For the ORIGINAL, vLLM loads the original itself (its own folder, quantized to 4 bits while loading) unless it fits the card in 16 bits; a pre-quantized checkpoint stands in only for the original it was made from or a copy you confirmed in Settings. If neither works, the standard engine checks the original and the result says so. The preflight below names the model vLLM would load.</p>` : ""}
    ${pfBox("think_check", be === "vllm" ? { backend: "vllm", name: rs.tk.name, target: "original" } : {})}<div class="row">${btn("original", "the original")}${btn("cut", "the cut model")}${m.healed ? btn("healed", "the healed model") : ""}</div>
    <div class="row">${gg}<input id="tk-gguf" type="text" placeholder="…or the path of a GGUF file (run with llama.cpp)" value="${esc(tk.gguf)}" style="min-width:18em">${btn("gguf", "this GGUF")}</div>
    ${tk.run && tk.run.name === m.name ? `<div class="stepline" aria-live="polite">${esc(tk.run.text)}</div>${tk.run.progress != null ? `<progress max="1" value="${tk.run.progress}"></progress>` : ""}<div class="row"><button class="secondary" id="tk-stop">Stop</button></div>` : ""}
    ${tk.err ? errDiv(tk.err, "p") : tk.ok ? `<p class="ok">${esc(tk.ok)}</p>` : ""}${note}${table}</div>`;
}

async function tkLoad(name) {
  rs.tk.name = name;
  try { rs.tk.data = await invoke("think_check_results", { name }); } catch (e) { rs.tk.data = null; rs.tk.err = String(e); }
  if (onReshape()) render();
}

function tkStart(m, target) {
  const tk = rs.tk;
  tk.run = { name: m.name, target, text: "Starting…", progress: null, done: 0 }; tk.err = null; tk.ok = null; render();
  invoke("think_check_start", { spec: { name: m.name, target, quick: tkQuick(m), gguf: target === "gguf" ? tk.gguf : null, ...tkSpecExtra() } }).catch((e) => { tk.run = null; tk.err = String(e); render(); });
}

function onThinkEvent(p) {
  const tk = rs.tk;
  if (p.kind === "progress") { if (tk.run) { tk.run.text = p.step === "load" ? (p.text || "Loading the model…") : (p.text || p.step); tk.run.progress = p.value; } }
  else if (p.kind === "think_item") { if (tk.run) tk.run.done++; }
  else if (p.kind === "log") rs.log.push(p.text);
  else if (p.kind === "error") { tk.run = null; tk.err = p.text; }
  else if (p.kind === "done") {
    const name = tk.run && tk.run.name;
    tk.run = null;
    tk.ok = p.data && p.data.stopped ? "Stopped. Nothing was saved from this check." : p.data && p.data.cached ? "Done (the saved numbers of this model were used)." : "Done. The numbers are in the table.";
    if (name) { tkLoad(name); return; }
  }
  if (onReshape()) render();
}

// "Make thinking answers with the original model" (worker job trace_gen, docs/VLLM.md): the original writes thinking answers for prompts; they are filtered and kept in a jsonl file that a
// heal can use as "My own file". The way (vLLM when it is switched on, else llama-server, else the standard engine) is chosen by the worker; speeds are estimates.
const tg = { name: null, state: null, source: "", column: "", limit: 200, run: null, err: null, ok: null };
function traceCard(m) {
  if (!m.summary || m.status !== "done") return "";
  const s = tg.name === m.name ? tg.state : null;
  const busy = rs.running || rs.testing || (rs.hlRun && rs.hlRun.running) || rs.tk.run || tg.run;
  const vOn = !!vllmSettings().enabled;
  return `<div class="card-in" id="tg-card"><h3>Make thinking answers with the original model</h3>
    <p class="dim small">The original model writes answers WITH its thinking for prompts you give (a file or a dataset), so a heal can learn from them. Each answer is checked: the thinking must be closed, the answer not empty, not cut by the length limit, no loops, no garbled text; the rest is thrown away and not asked again. A second run goes on where the first stopped; Stop keeps what was written. ${vOn ? "The experimental vLLM engine will be used when it works (estimated 5 to 10 times faster than the standard engine: an estimate from a research report, not measured on this PC); otherwise the standard engine." : "The standard engine is used (slow for a big original: the vLLM engine in Training setup is the experimental, faster way)."}</p>
    <div class="row"><label>Prompts: a file or a Hugging Face dataset <input type="text" id="tg-source" value="${esc(tg.source)}" placeholder="C:\\prompts.jsonl or mlabonne/open-perfectblend" style="min-width:300px"></label>
      <label>Column (optional) <input type="text" id="tg-column" value="${esc(tg.column)}" style="width:8em"></label>
      <label>Prompts to try <input type="number" id="tg-limit" min="1" value="${esc(String(tg.limit))}" style="width:6em"></label></div>
    ${pfBox("trace_gen", vOn ? { backend: "vllm" } : {})}
    <div class="row"><button class="primary" id="tg-go" ${busy || !rs.state.env_ready || !tg.source.trim() ? "disabled" : ""}>Make the answers</button>${tg.run && tg.run.name === m.name ? `<button class="secondary" id="tg-stop">Stop</button>` : ""}</div>
    ${tg.run && tg.run.name === m.name ? `<div class="stepline" aria-live="polite">${esc(tg.run.text)}</div>${tg.run.progress != null ? `<progress max="1" value="${tg.run.progress}"></progress>` : ""}` : ""}
    ${tg.err ? errDiv(tg.err, "p") : tg.ok ? `<p class="ok">${esc(tg.ok)}</p>` : ""}
    ${s && s.count ? `<p class="small" id="tg-state">${s.count} answers are kept in <code>${esc(s.file)}</code> ${copyBtn(s.file)} (${s.rejected} prompts were rejected). Use the file as "My own file" in the general text of a heal.</p>` : ""}</div>`;
}
async function tgLoad(name) {
  tg.name = name;
  try { tg.state = await invoke("trace_gen_state", { name }); } catch (_) { tg.state = null; }
  if (onReshape()) render();
}
function onTraceEvent(p) {
  if (p.kind === "progress") { if (tg.run) { tg.run.text = p.text || p.step; tg.run.progress = p.value; } }
  else if (p.kind === "error") { tg.run = null; tg.err = p.text; }
  else if (p.kind === "done") {
    const d = p.data || {}, name = tg.run && tg.run.name;
    tg.run = null;
    tg.ok = `${d.stopped ? "Stopped. " : "Done. "}${d.kept || 0} of ${d.attempted || 0} answers kept${d.backend ? " (way: " + d.backend + ")" : ""}${(d.notes || []).length ? ". " + d.notes.join(" ") : ""}${d.note ? " " + d.note : ""}`;
    if (name) { tgLoad(name); return; }
  }
  if (onReshape()) render();
}
function wireTrace() {
  const m = rs.state && rs.state.made.find((x) => x.name === rs.tk.name);
  if (m && tg.name !== m.name && !tg.run) tgLoad(m.name);
  const src = $("tg-source"); if (src) src.oninput = () => { tg.source = src.value; };
  const col = $("tg-column"); if (col) col.oninput = () => { tg.column = col.value; };
  const lim = $("tg-limit"); if (lim) lim.oninput = () => { tg.limit = Math.max(1, Math.floor(+lim.value) || 200); };
  const go = $("tg-go"); if (go) go.onclick = () => {
    if (!m) return;
    tg.source = (($("tg-source") || {}).value || "").trim(); tg.column = (($("tg-column") || {}).value || "").trim(); tg.limit = Math.max(1, Math.floor(+(($("tg-limit") || {}).value) || 200));
    tg.run = { name: m.name, text: "Starting…", progress: null }; tg.err = null; tg.ok = null; render();
    invoke("trace_gen_start", { spec: { name: m.name, source: tg.source, column: tg.column || null, limit: tg.limit } }).catch((e) => { tg.run = null; tg.err = String(e); render(); });
  };
  const st = $("tg-stop"); if (st) st.onclick = () => { if (tg.run) tg.run.text = "Stopping…"; invoke("reshape_stop"); render(); };
}

function wireThink() {
  const m = rs.state && rs.state.made.find((x) => x.name === rs.tk.name);
  const tk = rs.tk;
  document.querySelectorAll("input[name=tk-mode]").forEach((r) => r.onchange = () => { tk.quick = r.value === "quick"; render(); });
  document.querySelectorAll("[data-tkgo]").forEach((b) => b.onclick = () => { if (m) tkStart(m, b.dataset.tkgo); });
  const bsel = $("tk-backend"); if (bsel) bsel.onchange = () => { tk.backend = bsel.value; pfReset(); render(); };
  const pick = $("tk-gguf-pick"); if (pick) pick.onchange = () => { tk.gguf = pick.value; render(); };
  const gf = $("tk-gguf"); if (gf) gf.onchange = () => { tk.gguf = gf.value.trim(); render(); };
  const st = $("tk-stop"); if (st) st.onclick = () => { if (tk.run) tk.run.text = "Stopping…"; invoke("reshape_stop"); render(); };
}

// ------------------------------------------------------------------------------------------------------- try before cutting
// A "Try first" card: what a cut will cost, found out in memory on the project's own text BEFORE a model is written (worker job reshape_try). The numbers are the
// same text-prediction error as the loss_before / loss_after of a made model. Colour bands and the stability check come from the worker (one constant there).

const TRY_BAND_WORD = { small: "small", noticeable: "noticeable", large: "large" };
const TRY_WHAT = [["search", "Find the best cut for me (automatic)"], ["blocks", "Every block on its own"], ["runs", "Every run of the blocks on the slider"], ["widths", "Every width of the slider"], ["choice", "My choice (click blocks in the chart)"]];

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
  if (w === "search") return null;
  if (w === "blocks") { if (!tryDepthOn()) return null; spec.blocks = true; }
  else if (w === "runs") { if (!tryDepthOn() || !rs.blocks) return null; spec.runs = [rs.blocks]; }
  else if (w === "widths") {
    const ws = tryWidthOnShape() ? tryWidths() : []; if (!ws.length) return null; spec.widths = ws;
    if (rs.tr.scoring) spec.choices = ws.map((x) => ({ blocks: [], width: x, scoring: "output" }));          // the plain widths are tried as well: it is what they are compared with
    if (rs.tr.recon) spec.choices = spec.choices.concat(ws.flatMap((x) => ["pruned", "original"].map((r) => ({ blocks: [], width: x, reconstruct: r }))));
  }
  else {
    const c = tryChoice();
    if (!c.blocks.length && c.width == null) return null;
    spec.choices = [c.width == null ? { blocks: c.blocks } : { blocks: c.blocks, width: c.width }];
    if (rs.tr.scoring && c.width != null) spec.choices.push({ ...spec.choices[0], scoring: "output" });
    if (rs.tr.recon && c.width != null) for (const r of ["pruned", "original"]) spec.choices.push({ blocks: c.blocks, width: c.width, reconstruct: r });
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
      invoke("reshape_try_plan", { spec }).then((p) => { if (token === tr.planTok) { tr.plan = p; if (onReshape()) render(); } })
        .catch((e) => { if (token === tr.planTok) { tr.planErr = String(e); if (onReshape()) render(); } });
    }
  }
  const rsig = rs.model + "|" + trySamples() + "|" + rs.load;
  if (rsig !== tr.resFor && rs.state.dataset_version) {
    tr.resFor = rsig;
    invoke("reshape_try_results", { model: rs.model, samples: trySamples(), len: 512, load: rs.load }).then((r) => { if (rsig === tr.resFor && !tr.running) { tr.analysis = r && r.exists ? r.analysis : null; tr.sps = r && r.exists ? r.seconds_per_pass : null; if (onReshape()) render(); } })
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

// The cuts that rank the neurons another way, next to the same cut without it. "Better in both halves" is the only thing called a gain: one half alone can be luck.
function tryVariantsTable(a) {
  const rows = ((a && a.variants) || []).filter((r) => r.tried);
  if (!rows.length) return "";
  const pct = (x) => x == null ? "–" : (x >= 0 ? "+" : "−") + Math.abs(100 * x).toFixed(1) + " %";
  const both = (r) => r.better_both_halves === true ? "yes" : r.better_both_halves === false ? "no" : "–";
  return `<h4>Another way of choosing the neurons</h4><table class="tbl" id="tr-variants"><tr><th>Cut</th><th>Error</th><th>Against the original</th><th>Against the same cut by activation</th><th>Better in both halves</th><th>Given back by the refit (median / worst layer)</th></tr>${rows.map((r) =>
    `<tr><td>${esc(r.label)}</td><td>${(+r.loss).toFixed(3)}</td><td>${pct(r.rel)}</td><td>${pct(r.rel_vs_plain)}</td><td>${both(r)}</td><td>${r.recon && r.recon.median_removed_share != null ? `${pct(r.recon.median_removed_share).replace("+", "")} / ${pct(r.recon.worst_removed_share).replace("+", "")}${r.recon.overlap ? ` · ${r.recon.overlap} rows overlap Try first` : ""}` : "–"}</td></tr>`).join("")}</table>
    <p class="dim small">Experimental, and measured here only before healing: a gain on both halves is a reason to try the same cut healed, not a promise. The citations behind these options (the output ranking and the refit) have not been checked by this program; the refit is algebra, not a result. A gain here may not survive healing.</p>`;
}

// the plain verdict for one answer, with the honest part that never changes
function tryVerdict(a, r, what) {
  const [e1, e2] = a.bands.edges, band = r.band, base = a.base;
  const sentence = band === "small" ? "a small rise" : band === "noticeable" ? "a noticeable rise" : "a large rise";
  return `<div class="tr-verdict"><p><strong>${esc(what)}</strong>: the text-prediction error on your own text goes from ${base.loss.toFixed(3)} to ${r.loss.toFixed(3)} (<strong>${pct(r.rel)}</strong>; perplexity ${base.ppl} → ${r.ppl}). That is
    <span class="tr-chip" style="background:${bandCss(band)}"></span> <strong>${sentence}</strong>.</p></div>`;
}

const TRY_HONEST = (a) => `<p class="dim small">This is the error <strong>before</strong> healing: healing usually wins back part of it. A cheap cut now is not a guarantee that it heals well, and an expensive one can still heal into a usable model; nobody has measured how the two relate for your model yet. The colours are rules of thumb (under +${Math.round(a.bands.edges[0] * 100)} %, +${Math.round(a.bands.edges[0] * 100)} to +${Math.round(a.bands.edges[1] * 100)} %, over +${Math.round(a.bands.edges[1] * 100)} % of the original's error), not promises. Removing two blocks costs more than the two single costs added up; the "My choice" pass is the honest number. Small differences between two options are noise: the two halves of your text are compared below the chart.</p>`;

// For a model that thinks (Qwen3.5 / 3.8): the error before healing is not a thinking test. When the choice has a LOWER error than the original, both may have been measured differently.
function tryThinkNote(ins, a, hit) {
  if (!ins || ins.model_type !== "qwen3_5") return "";
  const lower = hit && hit.loss != null && a && a.base && a.base.loss != null && hit.loss < a.base.loss;
  return `<p class="warn small" id="tr-think-note">A lower text-prediction error does not mean the model can still think. Make the cut, then run “Check the cut model” in the Thinking check before healing.${lower ? " This choice's error is lower than the original's: check that both were measured the same way (a cut that predicts text better than the model it came from is a reason to doubt the measurement)." : ""}</p>`;
}

function tryCard() {
  const op = rs.options, ins = op && op.inspect, st = rs.state, tr = rs.tr;
  if (!ins || !ins.supported || rs.mode !== "prune" || !(op.options.length > 0 || tryWidthOnShape())) return "";
  if (!st.dataset_version) return `<div class="card wide"><h2>Try first</h2><p class="dim">Save a training-data version in Step 2 first: the cuts are tried on your own text.</p></div>`;
  const a = tr.analysis, spec = trySpec(), needM = tryNeedsMeasure(spec), run = tr.running;
  const se = tr.se, searchOn = tr.what === "search";
  const what = TRY_WHAT.filter(([k]) => (k === "blocks" || k === "runs") ? tryDepthOn() : k === "widths" ? tryWidthOnShape() : true)
    .map(([k, l]) => `<label class="opt"><input type="radio" name="tr-what" value="${k}" ${tr.what === k ? "checked" : ""} ${run || se.running ? "disabled" : ""}> ${esc(l)}${k === "runs" && rs.blocks ? ` <span class="dim">(${rs.blocks} block${rs.blocks === 1 ? "" : "s"})</span>` : ""}</label>`).join("");
  const q = trySamples();
  const plan = tr.plan;
  const planLine = !spec ? `<span class="dim">${tr.what === "choice" ? "Click blocks in the chart (or pick a width) to choose what to try." : "Nothing to try with the settings above."}</span>`
    : needM ? `<span class="dim">The width needs the neuron measurement of this model first. <button class="link" id="tr-measure" ${rs.running ? "disabled" : ""}>Measure first</button></span>`
    : tr.planErr ? `<span class="err">${esc(tr.planErr)}</span>`
    : !plan ? `<span class="dim">Counting…</span>`
    : plan.passes_todo === 0 ? `<span class="dim">All ${plan.passes_total} answers are already kept; showing them costs nothing.</span>`
    : `<strong>${plan.passes_todo}</strong> pass${plan.passes_todo === 1 ? "" : "es"} over your text${plan.passes_cached ? ` (${plan.passes_cached} already kept)` : ""} × ${fmtDur(plan.seconds_per_pass).replace("about ", "")} each = <strong>${fmtDur(plan.seconds)}</strong>${plan.guess ? ` <span class="dim">(a guess from the model's size; the real time is measured on the first pass and kept)</span>` : ` <span class="dim">(from a pass that was timed)</span>`}`;
  const disabled = !spec || needM || run || rs.running || rs.testing || !!tr.planErr || (plan && plan.passes_todo === 0 && false);
  const progress = run ? `<div class="stepline" aria-live="polite">${esc(tr.text)}${tr.total ? ` · ${tr.done} of ${tr.total} done` : ""}</div>${tr.progress != null ? `<progress max="1" value="${tr.progress}"></progress>` : ""}` : "";
  let body = "";
  if (a && a.base && a.base.loss != null) {
    const parts = [];
    if (tryDepthOn() && a.blocks.some((r) => r.tried)) parts.push(`<h4>Skipping one block at a time</h4>${tryBlockChart(a, ins)}${tryLegend(a)}${a.blocks_stability && !a.blocks_stability.agree ? `<div class="warn">The two halves of your text disagree about which blocks are cheapest to skip. Treat the order as uncertain.</div>` : ""}`);
    else if (tryDepthOn() && (tr.what === "choice" || Object.keys(a.runs || {}).length)) parts.push(`<h4>Choose blocks</h4>${tryBlockChart(a, ins)}${tryLegend(a)}`);
    if (tryDepthOn() && rs.blocks && a.runs && a.runs[String(rs.blocks)]) parts.push(tryRunsTable(a, rs.blocks));
    if (tryWidthOnShape() && a.widths.some((w) => w.tried)) parts.push(`<h4>Thinner layers: quality against size</h4>${tryWidthChart(a, ins)}${tryLegend(a)}${tryWidthTable(a)}`);
    if ((a.variants || []).length) parts.push(tryVariantsTable(a));
    const ch = tryChoice(), hit = tryLookup(a, ch);
    const label = (ch.blocks.length ? `blocks ${ranges(ch.blocks)}` : "") + (ch.blocks.length && ch.width != null ? " and " : "") + (ch.width != null ? `MLP ${thou(ch.width)}` : "");
    const verdict = hit ? tryVerdict(a, hit, "Your choice (" + label + ")") : (ch.blocks.length || ch.width != null) ? `<p class="dim">Your choice (${esc(label)}) has not been tried yet: choose “My choice” above and press Try.</p>` : "";
    parts.push(`<h4>Verdict</h4>${verdict}<p>Original: error ${a.base.loss.toFixed(3)}, perplexity ${a.base.ppl}.</p>${TRY_HONEST(a)}${tryThinkNote(ins, a, hit)}
      <div class="row"><button class="primary" id="tr-use" ${hit && !run && !rs.running ? "" : "disabled"}>Use this choice</button><span class="dim small">${hit ? "fills the Make form below so the model is cut exactly as it was tried" : "try the choice first, then it can be used"}</span></div>`);
    body = parts.join("");
  }
  return `<div class="card wide"><h2>Try first <span class="tag">optional</span></h2>
    <p>Before a new model is written (tens of GB), find out what a cut costs. The program imitates the cut inside the loaded model and measures the same text-prediction error as the “before / after” numbers of a made model. <strong>Nothing is saved except a small results file.</strong></p>
    <div class="row">${what}</div>
    <p>Text: <label class="opt" style="display:inline"><input type="radio" name="tr-q" value="quick" ${q === 64 ? "checked" : ""} ${run ? "disabled" : ""}> Quick (64 samples)</label> <label class="opt" style="display:inline"><input type="radio" name="tr-q" value="full" ${q === 128 ? "checked" : ""} ${run ? "disabled" : ""}> Full (128 samples)</label>
      <span class="dim small">${ins.total_params / 1e9 >= 15 ? "Quick is the default for models of 15 B or more." : "Full is the default for smaller models."}</span></p>
    ${tryWidthOnShape() && state.settings.expert_mode && !searchOn ? `<label class="opt" id="tr-scoring-row"><input type="checkbox" id="tr-scoring" ${tr.scoring ? "checked" : ""} ${run ? "disabled" : ""}> Also try ranking the neurons by their expected output (experimental; the plain cut is tried too, to compare)</label>
    <label class="opt" id="tr-recon-row"><input type="checkbox" id="tr-recon" ${tr.recon ? "checked" : ""} ${run ? "disabled" : ""}> Also try refitting the kept neurons without training (experimental; both ways: from the cut model's own activations and from the original's; the plain cut is tried too; about ${fmtDur(plan && plan.recon_seconds ? plan.recon_seconds : null).replace("about ", "")} on top, a formula and not a measurement)</label>` : ""}
    ${searchOn ? trySearchBlock(ins) : `<p id="tr-plan">${planLine}</p>
    <div class="row"><button class="primary" id="tr-go" ${disabled ? "disabled" : ""}>${tr.what === "choice" ? "Preview my choice" : "Try"}</button><button class="secondary" id="tr-stop" ${run ? "" : "disabled"}>Stop</button><span class="dim small">Stopping keeps the cuts already tried.</span></div>
    ${progress}${body}`}</div>`;
}


// ---------------------------------------------------------------------------------------------------- the automatic search ("Find the best cut for me")
// One job (reshape_try with want.search) measures many cuts in one model load, round by round; the program ranks them, says how sure the call is, and can check the thinking of
// the best two. It measures the error on your own text, NOT the quality after healing, and for a model that thinks a low error is not enough: the thinking check decides.
const SE_ROUNDS = { singles: "each block on its own", width_refs: "the widths", sets_k: "sets of blocks", combined: "the real cuts", refine1: "swapping a block", refine2: "swapping a block again" };
function seKeep() { const se = rs.tr.se, ins = rs.options && rs.options.inspect; return se.keep == null ? !!ins && ins.model_type === "qwen3_5" : se.keep; }
function seSpec() {
  const se = rs.tr.se, ins = rs.options && rs.options.inspect;
  if (!ins || !ins.supported || !rs.model) return null;
  const total = ins.total_params / 1e9;
  const target = se.target === "b" ? { b: +(se.b != null ? se.b : Math.round(total * 0.76 * 2) / 2) } : se.target === "gb" ? (+se.gb > 0 ? { gb: +se.gb } : null) : se.target === "pct" ? { kept_pct: +se.pct } : "tradeoff";
  if (!target) return null;
  const keep = seKeep();
  return { model_id: rs.model, target, budget_min: +se.budget, keep_thinking: keep, samples: trySamples(), axes: se.axes, allow_many_blocks: !!se.many, beyond_floor: !!se.beyond, think_check: keep && se.think ? "top2" : "off", confirm_full: !!se.confirm, thorough: !!se.thorough };
}
function seBand(rel, edges) { return rel == null ? null : rel < edges[0] ? "small" : rel < edges[1] ? "noticeable" : "large"; }
// the table of the best cuts so far, from the analysis the worker sends after every pass (`analysis.search`)
function seLiveRows(a) {
  if (!a || !a.search || !a.base || a.base.loss == null) return [];
  const ranking = (a.search.ranking || {})["0"] || [], rows = a.search.rows || {}, edges = (a.bands && a.bands.edges) || [0.03, 0.1];
  return ranking.slice(0, 10).map((k) => {
    const [d, w] = k.split("|"), blocks = d.slice(2).split(",").filter(Boolean).map(Number), r = rows[k] || {}, rel = r.loss != null ? r.loss / a.base.loss - 1 : null;
    const agree = r.loss_h1 != null && r.loss_h2 != null ? Math.abs(r.loss_h1 - r.loss_h2) < Math.max(0.002, 0.25 * Math.abs(r.loss - a.base.loss)) : null;
    return { key: k, blocks, width: w.slice(2) ? +w.slice(2) : null, rel, band: seBand(rel, edges), agree };
  });
}
function seRowHtml(r) {
  return `<tr data-serow="${esc(r.key)}"><td>${r.blocks.length ? esc(ranges(r.blocks)) : "none"}</td><td>${r.width ? thou(r.width) : "unchanged"}</td><td>${r.rel != null ? `<span class="tr-chip" style="background:${bandCss(r.band)}"></span> ${pct(r.rel)}` : "–"}</td><td>${r.agree == null ? "–" : r.agree ? "yes" : "no"}</td></tr>`;
}
const SE_CONF = { clear: "A clear winner.", close_call: "A close call: the second best is within the noise of the two halves of your text.", uncertain: "Uncertain: the two halves of your text name different winners." };
function seRecommendation(res) {
  const rec = res && res.recommendation, rows = (res && res.rows) || [];
  if (!rec) return `<p class="warn" id="se-none">The search measured no cut. Give it more time, or check that the model is measured (Measure first).</p>`;
  const r = rows.find((x) => x.key === rec.key) || {};
  const think = r.think ? `<p id="se-think">Thinking checked: ${r.think.verdict === "lost_much" ? "<strong class=\"err\">it lost much of its ability to finish its reasoning</strong>" : "<strong class=\"ok\">it kept its reasoning</strong>"} (closed ${Math.round(((r.think.main || {}).closed_rate || 0) * 100)} %, loops ${Math.round(((r.think.main || {}).loop_rate || 0) * 100)} %${r.think.original ? `; the original: closed ${Math.round((r.think.original.closed_rate || 0) * 100)} %` : ""}).</p>`
    : seKeep() ? `<p class="warn" id="se-think">Thinking not checked: a low error does not prove that the model can still think. <button class="secondary" id="se-think-go" ${rs.tr.se.running || rs.running ? "disabled" : ""}>Check the thinking of the top 2</button> <span class="dim small">writes one cut at a time (about ${r.params_b ? Math.round(r.params_b * 2) : "?"} GB) and removes it again; nothing is kept</span></p>` : "";
  const guard = (r.guardrails || []).length ? `<p class="warn small" id="se-guard">${esc(r.guardrails.join(" "))}</p>` : "";
  return `<div class="tr-verdict" id="se-rec"><h4>The best cut found</h4>
    <p><strong>${r.blocks && r.blocks.length ? `Blocks ${esc(ranges(r.blocks))} out` : "No blocks out"}, MLP ${r.intermediate ? thou(r.intermediate) : "unchanged"}</strong>${r.params_b ? ` (${r.params_b} B)` : ""}: ${r.rel != null ? `the text-prediction error rises by ${pct(r.rel)}` : ""}.</p>
    <p id="se-reason">${esc(rec.reason)}</p><p class="${rec.confidence === "clear" ? "ok" : "warn"} small" id="se-conf">${esc(SE_CONF[rec.confidence] || "")}</p>${think}${guard}
    <p class="dim small">This ranks cuts by how well they predict your text before healing. It does not measure the quality after healing${seKeep() ? "; for a model that thinks only the thinking check decides" : ""}.</p>
    <div class="row"><button class="primary" id="se-use-ac">Use this cut in Auto-cut</button><button class="secondary" id="se-cut-now">Cut it now</button><button class="secondary" id="se-cut-heal">Cut and heal it (Auto-cut, one stage)</button></div></div>`;
}
function trySearchBlock(ins) {
  const se = rs.tr.se, total = ins.total_params / 1e9, expert = state.settings && state.settings.expert_mode, lock = se.running ? "disabled" : "";
  const bv = se.b != null ? se.b : Math.round(total * 0.76 * 2) / 2;
  const tgt = (k, label, inner) => `<label class="opt"><input type="radio" name="se-target" value="${k}" ${se.target === k ? "checked" : ""} ${lock}> ${label} ${inner || ""}</label>`;
  const spec = seSpec(), pl = se.plan;
  const planLine = !spec ? `<span class="dim">Choose a size.</span>` : se.planErr ? `<span class="err" id="se-planerr">${esc(se.planErr)}</span>` : !pl ? `<span class="dim">Planning…</span>`
    : `About <strong>${pl.passes}</strong> measurements (${pl.rounds.filter((r) => r.count).length} rounds) × ${acStepMin(pl.pass_min[0], pl.pass_min[1])} each, plus loading ${acStepMin(pl.load_min[0], pl.load_min[1])} = ${acStepMin(pl.total_min[0], pl.total_min[1])} in all (an estimate). ${pl.complete ? "All of it fits the time you gave." : `${pl.fit_passes} fit the ${pl.budget_min} minutes: rounds 1 to ${Math.max(1, pl.fit_rounds)}; the last rounds are cut when the time is up.`}`;
  const live = se.running ? `<div class="stepline" aria-live="polite" id="se-live">${se.round ? `Round ${se.round.index} of ${(pl && pl.rounds.filter((r) => r.count).length) || 6} (${esc(se.round.label)}) · pass ${se.done} of about ${se.total}${se.left != null ? ` · up to ${acDuration(se.left / 3600)} left` : ""}` : esc(se.text || "Starting…")}</div>${se.total ? `<progress max="1" value="${Math.min(1, se.done / Math.max(1, se.total))}" id="se-bar"></progress>` : ""}` : "";
  const rows = se.rows.length ? `<table class="tbl" id="se-live-table"><tr><th>Blocks out</th><th>MLP</th><th>Error change</th><th>Halves agree</th></tr>${se.rows.map(seRowHtml).join("")}</table>${se.best ? `<p class="dim small">Best so far: ${esc(se.best.key)}</p>` : ""}` : "";
  const res = !se.running && (se.result || se.partial) ? seRecommendation(se.result || se.partial) : "";
  return `<div id="se-card"><p>The program picks the cuts to try, round by round, in one go with the model loaded once: each block on its own, then sets of blocks, then the real cuts at the width the size needs, then small swaps. It ranks them on your own text and tells you how sure it is.</p>
    <div class="grid">${tgt("b", "Aim for", `<input type="text" id="se-b" value="${esc(String(bv))}" style="width:70px;flex:0" ${lock}> B parameters`)}
      ${tgt("gb", "A file of about", `<input type="text" id="se-gb" value="${esc(se.gb)}" placeholder="12.5" style="width:70px;flex:0" ${lock}> GB (Q4_K_M)`)}
      ${tgt("pct", "Keep", `<input type="text" id="se-pct" value="${esc(String(se.pct))}" style="width:60px;flex:0" ${lock}> % of the model`)}
      ${tgt("tradeoff", "The best trade-off", `<span class="dim small">the smallest size whose best cut still costs little (a rule of thumb)</span>`)}</div>
    <p>Spend at most <select id="se-budget" ${lock}>${[10, 20, 60, 120].map((m) => `<option value="${m}" ${+se.budget === m ? "selected" : ""}>${m} min</option>`).join("")}</select> measuring
      <label class="opt" style="display:inline"><input type="checkbox" id="se-confirm" ${se.confirm ? "checked" : ""} ${lock}> ask a close call again with more text</label></p>
    <label class="opt"><input type="checkbox" id="se-keep" ${seKeep() ? "checked" : ""} ${lock || (ins.model_type === "qwen3_5" ? "" : "disabled")}> Keep it able to think: measure on reasoning text${seKeep() ? ` and <label class="opt" style="display:inline"><input type="checkbox" id="se-think-on" ${se.think ? "checked" : ""} ${lock}> check the thinking of the best 2 cuts afterwards (writes one cut at a time)</label>` : ""}</label>
    ${expert ? `<p class="small">Expert: <select id="se-axes" ${lock}>${[["both", "blocks and width"], ["depth", "blocks only"], ["width", "width only"]].map(([v, l]) => `<option value="${v}" ${se.axes === v ? "selected" : ""}>${l}</option>`).join("")}</select>
      <label class="opt" style="display:inline"><input type="checkbox" id="se-many" ${se.many ? "checked" : ""} ${lock}> more than 4 blocks</label> <label class="opt" style="display:inline"><input type="checkbox" id="se-beyond" ${se.beyond ? "checked" : ""} ${lock}> thinner than the width-first floor</label> <label class="opt" style="display:inline"><input type="checkbox" id="se-thorough" ${se.thorough ? "checked" : ""} ${lock}> every neighbouring run too</label></p>` : ""}
    <p id="se-plan">${planLine}</p>${pl ? (pl.notes || []).map((n) => `<p class="dim small">${esc(n)}</p>`).join("") : ""}
    <div class="row"><button class="primary" id="se-go" ${!spec || se.running || se.planErr || rs.running || rs.testing || rs.tr.running ? "disabled" : ""}>Find the best cut</button><button class="secondary" id="se-stop" ${se.running ? "" : "disabled"}>Stop</button><span class="dim small">Stopping keeps everything measured; a repeated search costs nothing for the cuts it has.</span></div>
    ${live}${se.err ? errDiv(se.err) : ""}${rows}${res}</div>`;
}
function seRefresh() {
  if (page !== "reshape" || !rs.state || !rs.options || rs.tr.what !== "search") return;
  const se = rs.tr.se, spec = seSpec(), sig = spec ? JSON.stringify(spec) : "";
  if (sig === se.planFor || se.running) return;
  se.planFor = sig; se.plan = null; se.planErr = null;
  if (!spec) return;
  const token = ++se.planTok;
  invoke("reshape_search_plan", { spec, inspect: rs.options.inspect }).then((p) => { if (token === se.planTok) { se.plan = p; if (onReshape()) render(); } })
    .catch((e) => { if (token === se.planTok) { se.planErr = String(e); if (onReshape()) render(); } });
}
function onSearchEvent(p) {
  const se = rs.tr.se;
  if (p.kind === "progress") { se.text = p.text || se.text; }
  else if (p.kind === "log") rs.log.push(p.text);
  else if (p.kind === "search") { se.text = ({ measuring: "Measuring the cuts…", confirming: "Asking the close call again with more text…", writing_cut: "Writing one cut to check its thinking…" })[p.status] || se.text; se.id = p.id || se.id; }
  else if (p.kind === "try_pass") {
    se.done = p.done; se.total = p.planned_total || p.total; se.left = p.budget_left_s; se.best = p.best || null;
    se.round = p.round ? { name: p.round, label: p.round_label || SE_ROUNDS[p.round] || p.round, index: p.round_index || 1 } : se.round;
    se.rows = seLiveRows(p.analysis);
  } else if (p.kind === "error") { se.running = false; se.err = p.text; se.text = ""; }
  else if (p.kind === "done") {
    se.running = false; se.text = "";
    if (p.data && p.data.rows) { se.result = p.data; se.partial = null; se.id = p.data.id; }
    if (p.data && p.data.status === "stopped") {
      rs.ok = "Stopped. What was measured is kept; Find the best cut goes on from there.";
      const spec = seSpec();
      if (spec) invoke("reshape_search_status", { spec, inspect: rs.options.inspect }).then((r) => { se.partial = r && r.partial && r.partial.rows && r.partial.rows.length ? r.partial : null; se.result = null; if (onReshape()) render(); }).catch(() => {});
    } else rs.ok = "The search is done. The answers are kept: another search over the same text costs nothing for the cuts it has.";
  }
  if (onReshape()) render();
}
// what a recommended cut says to the rest of the program
function seChosen() {
  const se = rs.tr.se, res = se.result || se.partial, rec = res && res.recommendation, row = rec && (res.rows || []).find((r) => r.key === rec.key);
  return row ? { blocks: row.blocks, intermediate: row.intermediate, loss: row.loss, search_id: se.id || (se.result && se.result.id) || null, think: row.think || null, key: row.key } : null;
}
function sePin(stages) {
  const c = seChosen(); if (!c) return;
  ac.pin = { blocks: c.blocks, intermediate: c.intermediate, loss: c.loss, search_id: c.search_id, think: c.think };
  if (stages) ac.stages = stages;
  go("auto");          // Auto-cut lives on its own page: the pinned cut is shown there
  rs.ok = `This cut is pinned for Auto-cut: ${c.blocks.length ? "blocks " + ranges(c.blocks) : "no blocks"}${c.intermediate ? `, MLP ${thou(c.intermediate)}` : ""}. Auto-cut will not search again.`;
  render();
  const el = document.getElementById("ac-pin") || document.getElementById("ac-card"); if (el && el.scrollIntoView) el.scrollIntoView({ behavior: "smooth" });
}
function seCutNow() {
  const c = seChosen(), ins = rs.options.inspect; if (!c) return;
  rs.tr.sel = [...c.blocks];
  if (c.blocks.length) {
    rs.dropText = ranges(c.blocks.flatMap((b) => Array.from({ length: ins.period }, (_, i) => b * ins.period + i)));
    if (rs.options.options.some((o) => o.blocks === c.blocks.length)) rs.blocks = c.blocks.length;
    if (c.intermediate != null && ins.width_dims && c.intermediate !== ins.width_dims.intermediate && rs.shape === "depth") rs.shape = "both";
  }
  if (c.intermediate != null && !(ins.width_dims && c.intermediate === ins.width_dims.intermediate)) rs.wi = c.intermediate;
  rs.error = null;
  rs.ok = `The Make form now cuts exactly what the search found${c.blocks.length ? `: blocks ${ranges(c.blocks)}` : ""}${c.intermediate ? `, MLP ${thou(c.intermediate)}` : ""}.`;
}
function wireSearch() {
  const se = rs.tr.se, on = (id, fn) => { const e = $(id); if (e) e.onclick = fn; };
  const set = (id, f) => { const e = $(id); if (e) e.onchange = () => { f(e); render(); }; };
  document.querySelectorAll("input[name=se-target]").forEach((r) => r.onchange = () => { se.target = r.value; render(); });
  set("se-b", (e) => { se.b = +e.value; se.target = "b"; }); set("se-gb", (e) => { se.gb = e.value.trim(); se.target = "gb"; }); set("se-pct", (e) => { se.pct = +e.value; se.target = "pct"; });
  set("se-budget", (e) => { se.budget = +e.value; }); set("se-confirm", (e) => { se.confirm = e.checked; }); set("se-keep", (e) => { se.keep = e.checked; }); set("se-think-on", (e) => { se.think = e.checked; });
  set("se-axes", (e) => { se.axes = e.value; }); set("se-many", (e) => { se.many = e.checked; }); set("se-beyond", (e) => { se.beyond = e.checked; }); set("se-thorough", (e) => { se.thorough = e.checked; });
  const start = async (resume) => {
    const spec = seSpec(); if (!spec) return;
    se.running = true; se.text = "Starting…"; se.err = null; se.done = 0; se.total = 0; se.round = null; se.left = null; se.best = null; se.rows = []; if (!resume) { se.result = null; se.partial = null; } rs.error = null; rs.ok = null; rs.log = []; render();
    try { await invoke("reshape_search_start", { spec, resume: resume || null }); } catch (e) { se.running = false; se.err = String(e); render(); }
  };
  on("se-go", () => start(null));
  on("se-think-go", () => start(se.id || (se.result && se.result.id) || null));
  on("se-stop", () => { se.text = "Stopping…"; invoke("reshape_stop"); render(); });
  on("se-use-ac", () => sePin(null));
  on("se-cut-now", () => { seCutNow(); render(); });
  on("se-cut-heal", () => sePin(1));
  seRefresh();
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
  if (onReshape()) render();
}

function wireTry() {
  const on = (id, fn) => { const e = $(id); if (e) e.onclick = fn; };
  const tr = rs.tr;
  document.querySelectorAll("input[name=tr-what]").forEach((r) => r.onchange = () => { tr.what = r.value; render(); });
  document.querySelectorAll("input[name=tr-q]").forEach((r) => r.onchange = () => { tr.quick = r.value === "quick"; tr.analysis = null; render(); });
  const trs = $("tr-scoring"); if (trs) trs.onchange = () => { tr.scoring = trs.checked; render(); };
  const trr = $("tr-recon"); if (trr) trr.onchange = () => { tr.recon = trr.checked; render(); };
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
  wireSearch();
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
  const msg = (rs.error ? errDiv(rs.error) : "") + (rs.ok ? `<div class="ok">${esc(rs.ok)}</div>` : "");
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
    ${pfBox("cut", {})}<div class="row"><button class="primary" id="rs-go" ${!ok || rs.running || rs.testing || rs.tr.running || why || pfBlocked("cut") ? "disabled" : ""}>Make the model</button><button class="secondary" id="rs-stop" ${rs.running ? "" : "disabled"}>Stop</button></div>
    ${ok && why ? `<div class="err">${esc(why)}</div>` : ""}${ok && (wo || (!rsWidthOn() && o)) ? `<div class="dim small">Needs about ${needGb} GB of free disk space${st.free_disk_gb != null ? " (" + st.free_disk_gb + " GB free)" : ""}. The new model is saved inside the project, in <code>models/</code>.</div>` : ""}
    ${rs.running ? `<div class="stepline" aria-live="polite">${esc(rs.text)}</div>${rs.progress != null ? `<progress max="1" value="${rs.progress}"></progress>` : ""}` : ""}
    ${rs.log.length ? `<details><summary>Log</summary><pre class="log">${rs.log.slice(-120).map(esc).join("\n")}</pre></details>` : ""}</div>`;
  const made = st.made.filter((m) => m.status === "done" && m.summary);
  const shown = made.find((m) => m.name === rs.view) || made[0];
  const result = shown ? madeCard(shown) : "";
  const list = st.made.length ? `<div class="card wide"><h2>Models made so far</h2><table class="tbl"><tr><th>Name</th><th>Status</th><th>Layers</th><th>Size</th><th>Tested</th><th></th></tr>${st.made.map((m) =>
    `<tr><td>${esc(m.name)}</td><td>${esc(m.status)}${m.error ? `<div class="dim small">${esc(m.error)}</div>` : ""}</td><td>${m.summary ? m.summary.layers_before + " → " + m.summary.layers_after + (m.summary.width ? ", MLP " + thou(m.summary.width.before.intermediate_size) + " → " + thou(m.summary.width.after.intermediate_size) : "") : ""}</td><td>${m.size_gb ? m.size_gb + " GB" : ""}${m.healed ? ` <span class="tag">healed</span>` : ""}</td><td>${m.test ? esc(m.test.verdict.headline) : "–"}</td>
      <td>${m.summary ? `<button class="link" data-rsshow="${esc(m.name)}">Show</button> ` : ""}<button class="link" data-rsdelete="${esc(m.name)}" ${rs.running || rs.testing ? "disabled" : ""}>Delete</button></td></tr>`).join("")}</table></div>` : "";
  const toAuto = `<div class="card wide"><p style="margin:0">Want it all in one click? <button class="link" id="rs-open-auto">Open Auto</button> <span class="dim small">(measures, cuts, heals, checks and exports, one step after the other)</span></p></div>`;
  return head + msg + intro + toAuto + pick + size + tryCard() + make + healSection() + result + list + rsxCard(st);
}

// The Auto page: the Auto-cut card (and what it needs: the model it starts from, the pinned cut, the run's panels) on a page of its own. It reads the Slim page's state (`rs`).
function renderAuto() {
  const head = `<h1>🚀 ${TITLES.auto[0]}</h1><p class="lead">${TITLES.auto[1]}</p>`;
  if (!state.bundle) return head + `<p class="lead">Import your world data first (Step 1).</p>`;
  const st = rs.state;
  if (!st) return head + `<p class="lead">Loading…</p>`;
  const msg = (rs.error ? errDiv(rs.error) : "") + (rs.ok ? `<div class="ok">${esc(rs.ok)}</div>` : "");
  const modelOpts = st.models.map((m) => `<option value="${esc(m.id)}" ${m.id === rs.model ? "selected" : ""}>${esc(m.name)} · ${(+m.params_b).toFixed(1)} B${m.reshape ? " · " + esc(m.reshape) : ""}</option>`).join("");
  const ins = rs.options && rs.options.inspect;
  let info = "";
  if (rs.loadingOpts) info = `<p class="dim">Looking at the model… (the first time this can take a moment; nothing is downloaded)</p>`;
  else if (rs.optErr) info = `<div class="err">${esc(rs.optErr)}</div>`;
  else if (ins && !ins.supported) info = `<div class="err">${esc(ins.why || "This model cannot be slimmed or grown.")}</div>`;
  const busy = rs.running || rs.testing || (ac && ac.running);
  const pick = `<div class="card wide"><h2>Start from</h2><div class="row"><select id="ac-model" ${busy ? "disabled" : ""}>${modelOpts}</select></div>${info}</div>`;
  return head + msg + pick + autoCard();
}

async function rsRefresh(stale) {
  try {
    rs.state = await invoke("reshape_state");
    if (!rs.model && rs.state.models.length) rs.model = (rs.state.models.find((m) => m.id === "qwen3.5-9b") || rs.state.models[0]).id;
  } catch (e) { rs.error = String(e); }
  try { rsx.state = await invoke("export_model_state"); rsxEnsurePick(); } catch (e) { /* the card says what is missing */ }
  if (onReshape()) render();
  if (rs.state && rs.model && !rs.options && !rs.loadingOpts) rsOptions();
  if (stale && rs.hl) rs.hl.stale = true;
  hsLoadCand();
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
      if (rs.lv) args.linVHeads = rsAxis("lin_v_heads");
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
  if (onReshape()) render();
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
  if (onReshape()) render();
}

// the person's Heal card needs the options of the model that is shown
function rsAfterState() {
  const st = rs.state; if (!st) return;
  const done = st.made.filter((m) => m.status === "done" && m.summary), on = done.find((m) => m.name === rs.view) || done[0];
  if (on && (rs.tk.name !== on.name || !rs.tk.data) && !rs.tk.run) tkLoad(on.name);                // the thinking check card shows the saved numbers of the shown model
  if (rs.hl && rs.hl.pair) { if (rs.hl.stale) hsReloadPair(); return; }          // a pair chosen in the Heal section: it is reloaded as a pair
  const made = st.made.filter((m) => m.status === "done" && m.summary);
  const shown = made.find((m) => m.name === rs.view) || made[0];
  if (shown && shown.summary.mode === "prune" && (!rs.hl || rs.hl.name !== shown.name || rs.hl.stale)) rsLoadHeal(shown.name);
}

async function rsLoadHeal(name) {
  rs.hl = { name, opts: null };
  try { rs.hl = { name, opts: await invoke("heal_options", { name }) }; } catch (e) { rs.hl = { name, opts: null, error: String(e) }; }
  try { const last = await invoke("heal_bench_last", { name }); rs.hb = { ...(rs.hb || {}), last: last || null }; } catch (e) { /* no earlier speed test */ }
  if (onReshape()) render();
  if (rs.hl && rs.hl.opts) rsHealEstimate();
}

function onHealEvent(p) {
  const run = rs.hlRun || (rs.hlRun = { running: true, step: "cache", text: "", progress: null, live: [] });
  if (p.kind === "progress") { run.step = p.heal_step || run.step; run.text = p.text || ""; run.progress = p.value; }
  else if (p.kind === "log") rs.log.push(p.text);
  else if (p.kind === "distill_metric" || p.kind === "distill_eval") { run.step = p.heal_step || run.step; run.live.push(p); }
  else if (p.kind === "heal_round") { (run.rounds || (run.rounds = [])).push(p); if (p.note) run.text = p.note; }
  else if (p.kind === "error") { run.running = false; rs.error = p.text; rsRefresh(true); return; }
  else if (p.kind === "done") {
    run.running = false;
    if (p.data && p.data.stopped) rs.ok = "Stopped. Healing can be continued where it stopped.";
    else if (p.data) rs.ok = `Healed: ${p.data.name} now has a healed version, and it is in the model list of Step 3.`;
    setState_(); rsRefresh(true); return;
  }
  if (onReshape()) render();
}

function onReshapeEvent(p) {
  if (p.phase === "export") return onRsxEvent(p);
  if (p.phase === "auto") return onAutoEvent(p);
  if (p.phase === "think") return onThinkEvent(p);
  if (p.phase === "vllm") return onVllmEvent(p);
  if (p.phase === "vllm_teacher_check") return onVcheckEvent(p);
  if (p.phase === "vllm_probe") return onVprobeEvent(p);
  if (p.phase === "healspeed") return onHealSpeedEvent(p);
  if (p.phase === "trace") return onTraceEvent(p);
  if (p.phase === "heal") return onHealEvent(p);
  if (p.phase === "healbench") return onHealBenchEvent(p);
  if (p.phase === "try") return onTryEvent(p);
  if (p.phase === "search") return onSearchEvent(p);
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
  if (onReshape()) render();
  const l = document.querySelector("details .log"); if (l) l.scrollTop = l.scrollHeight;
}

// ----------------------------------------------------------------------------------------------------------------- save a model as GGUF without training (docs/RESHAPE.md)
// The "Save a model as GGUF (no training needed)" card: the same files, sizes and options as the export of a training run (Step 5), but from a model that was never
// trained here: one this page made (cut, grow or healed), any model of the list, or a folder on this computer. The core does the work (export::runs::execute_no_train);
// a model of the list that is not on this computer yet is fetched by the worker first. Events: "reshape-event" tagged {"phase": "export"} (onRsxEvent).

// the entry the current pick resolves to (the same shapes export_model_state gives)
function rsxEntry() {
  const st = rsx.state;
  if (!st) return null;
  if (rsx.src === "made") { const [n, sub] = (rsx.pick || "").split("||"); return (st.made || []).find((x) => x.name === n && x.sub === sub) || null; }
  if (rsx.src === "hub") return (st.models || []).find((x) => x.source && x.source.repo === rsx.pick) || null;
  return null;
}

// a pick that exists (the first one of the chosen kind), so the select never shows an option the args do not name
function rsxEnsurePick() {
  const st = rsx.state;
  if (!st) return;
  const ok = rsxEntry() || rsx.src === "folder";
  if (ok) return;
  if (rsx.src === "made") { const e = (st.made || [])[0]; rsx.pick = e ? `${e.name}||${e.sub}` : ""; if (e) rsx.thinking = !!e.thinking; }
  else if (rsx.src === "hub") { const m = (st.models || [])[0]; rsx.pick = m ? m.source.repo : ""; }
}

// the export name: every size becomes exports/<name>-<size>/ (a healed copy becomes <name>-healed, so the two copies never collide)
function rsxExportName() {
  const e = rsxEntry();
  if (e) return e.export_name;
  if (rsx.src === "folder") { const parts = rsx.folder.trim().replace(/[\\/]+$/, "").split(/[\\/]/); return parts[parts.length - 1] || "model"; }
  if (rsx.src === "hub") return (rsx.pick || "").split("/").pop() || "";
  return "";
}

function rsxSource() {
  if (rsx.src === "folder") return { kind: "folder", dir: rsx.folder.trim(), name: null };
  if (rsx.src === "hub") return { kind: "hub", repo: rsx.pick };
  const [name, sub] = (rsx.pick || "").split("||");
  return { kind: "made", name, sub };
}

// what "Save the GGUF" sends to export_model_start (the queue keeps the same); paramsB / downloadGb say nothing to the command (the queue's checklist reads them)
function rsxMakeArgs() {
  const e = rsxEntry(), saved = ((rsx.state || {}).imatrix || {}).saved || {};
  return { source: rsxSource(), quant: rsx.quants[0] || "q4_k_m", quants: rsx.quants, numCtx: rsx.numCtx, thinking: !!rsx.thinking,
    paramsB: e ? e.params_b : null, downloadGb: e ? (e.download_gb || 0) : 0,
    imatrix: rsx.im ? { enabled: true, chunks: rsx.imChunks, reuse: rsx.imReuse === null ? !!saved[rsxExportName()] : rsx.imReuse, nextn: rsxNextn((rsx.state || {}).imatrix).on, general: gcSource("imatrix"), ...(rsx.imSaved ? { use_saved: rsx.imSaved } : {}) } : null,
    mixed: (rsx.quants.includes("mixed_fit") || rsx.quants.includes("mixed_measured"))
      ? { ...(rsx.quants.includes("mixed_fit") ? { fit: { target_gb: Number(rsx.mx.target), matrix: rsx.mx.matrix || null } } : {}),
          ...(rsx.quants.includes("mixed_measured") ? { measured: { target_gb: Number(rsx.mm.target), matrix: rsx.mm.matrix || null } } : {}) }
      : null };
}

function rsxPfSpec() {
  const e = rsxEntry();
  return { what: "gguf", quants: rsx.quants, imatrix: rsx.im, params_b: e ? e.params_b : null, download_gb: e ? (e.download_gb || 0) : 0 };
}

// the speculative-decoding (MTP) head of the model being saved (the same rule as the export page, keyed by the export name)
function rsxNextn(im) {
  const head = ((im && im.mtp) || {})[rsxExportName()] === true;
  const unsupported = head && im.nextn_supported === false;
  return { head, unsupported, on: head && !unsupported && rsx.imNextn };
}

async function rsxRun(kind, cmd, args) {
  rsx.running = true; rsx.error = null; rsx.ok = null; rsx.log = []; rsx.text = "Starting…"; rsx.progress = null; rsx.pending = kind;
  render();
  try { await invoke(cmd, args); } catch (e) { rsx.running = false; rsx.error = String(e); rsx.text = ""; render(); }
}

function onRsxEvent(p) {
  if (p.kind === "progress") { rsx.text = p.text || p.step; rsx.progress = p.value; }
  else if (p.kind === "log") rsx.log.push(p.text);
  else if (p.kind === "error") { rsx.running = false; rsx.text = ""; rsx.progress = null; rsx.error = p.text; render(); return; }
  else if (p.kind === "done") {
    rsx.running = false; rsx.text = ""; rsx.progress = null;
    const d = p.data || {};
    const names = d.exports && d.exports.length ? d.exports.map((m) => m.name).join(", ") : d.name || "the files";
    rsx.ok = d.stopped ? "Stopped. Nothing was kept."
      : `Saved: ${names}. The files are in the exports list on Step 5, next to a Modelfile for Ollama.`;
    render();
    return;
  }
  if (onReshape()) render();
  const l = document.querySelector("details .log"); if (l) l.scrollTop = l.scrollHeight;
}

const rsxQuantBox = (q) => `<label class="opt" title="${esc(q.name)}"><input type="checkbox" class="rsx-q" value="${q.id}" ${rsx.quants.includes(q.id) ? "checked" : ""}> ${esc(q.label || q.name.split(":")[0])}</label>`;

function rsxCard(st) {
  const stx = rsx.state;
  const head = `<h2>Save a model as GGUF (no training needed)</h2>
    <p class="dim">Makes the GGUF files of a model that was never trained here, with a Modelfile for Ollama: a model this page made (a cut or a grow, or its healed copy), any model of the list, or a folder on this computer. The files join the same exports list as Step 5, and the sizes and options are the ones the export of a training run has. Nothing is merged (there is no trained add-on): the model is converted as it is.</p>`;
  if (!stx) return `<div class="card wide">${head}<p class="dim">Loading…</p></div>`;
  const made = stx.made || [], catalog = stx.models || [];
  const key = (e) => `${e.name}||${e.sub}`;
  const madeOpts = made.map((e) => `<option value="${esc(key(e))}" ${rsx.pick === key(e) ? "selected" : ""}>${esc(e.label)} · ${(+e.params_b).toFixed(1)} B${e.size_gb ? " · " + e.size_gb + " GB" : ""}</option>`).join("");
  const hubOpts = catalog.map((m) => `<option value="${esc(m.source.repo)}" ${rsx.pick === m.source.repo ? "selected" : ""}>${esc(m.name)} · ${(+m.params_b).toFixed(1)} B${m.local ? "" : " · downloads first (≈ " + m.download_gb + " GB)"}</option>`).join("");
  const radio = (id, label, extra) => `<label class="opt"><input type="radio" name="rsx-src" value="${id}" ${rsx.src === id ? "checked" : ""} ${extra || ""}> ${label}</label>`;
  const pickRow = rsx.src === "made" ? (made.length ? `<div class="row"><label>Model made here <select id="rsx-pick">${madeOpts}</select></label></div>`
      : `<p class="dim">No models made here yet: make one above (or choose one of the list, or a folder).</p>`)
    : rsx.src === "hub" ? `<div class="row"><label>Model of the list <select id="rsx-pick">${hubOpts}</select></label></div>`
    : `<div class="row"><label>Folder with config.json <input type="text" id="rsx-folder" value="${esc(rsx.folder)}" placeholder="C:\\models\\Qwen3.5-4B or /home/me/models/Qwen3.5-4B" style="min-width:320px"></label></div>`;
  const e = rsxEntry(), name = rsxExportName();
  const mixed = stx.mixed_quants || [], lib = ((stx.imatrix || {}).library) || [], saved = ((stx.imatrix || {}).saved) || {};
  const busy = rsx.running || rs.running;
  const mixedOn = rsx.quants.includes("mixed_fit") || rsx.quants.includes("mixed_measured");
  const imSavedOpts = lib.map((x) => `<option value="${esc(x.id)}" ${rsx.imSaved === x.id ? "selected" : ""}>${esc(x.label || x.id)} · ${esc(x.model || "other model")} · ${x.chunks != null ? x.chunks + " chunks" : "unknown amount"}</option>`).join("");
  const savedOne = saved[name];
  const why = !stx.env_ready ? "The training software is not installed yet (Training setup)."
    : rsx.src === "folder" && !rsx.folder.trim() ? "Type the folder of the model first."
    : rsx.src !== "folder" && !rsx.pick ? "Choose a model first."
    : rsx.src === "made" && !made.length ? "No models made here yet."
    : !rsx.quants.length ? "Choose at least one file size."
    : rsx.quants.includes("mixed_fit") && !(+rsx.mx.target > 0) ? "Give the size the fit should aim for."
    : rsx.quants.includes("mixed_measured") && !(+rsx.mm.target > 0) ? "Give the size the measured fit should aim for."
    : rsx.im && !rsx.imSaved && gcBlocked("imatrix") ? gcBlocked("imatrix") : "";
  const matRow = (who, st2) => `<label>Statistics from <select id="rsx-m${st2}matrix"><option value="">the matrix of this export</option>${lib.map((x) => `<option value="${esc(x.id)}" ${st2 === "x" ? rsx.mx.matrix === x.id : rsx.mm.matrix === x.id ? "selected" : ""}>${esc(x.label || x.id)}</option>`).join("")}</select></label>`;
  const mixedDetail = mixedOn ? `<p class="dim small">Made with llama.cpp (<code>llama-quantize</code> gives single tensors their own types), the same recipes as Step 5. The importance matrix, when you use one, is applied to these sizes too.</p>
    ${rsx.quants.includes("mixed_fit") ? `<div class="row"><strong>Fit to a size (experimental)</strong> <label>Target size <input type="text" id="rsx-mxtarget" value="${esc(rsx.mx.target)}" placeholder="12.5" style="min-width:80px;flex:0"> GB</label>${matRow("fit", "x")}</div>${!(+rsx.mx.target > 0) ? `<p class="warn small">Give the size in GB that the fit should aim for.</p>` : ""}` : ""}
    ${rsx.quants.includes("mixed_measured") ? `<div class="row"><strong>Mixed: measured fit to a size</strong> <label>Target size <input type="text" id="rsx-mmtarget" value="${esc(rsx.mm.target)}" placeholder="10.6" style="min-width:80px;flex:0"> GB</label>${matRow("measured", "m")}</div>${!(+rsx.mm.target > 0) ? `<p class="warn small">Give the size in GB that the measured fit should aim for.</p>` : ""}` : ""}` : "";
  const imatrixDetail = !rsx.im ? "" : rsx.imSaved ? `<div class="row"><label>Use a saved matrix <select id="rsx-imsaved"><option value="">Measure a new one now</option>${imSavedOpts}</select></label><button class="link" id="rsx-imlib">Open the matrix library</button></div>
      <p class="dim small">Nothing is measured: this matrix is copied into the export and given to <code>llama-quantize</code> (or Unsloth) as it is, for the sizes below Q8_0. It should have been made from the same model.</p>`
    : `<div class="row"><label>Calibration text <select id="rsx-imchunks">${(stx.imatrix.chunks || [64, 128, 256]).map((n) => `<option value="${n}" ${rsx.imChunks === n ? "selected" : ""}>${n} chunks${n === 128 ? " (recommended)" : ""}</option>`).join("")}</select></label>
      ${rsxNextn(stx.imatrix).head ? `<label class="opt ${rsxNextn(stx.imatrix).unsupported ? "dim" : ""}"><input type="checkbox" id="rsx-imnextn" ${rsxNextn(stx.imatrix).on ? "checked" : ""} ${rsxNextn(stx.imatrix).unsupported ? "disabled" : ""}> Include the speculative-decoding head (MTP) in the matrix</label>` : ""}
      ${savedOne ? `<label class="opt"><input type="checkbox" id="rsx-imreuse" ${rsx.imReuse === null ? "checked" : ""}> Reuse the saved matrix (${savedOne.chunks} chunks, made ${esc(String(savedOne.created || "").slice(0, 10))})</label>` : ""}</div>
      ${gcHtml("imatrix")}`;
  const prog = (rsx.running || rsx.text) ? `<div class="stepline" aria-live="polite">${esc(rsx.text)}</div>${rsx.progress != null ? `<progress max="1" value="${rsx.progress}"></progress>` : ""}` : "";
  return `<div class="card wide" id="rsx-card">${head}
    <div class="row">${radio("made", `A model made here (${made.length})`, rsx.running || rs.running ? "disabled" : "")}${radio("hub", "A model of the list", rsx.running || rs.running ? "disabled" : "")}${radio("folder", "A folder on this computer", rsx.running || rs.running ? "disabled" : "")}</div>
    ${pickRow}
    ${rsx.src === "hub" && e && !e.local ? `<p class="warn small">This model is not on this computer yet: the export downloads it first (about ${e.download_gb} GB into the Hugging Face cache), then makes the sizes. Needs an internet connection.</p>` : ""}
    <div class="row"><span class="quants">File sizes: ${stx.quants.concat(stx.mixed_quants).map(rsxQuantBox).join(" ")}</span>
      <label>Context <select id="rsx-ctx">${[4096, 8192, 16384, 32768].map((n) => `<option ${rsx.numCtx === n ? "selected" : ""}>${n}</option>`).join("")}</select></label></div>
    <div class="row"><label class="opt" title="${rsx.thinking ? "The exported chat template keeps the model's own thinking (a model healed with the thinking recipe)" : "The exported chat template answers without thinking unless asked"}"><input type="checkbox" id="rsx-thinking" ${rsx.thinking ? "checked" : ""} ${busy ? "disabled" : ""}> The exported file keeps the model's own thinking</label></div>
    <div class="row"><label class="opt"><input type="checkbox" id="rsx-im" ${rsx.im ? "checked" : ""} ${busy ? "disabled" : ""}> Better small sizes (importance matrix)</label>
      <span class="dim small">Makes Q6_K, Q5_K_M and Q4_K_M lose less quality; needs a saved training-data version (Step 2) and llama.cpp, and adds a few minutes.</span></div>
    ${imatrixDetail}
    ${mixed.length ? `<div class="row"><span class="quants"><strong>Mixed precision (our own recipe):</strong> ${mixed.map(rsxQuantBox).join(" ")}</span></div>${mixedDetail}` : ""}
    ${pfBox("export", rsxPfSpec())}
    <div class="row"><button class="primary" id="rsx-go" ${busy || !stx.env_ready || why || pfBlocked("export") ? "disabled" : ""}>Save the GGUF</button>${quBtn("export_model", why || busy ? "disabled" : "")}${rsx.running ? `<button class="secondary" id="rsx-cancel" title="Stops the export; while the importance matrix is made, llama.cpp is stopped at once">Cancel</button>` : ""}</div>
    ${why && !busy ? `<div class="err">${esc(why)}</div>` : ""}
    ${name ? `<p class="dim small">Saves as <code>exports/${esc(name)}-&lt;size&gt;/</code> (every ticked size of its own).</p>` : ""}
    ${prog}${rsx.log.length ? `<details><summary>Log</summary><pre class="log">${rsx.log.slice(-120).map(esc).join("\n")}</pre></details>` : ""}
    ${rsx.error ? errDiv(rsx.error) : ""}${rsx.ok ? `<div class="ok">${esc(rsx.ok)}</div>` : ""}</div>`;
}

function rsxWire() {
  const on = (id, fn) => { const e = $(id); if (e) e.onclick = fn; };
  const bind = (id, key, num) => { const e = $(id); if (e) e.onchange = () => { rsx[key] = num ? Number(e.value) : e.value; }; };
  document.querySelectorAll('input[name="rsx-src"]').forEach((r) => r.onchange = () => { rsx.src = r.value; rsx.pick = ""; rsx.imSaved = ""; rsx.imReuse = null; rsxEnsurePick(); render(); });
  const pk = $("rsx-pick"); if (pk) pk.onchange = () => { rsx.pick = pk.value; rsx.imSaved = ""; rsx.imReuse = null; const e = rsxEntry(); if (e && e.thinking != null) rsx.thinking = !!e.thinking; render(); };
  const fd = $("rsx-folder"); if (fd) fd.onchange = () => { rsx.folder = fd.value; render(); };
  document.querySelectorAll(".rsx-q").forEach((c) => c.onchange = () => { rsx.quants = [...document.querySelectorAll(".rsx-q")].filter((x) => x.checked).map((x) => x.value); render(); });
  bind("rsx-ctx", "numCtx", true);
  const th = $("rsx-thinking"); if (th) th.onchange = () => { rsx.thinking = th.checked; render(); };
  const imc = $("rsx-im"); if (imc) imc.onchange = () => { rsx.im = imc.checked; rsx.imSaved = ""; render(); };
  const imch = $("rsx-imchunks"); if (imch) imch.onchange = () => { rsx.imChunks = Number(imch.value); render(); };
  const imn = $("rsx-imnextn"); if (imn) imn.onchange = () => { rsx.imNextn = imn.checked; render(); };
  const imr = $("rsx-imreuse"); if (imr) imr.onchange = () => { rsx.imReuse = imr.checked; render(); };
  const ims = $("rsx-imsaved"); if (ims) ims.onchange = () => { rsx.imSaved = ims.value; render(); };
  const mxt = $("rsx-mxtarget"); if (mxt) mxt.onchange = () => { rsx.mx.target = mxt.value.trim(); render(); };
  const mxm = $("rsx-mxmatrix"); if (mxm) mxm.onchange = () => { rsx.mx.matrix = mxm.value; };
  const mmt = $("rsx-mmtarget"); if (mmt) mmt.onchange = () => { rsx.mm.target = mmt.value.trim(); render(); };
  const mmm = $("rsx-mmmatrix"); if (mmm) mmm.onchange = () => { rsx.mm.matrix = mmm.value; };
  if (rsx.im) gcWire("imatrix", render);
  on("rsx-imlib", () => go("imatrix"));
  on("rsx-go", () => rsxRun("make", "export_model_start", rsxMakeArgs()));
  on("rsx-cancel", () => invoke("env_cancel"));
}

function rsSpec() {
  const drop = parseLayers(rs.dropText);
  const spec = { source_model_id: rs.model, mode: rs.mode, blocks: rs.blocks, strategy: rs.strategy, protect_first: rs.pf, protect_last: rs.pl, init: rs.init, calib_samples: rs.samples, calib_len: 512, load: rs.load,
    drop_layers: rs.mode === "prune" && drop && drop.length ? drop : null };
  if (rs.mode === "prune" && rs.shape !== "depth" && rs.wi != null) {          // thinner layers: width only has no blocks
    spec.blocks = rs.shape === "width" ? 0 : rs.blocks;
    spec.width = { intermediate: rs.wi, lin_v_heads: rs.lv ? rsAxis("lin_v_heads") : null, heads: rs.hh || null };
    spec.width_samples = rs.wsamples;
    if (rs.scoring === "output") spec.width_scoring = "output";          // only when chosen: the default sends nothing
    if (rs.recon) spec.reconstruct = { variant: "pruned" };            // only when chosen (experimental)
    if (rs.shape === "width") spec.drop_layers = null;
  }
  return spec;
}

// "Keep thinking": the recipe is offered for Qwen3.5 / 3.8 models (`opts.recipe`) and is on unless the person unticked it
function hlRecipeOn(o) { return !!(o && o.recipe) && (rs.hlSpec.recipe == null ? !!o.recipe.default_on : !!rs.hlSpec.recipe); }

// what the Heal card sends
function rsHealSpec() {
  const sp = rs.hlSpec, o = rs.hl && rs.hl.opts, rc = hlRecipeOn(o) ? o.recipe : null;
  return { tokens_m: sp.tokens_m || (rc ? rc.tokens_m : o && o.default_tokens_m) || 20, general_share: sp.general ? +(rc && !sp.shareSet ? rc.general_share : sp.share) : 0, recipe: rc ? rc.id : null,
    general: sp.general ? gcSource("heal") : null, alpha: +sp.alpha, lr: +sp.lr, mode: sp.mode || null,
    rank: sp.rank, seq: sp.seq, batch: sp.batch, per_update: sp.per_update, checkpointing: sp.checkpointing || "auto", method: sp.method || null, optimizer: sp.optimizer || null,
    kl_chunk: sp.kl_chunk, k: sp.k, teacher_load: sp.teacher_load || null, warmup: sp.warmup, kbit_upcast: sp.kbit_upcast === true ? true : null, head_checkpoint: sp.head_checkpoint === "off" ? "off" : null,
    ...(sp.student_batch_tokens ? { student_batch_tokens: +sp.student_batch_tokens } : {}),
    ...(sp.rounds ? { rounds: hlRoundsSpec(sp.tokens_m || (rc ? rc.tokens_m : o && o.default_tokens_m) || 20, null), mode: null } : {}),
    ...(hlVllmAvailable() && sp.vllm ? { vllm: { teacher_cache: true } } : {}) };
}

async function rsMeasure(force) {
  rs.running = true; rs.ok = null; rs.error = null; rs.log = []; rs.text = "Starting…"; rs.progress = null; render();
  try { await invoke("reshape_measure", { model: rs.model, samples: rs.wsamples, len: 512, load: rs.load, force }); } catch (e) { rs.running = false; rs.error = String(e); render(); }
}


function wireHealSection() {
  const h = rs.hs, on = (id, fn) => { const e = $(id); if (e) e.onclick = fn; };
  const text = (id, key) => { const e = $(id); if (e) e.onchange = () => { h[key] = e.value; hsChanged(); }; };
  const sel = $("hs-student"); if (sel) sel.onchange = () => { h.studentPick = sel.value; h.guess = null; h.teacherPick = ""; hsChanged(); };
  const tsel = $("hs-teacher"); if (tsel) tsel.onchange = () => { h.teacherPick = tsel.value; hsChanged(); };
  text("hs-sfolder", "studentFolder"); text("hs-shub", "studentHub"); text("hs-tfolder", "teacherFolder"); text("hs-thub", "teacherHub");
  const browse = (key, title) => async () => { try { const dir = await tauri.dialog.open({ directory: true, title }); if (dir) { h[key] = dir; hsChanged(); } } catch (_) { /* no folder window: the field is typed */ } };
  on("hs-sbrowse", browse("studentFolder", "Choose the folder of the model to heal")); on("hs-tbrowse", browse("teacherFolder", "Choose the folder of the original model"));
  on("hs-check", () => hsCheck(true));
  on("hs-delrun", async () => {
    const name = $("hs-delrun").dataset.name;
    if (!confirm("Delete this heal and its files (the teacher's saved answers and the healed copy)? The model that was healed is not touched.")) return;
    try { rs.state = await invoke("reshape_remove", { name }); h.studentPick = ""; rs.hl = null; rs.ok = "Deleted."; } catch (e) { rs.error = String(e); }
    hsLoadCand(); render();
  });
}

function wireAutoPage() {
  if (!onAutoPage() || !rs.state) return;
  const m = $("ac-model"); if (m) m.onchange = () => { rs.model = m.value; rs.tr.sel = []; rs.tr.analysis = null; rs.tr.planFor = null; rs.tr.resFor = null; rs.dropText = ""; rs.blocks = null; rs.options = null; rs.ok = null; rs.error = null; rsOptions(); };
  wireAuto();
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
  const wrc = $("rs-recon"); if (wrc) wrc.onchange = () => { rs.recon = wrc.checked; };
  const wsc = $("rs-scoring"); if (wsc) wsc.onchange = () => { rs.scoring = wsc.value === "output" ? "output" : "activation"; };
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
  hv("hl-vllm", (e) => { hs.vllm = e.checked; });
  wireVllmFiles();
  const vck = $("hl-vllm-check"); if (vck) vck.onclick = () => { vch.run = { text: "Starting…", progress: null }; vch.err = null; render(); invoke("vllm_teacher_check_start", { name: rs.hl.name, spec: rsHealSpec() }).catch((e) => { vch.run = null; vch.err = String(e); render(); }); };
  const hst = $("hl-speed-teacher"); if (hst) hst.onchange = () => { hsp.teacher = hst.checked; };
  const hsg = $("hl-speed-go"); if (hsg) hsg.onclick = () => { hsp.teacher = !!($("hl-speed-teacher") || {}).checked; hsp.run = { text: "Starting…" }; hsp.err = null; render(); invoke("heal_speed_start", { name: rs.hl.name, spec: rsHealSpec(), withTeacher: hsp.teacher }).catch((e) => { hsp.run = null; hsp.err = String(e); render(); }); };
  const hsu = $("hl-speed-use"); if (hsu) hsu.onclick = () => { const r = hsp.files && hsp.files.recommended; if (r && r.value) { rs.hlSpec.student_batch_tokens = +r.value; render(); rsHealEstimate(); } };
  const hss = $("hl-speed-stop"); if (hss) hss.onclick = () => { if (hsp.run) hsp.run.text = "Stopping…"; invoke("reshape_stop"); render(); };
  const vcm = $("hl-vllm-checkcmp"); if (vcm) vcm.onclick = () => { vch.run = { text: "Starting…", progress: null }; vch.err = null; render(); invoke("vllm_teacher_check_start", { name: rs.hl.name, spec: { ...rsHealSpec(), vllm: { teacher_cache: true, kv_cache_dtype: "compare" } } }).catch((e) => { vch.run = null; vch.err = String(e); render(); }); };
  const vkv = $("hl-vllm-checkkv"); if (vkv) vkv.onclick = () => { vch.run = { text: "Starting…", progress: null }; vch.err = null; render(); invoke("vllm_teacher_check_start", { name: rs.hl.name, spec: { ...rsHealSpec(), vllm: { teacher_cache: true, kv_cache_dtype: "fp8" } } }).catch((e) => { vch.run = null; vch.err = String(e); render(); }); };
  const vpr = $("hl-vllm-probe"); if (vpr) vpr.onclick = () => { vch.probing = true; vch.probe = null; vch.probeText = "Starting…"; render(); invoke("vllm_container_probe_start").catch((e) => { vch.probing = false; vch.probe = { ok: false, text: String(e) }; render(); }); };
  const vcs = $("hl-vllm-checkstop"); if (vcs) vcs.onclick = () => { if (vch.run) vch.run.text = "Stopping…"; invoke("reshape_stop"); render(); };
  hv("hl-general", (e) => { hs.general = e.checked; }); hv("hl-tokens", (e) => { hs.tokens_m = +e.value; });
  hv("hl-recipe", (e) => {                                                                // the recipe's reasoning text, or the program's default text again
    hs.recipe = e.checked; hs.tokens_m = null; hs.shareSet = false;
    const o = rs.hl && rs.hl.opts, sc = gcScope("heal");
    if (gc.state && o && o.recipe) { sc.sel = e.checked ? o.recipe.general : gc.state.default; sc.check = null; sc.project = state && state.project ? state.project.folder : ""; gcPersist("heal"); }
  });
  hv("hl-share", (e) => { const v = parseFloat(e.value); hs.share = Number.isFinite(v) && v >= 0 && v <= 0.9 ? v : 0.5; hs.shareSet = true; });
  hv("hl-alpha", (e) => { const v = parseFloat(e.value); hs.alpha = Number.isFinite(v) && v >= 0 && v <= 1 ? v : 1; });
  const hnum = (id, key, lo, hi) => hv(id, (e) => { const v = parseInt(e.value, 10); hs[key] = Number.isFinite(v) && v >= lo && v <= hi ? v : null; });
  hnum("hl-rank", "rank", 8, 128); hnum("hl-klchunk", "kl_chunk", 64, 512); hnum("hl-k", "k", 8, 128); hnum("hl-warmup", "warmup", 0, 1000); hnum("hl-perupdate", "per_update", 1, 256);
  hv("hl-seq", (e) => { hs.seq = +e.value; }); hv("hl-batch", (e) => { hs.batch = +e.value; }); hv("hl-ckpt", (e) => { hs.checkpointing = e.value; }); hv("hl-method", (e) => { hs.method = e.value || null; });
  hv("hl-optim", (e) => { hs.optimizer = e.value; }); hv("hl-tload", (e) => { hs.teacher_load = e.value === "auto" ? null : e.value; });
  hv("hl-upcast", (e) => { hs.kbit_upcast = e.checked ? true : null; });
  hv("hl-headkeep", (e) => { hs.head_checkpoint = e.checked ? "off" : null; });
  hv("hl-sbt", (e) => { const n = Math.round(+e.value || 0); hs.student_batch_tokens = n >= 512 && n <= 65536 ? n : null; });
  const hbGo = $("hb-go");
  if (hbGo) hbGo.onclick = async () => {
    rs.hb = { ...(rs.hb || {}), running: true, rows: [], text: "Starting…", err: null, result: null };
    render();
    try { await invoke("heal_bench_start", { name: rs.hl.name, spec: rsHealSpec() }); } catch (e) { rs.hb.running = false; rs.hb.err = String(e); render(); }
  };
  const hbStop = $("hb-stop"); if (hbStop) hbStop.onclick = () => { if (rs.hb) rs.hb.text = "Stopping…"; invoke("heal_stop"); render(); };
  const hbUse = $("hb-use");
  if (hbUse) hbUse.onclick = () => {
    const b = rs.hb && (rs.hb.result || rs.hb.last) && (rs.hb.result || rs.hb.last).best_settings; if (!b) return;
    hs.kl_chunk = b.kl_chunk; hs.head_checkpoint = b.head_checkpoint === "off" ? "off" : null; render(); rsHealEstimate();
  };
  on("hl-reset", () => { Object.assign(hs, HEAL_AUTO); render(); rsHealEstimate(); });
  on("hl-safer", () => { if (!healUseSafer()) return; render(); rsHealEstimate(); });
  hv("hl-lr", (e) => { const v = parseFloat(e.value); hs.lr = Number.isFinite(v) && v >= 1e-6 && v <= 1e-2 ? v : 1e-4; });
  hv("hl-mode", (e) => { hs.mode = e.value; });
  hv("hl-rounds", (e) => { hs.rounds = e.checked; });
  hv("hl-round", (e) => { hs.roundM = +e.value; if (hs.firstCheck < hs.roundM) hs.firstCheck = hs.roundM; });
  hv("hl-firstcheck", (e) => { hs.firstCheck = +e.value; }); hv("hl-roundcheck", (e) => { hs.roundCheck = e.value; }); hv("hl-stoprule", (e) => { hs.stopRule = e.value; });
  on("hl-more", async () => {
    const r = rs.hl && rs.hl.opts && rs.hl.opts.heal && rs.hl.opts.heal.rounds; if (!r) return;
    const cap = Math.round(r.cap_m) + 10, spec = rsHealSpec();
    spec.rounds = { ...(r.spec || {}), round_m: (r.spec && r.spec.round_m) || hs.roundM, max_tokens_m: cap, extend: true };
    rs.hlRun = { running: true, step: "cache", text: "", progress: null, live: [] }; rs.ok = null; rs.error = null; rs.log = []; render();
    try { await invoke("heal_start", { name: rs.hl.name, spec }); } catch (e) { rs.hlRun.running = false; rs.error = String(e); render(); }
  });
  on("hl-go", async () => {
    rs.hlRun = { running: true, step: "cache", text: "", progress: null, live: [] }; rs.ok = null; rs.error = null; rs.log = []; render();
    try {
      if (rs.hl.pair) await invoke("heal_source_start", { student: rs.hl.pair.student, teacher: rs.hl.pair.teacher, spec: rsHealSpec() });
      else await invoke("heal_start", { name: rs.hl.name, spec: rsHealSpec() });
    } catch (e) { rs.hlRun.running = false; rs.error = String(e); render(); }
  });
  if (rs.hl && rs.hl.opts && hs.general) gcWire("heal", render);
  on("hl-usedefault", () => { if (gc.state) { gcScope("heal").sel = gc.state.default; gcScope("heal").check = null; gcPersist("heal"); render(); } });
  on("hl-stop", () => { if (rs.hlRun) rs.hlRun.text = "Stopping…"; invoke("heal_stop"); render(); });
  on("rs-cleardrop", () => { rs.dropText = ""; render(); });
  wireHealSection();
  wireThink();
  wireTrace();
  wireAuto();
  on("rs-open-auto", () => go("auto"));
  wireTry();
  rsxWire();
  document.querySelectorAll("[data-rstrain]").forEach((b) => b.onclick = () => { tr.model = b.dataset.rstrain; tr.over = {}; tr.plan = null; go("train"); });
}

// ------------------------------------------------------------------------------------------------------------ guide
let gd = { html: null, report: null, copied: false };
function renderGuide() {
  if (gd.html == null) return `<h1>Guide</h1><p class="lead">Loading…</p>`;
  return `<div class="guide">${gd.html}</div>
    <div class="card wide"><h2>Something is wrong?</h2><p>The help panel on the right lists the usual problems. If it is not there, make a report and send it with your question. It contains no passwords or access keys.</p>
      <div class="row"><button class="primary" id="gd-report">Make a problem report</button><button class="secondary" id="gd-bundle" title="One zip with the report, the last job logs, the settings (without keys), the versions of the training software and the newest Auto-cut, heal and export records: to attach to a message. Keys and your home folder name are taken out.">Save a diagnostics file…</button>${gd.report ? `<button class="secondary" id="gd-copy">${gd.copied ? "Copied" : "Copy to the clipboard"}</button>` : ""}</div>
      ${gd.bundle ? `<p class="ok small" id="gd-bundle-ok">Saved: ${esc(gd.bundle)}</p>` : ""}${gd.bundleErr ? `<div class="err">${esc(gd.bundleErr)}</div>` : ""}${gd.report ? `<textarea id="gd-text" rows="14" readonly>${esc(gd.report)}</textarea>` : ""}</div>`;
}
async function gdLoad() { gd.html = await invoke("help", { key: "guide" }); if (page === "guide") render(); }
function wireGuide() {
  if (page !== "guide") return;
  const on = (id, fn) => { const e = $(id); if (e) e.onclick = fn; };
  on("gd-report", async () => { gd.report = await invoke("problem_report"); gd.copied = false; render(); });
  on("gd-bundle", async () => {
    const dir = await tauri.dialog.open({ directory: true, title: "Choose where to save the diagnostics file" });
    if (!dir) return;
    try { gd.bundle = await invoke("diagnostics_save", { dest: dir }); gd.bundleErr = null; } catch (e) { gd.bundle = null; gd.bundleErr = String(e); }
    render();
  });
  on("gd-copy", async () => { try { await navigator.clipboard.writeText(gd.report); gd.copied = true; } catch (_) { $("gd-text").select(); } render(); });
}

function renderSoon(key) {
  const [t, lead] = TITLES[key];
  return `<h1>${t}</h1><p class="lead">${lead}</p><span class="soon">Coming in a later version. The help panel shows what this step will do.</span>`;
}


// ------------------------------------------------------------------------------------------------------------ Storage (docs/STORAGE.md)
// Where the disk went and a guarded way to give it back. The window only ever holds OPAQUE ids from the latest scan and sends those; the program checks every one again against a fresh scan.
const sg = { items: [], scanning: false, scanned: false, sel: {}, sized: {}, free: null, busy: false, sug: null, err: null, conf: null, run: null, result: null, want: null, total: null };
const stGroups = [["models", "Models in this project"], ["exports", "Exports"], ["heal", "Heal data and caches"], ["downloads", "Downloaded models"], ["leftovers", "Old logs and leftovers of failed jobs"]];
function stBytes(n) {
  if (n == null) return "";
  const u = [["GB", 1e9], ["MB", 1e6], ["KB", 1e3]];
  for (const [l, d] of u) if (n >= d) return (n / d).toFixed(n >= 1e10 ? 0 : 1) + " " + l;
  return n + " B";
}
function stSize(it) { return it.size != null ? stBytes(it.size) + (it.size_cached && !sg.sized[it.id] ? " (earlier measurement)" : "") : sg.sized[it.id] ? "size not available" : "size not measured yet"; }
// the ticked items without the ones that lie inside another ticked item (an export and its 16-bit file): what goes, and how much
function stChosen() {
  const picked = sg.items.filter((i) => sg.sel[i.id] && i.status !== "in_use");
  const inside = (a, b) => a !== b && a.path_short.startsWith(b.path_short) && /[\\/]/.test(a.path_short.charAt(b.path_short.length));
  return picked.filter((a) => !picked.some((b) => inside(a, b)));
}
function stTotals() {
  const ch = stChosen();
  const bytes = ch.reduce((a, i) => a + (i.size || 0), 0);
  return { n: ch.length, bytes, unmeasured: ch.filter((i) => i.size == null).length, model: ch.some((i) => i.model), free: sg.free };
}
function stRefresh() {
  sg.scanning = true; sg.err = null; sg.items = []; sg.sized = {}; sg.sel = {}; sg.conf = null;
  Promise.resolve(invoke("storage_scan")).catch((e) => { sg.scanning = false; sg.err = String(e); if (page === "storage") render(); });
  if (page === "storage") render();
}
function onStorageEvent(p) {
  if (!p) return;
  if (p.kind === "item") { sg.items.push(p.item); }
  else if (p.kind === "size") { const it = sg.items.find((i) => i.id === p.id); if (it) { it.size = p.size; sg.sized[p.id] = true; } }
  else if (p.kind === "progress") { sg.run = { text: p.text, value: p.value }; }
  else if (p.kind === "done" && p.phase === "scan") {
    const d = p.data || {};
    sg.items = d.items || sg.items; sg.sug = d.suggestions || null; sg.free = d.free_gb; sg.busy = !!d.busy; sg.scanning = false; sg.scanned = true; sg.total = d.total_bytes;
    sg.items.forEach((i) => { sg.sized[i.id] = true; });
    if (sg.want) { const w = sg.want; sg.want = null; const it = sg.items.find((i) => i.kind === w.kind && i.label.startsWith(w.name) && i.status !== "in_use"); if (it) { sg.sel[it.id] = true; if (page === "storage") { stAsk(); return; } } else sg.err = `${w.name} was not found (it may be deleted already) or it is in use right now.`; }
  }
  else if (p.kind === "done") { sg.run = null; sg.result = p.data || null; sg.conf = null; sg.sel = {}; if (page === "storage") stRefresh(); return; }
  else if (p.kind === "error") { sg.run = null; sg.err = p.text; sg.conf = null; }
  if (page === "storage") render();
}
function stPinned() {
  const m = rs.state && rs.state.models && rs.state.models.find((x) => x.id === rs.model);
  return m && m.repo ? [m.repo] : [];
}
// asks the program what exactly would go (a dry run) and shows it in the confirmation
async function stAsk(allowPinned) {
  const ids = stChosen().map((i) => i.id);
  if (!ids.length) return;
  sg.err = null;
  try {
    const dry = await invoke("storage_delete", { ids, dryRun: true, confirm: null, pinned: stPinned(), allowPinned: !!allowPinned, token: null });
    sg.conf = { ids, dry, typed: "", allowPinned: !!allowPinned };
  } catch (e) {
    const t = String(e);
    if (/chosen on the Slim or Auto page/.test(t) && !allowPinned && confirm(t + "\n\nDelete it anyway?")) return stAsk(true);
    sg.err = t; sg.conf = null;
  }
  render();
}
async function stGo() {
  const c = sg.conf; if (!c) return;
  const need = c.dry.needs_delete_word;
  if (need && c.typed !== "DELETE") return;
  sg.run = { text: "Starting…", value: 0 };
  try { await invoke("storage_delete", { ids: c.ids, dryRun: false, confirm: c.typed, pinned: stPinned(), allowPinned: c.allowPinned, token: c.dry.token || null }); sg.conf = null; }
  catch (e) { sg.run = null; sg.err = String(e); }
  render();
}
function stRow(it) {
  const chip = { in_use: ["in use", "err"], referenced: ["referenced", "warn"], safe: ["safe", "ok"] }[it.status] || ["", ""];
  const off = it.status === "in_use" || sg.run || sg.busy;
  return `<tr class="st-row st-${esc(it.status)}"><td><input type="checkbox" data-stsel="${esc(it.id)}" ${sg.sel[it.id] ? "checked" : ""} ${off ? "disabled" : ""} aria-label="Delete ${esc(it.label)}"></td>
    <td><strong>${esc(it.label)}</strong> <span class="tag st-chip st-chip-${esc(it.status)}">${chip[0]}</span>${it.link ? ` <span class="tag" title="It is a link: only the link is removed, never what it points to.">link</span>` : ""}
      ${it.from ? `<div class="dim small">${esc(it.from)}</div>` : ""}${it.note ? `<div class="dim small">${esc(it.note)}</div>` : ""}${(it.refs || []).map((r) => `<div class="${it.status === "in_use" ? "warn" : "dim"} small">${esc(r)}</div>`).join("")}
      ${it.suggest ? `<div class="small st-why">Safe to delete: ${esc(it.suggest)}</div>` : ""}<div class="dim small st-path">${esc(it.path_short)}</div></td>
    <td class="num">${esc(stSize(it))}</td><td class="dim small">${esc(it.modified || "")}</td></tr>`;
}
function stConfirm() {
  const c = sg.conf; if (!c) return "";
  const d = c.dry;
  const rows = (d.items || []).map((i) => `<li><strong>${esc(i.label)}</strong> · ${esc(stBytes(i.size) || "size not available")}<div class="dim small">${esc(i.path_short)}</div></li>`).join("");
  return `<div class="card wide st-confirm" id="st-confirm" role="alertdialog" aria-label="Confirm the deletion"><h2>Delete ${(d.items || []).length} item${(d.items || []).length === 1 ? "" : "s"} (${esc(stBytes(d.total_bytes))})?</h2>
    <p class="err"><strong>This is permanent.</strong> The files are not moved to the Recycle Bin (a bin would not free the space). They cannot be brought back.</p>
    <ul>${rows}</ul>${(d.warnings || []).map((w) => `<p class="warn small">${esc(w)}</p>`).join("")}
    ${d.needs_delete_word ? `<p>Type <strong>DELETE</strong> to confirm: <input type="text" id="st-typed" value="${esc(c.typed)}" autocomplete="off" spellcheck="false" style="width:10em"></p>` : ""}
    <div class="row"><button class="primary" id="st-go" ${d.needs_delete_word && c.typed !== "DELETE" ? "disabled" : ""}>Delete permanently</button><button class="secondary" id="st-no">Cancel</button></div></div>`;
}
function renderStorage() {
  const head = `<h1>🧹 ${TITLES.storage[0]}</h1><p class="lead">${TITLES.storage[1]}</p>`;
  if (!state.project) return head + `<p class="lead">Open a project first.</p>`;
  const t = stTotals();
  const msg = sg.err ? errDiv(sg.err) : "";
  const busy = sg.busy ? `<div class="warn">A job is running. Nothing can be deleted until it has ended.</div>` : "";
  const res = sg.result ? `<div class="card wide st-result" id="st-result"><h2>${sg.result.cancelled ? "Stopped" : "Done"}: ${esc(stBytes(sg.result.freed_bytes))} freed</h2>
      ${(sg.result.deleted || []).map((d) => `<div class="small">✓ ${esc(d.label)} (${esc(stBytes(d.freed))})</div>`).join("")}
      ${(sg.result.failed || []).map((f) => `<div class="err small">✕ ${esc(f.label)}: ${(f.errors || []).map((e) => esc(e.path) + " - " + esc(e.error)).join("; ")}</div>`).join("")}
      ${sg.result.free_gb != null ? `<div class="dim small">Free on this drive now: ${(+sg.result.free_gb).toFixed(1)} GB</div>` : ""}<button class="link" id="st-close-result">Close</button></div>` : "";
  const sug = sg.sug && sg.sug.ids && sg.sug.ids.length ? `<div class="card wide st-sug" id="st-sug"><h2>Safe to delete</h2>
      <p>These are only the steps between your models and files (nothing is ticked until you do it). Together they free about <strong>${esc(stBytes(sg.sug.total_bytes))}</strong>.</p>
      ${sg.items.filter((i) => sg.sug.ids.includes(i.id)).map((i) => `<div class="small"><label><input type="checkbox" data-stsel="${esc(i.id)}" ${sg.sel[i.id] ? "checked" : ""} ${sg.run || sg.busy ? "disabled" : ""}> <strong>${esc(i.label)}</strong> (${esc(stSize(i))}): ${esc(i.suggest)}</label></div>`).join("")}
      <div class="row"><button class="secondary" id="st-preset" ${sg.run || sg.busy || !sg.sug.preset.ids.length ? "disabled" : ""} title="Ticks the steps between: the cuts before healing, the teacher's answers, the 16-bit GGUF, the merged models and leftovers. The final healed model and the GGUF files are not ticked.">${esc(sg.sug.preset.label)}</button></div></div>` : "";
  const groups = stGroups.map(([k, name]) => {
    const rows = sg.items.filter((i) => i.group === k);
    if (!rows.length) return "";
    const tot = rows.reduce((a, i) => a + (i.size || 0), 0);
    return `<div class="card wide st-group" data-stgroup="${k}"><h2>${esc(name)} <span class="dim small">${rows.length} · ${esc(stBytes(tot))}</span></h2><table class="tbl st-tbl">${rows.map(stRow).join("")}</table></div>`;
  }).join("");
  const bar = `<div class="card wide st-bar" id="st-bar"><div id="st-total">${t.n ? `Frees <strong>${esc(stBytes(t.bytes))}</strong>${t.unmeasured ? ` (and ${t.unmeasured} not measured yet)` : ""}${t.free != null ? ` · disk free now ${(+t.free).toFixed(1)} GB → ${(+t.free + t.bytes / 1e9).toFixed(1)} GB` : ""}` : "Nothing is ticked."}</div>
      <div class="row"><button class="primary" id="st-del" ${!t.n || sg.run || sg.busy ? "disabled" : ""}>Delete the ticked items…</button><button class="secondary" id="st-rescan" ${sg.scanning || sg.run ? "disabled" : ""}>Scan again</button>
      ${sg.scanning ? `<button class="secondary" id="st-cancel-scan">Stop scanning</button>` : ""}</div>
      ${sg.scanning ? `<div class="stepline" aria-live="polite">Looking at the disk… ${sg.items.length} found; sizes appear as they are measured.</div>` : ""}
      ${sg.run ? `<div class="stepline" aria-live="polite">${esc(sg.run.text)}</div><progress max="1" value="${sg.run.value || 0}"></progress><div class="row"><button class="secondary" id="st-cancel">Cancel the deletion</button></div>` : ""}</div>`;
  return head + msg + busy + res + stConfirm() + bar + sug + groups + (sg.scanned && !sg.items.length ? `<p class="dim">Nothing found.</p>` : "");
}
function wireStorage() {
  if (page !== "storage") return;
  const on = (id, fn) => { const e = $(id); if (e) e.onclick = fn; };
  document.querySelectorAll("[data-stsel]").forEach((b) => b.onchange = () => { if (b.checked) sg.sel[b.dataset.stsel] = true; else delete sg.sel[b.dataset.stsel]; render(); });
  on("st-del", () => stAsk(false));
  on("st-rescan", () => stRefresh());
  on("st-cancel-scan", () => invoke("storage_cancel"));
  on("st-cancel", () => invoke("storage_cancel"));
  on("st-no", () => { sg.conf = null; render(); });
  on("st-go", () => stGo());
  on("st-close-result", () => { sg.result = null; render(); });
  on("st-preset", () => { sg.sel = {}; (sg.sug.preset.ids || []).forEach((id) => { sg.sel[id] = true; }); render(); });
  const ty = $("st-typed"); if (ty) ty.oninput = () => { sg.conf.typed = ty.value; const g = $("st-go"); if (g) g.disabled = ty.value !== "DELETE"; };
}
// a "Delete…" button on another page opens the Storage page with that model or export ticked; nothing goes before the confirmation
function stDeleteFor(kind, name) { sg.want = { kind, name }; sg.result = null; go("storage"); }
try { document.addEventListener("click", (ev) => { const t = ev.target && ev.target.closest && ev.target.closest("[data-stdel]"); if (t) { const [kind, ...rest] = String(t.dataset.stdel).split("|"); stDeleteFor(kind, rest.join("|")); } }); } catch (_) { /* a preview without a window */ }

function render() {
  if (!state) return;
  renderSteps();
  const el = $("page");
  el.innerHTML = page === "welcome" ? renderWelcome() : page === "data" ? renderData() : page === "setup" ? renderSetup() : page === "dataset" ? renderDataset() : page === "train" ? renderTrain() : page === "storage" ? renderStorage() : page === "tune" ? renderTune() : page === "auto" ? renderAuto() : page === "reshape" ? renderReshape() : page === "use" ? renderUse() : page === "imatrix" ? renderImatrix() : page === "quantcheck" ? renderQuantcheck() : page === "test" ? renderTest() : page === "guide" ? renderGuide() : renderSoon(page);
  $("status").textContent = state.project ? `Project: ${state.project.name}  (${state.project.folder})` : "No project open";
  applyTheme();
  $("expert").checked = state.settings.expert_mode;
  $("helptoggle").checked = state.settings.show_help_panel;
  $("layout").classList.toggle("nohelp", !state.settings.show_help_panel);
  wire();
  wireTune();
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
  document.querySelectorAll("[data-fix]").forEach((b) => b.onclick = () => fixDo(b.dataset.fix));
  wireQueue();
  const kd = $("keep-display"); if (kd) kd.onchange = async () => { try { setState(await invoke("set_setting", { key: "keep_display_on", value: kd.checked })); } catch (e) { alert(String(e)); } };
  const nf = $("notify-on"); if (nf) nf.onchange = async () => { try { setState(await invoke("set_setting", { key: "notifications", value: nf.checked })); } catch (e) { alert(String(e)); } };
  const wh = $("webhook-save"); if (wh) wh.onclick = async () => { try { setState(await invoke("webhook_set", { url: ($("webhook-url") || {}).value || "" })); } catch (e) { alert(String(e)); } };
  const ac2 = $("auto-continue"); if (ac2) ac2.onchange = async () => { try { setState(await invoke("set_setting", { key: "auto_continue_interrupted", value: ac2.checked })); } catch (e) { alert(String(e)); } };
  on("choose", async () => {
    const f = await tauri.dialog.open({ multiple: false, title: "Choose a training file", filters: [{ name: "Training bundle", extensions: ["ndtrain"] }, { name: "All files", extensions: ["*"] }] });
    if (f) checkFile(f);
  });
  wireDataset();
  wireTrain();
  wireUse();
  wireImatrix();
  wireQuantcheck();
  wireTest();
  wireReshape();
  wireAutoPage();
  wireStorage();
  wireGuide();
  wireRetrain();
  wireUnsloth();
  wireLlama();
  wireVllm();
  on("recheck", refreshEnv);
  const nu = $("no-unsloth"); if (nu) nu.onchange = () => { env.noUnsloth = nu.checked; };
  on("install", () => startEnv("env_install", { unsloth: !env.noUnsloth }));
  on("selftest", () => startEnv("env_selftest"));
  on("tm-load", () => tmLoad());
  on("kernels", () => startEnv("env_install_kernels"));
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
  if (["setup", "use", "imatrix", "quantcheck"].includes(key)) lcRefresh();
  if (key === "guide") gdLoad();
  if (key === "dataset" && state.bundle) { ds.ok = null; dsRefresh(false); }
  if (key === "train" && state.bundle) trRefresh();
  if ((key === "reshape" || key === "auto") && state.bundle) { rs.ok = null; rs.error = null; if (key === "auto") ac.loaded = false; rsRefresh(); }
  if (key === "storage" && state.project) stRefresh();
  if (key === "tune" && state.bundle) { rsRefresh(); tn.ok = null; tn.err = null; tn.view = null; tn.model = ""; }
  if (key === "use" && state.bundle) usRefresh();
  if (key === "imatrix" && state.bundle) { im.ok = null; im.error = null; imRefresh(); }
  if (key === "quantcheck" && state.bundle) { qk.ok = null; qk.error = null; qkRefresh(); }
  if (key === "test" && state.bundle) { te.ok = null; teRefresh(); }
  if (key === "data" && state.project) { if (!us.state) usRefresh(); rtRefresh(); }
  $("help").innerHTML = await invoke("help", { key: key === "guide" ? "troubleshooting" : key });
  $("help").scrollTop = 0;
}

// A job that has written nothing for a long time (the program sends {kind: "stall", minutes} on the job's own channel): one banner for every page. It only warns.
const STALL_CHANNELS = ["train-event", "reshape-event", "env-event", "export-event", "imatrix-event", "quantcheck-event", "quantfit-event", "unsloth-event", "server-event", "test-event", "teacher-event", "bench-event"];
function onStallEvent(ev) {
  const box = $("stall");
  if (!box || !ev || typeof ev !== "object") return;
  const kind = ev.kind || ev.type;
  if (kind === "stall") {
    const n = Math.max(1, Math.round(Number(ev.minutes) || 0));
    box.textContent = `No sign of progress for ${n} minute${n === 1 ? "" : "s"}. The job may be stuck; Stop is safe.`;
    box.hidden = false;
  } else if (!box.hidden) {
    box.hidden = true;                  // any other event: it is moving again (or it ended)
    box.textContent = "";
  }
}

async function init() {
  setState(await invoke("get_state"));
  // a damaged settings file was set aside (core: settings.rs); say so once
  if (state && state.settings && state.settings.load_note) message = { kind: "err", text: state.settings.load_note };
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
  try { await tauri.event.listen("imatrix-event", (e) => onImatrixEvent(e.payload)); } catch (_) { /* previews */ }
  try { await tauri.event.listen("quantcheck-event", (e) => onQuantcheckEvent(e.payload)); } catch (_) { /* previews */ }
  try { await tauri.event.listen("quantfit-event", (e) => onQuantfitEvent(e.payload)); } catch (_) { /* previews */ }
  try { await tauri.event.listen("train-event", (e) => onTrainEvent(e.payload)); } catch (_) { /* previews */ }
  try { await tauri.event.listen("reshape-event", (e) => onReshapeEvent(e.payload)); } catch (_) { /* previews */ }
  try { await tauri.event.listen("storage-event", (e) => onStorageEvent(e.payload)); } catch (_) { /* previews */ }
  try { await tauri.event.listen("tune-event", (e) => onTuneEvent(e.payload)); } catch (_) { /* previews */ }
  try { await tauri.event.listen("queue-event", () => quLoad()); } catch (_) { /* previews */ }
  quLoad();
  memPoll();
  try { await tauri.event.listen("env-event", (e) => onEnvEvent(e.payload)); } catch (_) { /* events are optional in previews */ }
  for (const ch of STALL_CHANNELS) { try { await tauri.event.listen(ch, (e) => onStallEvent(e.payload)); } catch (_) { /* previews */ } }
  go("welcome");
}
init();
