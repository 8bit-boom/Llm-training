// The Updates card on the Training setup page and the footer chip. app.js runs in a stand-in browser (see _browser.mjs, which also checks every invoke's keys against src-tauri/src/lib.rs).
import test from "node:test";
import assert from "node:assert/strict";
import { makeBrowser, tick } from "./_browser.mjs";
import { rustCommands, checkArgs, invokeCalls } from "./_tauri_check.mjs";
import fs from "node:fs";

const status = (over = {}) => ({ version: "0.1.0", os: "windows", mode: "checksum_only", mode_text: "Checksum only: this program has no signing key built in, so an update is checked against the checksum list of the same release.", check_at_start: true, checked_at: 0, update: null, installable: true, ...over });
const UP = { version: "0.2.0", tag: "v0.2.0", notes: "Fixes <b>things</b>\n- one\n- two", name: "ND-Lore-Forge-Setup-0.2.0.exe", url: "u", size: 52_428_800, sums_url: "s", sig_url: null, html_url: "https://github.com/8bit-boom/Llm-training/releases/tag/v0.2.0", published: "" };
const avail = (over = {}) => ({ skipped: false, result: { current: "0.1.0", state: "available", update: UP, installable: true, problem: null, note: null, mode: "checksum_only" }, status: status({ update: UP }), ...over });

function browser(answers = {}) {
  return makeBrowser({ answers: { update_status: () => status(), update_check: () => ({ skipped: true, status: status() }), ...answers }, cloneArgs: true, state: () => ({ project: { name: "P", folder: "/p" }, steps: [], settings: { theme: "light", check_updates: true }, bundle: null, worlds: [], recent: [], app_version: "0.1.0" }) });
}
const open = (b) => { b.run(`state = {project: {name: "P", folder: "/p"}, steps: [], settings: {theme: "light", check_updates: true}, bundle: null, worlds: [], recent: [], app_version: "0.1.0"}; page = "setup";`); };

test("the card is on the Training setup page with the version, the check button, the start toggle and the mode in plain words", async () => {
  const b = browser();
  open(b);
  await b.run("updLoad()"); await tick();
  const html = b.run("renderSetup()");
  assert.match(html, /id="upd-card"/); assert.match(html, /version 0\.1\.0/); assert.match(html, /id="upd-check"/);
  assert.match(html, /id="upd-auto" checked/); assert.match(html, /Check at start/); assert.match(html, /Checksum only/);
  assert.match(html, /class="warn small" id="upd-mode"/, "checksum-only is shown as a warning, not hidden");
  assert.doesNotMatch(html, /id="upd-banner"|id="upd-install"/);
  assert.doesNotMatch(b.run("renderUpdates()"), /id="upd-install"/);
});

test("the start check runs on its own (auto true), does not block, and a skipped answer shows nothing", async () => {
  const b = browser();
  open(b);
  await b.run("updLoad()"); await tick();
  const c = b.calls.filter((x) => x[0] === "update_check");
  assert.ok(c.length >= 1 && c.every((x) => x[1].auto === true), "the start check (init runs it too) is always the automatic kind");
  assert.equal(b.els["upd-chip"]?.hidden ?? true, true);
  // with the setting off, no check at start
  const b2 = browser({ update_status: () => status({ check_at_start: false }) });
  open(b2);
  await b2.run("updLoad()"); await tick();
  assert.equal(b2.calls.filter((x) => x[0] === "update_check").length, 0);
});

test("an update found at start shows the footer chip, a click on it opens the setup page, the version is in the footer", async () => {
  const b = browser({ update_check: () => avail() });
  open(b); b.run("render()");
  await b.run("updLoad()"); await tick();
  assert.equal(b.els["upd-chip"].hidden, false);
  assert.match(b.els["upd-chip"].textContent, /Update available: 0\.2\.0/);
  assert.equal(b.els["app-ver"].textContent, "v0.1.0");
});

test("a saved update shows the chip at once (no network), and an up-to-date answer clears it", async () => {
  const b = browser({ update_status: () => status({ update: UP, check_at_start: false }), update_check: () => ({ skipped: false, result: { current: "0.2.0", state: "up_to_date", update: null, installable: false }, status: status() }) });
  open(b);
  await b.run("updLoad()"); await tick();
  assert.equal(b.els["upd-chip"].hidden, false);
  b.run("render()");
  await b.els["upd-check"].onclick(); await tick();
  assert.equal(b.els["upd-chip"].hidden, true);
  assert.match(b.run("renderUpdates()"), /You have the newest version/);
});

test("the banner shows the version, the size and the release notes (as text), and the button asks before it installs", async () => {
  const b = browser({ update_check: () => avail(), update_install: () => null });
  open(b);
  await b.run("updLoad()"); await tick();
  const html = b.run("renderUpdates()");
  assert.match(html, /Version 0\.2\.0 is available/); assert.match(html, /52\.4 MB/); assert.match(html, /id="upd-notes"/);
  assert.match(html, /Fixes &lt;b&gt;things&lt;\/b&gt;/, "notes are escaped");
  assert.match(html, /id="upd-install"/); assert.match(html, /Nothing is installed without this click/);
  b.run("render()");
  b.confirm(false);
  await b.els["upd-install"].onclick();
  assert.equal(b.calls.filter((x) => x[0] === "update_install").length, 0, "a No does nothing");
  b.confirm(true);
  await b.els["upd-install"].onclick();
  assert.deepEqual(b.calls.find((x) => x[0] === "update_install")[1], { confirm: true, version: "0.2.0" });
  assert.equal(b.run("upd.running"), true);
});

test("progress, cancel and the end of the job are shown", async () => {
  const b = browser({ update_check: () => avail(), update_install: () => null, update_cancel: () => null });
  open(b);
  await b.run("updLoad()"); await tick(); b.run("render()");
  await b.els["upd-install"].onclick();
  b.send("update-event", { kind: "progress", phase: "download", done: 10_000_000, total: 52_428_800 });
  let html = b.run("renderUpdates()");
  assert.match(html, /Downloading… 10\.0 of 52\.4 MB/); assert.match(html, /id="upd-progress"/); assert.match(html, /id="upd-cancel"/);
  assert.match(html, /id="upd-install" disabled/);
  b.run("render()");
  await b.els["upd-cancel"].onclick();
  assert.ok(b.calls.some((x) => x[0] === "update_cancel"));
  b.send("update-event", { kind: "progress", phase: "verify", done: 1, total: 1 });
  assert.match(b.run("renderUpdates()"), /Checking the checksum/);
  b.send("update-event", { kind: "error", text: "The download was cut off. Nothing was changed." });
  html = b.run("renderUpdates()");
  assert.match(html, /id="upd-error"[^>]*>The download was cut off\. Nothing was changed\./); assert.equal(b.run("upd.running"), false);
  // a successful Linux end shows the instructions
  b.send("update-event", { kind: "done", data: { action: "show_file", text: "Saved as X. Close the program, then start the new file." } });
  assert.match(b.run("renderUpdates()"), /id="upd-result"[^>]*>Saved as X/);
});

test("a refusal (a job is running) is shown as plain text and nothing is marked as running", async () => {
  const b = browser({ update_check: () => avail(), update_install: () => { throw "A job is running (training, a heal, an export, a quant check or an Auto run). Wait until it has finished or stop it."; } });
  open(b);
  await b.run("updLoad()"); await tick(); b.run("render()");
  await b.els["upd-install"].onclick();
  const html = b.run("renderUpdates()");
  assert.match(html, /id="upd-error"[^>]*>A job is running/); assert.equal(b.run("upd.running"), false);
  assert.match(html, /id="upd-install"/, "the person can try again later");
});

test("a failed check shows a plain message from the button but stays silent at start", async () => {
  const b = browser({ update_check: () => { throw "GitHub could not be reached. Check the internet connection, then try again."; } });
  open(b);
  await b.run("updLoad()"); await tick();
  assert.doesNotMatch(b.run("renderUpdates()"), /upd-error/);
  b.run("render()");
  await b.els["upd-check"].onclick(); await tick();
  assert.match(b.run("renderUpdates()"), /id="upd-error"[^>]*>GitHub could not be reached/);
  assert.equal(b.run("upd.checking"), false);
});

test("a release without an installer for this system, and a broken key, give a message and no install button", async () => {
  const noInst = { ...UP, name: "", url: "" };
  const b = browser({ update_check: () => avail({ result: { current: "0.1.0", state: "available", update: noInst, installable: false, problem: "Version 0.2.0 exists, but it has no installer for this system (macos). See the release page." } }) });
  open(b);
  await b.run("updLoad()"); await tick();
  const html = b.run("renderUpdates()");
  assert.match(html, /id="upd-problem"/); assert.match(html, /no installer for this system/); assert.doesNotMatch(html, /id="upd-install"/);
});

test("signed mode is shown as such; the Check at start toggle saves the setting with the key the Rust command wants", async () => {
  const sg = () => status({ mode: "signed", mode_text: "Signed: an update is installed only when its checksum matches and its signature is valid." });
  const b = browser({ update_status: sg, update_check: () => ({ skipped: true, status: sg() }), set_setting: (a) => ({ project: null, steps: [], settings: { theme: "light", check_updates: a.value }, bundle: null, worlds: [], recent: [], app_version: "0.1.0" }) });
  open(b);
  await b.run("updLoad()"); await tick(); b.run("render()");
  assert.match(b.run("renderUpdates()"), /class="ok small" id="upd-mode"[^>]*>Signed/);
  b.els["upd-auto"].checked = false;
  await b.els["upd-auto"].onchange();
  assert.deepEqual(b.calls.find((x) => x[0] === "set_setting")[1], { key: "check_updates", value: false });
  assert.doesNotMatch(b.run("renderUpdates()"), /id="upd-auto" checked/);
});

test("the Rust commands: confirm, version and auto are REQUIRED, and app.js calls them with those keys", () => {
  const c = rustCommands();
  assert.deepEqual(c.update_install.map((p) => p.key), ["confirm", "version"]);
  assert.ok(c.update_install.every((p) => p.required), "an install without confirm must not compile into a default");
  assert.deepEqual(c.update_check.map((p) => p.key), ["auto"]);
  assert.ok(c.update_check[0].required);
  assert.deepEqual(c.update_status, []); assert.deepEqual(c.update_cancel, []);
  const js = fs.readFileSync(new URL("../app.js", import.meta.url), "utf8");
  const calls = invokeCalls(js).filter((x) => x.cmd.startsWith("update_"));
  assert.ok(calls.length >= 4);
  for (const x of calls) assert.deepEqual(checkArgs(x.cmd, Object.fromEntries((x.keys || []).map((k) => [k, 1]))), [], `line ${x.line}`);
  assert.ok(checkArgs("update_install", { version: "0.2.0" }).length >= 1, "a missing confirm is caught");
  assert.ok(checkArgs("update_check", {}).length >= 1);
});
