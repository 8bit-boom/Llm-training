// "Upgrade transformers": the button next to the software finding, the one on the failure message, the "Python packages" card on the Training setup page, the confirmation dialog, progress, result and "Go back".
import test from "node:test";
import assert from "node:assert/strict";
import { makeBrowser, baseState, tick } from "./_browser.mjs";
import { rustCommands, checkArgs } from "./_tauri_check.mjs";

const plan = (over = {}) => ({ package: "transformers", target: "transformers>=5.24.0,<6", min: "5.24.0", python: "/env/bin/python", env_dir: "/env", installer: "uv pip install", current: "5.17.0", needed: true, ready: true, problem: null,
  previous: null, can_go_back: false, will_do: ["Install transformers>=5.24.0,<6 into the program's own Python environment: /env/bin/python", "Qwen and Gemma keep working: the newer transformers only adds the Inkling model type."], go_back_to: null, ...over });
const finding = (status) => ({ tool: "transformers", need: "5.24.0 or newer", have: "5.17.0", status, message: "old", fix: "do this", command: "uv pip install \"transformers>=5.24.0,<6\"", rollback: null, blocks: "all" });
const moeInspect = { model_type: "inkling2_loop", supported: true, layers: 24, layer_types: [], period: 6, blocks: 4, total_params: 26.4e9, has_vision: true, moe: { arch: "inkling2_loop", experts: 128, top_k: 8, passes: 3, varloop: false, core: [6, 18], visits: 48, active_params: 5.4e9, packed_bytes: 19.5e9, sizes: [] } };

function open(answers = {}) {
  const b = makeBrowser({ cloneArgs: true, answers: { transformers_upgrade_plan: () => plan(), moe_compat: { family: "inkling2", findings: [finding("too_old")], summary: "Upgrade needed", repo: null }, env_status: { checks: [], can_install: false }, ...answers }, state: () => baseState() });
  b.run(`state = ${JSON.stringify(baseState())}; page = "reshape"; rs.state = {models: [{id: "inkling-2-mini", name: "I", params_b: 26}], made: [], dataset_version: 1, env_ready: true, free_disk_gb: 500}; rs.model = "inkling-2-mini"; rs.mode = "prune"; rs.options = ${JSON.stringify({ model: { id: "inkling-2-mini" }, mode: "prune", inspect: moeInspect, options: [], default_blocks: 0, load: "auto", vram_gb: 32 })};`);
  return b;
}

test("the Slim page shows the Upgrade button next to the old transformers, and nothing is asked or installed until Confirm", async () => {
  const b = open();
  await b.run("tfLoad(); moeLoad(false)"); await tick();
  const html = b.run("renderReshape()");
  assert.match(html, /id="tf-upgrade"[^>]*>Upgrade transformers</);
  assert.doesNotMatch(html, /id="tf-back"/);
  b.run("render()");
  b.els["tf-upgrade"].onclick(); await tick();
  const dlg = b.run("tfDialogHtml()");
  for (const w of [/transformers&gt;=5\.24\.0,&lt;6/, /\/env\/bin\/python/, /Installed now<\/td><td>5\.17\.0/, /Qwen and Gemma keep working/, /id="tf-confirm"/, /id="tf-dismiss"/]) assert.match(dlg, w);
  assert.doesNotMatch(dlg, /id="tf-confirm" disabled/);
  assert.equal(b.calls.filter((c) => c[0] === "transformers_upgrade_start").length, 0, "the dialog alone installs nothing");
  assert.match(b.els.page.innerHTML, /id="tf-dialog"/);
});

test("Confirm calls the command with confirm: true; events show progress; done shows the changed packages and re-runs the software check", async () => {
  const b = open();
  await b.run("moeLoad(false)"); await tick();
  await b.run(`tfAsk("upgrade")`);
  b.run("render()");
  await b.els["tf-confirm"].onclick(); await tick();
  assert.deepEqual(b.calls.filter((c) => c[0] === "transformers_upgrade_start").map((c) => c[1]), [{ confirm: true }]);
  const before = b.calls.filter((c) => c[0] === "moe_compat").length;
  b.send("env-event", { kind: "step", task: "transformers", index: 3, total: 4, text: "Installing" });
  b.send("env-event", { kind: "log", task: "transformers", text: "uv says hi" });
  const running = b.run("tfDialogHtml()");
  assert.match(running, /Step 3 of 4: Installing/); assert.match(running, /uv says hi/); assert.match(running, /id="tf-cancel"/); assert.doesNotMatch(running, /id="tf-confirm"/);
  b.send("env-event", { kind: "done", data: { task: "transformers", ok: true, text: "transformers 5.24.1 is installed (it was 5.17.0).", changed: [{ name: "transformers", from: "5.17.0", to: "5.24.1" }, { name: "tokenizers", from: "0.21.0", to: "0.30.0" }], rolled_back: false } });
  await tick();
  const done = b.run("tfDialogHtml()");
  assert.match(done, /class="ok" id="tf-result">transformers 5\.24\.1 is installed/); assert.match(done, /tokenizers<\/code> 0\.21\.0 → 0\.30\.0/); assert.match(done, /id="tf-close"/);
  assert.ok(b.calls.filter((c) => c[0] === "moe_compat").length > before, "the software check ran again");
  assert.ok(b.calls.filter((c) => c[0] === "transformers_upgrade_plan").length >= 3, "the plan was read again");
});

test("a failure with the automatic rollback says so plainly; Cancel calls env_cancel; a refusal is shown", async () => {
  const b = open();
  await b.run(`tfAsk("upgrade")`); b.run("render()");
  await b.els["tf-confirm"].onclick(); await tick();
  await b.els["tf-cancel"].onclick(); await tick();
  assert.equal(b.calls.filter((c) => c[0] === "env_cancel").length, 1);
  b.send("env-event", { kind: "done", data: { task: "transformers", ok: false, stopped: true, rolled_back: true, rollback_ok: true, text: "The upgrade was cancelled while it was installing. The previous version, transformers 5.17.0, was put back automatically and checked.", changed: [] } });
  const html = b.run("tfDialogHtml()");
  assert.match(html, /class="err" id="tf-result">The upgrade was cancelled/); assert.match(html, /The previous version was put back and checked/);
  const b2 = open({ transformers_upgrade_start: new Error("A job is running (training). Wait until it has finished") });
  await b2.run(`tfAsk("upgrade")`); b2.run("render()");
  await b2.els["tf-confirm"].onclick(); await tick();
  assert.match(b2.run("tfDialogHtml()"), /A job is running/);
});

test("Go back appears when a record exists, asks to confirm, and sends confirm and the version on the button", async () => {
  const b = open({ transformers_upgrade_plan: () => plan({ current: "5.24.1", needed: false, previous: "5.17.0", can_go_back: true, go_back_to: "transformers==5.17.0" }), moe_compat: { family: "inkling2", findings: [finding("ok")], summary: null, repo: null } });
  await b.run("tfLoad(); moeLoad(false)"); await tick();
  const html = b.run("renderReshape()");
  assert.match(html, /id="tf-back"[^>]*>Go back to 5\.17\.0</); assert.doesNotMatch(html, /id="tf-upgrade"/);
  b.run("render()");
  await b.els["tf-back"].onclick(); await tick();
  assert.match(b.run("tfDialogHtml()"), /Go back to transformers 5\.17\.0/);
  b.run("render()");
  await b.els["tf-confirm"].onclick(); await tick();
  assert.deepEqual(b.calls.filter((c) => c[0] === "transformers_rollback_start").map((c) => c[1]), [{ confirm: true, version: "5.17.0" }]);
});

test("the Training setup page has the Python packages card with the installed and the needed version, Update and Go back", async () => {
  const b = open();
  b.run(`page = "setup"; env.status = {checks: [], can_install: false, env_installed: true}`);
  await b.run("tfLoad()"); await tick();
  let html = b.run("renderSetup()");
  assert.match(html, /Python packages/); assert.match(html, /<strong>5\.17\.0<\/strong>/); assert.match(html, /<strong>5\.24\.0<\/strong> or newer/); assert.match(html, /older than needed/);
  assert.match(html, /id="tf-update" >Update/); assert.doesNotMatch(html, /id="tf-back2"/);
  b.run("render()");
  await b.els["tf-update"].onclick(); await tick();
  assert.match(b.run("tfDialogHtml()"), /id="tf-confirm"/);
  b.run("tf.open = false");
  const b2 = open({ transformers_upgrade_plan: () => plan({ current: "5.24.1", needed: false, previous: "5.17.0", can_go_back: true }) });
  b2.run(`page = "setup"; env.status = {checks: [], can_install: false}`);
  await b2.run("tfLoad()"); await tick();
  html = b2.run("renderSetup()");
  assert.match(html, /new enough/); assert.match(html, /id="tf-update" disabled/); assert.match(html, /id="tf-back2"[^>]*>Go back to 5\.17\.0/);
});

test("the failure message of a job that needs a newer transformers carries the button; other errors do not", async () => {
  const b = open();
  const msg = "Inkling-2-Mini needs the training software updated: transformers 5.24.0 or newer (this computer has 5.17.0). The Slim page shows the command; Qwen and Gemma keep working with the newer version.";
  const html = b.run(`errDiv(${JSON.stringify(msg)})`);
  assert.match(html, /data-fix="tf_upgrade">Upgrade transformers</); assert.match(html, /transformers 5\.24\.0 or newer/);
  assert.doesNotMatch(b.run(`errDiv("The disk is full.")`), /tf_upgrade/);
  await b.run(`fixDo("tf_upgrade")`); await tick();
  assert.equal(b.run("tf.open"), true);
  assert.equal(b.calls.filter((c) => c[0] === "transformers_upgrade_start").length, 0);
});

test("the commands need confirm (and the version for going back), spelled in camelCase", () => {
  const c = rustCommands();
  assert.deepEqual(c.transformers_upgrade_start.map((p) => [p.key, p.required]), [["confirm", true]]);
  assert.deepEqual(c.transformers_rollback_start.map((p) => [p.key, p.required]), [["confirm", true], ["version", true]]);
  assert.deepEqual(c.transformers_upgrade_plan, []);
  assert.ok(checkArgs("transformers_upgrade_start", {}).length >= 1, "a call without confirm is caught");
  assert.ok(checkArgs("transformers_upgrade_start", { confirmed: true }).length >= 1);
  assert.deepEqual(checkArgs("transformers_upgrade_start", { confirm: true }), []);
  assert.deepEqual(checkArgs("transformers_rollback_start", { confirm: true, version: "5.17.0" }), []);
});
