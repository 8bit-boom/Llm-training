// The app-wide Hugging Face token card on the Training setup page: saved / not saved, the password field, Save, Clear, the notes, and that it shares its saved state with the
// General-text chooser. Only a made-up token is used. app.js runs in a stand-in browser (see _browser.mjs, which also checks every invoke's keys against src-tauri/src/lib.rs).
import test from "node:test";
import assert from "node:assert/strict";
import { makeBrowser, tick } from "./_browser.mjs";

const FAKE = "hf_FAKETOKEN123";
const P = (key) => ({ key, id: key, name: key, license: "MIT", license_note: "", kind: "chat", note: "", config: null, split: "train", gated: false, needs_ack: false, url: "https://huggingface.co/datasets/" + key });
const genState = (over = {}) => ({ default: "a/b", has_token: false, acknowledged: [], custom: [], choice: {}, presets: [P("a/b")], ...over });

function browser(answers = {}) {
  return makeBrowser({ answers, cloneArgs: true, state: () => ({ project: { name: "P", folder: "/p" }, steps: [], settings: { theme: "light" }, bundle: null, worlds: [], recent: [] }) });
}
const open = (b) => { b.run(`state = {project: {name: "P", folder: "/p"}, steps: [], settings: {theme: "light"}, bundle: null, worlds: [], recent: []}; page = "setup";`); };
const card = (b) => b.run("renderHfToken()");

test("the card is on the Training setup page and says what it is for, where to make a token and where it is kept", async () => {
  const b = browser({ general_state: () => genState() });
  open(b);
  assert.match(b.run("renderSetup()"), /id="hf-card"/);
  let html = card(b);
  assert.match(html, /Hugging Face token/); assert.match(html, /\(checking…\)/);
  await tick();                                         // general_state answered
  html = card(b);
  assert.deepEqual(b.calls.filter((c) => c[0] === "general_state").length, 1);
  assert.match(html, /\(not saved\)/); assert.match(html, /type="password" id="hf-token"/); assert.match(html, /id="hf-token-save"/); assert.doesNotMatch(html, /hf-token-clear/);
  assert.match(html, /Inkling-2-Mini/); assert.match(html, /read token is enough/i); assert.match(html, /huggingface\.co\/settings\/tokens/);
  assert.match(html, /only in this program's secrets file/); assert.match(html, /HF_TOKEN/); assert.match(html, /never written into job files, logs, reports or problem reports/);
});

test("Save sends the token once with the key the Rust command wants, shows saved, and never shows the value again", async () => {
  const b = browser({ general_state: () => genState(), hf_token_set: () => genState({ has_token: true }), hf_token_clear: () => genState() });
  open(b); card(b); await tick();
  b.run("render()"); await tick();
  b.els["hf-token"].value = "  " + FAKE + " "; b.els["hf-token"].onchange();
  await b.els["hf-token-save"].onclick();
  assert.deepEqual(b.calls.find((c) => c[0] === "hf_token_set")[1], { token: FAKE });
  const html = card(b);
  assert.match(html, /\(saved\)/); assert.match(html, /hf-token-clear/); assert.ok(!html.includes(FAKE) && !html.includes("FAKETOKEN"));
  assert.ok(!b.run("renderSetup()").includes("FAKETOKEN"));
  // the General-text chooser shows the same saved state
  b.run(`gcUse(gc.state)`);
  assert.match(b.run(`gcHtml("heal")`), /Hugging Face token \(saved\)/);
  // Clear
  b.run("render()");
  await b.els["hf-token-clear"].onclick();
  assert.ok(b.calls.some((c) => c[0] === "hf_token_clear"));
  assert.match(card(b), /\(not saved\)/); assert.match(b.run(`gcHtml("heal")`), /Hugging Face token \(none\)/);
});

test("a token the program refuses is shown as a plain message and nothing is shown as saved", async () => {
  const b = browser({ general_state: () => genState(), hf_token_set: () => { throw "That does not look like a Hugging Face token (it starts with hf_)."; } });
  open(b); card(b); await tick();
  b.run("render()");
  b.els["hf-token"].value = "nope"; b.els["hf-token"].onchange();
  await b.els["hf-token-save"].onclick();
  const html = card(b);
  assert.match(html, /id="hf-error"/); assert.match(html, /starts with hf_/); assert.match(html, /\(not saved\)/);
});

test("saving in the General-text chooser shows up on the card (one saved state)", async () => {
  const b = browser({ general_state: () => genState(), hf_token_set: () => genState({ has_token: true }) });
  open(b);
  b.run(`gcUse(${JSON.stringify(genState())})`);
  b.run(`gcWire("heal", () => {})`); b.run(`gcHtml("heal")`); b.run(`gcWire("heal", () => {})`);
  b.els["gc-token"].value = FAKE; b.els["gc-token"].onchange();
  await b.els["gc-token-save"].onclick();
  assert.match(card(b), /\(saved\)/);
  assert.equal(b.calls.filter((c) => c[0] === "general_state").length, 0, "the card did not need to ask again");
});

test("the card says what the token is for", () => {
  const b = browser();
  open(b);
  assert.match(b.run("renderHfToken()"), /gated or very large model downloads/);
});
