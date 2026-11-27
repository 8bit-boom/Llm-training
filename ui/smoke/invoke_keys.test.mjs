// Every invoke call of app.js against the Rust command signatures (src-tauri/src/lib.rs): Tauri 2 wants the camelCase of each parameter name, ignores unknown keys, and gives a missing Option
// parameter the value None. A mis-spelled `dry_run` once turned a preview into a real deletion; the smoke tests' stub never saw it. This reads both sides, so it cannot happen again.
import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { rustCommands, checkArgs, invokeCalls } from "./_tauri_check.mjs";

const here = path.dirname(fileURLToPath(import.meta.url));
const js = fs.readFileSync(path.join(here, "..", "app.js"), "utf8");

test("the parser finds the commands and their parameters", () => {
  const c = rustCommands();
  assert.ok(Object.keys(c).length > 100);
  assert.deepEqual(c.heal_speed_start.map((p) => p.key), ["name", "spec", "withTeacher"]);
  assert.ok(c.storage_delete.find((p) => p.key === "dryRun").required, "dry_run is a required bool, not an Option");
  assert.ok(c.storage_delete.find((p) => p.key === "allowPinned").required && c.storage_delete.find((p) => p.key === "pinned").required);
});

test("every invoke call in app.js uses the camelCase keys of a parameter and gives every required one", () => {
  const calls = invokeCalls(js);
  assert.ok(calls.length > 150, `only ${calls.length} calls were found`);
  const problems = [];
  for (const c of calls) {
    if (c.keys === null) { problems.push(`line ${c.line}: ${c.cmd}: the arguments are not a literal object (spell them out, or build a literal \`const args = {...}\` above)`); continue; }
    for (const p of checkArgs(c.cmd, Object.fromEntries(c.keys.map((k) => [k, 1])))) {
      if (c.spread && /required parameter/.test(p)) continue;          // a spread may bring the key
      problems.push(`line ${c.line}: ${p}`);
    }
  }
  assert.deepEqual(problems, []);
});

test("the checker catches the two mistakes that happened", () => {
  assert.ok(checkArgs("heal_speed_start", { name: "m", spec: {}, with_teacher: true }).length === 2);
  assert.ok(checkArgs("storage_delete", { ids: [], dry_run: true, allow_pinned: false }).length >= 3);
  assert.deepEqual(checkArgs("storage_delete", { ids: [], dryRun: true, pinned: [], allowPinned: false }), []);
});

test("the tokens and flags of the destructive commands are never an Option with a permissive default", () => {
  const c = rustCommands();
  for (const [cmd, keys] of Object.entries({ storage_delete: ["ids", "dryRun", "pinned", "allowPinned"], imatrix_delete: ["id"], quantcheck_delete: ["id"], quantcheck_delete_logits: ["id"], reshape_remove: ["name"], model_remove: ["id"] }))
    for (const k of keys) assert.ok(c[cmd].find((p) => p.key === k).required, `${cmd}.${k} must be required`);
});

test("the Hugging Face token commands: `token` for the save, nothing for the clear, and the Settings card calls them with those keys", () => {
  const c = rustCommands();
  assert.deepEqual(c.hf_token_set.map((p) => p.key), ["token"]);
  assert.ok(c.hf_token_set[0].required);
  assert.deepEqual(c.hf_token_clear, []);
  const calls = invokeCalls(js).filter((x) => x.cmd === "hf_token_set" || x.cmd === "hf_token_clear");
  assert.ok(calls.length >= 2, "the card saves and clears with literal keys");
  for (const x of calls) assert.deepEqual(checkArgs(x.cmd, Object.fromEntries((x.keys || []).map((k) => [k, 1]))), [], `line ${x.line}`);
  assert.deepEqual(checkArgs("hf_token_set", { token: "hf_x" }), []);
  assert.ok(checkArgs("hf_token_set", { hf_token: "hf_x" }).length >= 1, "a wrong key is caught");
});
