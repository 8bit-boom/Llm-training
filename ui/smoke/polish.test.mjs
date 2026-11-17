// Smoke tests of the polish items: the Auto-cut finish time, the live regions, and that every command the window calls exists in the program. Run with `node --test ui/smoke`.
import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { makeBrowser, source } from "./_browser.mjs";

const here = path.dirname(fileURLToPath(import.meta.url));
const step = (id, status) => ({ id, status });
const run = (steps) => ({ status: "running", steps, plan: { stages: [{ hours: [2, 4] }, { hours: [1, 3] }] } });

test("the finish time is the rest of the plan's hours plus fixed allowances for the checks and the export", () => {
  const b = makeBrowser();
  const r = (steps, eta, now) => JSON.parse(JSON.stringify(b.ctx.acEtaRange(run(steps), eta, now)));
  const all = [step("measure", "done"), step("s1.cut", "done"), step("s1.check_cut", "done"), step("s1.heal", "pending"), step("s1.check_heal", "pending"), step("s2.cut", "pending"), step("s2.check_cut", "pending"), step("s2.heal", "pending"), step("s2.check_heal", "pending"), step("export", "pending"), step("check_gguf", "pending")];
  // heal 1: 2-4 h, heal 2: 1-3 h, four checks left (1 h), the export (0.5 h), the gguf check (0.25 h)
  assert.deepEqual(r(all), { lo: 4.5, hi: 8.5 });
  // the running heal reports its own time left: that replaces the plan's hours for it, minus the time that has passed since
  const live = all.map((x) => (x.id === "s1.heal" ? step(x.id, "running") : x));
  const now = 1_000_000;
  const out = r(live, { s: 7200, step: "s1.heal", at: now - 3600e3 }, now);
  assert.deepEqual(out, { lo: 3.5, hi: 5.5 }, "1 h left of the first heal, the second heal's 1-3 h, 4 checks, the export");
  // another step's estimate is not used for this one, and a finished run has none
  assert.equal(b.ctx.acEtaRange({ status: "done", steps: [] }, null), null);
  assert.equal(b.ctx.acEtaRange(run([step("s1.cut", "done")]), null), null);
});

test("the text names the clock times of the range, or one time when there is no range", () => {
  const b = makeBrowser();
  const at = new Date(2026, 0, 1, 12, 0, 0).getTime();
  const text = (steps, eta) => b.ctx.acEtaText(run(steps), eta, at);
  assert.equal(text([step("export", "pending")]), "done around 12:30");
  assert.equal(text([step("s1.heal", "pending"), step("export", "pending")]), "done around 14:30–16:30");
  assert.equal(text([step("s1.heal", "running")], { s: 3600, step: "s1.heal", at }), "done around 13:00");
  assert.equal(text([]), "");
});

test("an Auto-cut event with a time left is remembered for the estimate", () => {
  const b = makeBrowser();
  b.run(`page = "welcome"; ac.running = true; onAutoEvent({ kind: "metric", eta_s: 1800, auto_step: "s1.heal" })`);
  const eta = JSON.parse(JSON.stringify(b.run("ac.eta")));
  assert.equal(eta.s, 1800); assert.equal(eta.step, "s1.heal");
});

test("progress and status lines are live regions", () => {
  const n = (source.match(/class="stepline" aria-live="polite"/g) || []).length;
  assert.ok(n >= 15, `${n} progress lines`);
  assert.doesNotMatch(source, /class="stepline"(?! aria-live)/);
  assert.match(source, /id="ac-eta" aria-live="polite"/);
});

test("every command the window calls is one the program has", () => {
  const lib = fs.readFileSync(path.join(here, "..", "..", "src-tauri", "src", "lib.rs"), "utf8");
  const handler = lib.match(/generate_handler!\[([^\]]*)\]/)[1].split(",").map((s) => s.trim()).filter(Boolean);
  assert.ok(handler.length > 100, "the list of commands was found");
  const called = new Set();
  for (const m of source.matchAll(/\b(?:invoke|call|startEnv)\(\s*"([a-z0-9_]+)"/g)) called.add(m[1]);
  for (const m of source.matchAll(/\busRun\(\s*"[a-z]+",\s*"([a-z0-9_]+)"/g)) called.add(m[1]);
  assert.ok(called.size > 80, `the calls were found: ${called.size}`);
  const missing = [...called].filter((c) => !handler.includes(c));
  assert.deepEqual(missing, [], "commands called by the window that generate_handler! does not list");
  // and the commands the program lists but no window code calls are only the ones another way reaches (kept short on purpose)
  const unused = handler.filter((h) => !called.has(h));
  assert.ok(unused.length < 40, `commands nobody calls: ${unused.join(", ")}`);
});
