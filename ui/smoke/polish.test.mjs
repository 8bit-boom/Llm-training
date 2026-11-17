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
  assert.equal(text([step("export", "pending")]), "about 30 min left (done around 12:30)");
  assert.equal(text([step("s1.heal", "pending"), step("export", "pending")]), "about 2 h 30 min to 4 h 30 min left (done around 14:30–16:30)");
  assert.equal(text([step("s1.heal", "running")], { s: 3600, step: "s1.heal", at }), "about 1 h left (done around 13:00)");
  assert.equal(text([]), "");
});

test("a finish time on another calendar day says so (local dates, not 24 hours)", () => {
  const b = makeBrowser();
  const DAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
  const text = (now, lo, hi) => b.ctx.acEtaText({ status: "running", steps: [step("s1.heal", "pending")], plan: { stages: [{ hours: [lo, hi] }] } }, null, now);
  const noon = new Date(2026, 0, 1, 12, 0, 0).getTime();
  assert.equal(text(noon, 1, 3), "about 1 h to 3 h left (done around 13:00–15:00)", "same day: no day word");
  const evening = new Date(2026, 0, 1, 18, 49, 0).getTime();
  assert.equal(text(evening, 3 + 1 / 3, 20), "about 3 h 20 min to 20 h left (done around 22:09–tomorrow 14:49)", "the owner's case: the day is named");
  assert.equal(text(noon, 12.5, 14.5), "about 12 h 30 min to 14 h 30 min left (done around tomorrow 00:30–02:30)", "both ends tomorrow: the day word once");
  const late = new Date(2026, 0, 1, 23, 30, 0).getTime();
  assert.equal(text(late, 1.5, 1.5), "about 1 h 30 min left (done around tomorrow 01:00)", "a single value that crosses midnight");
  const far = new Date(noon + 50 * 3600e3);                 // two days on
  assert.equal(text(noon, 4, 50), `about 4 h to 50 h left (done around 16:00–${DAYS[far.getDay()]} 14:00)`);
  assert.equal(text(noon, 20, 30), "about 20 h to 30 h left (done around tomorrow 08:00–18:00)");
});

test("the time left is told in hours and minutes without false precision", () => {
  const b = makeBrowser();
  const noon = new Date(2026, 0, 1, 12, 0, 0).getTime();
  const left = (h) => b.ctx.acEtaText({ status: "running", steps: [step("s1.heal", "pending")], plan: { stages: [{ hours: [h, h] }] } }, null, noon).replace(/ \(done.*$/, "");
  assert.equal(left(0.75), "about 45 min left");
  assert.equal(left(0.02), "about 5 min left", "never claims less than 5 minutes");
  assert.equal(left(1), "about 1 h left");
  assert.equal(left(2 + 5 / 60), "about 2 h 05 min left");
  assert.equal(left(2 + 7 / 60), "about 2 h 05 min left", "rounded to 5 minutes");
  assert.equal(left(10 + 10 / 60), "about 10 h left", "from 10 hours on to 30 minutes");
  assert.equal(left(10.3), "about 10 h 30 min left");
  assert.equal(left(20), "about 20 h left");
  // nothing for a finished run
  assert.equal(b.ctx.acEtaText({ status: "done", steps: [] }, null, noon), "");
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

test("the setup page offers the fast-kernels button only when that light is not green", () => {
  const b = makeBrowser();
  const page = (checks, installed = true) => {
    b.run(`state = {project: {name: "P", folder: "/p"}, steps: [], bundle: {world: "W"}, settings: {theme: "light", expert_mode: false}, worlds: [], recent: []}; page = "setup"; env.status = ${JSON.stringify({ checks, can_install: true, env_installed: installed })}; lc.status = null;`);
    return b.run("renderSetup()");
  };
  const k = (light) => ({ id: "kernels", label: "Fast kernels for Qwen3.5 linear attention", light, detail: "d", fix: "f" });
  assert.match(page([k("warn")]), /id="kernels"[^>]*>Install the fast kernels<\/button>/);
  assert.doesNotMatch(page([k("ok")]), /id="kernels"/);
  assert.doesNotMatch(page([]), /id="kernels"/, "no light, no button");
  assert.doesNotMatch(page([k("warn")], false), /id="kernels"/, "nothing to add to before the install");
});
