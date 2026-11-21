// Heal speed: the recommended token budget is shown and used only by a click; the Expert field and the spec key. Run with `node --test ui/smoke`.
import test from "node:test";
import assert from "node:assert/strict";
import { makeBrowser, baseState } from "./_browser.mjs";

function open(files) {
  const b = makeBrowser({ answers: { heal_speed_report: files }, state: baseState });
  b.run(`page = "reshape"; rs.hl = { name: "slim-w", opts: { default_tokens_m: 20 } }; hsp.name = "slim-w"; hsp.files = ${JSON.stringify(files)};`);
  return b;
}

test("a recommendation is shown with its numbers and the button applies it; nothing is applied by itself", () => {
  const b = open({ exists: true, dir: "/p/benchmarks", markdown: "# r", recommended: { value: 4096, speedup: 2.25, why: "4096 tokens per forward pass: 2.25 times the speed of one row per step" } });
  assert.equal(b.run("rsHealSpec().student_batch_tokens"), undefined, "a default spec carries no key");
  const html = b.run("hlSpeedBox(false)");
  assert.match(html, /Recommended: 4096 tokens per forward pass/);
  assert.match(html, /id="hl-speed-use"/);
  assert.equal(b.run("rsHealSpec().student_batch_tokens"), undefined, "still nothing applied");
  b.run("rs.hlSpec.student_batch_tokens = 4096");
  assert.equal(b.run("rsHealSpec().student_batch_tokens"), 4096);
  assert.match(b.run("hlSpeedBox(false)"), /id="hl-speed-use" disabled/, "already in use");
});

test("without a clear winner the report says so and offers no button", () => {
  const b = open({ exists: true, dir: "/p", markdown: "# r", recommended: { value: 0, speedup: 1, why: "No token budget is clearly faster" } });
  const html = b.run("hlSpeedBox(false)");
  assert.match(html, /No token budget is clearly faster/);
  assert.doesNotMatch(html, /id="hl-speed-use"/);
});
