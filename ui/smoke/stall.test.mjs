// Smoke test of the "no sign of progress" banner (any job channel), without a window. Run with `node --test ui/smoke`.
import test from "node:test";
import assert from "node:assert/strict";
import { makeBrowser, plain as plainOf, tick as tickOf } from "./_browser.mjs";


async function browser() {
  const b = makeBrowser({ state: () => ({ project: null, steps: [], bundle: null, settings: { theme: "light", expert_mode: false, show_help_panel: true }, worlds: [], recent: [] }) });
  await new Promise((r) => setTimeout(r, 20));
  return b;
}

test("a stall event on any job channel shows the banner and the next event hides it", async () => {
  const b = await browser();
  for (const ch of ["train-event", "reshape-event", "export-event", "imatrix-event"]) assert.ok((b.listeners[ch] || []).length >= 1, ch);
  assert.equal(b.el("stall").hidden, true);
  b.send("reshape-event", { kind: "stall", minutes: 20 });
  assert.equal(b.el("stall").hidden, false);
  assert.match(b.el("stall").textContent, /^No sign of progress for 20 minutes\. The job may be stuck; Stop is safe\.$/);
  b.send("train-event", { kind: "stall", minutes: 40 });
  assert.match(b.el("stall").textContent, /for 40 minutes/);
  b.send("export-event", { type: "stall", minutes: 1 });                  // a raw worker event has "type" only
  assert.match(b.el("stall").textContent, /for 1 minute\./);
  b.send("train-event", { kind: "progress", step: "x", value: 0.1, text: "" });
  assert.equal(b.el("stall").hidden, true);
  assert.equal(b.el("stall").textContent, "");
});
