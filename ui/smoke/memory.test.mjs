// The memory panel: the footer numbers, the popover, the confirmation that lists exactly what will go, the stop-the-job tick, before/after, and the real invoke keys (the stub checks them against lib.rs).
import test from "node:test";
import assert from "node:assert/strict";
import { makeBrowser, plain, tick } from "./_browser.mjs";

const status = (over = {}) => ({ gpu: { total_mb: 32607, used_mb: 20000, free_mb: 12607, per_process: false, app_mb: null, others: [] }, ram: { total_mb: 65536, used_mb: 40000, free_mb: 25536, app_mb: 3072 }, containers: [{ name: "ndforge-vllm-1", running: false }], busy: false, wsl: true, ...over });
const planOf = (over = {}) => ({ kind: "gpu", token: "p1", notes: [], others: [{ pid: 777, name: "chrome.exe", used_mb: 900, note: "another program: close it yourself, this program never ends it" }],
  items: [{ id: "m1", kind: "container", label: "Engine container ndforge-vllm-1", detail: "stopped, left behind by an earlier run", refused: null, needs_stop: false }, { id: "m2", kind: "worker", label: "Worker process 200", detail: "idle. About 1000 MB of RAM.", refused: null, needs_stop: false }], ...over });

function make(answers) {
  const b = makeBrowser({ answers });
  return b;
}

test("the footer shows the free numbers and the popover opens with the two buttons; the driver without per-process numbers says so", async () => {
  const b = make({ memory_status: status() });
  await b.run("memPoll()");
  assert.match(b.els["mem-btn"].textContent, /GPU 12\.3 of 31\.8 GB free · RAM 24\.9 of 64\.0 GB free · this app 3\.0 GB/);
  b.run("mem.open = true; renderMemory()");
  const html = b.els["mem-pop"].innerHTML;
  assert.match(html, /id="mem-gpu"/); assert.match(html, /id="mem-ram"/); assert.match(html, /id="mem-wsl"/);
  assert.match(html, /does not say which program uses how much graphics memory/);
  assert.match(html, /ndforge-vllm-1 \(stopped\)/);
  assert.equal(b.els["mem-pop"].hidden, false);
});

test("Free GPU asks for the plan with the real keys, lists exactly what will go and the other programs read-only, then frees with the token and shows before and after", async () => {
  const b = make({ memory_status: status(), memory_free_plan: planOf(), memory_free: (a) => (a.planToken === "p1" && a.kind === "gpu" && a.stopRunning === false
    ? { kind: "gpu", before: { gpu: { total_mb: 32607, free_mb: 12607 }, ram: { total_mb: 65536, free_mb: 25536 } }, after: { gpu: { total_mb: 32607, free_mb: 30000 }, ram: { total_mb: 65536, free_mb: 25600 } },
        done: [{ label: "Engine container ndforge-vllm-1", freed_mb: 0 }], skipped: [{ label: "Worker process 200", why: "it belongs to the job that is running", needs_stop: true }], failed: [{ label: "Worker process 201", why: "access denied" }], others: [{ pid: 777, name: "chrome.exe", used_mb: 900 }], notes: [] }
    : (() => { throw new Error("wrong arguments"); })()) });
  await b.run("memPoll()");
  await b.run(`memAsk("gpu")`);
  let html = b.els["mem-pop"].innerHTML;
  assert.match(html, /Free the graphics memory this program holds\?/);
  assert.match(html, /Engine container ndforge-vllm-1/); assert.match(html, /Worker process 200/);
  assert.match(html, /chrome\.exe \(pid 777\), 0\.9 GB/); assert.match(html, /never ends them|close them yourself/);
  assert.doesNotMatch(html, /id="mem-stop"/, "no job runs: nothing to stop");
  assert.deepEqual(plain(b.calls.find((c) => c[0] === "memory_free_plan")[1]), { kind: "gpu" });
  await b.run("memGo()");
  html = b.els["mem-pop"].innerHTML;
  assert.match(html, /id="mem-result"/); assert.match(html, /12\.3 of 31\.8 GB free/); assert.match(html, /29\.3 of 31\.8 GB free/);
  assert.match(html, /Left alone: Worker process 200 — it belongs to the job that is running/); assert.match(html, /Could not: Worker process 201 — access denied/);
  assert.match(html, /Still using memory and not ours: chrome\.exe 0\.9 GB/);
  assert.deepEqual(plain(b.calls.find((c) => c[0] === "memory_free")[1]), { kind: "gpu", planToken: "p1", stopRunning: false });
});

test("a running job's items are refused in the list and the person can tick 'Stop the running job too', which is sent as stopRunning", async () => {
  const items = [{ id: "m2", kind: "worker", label: "Worker process 200", detail: "idle.", refused: "it belongs to the job that is running", needs_stop: true }];
  const b = make({ memory_status: status({ busy: true }), memory_free_plan: planOf({ items, others: [] }), memory_free: { kind: "gpu", before: { gpu: null, ram: null }, after: { gpu: null, ram: null }, done: [], skipped: [], failed: [], others: [] } });
  await b.run("memPoll()");
  await b.run(`memAsk("gpu")`);
  assert.match(b.els["mem-pop"].innerHTML, /Not now: it belongs to the job that is running/);
  assert.match(b.els["mem-pop"].innerHTML, /id="mem-stop"/);
  b.run("wireMemory()");
  b.els["mem-stop"].checked = true; b.els["mem-stop"].onchange();
  await b.run("memGo()");
  assert.equal(plain(b.calls.find((c) => c[0] === "memory_free")[1]).stopRunning, true);
});

test("Release WSL memory carries a clear warning, and a refusal from the program is shown as it is", async () => {
  const b = make({ memory_status: status(), memory_free_plan: planOf({ kind: "wsl", others: [], items: [{ id: "m9", kind: "wsl", label: "Release the memory of the WSL virtual machine", detail: "Runs `wsl --shutdown`: this ENDS EVERY WSL distribution.", refused: null, needs_stop: false }] }),
                    memory_free: new Error("The list has changed since it was made. Nothing was done.") });
  await b.run("memPoll()");
  await b.run(`memAsk("wsl")`);
  assert.match(b.els["mem-pop"].innerHTML, /ends every WSL distribution/);
  await b.run("memGo()");
  assert.match(b.els["mem-pop"].innerHTML, /The list has changed since it was made\. Nothing was done\./);
});

test("a vLLM error about memory offers Free GPU first, other errors do not", () => {
  const b = make({});
  assert.match(b.run(`memHint("Free memory on device (15.1/31.8 GiB) is less than desired GPU memory utilization")`), /data-memfree="gpu"/);
  assert.equal(b.run(`memHint("The model name is not valid")`), "");
});
