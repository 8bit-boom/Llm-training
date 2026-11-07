// Takes the guide's screenshots of the real window code (ui/) with the real data of a demo project (see make_guide_screenshots.sh).
// usage: NODE_PATH=$(npm root -g) node scripts/guide_shots.js <states.json> <out dir>
const fs = require('fs');
const { chromium } = require('playwright');
const states = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const out = process.argv[3];
const sc = (n, m) => ({ n, mean: m });
// Illustration only: a test result with example numbers (a real one needs a trained model)
const example_result = { run: "20261107-120000-qwen3.5-4b", created: "2026-11-07T12:30:00Z", dataset_version: 1, n: 54, method: "overlap", judged: 0,
  scores: { original: { lore: sc(30, 0.12), rules: sc(10, 0.1), made_up: sc(8, 0.9), general: sc(6, 1) }, trained: { lore: sc(30, 0.64), rules: sc(10, 0.5), made_up: sc(8, 0.2), general: sc(6, 0.83) } },
  verdict: { headline: "Better than the original on 86 % of lore questions (worse on 3 %).", better: true, lines: ["Rules questions: better on 70 % of 10.", "General chat: 100 % sensible answers before, 83 % after."] },
  questions: [{ id: "q1", category: "lore", question: "Who is Anders de Vallière?", reference: "Anders is a Widow-Bound Hunter.", original: "I am not sure who that is.", trained: "Anders de Vallière is a Widow-Bound Hunter, a captain.", score_original: 0, score_trained: 0.8 }] };
(async () => {
  const b = await chromium.launch();
  const pg = await b.newPage({ viewport: { width: 1250, height: 800 } });
  let current = 'state';
  await pg.exposeFunction('__which', () => current);
  await pg.addInitScript(([S, result]) => {
    window.__TAURI__ = { core: { invoke: async (cmd, args) => {
      const w = await window.__which();
      const map = { get_state: S[w === 'welcome' ? 'welcome' : 'state'], dataset_state: S.dataset_state, dataset_examples: S.dataset_examples, train_state: S.train_state, train_plan: S.train_plan,
        export_state: S.export_state, test_state: S.test_state, test_result: result, retrain_info: S.retrain_info, env_status: S.env_status, help: S.help[args && args.key] || "" };
      return map[cmd] ?? {};
    } }, event: { listen: async () => {} }, dialog: { open: async () => null } };
  }, [states, example_result]);
  const shot = async (name) => { await pg.waitForTimeout(500); await pg.screenshot({ path: `${out}/${name}.png` }); console.log(name); };
  const open = async (key) => { await pg.click(`[data-key="${key}"]`); };
  current = 'welcome';
  await pg.goto('file://' + process.cwd() + '/ui/index.html');
  await shot('welcome');
  current = 'state'; await pg.reload();
  await open('data'); await shot('data');
  await open('dataset'); await shot('dataset');
  await open('train'); await pg.waitForTimeout(300); await pg.click('[data-model="qwen3.5-4b"]').catch(() => {}); await shot('train');
  await open('test'); await shot('test');
  await open('use'); await shot('use');
  await open('setup'); await shot('setup');
  await b.close();
})();
