// What Tauri 2 does with the arguments of `invoke(cmd, args)`, as a function: every top-level key must be the camelCase of a parameter name of the Rust command, and every REQUIRED
// (not Option) parameter must be there. A wrong key is NOT an error in Tauri when the parameter is an Option (it arrives as None): that is how a preview once became a real deletion.
// `rustCommands()` reads src-tauri/src/lib.rs; `checkArgs(cmd, args)` returns a list of problems (empty: fine); `invokeCalls(source)` finds the invoke calls of a JS text.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
export const libPath = path.join(here, "..", "..", "src-tauri", "src", "lib.rs");
export const camel = (s) => s.replace(/_([a-z0-9])/g, (_, c) => c.toUpperCase());

function splitTop(text) {
  const parts = []; let depth = 0, cur = "";
  for (const ch of text) {
    if ("<([{".includes(ch)) depth++;
    if (">)]}".includes(ch)) depth--;
    if (ch === "," && depth === 0) { parts.push(cur); cur = ""; } else cur += ch;
  }
  if (cur.trim()) parts.push(cur);
  return parts;
}

let cache = null;
export function rustCommands() {
  if (cache) return cache;
  const src = fs.readFileSync(libPath, "utf8");
  const out = {};
  const re = /#\[tauri::command[^\]]*\]\s*(?:\/\/[^\n]*\n\s*|\/\*[\s\S]*?\*\/\s*|#\[[^\]]*\]\s*)*(?:pub\s+)?(?:async\s+)?fn\s+(\w+)\s*(?:<[^>]*>)?\s*\(/g;
  let m;
  while ((m = re.exec(src))) {
    let i = re.lastIndex, depth = 1;
    while (depth > 0 && i < src.length) { if (src[i] === "(") depth++; else if (src[i] === ")") depth--; i++; }
    const params = [];
    for (const p of splitTop(src.slice(re.lastIndex, i - 1))) {
      const mm = p.trim().match(/^(?:mut\s+)?(\w+)\s*:\s*([\s\S]+)$/);
      if (!mm) continue;
      const ty = mm[2].trim();
      if (/\b(State|AppHandle|Window|WebviewWindow|Webview|tauri::ipc)\b/.test(ty)) continue;
      params.push({ name: mm[1], key: camel(mm[1]), required: !/^Option\s*</.test(ty), type: ty });
    }
    out[m[1]] = params;
  }
  cache = out;
  return out;
}

export function checkArgs(cmd, args) {
  const cmds = rustCommands();
  if (!(cmd in cmds)) return [`${cmd}: no such command in lib.rs`];
  const params = cmds[cmd], keys = args && typeof args === "object" ? Object.keys(args) : [], problems = [];
  for (const k of keys) if (!params.some((p) => p.key === k)) problems.push(`${cmd}: the key \`${k}\` is not a parameter (${params.map((p) => p.key).join(", ") || "none"}); Tauri ignores it`);
  for (const p of params) if (p.required && !keys.includes(p.key)) problems.push(`${cmd}: the required parameter \`${p.key}\` is missing`);
  return problems;
}

// ---- the invoke calls of a JS text: [{cmd, keys: [..] | null (not a literal), line, spread: bool}]
export function invokeCalls(source) {
  const calls = [];
  // the helpers that forward `(cmd, args)` to invoke are calls too: `call("name", {...})`
  const wrappers = new Set(["invoke"]);
  for (const w of source.matchAll(/(?:function\s+(\w+)|(?:const|let)\s+(\w+)\s*=\s*(?:async\s*)?)\(\s*cmd\b/g)) wrappers.add(w[1] || w[2]);
  const re = new RegExp(`\\b(?:${[...wrappers].join("|")})\\(\\s*(["'])(\\w+)\\1\\s*(,|\\))`, "g");
  let m;
  while ((m = re.exec(source))) {
    const line = source.slice(0, m.index).split("\n").length;
    if (m[3] === ")") { calls.push({ cmd: m[2], keys: [], line, spread: false }); continue; }
    let i = re.lastIndex;
    while (/\s/.test(source[i])) i++;
    if (source[i] !== "{") {
      const id = (source.slice(i).match(/^([A-Za-z_$][\w$]*)\s*\)/) || [])[1];            // `invoke("x", args)`: the literal `const args = {...}` above and its `args.key =` lines
      const before = source.slice(0, m.index).split("\n").slice(-40).join("\n");
      const decl = id && before.match(new RegExp(`(?:const|let)\\s+${id}\\s*=\\s*\\{`));
      if (decl) {
        const at = source.slice(0, m.index).lastIndexOf(decl[0]);
        const { keys, spread } = objectKeys(source, at + decl[0].length - 1);
        for (const a of source.slice(at, m.index).matchAll(new RegExp(`\\b${id}\\.(\\w+)\\s*=`, "g"))) keys.push(a[1]);
        calls.push({ cmd: m[2], keys, line, spread });
      } else calls.push({ cmd: m[2], keys: null, line, spread: false });
      continue;
    }
    const { keys, spread } = objectKeys(source, i);
    calls.push({ cmd: m[2], keys, line, spread });
  }
  return calls;
}

function objectKeys(s, start) {
  const keys = []; let spread = false;
  let i = start + 1, depth = 0, expectKey = true, tok = "";
  const flush = (end) => { // a shorthand key (`name,` / `name }`) ends here
    if (expectKey && /^[A-Za-z_$][\w$]*$/.test(tok.trim())) keys.push(tok.trim());
    tok = "";
  };
  while (i < s.length) {
    const ch = s[i];
    if (ch === '"' || ch === "'" || ch === "`") {
      const q = ch; let j = i + 1, str = "";
      while (j < s.length && s[j] !== q) { if (s[j] === "\\") j++; else if (q === "`" && s[j] === "$" && s[j + 1] === "{") { let d = 1; j += 2; while (j < s.length && d) { if (s[j] === "{") d++; else if (s[j] === "}") d--; j++; } continue; } str += s[j]; j++; }
      if (depth === 0 && expectKey && q !== "`") { let k = j + 1; while (/\s/.test(s[k])) k++; if (s[k] === ":") { keys.push(str); expectKey = false; i = k + 1; tok = ""; continue; } }
      i = j + 1; continue;
    }
    if (ch === "/" && s[i + 1] === "/") { while (i < s.length && s[i] !== "\n") i++; continue; }
    if ("([{".includes(ch)) { if (depth === 0 && ch === "[" && expectKey) keys.push("[computed]"); depth++; i++; continue; }
    if (")]}".includes(ch)) {
      if (depth === 0) { flush(); break; }
      depth--; i++; continue;
    }
    if (depth === 0) {
      if (ch === ",") { flush(); expectKey = true; i++; continue; }
      if (ch === ":" && expectKey) { keys.push(tok.trim()); tok = ""; expectKey = false; i++; continue; }
      if (ch === "." && s.startsWith("...", i) && expectKey) { spread = true; expectKey = false; i += 3; continue; }
      if (expectKey) tok += ch;
    }
    i++;
  }
  return { keys, spread };
}
