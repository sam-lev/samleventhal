// Integration tests for the assembled app logic (DOM-free portion).
// Evaluates app.js up to the THREE section inside a vm sandbox with the
// core, GeoCells and GeoAI modules as globals, then exercises
// buildBoard + deriveBoard across every surface x mesh x incidence combo.
// Run: node test_app.mjs
import { readFileSync } from "fs";
import vm from "vm";
import { createRequire } from "module";
const require = createRequire(import.meta.url);

const core = require("./core.js");
const GeoCells = require("./cells.js");
const GeoAI = require("./ai.js");

const src = readFileSync("app.js", "utf8");
const cut = src.indexOf("// ---------- THREE scene");
if (cut < 0) throw new Error("THREE marker not found in app.js");
const sandbox = { ...core, GeoCells, GeoAI, console };
vm.createContext(sandbox);
vm.runInContext(src.slice(0, cut), sandbox, { filename: "app-head.js" });
const { buildBoard, deriveBoard, SURFACES, INC_OK, INC_MODES } = vm.runInContext(
  "({ buildBoard, deriveBoard, SURFACES, INC_OK, INC_MODES })", sandbox);

let passed = 0, failed = 0;
function ok(cond, name, extra) {
  if (cond) { passed++; console.log(`PASS  ${name}`); }
  else { failed++; console.log(`FAIL  ${name}${extra ? " — " + extra : ""}`); }
}

function checkGraph(B) {
  const n = B.adj.length;
  for (let v = 0; v < n; v++) {
    const seen = new Set();
    for (const u of B.adj[v]) {
      if (u === v) return "self-loop at " + v;
      if (u < 0 || u >= n) return "out of range " + v + "->" + u;
      if (seen.has(u)) return "duplicate " + v + "->" + u;
      seen.add(u);
      if (!B.adj[u].includes(v)) return "asymmetric " + v + "->" + u;
    }
  }
  for (const p of B.pos)
    if (p.length !== 3 || p.some(x => !isFinite(x))) return "bad position";
  return null;
}

function seamCheck(B, tol) {
  // both ends of every rendered edge curve must land exactly on the sites
  let worst = 0;
  for (const [a, b] of B.edges) {
    const pts = B.edgeCurve(a, b, 4);
    for (const [end, site] of [[pts[0], B.pos[a]], [pts[4], B.pos[b]]]) {
      const d = Math.hypot(end[0] - site[0], end[1] - site[1], end[2] - site[2]);
      if (d > worst) worst = d;
    }
  }
  return worst <= tol ? null : "curve endpoint off by " + worst.toExponential(2);
}

// ---------- every combination builds a sound board -----------------------------
{
  let bad = [];
  let combos = 0;
  for (const surf of Object.keys(SURFACES))
    for (const mesh of Object.keys(SURFACES[surf].meshes)) {
      const modes = INC_OK[surf + ":" + mesh] ? INC_MODES : ["vertices"];
      for (const mode of modes) {
        combos++;
        const B = deriveBoard(buildBoard(surf, mesh, 0), mode);
        const g = checkGraph(B) || seamCheck(B, 1e-9);
        if (g) bad.push(`${surf}:${mesh}:${mode} — ${g}`);
        if (mode !== "vertices") {
          const m = B.inc.meta;
          const chi = m.nV - m.nE + m.nF;
          const want = surf === "sphere" ? 2 : surf === "plane" ? 1 : 0;
          if (chi !== want) bad.push(`${surf}:${mesh}:${mode} — chi ${chi}`);
        }
      }
    }
  ok(bad.length === 0, `all ${combos} surface×mesh×incidence combos are sound`,
     bad.join("; "));
}

// ---------- known counts --------------------------------------------------------
{
  const B = deriveBoard(buildBoard("sphere", "tri", 1), "faces"); // f=3
  ok(B.adj.length === 180 && B.adj.every(l => l.length === 3),
     "geodesic f=3 faces mode: 180 deg-3 sites (Goldberg-like dual)",
     "n=" + B.adj.length);
}
{
  const B = deriveBoard(buildBoard("torus", "square", 0), "faces"); // 7×7
  ok(B.adj.length === 49 && B.adj.every(l => l.length === 4),
     "7×7 square torus faces mode: self-dual 49 deg-4 sites");
}
{
  const B = deriveBoard(buildBoard("mobius", "square", 0), "cells"); // 12×5
  const m = B.inc.meta;
  ok(B.adj.length === 60 + 108 + 48 && m.nV === 60 && m.nE === 108 && m.nF === 48,
     "12×5 Möbius cells mode: 216 sites over V60 E108 F48 (χ=0)");
}
{
  const B = deriveBoard(buildBoard("torus", "tri", 0), "edges"); // 7×7 tri
  ok(B.adj.length === 147, "7×7 tri torus edges mode: 147 line-graph sites",
     "n=" + B.adj.length);
}

// ---------- classical plane boards ----------------------------------------------
{
  const B = deriveBoard(buildBoard("plane", "square", 2), "vertices"); // 19×19
  const corners = B.adj.filter(l => l.length === 2).length;
  ok(B.adj.length === 361 && corners === 4 && B.defects.size === 72,
     "19×19 plane: 361 points, 4 corners, brass rim of 72");
}
{
  const B = deriveBoard(buildBoard("plane", "square", 0), "faces"); // 9×9
  const m = B.inc.meta;
  ok(B.adj.length === 64 && m.nV - m.nE + m.nF === 1 &&
     B.adj.filter(l => l.length === 2).length === 4,
     "9×9 plane faces mode: 64 cells, chi = 81 - 144 + 64 = 1");
}
{
  const eng0 = core.Engine(deriveBoard(buildBoard("plane", "square", 2),
                                       "vertices").adj);
  const B = buildBoard("plane", "square", 2);
  const ai = GeoAI.models[0].create(B.adj, { level: "strong", seed: 5 });
  const r = ai.pickMove(new Int8Array(361), 1,
                        { select: { temp: 0, noise: 0 } });
  ok(r.move >= 0 && B.adj[r.move].length >= 3 && !eng0.play(r.move).err,
     "AI opens legally on 19×19, away from the corners (site " + r.move + ")");
}

// ---------- honeycomb and Goldberg incidence ---------------------------------------
{
  const B = deriveBoard(buildBoard("torus", "hex", 0), "edges"); // 8×6 kagome
  ok(B.adj.length === 72 && B.adj.every(l => l.length === 4),
     "8×6 honeycomb torus edges mode: the kagome lattice (72 deg-4 sites)");
}
{
  const B = deriveBoard(buildBoard("torus", "hex", 0), "faces");
  ok(B.adj.length === 24 && B.adj.every(l => l.length === 6),
     "8×6 honeycomb torus faces mode: the triangular dual (24 deg-6 sites)");
}
{
  const B = deriveBoard(buildBoard("sphere", "hex", 0), "faces"); // GP(2) dual
  const d5 = B.adj.filter(l => l.length === 5).length;
  const d6 = B.adj.filter(l => l.length === 6).length;
  ok(B.adj.length === 42 && d5 === 12 && d6 === 30,
     "Goldberg GP(2) faces mode: the geodesic returns (42 sites, 12 pentagons)");
}
{
  ok(!INC_OK["mobius:hex"],
     "Möbius honeycomb stays a vertex board (flip shears the brick parity)");
}

// ---------- Möbius seam geometry: derived sites respect the deck transform ------
{
  const B = deriveBoard(buildBoard("mobius", "tri", 0), "edges");
  ok(seamCheck(B, 1e-9) === null && B.defects.size > 0,
     "Möbius tri edge-sites: curves close across the flip seam; boundary sites brass");
}

// ---------- rules + AI on derived boards -----------------------------------------
{
  const B = deriveBoard(buildBoard("sphere", "tri", 0), "faces"); // 80 sites
  const eng = core.Engine(B.adj, { komi: 7.5 });
  const ai = GeoAI.models[0].create(B.adj, { level: "standard", seed: 7 });
  let legalAll = true, moves = 0;
  for (let t = 0; t < 12 && !eng.gameOver(); t++) {
    const color = eng.toMove;
    const mask = new Uint8Array(B.adj.length);
    for (let v = 0; v < B.adj.length; v++)
      if (eng.colors[v] === 0 && !eng.trySim(v, color).err) mask[v] = 1;
    const r = ai.pickMove(eng.colors, color, { legalMask: mask });
    if (r.move < 0) { eng.pass(); continue; }
    const res = eng.play(r.move);
    if (res.err) { legalAll = false; break; }
    moves++;
  }
  ok(legalAll && moves >= 10,
     `AI vs itself on the geodesic dual stays legal (${moves} moves)`);
}
{
  const B = deriveBoard(buildBoard("torus", "square", 0), "cells"); // 196 sites
  const eng = core.Engine(B.adj, { komi: 7.5 });
  const ai = GeoAI.models[0].create(B.adj, { level: "casual", seed: 3 });
  const mask = new Uint8Array(B.adj.length).fill(1);
  const r = ai.pickMove(eng.colors, 1, { legalMask: mask });
  ok(r.move >= 0 && !eng.play(r.move).err,
     "AI opens legally on the torus Hasse-cells board (196 cross-dimensional sites)");
}

// ---------- models/ folder bundling + supports gating ------------------------------
{
  // replicate the build wrapper for models/random.js and register it
  const before = GeoAI.models.length;
  const wrapped = "(function () {\nconst WEIGHTS = null;\n" +
    readFileSync("models/random.js", "utf8") + "\n})();";
  vm.runInContext(wrapped, sandbox);
  const rand = GeoAI.models.find(m => m.id === "random");
  let legalOk = false;
  if (rand) {
    const N = [[1],[0,2],[1,3],[2,4],[3]];
    const e = rand.create(N, {});
    const stones = new Int8Array(5); stones[2] = 1;
    const mask = new Uint8Array([1,1,0,1,1]);
    const r = e.pickMove(stones, 1, { legalMask: mask });
    legalOk = r.move >= 0 && mask[r.move] === 1;
  }
  ok(GeoAI.models.length === before + 1 && rand && legalOk,
     "models/ file registers via the build wrapper and respects the mask");
}
{
  const kata = { supports: { surfaces: ["plane"], meshes: ["square"],
                             incidence: ["vertices"] } };
  const anyb = { supports: null };
  const ms = vm.runInContext("modelSupports", sandbox);
  const k9 = { supports: { surfaces: ["plane"], meshes: ["square"],
                           incidence: ["vertices"], scaleIdx: [0] } };
  ok(ms(kata, "plane", "square", "vertices", 2) &&
     !ms(kata, "torus", "square", "vertices", 0) &&
     !ms(kata, "plane", "square", "faces", 0) &&
     ms(anyb, "mobius", "hex", "vertices", 1) &&
     ms(k9, "plane", "square", "vertices", 0) &&
     !ms(k9, "plane", "square", "vertices", 2),
     "supports gating: board whitelist, null = any, scaleIdx pins a 9\u00D79-only net");
}

// ---------- share spec round-trip with incidence ---------------------------------
{
  const code = core.encodeShare(["torus", "square", 0, "faces"], [3, 17, -1]);
  const st = core.decodeShare(code);
  ok(st.s[3] === "faces" && st.m.length === 3 && st.m[2] === -1,
     "share codec carries the incidence mode (4-element spec)");
  const old = core.decodeShare(core.encodeShare(["sphere", "tri", 1], [0]));
  ok(old.s.length === 3, "legacy 3-element codes still decode");
}



// ---------- new quotient surfaces: python-package parity + build smoke ---------
{
  // edge sets exported from the python package (make_board, resolution 1, 5x5):
  const PY_EDGES = {"cylinder": ["0,1", "0,4", "0,5", "1,2", "1,6", "10,11", "10,14", "10,15", "11,12", "11,16", "12,13", "12,17", "13,14", "13,18", "14,19", "15,16", "15,19", "15,20", "16,17", "16,21", "17,18", "17,22", "18,19", "18,23", "19,24", "2,3", "2,7", "20,21", "20,24", "21,22", "22,23", "23,24", "3,4", "3,8", "4,9", "5,10", "5,6", "5,9", "6,11", "6,7", "7,12", "7,8", "8,13", "8,9", "9,14"], "klein": ["0,1", "0,20", "0,24", "0,5", "1,2", "1,21", "1,6", "10,11", "10,14", "10,15", "11,12", "11,16", "12,13", "12,17", "13,14", "13,18", "14,19", "15,16", "15,20", "16,17", "16,21", "17,18", "17,22", "18,19", "18,23", "19,24", "2,22", "2,3", "2,7", "20,21", "21,22", "22,23", "23,24", "3,23", "3,4", "3,8", "4,20", "4,24", "4,9", "5,10", "5,19", "5,6", "6,11", "6,7", "7,12", "7,8", "8,13", "8,9", "9,14", "9,15"], "rp2": ["0,1", "0,24", "0,5", "1,2", "1,23", "1,6", "10,11", "10,14", "10,15", "11,12", "11,16", "12,13", "12,17", "13,14", "13,18", "14,19", "15,16", "15,20", "16,17", "16,21", "17,18", "17,22", "18,19", "18,23", "19,24", "2,22", "2,3", "2,7", "20,21", "21,22", "22,23", "23,24", "3,21", "3,4", "3,8", "4,20", "4,9", "5,10", "5,19", "5,6", "6,11", "6,7", "7,12", "7,8", "8,13", "8,9", "9,14", "9,15"]};
  const edges = adj => { const s = new Set();
    adj.forEach((l, a) => l.forEach(b => s.add(a < b ? a + "," + b : b + "," + a)));
    return [...s].sort(); };
  const flags = { cylinder: [true, false, false, false],
                  klein: [true, true, true, false],
                  rp2: [true, true, true, true] };
  let parity = true;
  for (const [surf, fl] of Object.entries(flags)) {
    const e = edges(core.gridQuotient(5, 5, ...fl, "square").adj);
    const t = PY_EDGES[surf];
    parity = parity && e.length === t.length && e.every((x, i) => x === t[i]);
  }
  ok(parity, "cylinder/klein/rp2 quotient adjacency identical to the python " +
     "package edge-for-edge (5\u00D75) \u2014 trained models transfer");
}

{
  const expect = { cylinder: { n: 45, e: 81 },     // 9x5: 45 + 36
                   klein:    { n: 49, e: 98 },     // 7x7: 4-regular closed
                   rp2:      { n: 49, e: 96 } };   // 7x7: two corner merges
  for (const surf of Object.keys(expect)) {
    const B = buildBoard(surf, "square", 0);
    const E = B.adj.reduce((s, l) => s + l.length, 0) / 2;
    const finite = B.pos.every(p => p.every(Number.isFinite));
    let seam = true;
    outer: for (let a = 0; a < B.adj.length; a++)
      for (const b of B.adj[a]) {
        const [x1, y1] = B.uv[a], [x2, y2] = B.uv[b];
        if (Math.abs(x1 - x2) > 1.5 || Math.abs(y1 - y2) > 1.5) {
          seam = B.edgeCurve(a, b, 6).every(p => p.every(Number.isFinite));
          break outer;
        }
      }
    ok(B.adj.length === expect[surf].n && E === expect[surf].e && finite && seam,
       `${surf} board builds: ${expect[surf].n} vertices, ${expect[surf].e} ` +
       "edges, finite embedding, seam edges drawable",
       `n=${B.adj.length} E=${E}`);
  }
}

// ---------- boot smoke: execute the BUILT page end-to-end -----------------------
// This is the test that guards against exactly one class of shipping accident:
// a page that parses but dies at boot (e.g. a call to a function a bad merge
// removed). It runs every inline script block of ../geodesics.html in order
// inside a stub DOM, with a live bridge server on the side, and asserts the
// board state exists, no block threw, and the remote engine round-trip
// completed all the way into the Opponent menu.
{
  const { readFileSync } = await import("node:fs");
  const vm = await import("node:vm");
  const { spawn } = await import("node:child_process");
  const path = await import("node:path");
  const html = readFileSync(new URL("../geodesics.html", import.meta.url), "utf8");
  const blocks = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]);

  const Anything = new Proxy(function () {}, {
    get(t, p) {
      if (p === Symbol.toPrimitive) return () => 0;
      if (p === Symbol.iterator) return function* () {};
      if (p === "then" || p === "toJSON") return undefined;
      if (p === "length" || p === "count") return 0;
      return Anything;
    },
    set() { return true; },
    apply() { return Anything; },
    construct() { return Anything; },
  });
  const elements = new Map();
  const makeEl = (id) => ({
    id, style: {}, dataset: {}, children: [], _html: "",
    get innerHTML() { return this._html; },
    set innerHTML(v) { this._html = String(v); },
    textContent: "", value: "", hidden: false, disabled: false,
    clientWidth: 800, clientHeight: 600, width: 800, height: 600,
    classList: { add() {}, remove() {}, toggle() {}, contains: () => false },
    addEventListener() {}, removeEventListener() {},
    appendChild(c) { this.children.push(c); return c; },
    removeChild() {}, setAttribute() {}, getAttribute: () => null,
    querySelector: () => makeEl("q"), querySelectorAll: () => [],
    getBoundingClientRect: () => ({ left: 0, top: 0, width: 800, height: 600 }),
    getContext: () => Anything, focus() {}, blur() {}, select() {},
    setPointerCapture() {}, releasePointerCapture() {},
  });
  const sandbox = {
    console, setTimeout, clearTimeout, setInterval, clearInterval,
    performance: { now: () => Date.now() },
    requestAnimationFrame: () => 0, cancelAnimationFrame() {},
    addEventListener() {}, removeEventListener() {}, dispatchEvent() {},
    alert() {}, confirm: () => true,
    devicePixelRatio: 1,
    location: { hash: "", href: "http://localhost/geodesics.html" },
    history: { replaceState() {} },
    navigator: { clipboard: { writeText: async () => {} } },
    document: {
      getElementById: (id) => (elements.has(id) ? elements.get(id)
        : (elements.set(id, makeEl(id)), elements.get(id))),
      createElement: (tag) => makeEl(tag),
      addEventListener() {}, removeEventListener() {},
      body: makeEl("body"), documentElement: makeEl("html"),
      execCommand: () => true,
    },
    THREE: Anything,
    WebSocket: globalThis.WebSocket,          // real socket: full-stack bridge test
    Int8Array, Float32Array, Float64Array, Uint32Array, Uint8Array,
    Math, JSON, Promise, Map, Set, Array, Object,
  };
  sandbox.window = sandbox; sandbox.self = sandbox; sandbox.globalThis = sandbox;
  vm.createContext(sandbox);

  const serveDir = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..", "bridge");
  const serve = spawn("python3", ["serve.py", "8765"], { cwd: serveDir, stdio: "ignore" });
  await new Promise(r => setTimeout(r, 700));

  let bootErr = null;
  try {
    for (const src of blocks) vm.runInContext(src, sandbox, { timeout: 30000 });
  } catch (e) { bootErr = e; }
  ok(!bootErr, "boot smoke: every inline block of the built page executes " +
     "without throwing" + (bootErr ? " -- " + bootErr.message : ""));
  ok(sandbox.GeoAI && sandbox.GeoAI.models.some(m => m.id === "random"),
     "boot smoke: model registry populated inside the page");

  await new Promise(r => setTimeout(r, 1500));
  const remote = sandbox.GeoAI && sandbox.GeoAI.models.find(m => m.id === "rx-py-random");
  const menu = elements.get("opponent");
  ok(!!remote && !!menu && menu._html.includes("rx-py-random"),
     "boot smoke: bridge client connected to a live server and the remote " +
     "engine reached the Opponent menu");
  serve.kill();
}

console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
