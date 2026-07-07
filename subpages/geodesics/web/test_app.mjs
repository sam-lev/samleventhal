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
          const want = surf === "sphere" ? 2 : 0;
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

// ---------- share spec round-trip with incidence ---------------------------------
{
  const code = core.encodeShare(["torus", "square", 0, "faces"], [3, 17, -1]);
  const st = core.decodeShare(code);
  ok(st.s[3] === "faces" && st.m.length === 3 && st.m[2] === -1,
     "share codec carries the incidence mode (4-element spec)");
  const old = core.decodeShare(core.encodeShare(["sphere", "tri", 1], [0]));
  ok(old.s.length === 3, "legacy 3-element codes still decode");
}

console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
