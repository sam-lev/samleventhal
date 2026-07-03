const assert = require("assert");
const C = require("./core.js");

// icosphere counts
for (const f of [2, 3, 4]) {
  const s = C.geodesicSphere(f);
  assert.strictEqual(s.adj.length, 10 * f * f + 2, "V");
  const E = s.adj.reduce((a, l) => a + l.length, 0) / 2;
  assert.strictEqual(E, 30 * f * f, "E");
  const d5 = s.adj.filter(l => l.length === 5).length;
  assert.strictEqual(d5, 12, "pentagons");
  for (const p of s.positions)
    assert(Math.abs(Math.hypot(...p) - 1) < 1e-9, "unit sphere");
}

// Möbius seam adjacency: (nx-1, 0) ~ (0, ny-1)
{
  const m = C.gridQuotient(6, 5, true, false, true, false);
  const vid = (x, y) => y * 6 + x;
  assert(m.adj[vid(5, 0)].includes(vid(0, 4)), "seam flip");
  // Möbius embedding: P(nx, v) coincides with P(0, ny-1-v)
  const a = C.mobiusPoint(6, 1, 6, 5, 1.5, 0.55);
  const b = C.mobiusPoint(0, 3, 6, 5, 1.5, 0.55);
  for (let d = 0; d < 3; d++) assert(Math.abs(a[d] - b[d]) < 1e-9, "seam geo");
}

// capture: plane corner needs 2, torus same point needs 4
{
  const p = C.gridQuotient(5, 5, false, false, false, false);
  const e = C.Engine(p.adj);
  const vid = (x, y) => y * 5 + x;
  e.play(vid(1, 0)); e.play(vid(0, 0)); e.play(vid(0, 1));
  assert.strictEqual(e.colors[vid(0, 0)], C.EMPTY, "plane capture");
  assert.strictEqual(e.captures[C.BLACK], 1);

  const t = C.gridQuotient(5, 5, true, true, false, false);
  const et = C.Engine(t.adj);
  et.play(vid(1, 0)); et.play(vid(0, 0));
  et.play(vid(4, 0)); et.play(vid(2, 2));
  et.play(vid(0, 1)); assert.strictEqual(et.colors[vid(0, 0)], C.WHITE);
  et.play(vid(2, 3)); et.play(vid(0, 4));
  assert.strictEqual(et.colors[vid(0, 0)], C.EMPTY, "torus capture");
}

// superko: immediate ko retake refused, allowed after exchange
{
  const p = C.gridQuotient(5, 5, false, false, false, false);
  const vid = (x, y) => y * 5 + x;
  const e = C.Engine(p.adj);
  const setup = { [vid(1, 0)]: 1, [vid(0, 1)]: 1, [vid(1, 2)]: 1,
                  [vid(2, 0)]: 2, [vid(3, 1)]: 2, [vid(2, 2)]: 2, [vid(1, 1)]: 2 };
  for (const [v, c] of Object.entries(setup)) e.colors[+v] = c;
  e.posHist = [e.colors.join("")];
  const r1 = e.play(vid(2, 1));
  assert(r1.ok && e.colors[vid(1, 1)] === C.EMPTY, "ko capture");
  const r2 = e.play(vid(1, 1));
  assert.strictEqual(r2.err, "superko", "retake refused");
  e.play(vid(4, 4)); e.play(vid(0, 4));
  const r3 = e.play(vid(1, 1));
  assert(r3.ok, "retake after exchange");
}

// Benson: two-eye corner group pass-alive
{
  const p = C.gridQuotient(7, 7, false, false, false, false);
  const vid = (x, y) => y * 7 + x;
  const e = C.Engine(p.adj);
  for (const [x, y] of [[1, 0], [0, 1], [1, 1], [2, 1], [3, 1], [3, 0]])
    e.colors[vid(x, y)] = C.BLACK;
  const alive = e.passAlive(C.BLACK);
  assert.strictEqual(alive.size, 6, "pass-alive");
}

console.log("all JS core checks passed");

// ---------- v0.2: new builders, colorings, share codec ----------
(function () {
  const C = require("./core.js");

  // Goldberg GP(3,0): 180 vertices, all degree 3, 270 edges
  const gb = C.goldbergSphere(3);
  assert(gb.adj.length === 180, "goldberg V");
  assert(gb.adj.every(l => l.length === 3), "goldberg 3-regular");
  assert(C.edgeList(gb.adj).length === 270, "goldberg E");

  // cube-sphere f=3: 56 vertices, 8 corners of degree 3
  const cs = C.cubeSphere(3);
  assert(cs.adj.length === 56, "cubesphere V");
  const d3 = cs.adj.filter(l => l.length === 3).length;
  assert(d3 === 8 && cs.adj.every(l => l.length === 3 || l.length === 4),
         "cubesphere corners");

  // hex torus 8x6: 3-regular; tri torus 7x7: 6-regular
  const ht = C.gridQuotient(8, 6, true, true, false, false, "hex");
  assert(ht.adj.every(l => l.length === 3), "hex torus 3-regular");
  const tt = C.gridQuotient(7, 7, true, true, false, false, "tri");
  assert(tt.adj.every(l => l.length === 6), "tri torus 6-regular");

  // Mobius hex 12x5: degree <= 3, seam adjacency present
  const mh = C.gridQuotient(12, 5, true, false, true, false, "hex");
  assert(Math.max(...mh.adj.map(l => l.length)) === 3, "mobius hex deg");
  assert(mh.adj[11].includes(4 * 12 + 0), "mobius hex seam flip"); // (11,0)~(0,4)

  // box lattice 4^3: corner degree 3, interior degree 6
  const bx = C.boxLattice(4);
  assert(bx.adj[0].length === 3, "box corner");
  assert(bx.adj[(1 * 4 + 1) * 4 + 1].length === 6, "box interior");

  // colorings are proper
  const vc = C.greedyVertexColoring(tt.adj);
  for (let v = 0; v < tt.adj.length; v++)
    for (const w of tt.adj[v]) assert(vc[v] !== vc[w], "vertex coloring proper");
  const ec = C.greedyEdgeColoring(ht.adj);
  const atV = ht.adj.map(() => []);
  ec.edges.forEach(([a, b], i) => {
    assert(!atV[a].includes(ec.colors[i]) && !atV[b].includes(ec.colors[i]),
           "edge coloring proper");
    atV[a].push(ec.colors[i]); atV[b].push(ec.colors[i]);
  });
  const fc = C.greedyFaceColoring(C.geodesicSphere(2).faces);
  assert(fc.length === 80, "face coloring size");

  // share codec round-trip, including URL form and validation
  const code = C.encodeShare(["torus", "hex", 1], [0, 5, -1, 12]);
  const st = C.decodeShare("https://example.com/geodesics.html#g=" + code);
  assert(st.s[0] === "torus" && st.m.length === 4 && st.m[2] === -1,
         "share roundtrip");
  let threw = false;
  try { C.decodeShare(C.encodeShare(["x"], []).slice(0, 4) + "AAAA"); }
  catch (e) { threw = true; }
  assert(threw, "share rejects garbage");

  console.log("v0.2 core tests passed");
})();
