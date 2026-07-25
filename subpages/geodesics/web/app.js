// Geodesics web app v0.2 — the board design space, playable.
// Depends on core.js (topologies, engine, colorings, share codec) and THREE.

/* eslint-disable no-undef */

// ---------- configuration: the design space exposed to the player -----------

const RS = 1.55;                        // sphere embedding radius
const TOR = { R: 1.12, r: 0.56 };       // torus embedding
const MOB = { R: 1.30, w: 0.50 };       // Möbius embedding

// Every surface exposes every mesh its identifications admit. Grid scales
// are (nx, ny) pairs; honeycombs carry parity conditions (even sides across
// plain wraps — see gridFaces / the python package for the mathematics), and
// the Klein bottle / RP2 refuse the honeycomb outright (the brick parity
// cannot be made consistent across a flipped seam meeting a second gluing).
const SURFACES = {
  sphere: {
    label: "Sphere S\u00B2",
    meshes: {
      tri:    { scales: [2, 3, 4] },        // geodesic frequency
      square: { scales: [3, 4, 5] },        // cube-sphere frequency
      hex:    { scales: [2, 3, 4] },        // Goldberg frequency
    },
  },
  plane: {
    label: "Plane D\u00B2",
    meshes: {
      tri:    { scales: [[9, 9], [13, 13], [19, 19]] },  // + one diagonal/cell
      square: { scales: [[9, 9], [13, 13], [19, 19]] },  // the classical boards
      hex:    { scales: [[10, 10], [14, 14], [20, 20]] },// brick-wall honeycomb
    },
  },
  torus: {
    label: "Torus T\u00B2",
    meshes: {
      tri:    { scales: [[7, 7], [9, 9], [11, 11]] },
      square: { scales: [[7, 7], [9, 9], [11, 11]] },
      hex:    { scales: [[8, 6], [10, 8], [12, 10]] },   // parity: even sides
    },
  },
  mobius: {
    label: "M\u00F6bius band",
    meshes: {
      tri:    { scales: [[12, 5], [14, 5], [16, 7]] },
      square: { scales: [[12, 5], [14, 5], [16, 7]] },
      hex:    { scales: [[12, 5], [14, 5], [16, 7]] },   // nx even, ny odd
    },
  },
  box: {
    label: "Box E\u00B3 (3D)",
    meshes: {
      square: { scales: [3, 4, 5] },        // n x n x n lattice
    },
  },
  cylinder: {
    label: "Cylinder",
    meshes: {
      tri:    { scales: [[9, 5], [11, 7], [13, 9]] },
      square: { scales: [[9, 5], [11, 7], [13, 9]] },
      hex:    { scales: [[10, 6], [12, 8], [14, 10]] },  // parity: even nx
    },
  },
  klein: {
    label: "Klein bottle K\u00B2",
    meshes: {
      tri:    { scales: [[7, 7], [9, 9], [11, 11]] },
      square: { scales: [[7, 7], [9, 9], [11, 11]] },
      // hex: obstructed — flip_x with wrap_y shears the brick parity
    },
  },
  rp2: {
    label: "Projective plane \u211DP\u00B2",
    meshes: {
      tri:    { scales: [[7, 7], [9, 9], [11, 11]] },
      square: { scales: [[7, 7], [9, 9], [11, 11]] },
      // hex: obstructed — the doubly-flipped gluing refuses the honeycomb
    },
  },
};

// quotient-grid surfaces: identification flags + display style + the plate
// invariants (surface symbol, Euler characteristic, boundary-circle count).
// Adjacency conventions verified edge-for-edge against the python package
// (test suite).
const QUOT = {
  torus:    { wrapY: true,  flipX: false, flipY: false, flat: false,
              sym: "T\u00B2", chi: 0, bnd: 0 },
  mobius:   { wrapY: false, flipX: true,  flipY: false, flat: true,
              sym: "M\u00B2", chi: 0, bnd: 1 },
  cylinder: { wrapY: false, flipX: false, flipY: false, flat: false,
              sym: "S\u00B9\u00D7I", chi: 0, bnd: 2 },
  klein:    { wrapY: true,  flipX: true,  flipY: false, flat: true,
              sym: "K\u00B2", chi: 0, bnd: 0 },
  rp2:      { wrapY: true,  flipX: true,  flipY: true,  flat: true,
              sym: "\u211DP\u00B2", chi: 1, bnd: 0 },
};
const CYL = { R: 1.7, sp: 0.62 };
const KLN = { a: 2.1, b: 0.85 };
const RP2 = { S: 2.9 };
function quotPN(surface, nx, ny) {
  let P;
  if (surface === "torus") P = (u, v) => torusPoint(u, v, nx, ny, TOR.R, TOR.r);
  else if (surface === "mobius")
    P = (u, v) => mobiusPoint(u, v, nx, ny, MOB.R, MOB.w);
  else if (surface === "cylinder")
    P = (u, v) => cylinderPoint(u, v, nx, ny, CYL.R, CYL.sp);
  else if (surface === "klein")
    P = (u, v) => kleinPoint(u, v, nx, ny, KLN.a, KLN.b);
  else P = (u, v) => rp2Point(u, v, nx, ny, RP2.S);
  const N = surface === "torus" ? (u, v) => torusNormal(u, v, nx, ny)
    : surface === "mobius" ? (u, v) => mobiusNormal(u, v, nx, ny, MOB.R, MOB.w)
    : surfNormal(P);
  return { P, N };
}
const MESH_LABELS = { tri: "Triangular \u00B7 deg 6",
                      square: "Square \u00B7 deg 4",
                      hex: "Hexagonal \u00B7 deg 3" };
const SCALE_LABELS = ["I", "II", "III"];
// ---------- availability: which boards support cell paint / incidence play ---
// Both tables are COMPUTED, not hand-maintained: for every grid combo in
// SURFACES we build the smallest scale's quotient and its face complex
// (gridFaces, shared with the python package) and test coverage. Coverage is
// scale-independent for these lattices — the parity conditions are baked
// into the scale tables above — so the smallest scale decides for all three.
//
//   INC_OK    incidence play (stones on edges / faces / all cells) needs the
//             face complex to cover EVERY edge of the graph. Where it covers
//             only part (an open honeycomb rim; the sheared Möbius-honeycomb
//             seam) the board honestly stays a vertex board — a partial
//             complex would silently drop playable structure.
//   CELLS_OK  the cell-paint layer additionally needs square unit cells
//             (curved patches are subdivided per quad). Sphere meshes paint
//             their native faces directly.
const { INC_OK, CELLS_OK } = (() => {
  const inc = {}, cel = {};
  for (const surf of Object.keys(SURFACES)) {
    for (const mesh of Object.keys(SURFACES[surf].meshes)) {
      const k = surf + ":" + mesh;
      if (surf === "sphere") { inc[k] = 1; cel[k] = 1; continue; }
      if (surf === "box") continue;                 // 3D lattice: no 2-cells
      const [nx, ny] = SURFACES[surf].meshes[mesh].scales[0];
      const q = QUOT[surf] || { wrapY: false, flipX: false, flipY: false };
      const wrapX = surf !== "plane";
      const g = gridQuotient(nx, ny, wrapX, q.wrapY, q.flipX, q.flipY, mesh);
      const F = gridFaces(nx, ny, wrapX, q.wrapY, q.flipX, q.flipY, mesh,
                          g.adj);
      if (F.fullCover) inc[k] = 1;
      if (mesh === "square" && F.quads.length) cel[k] = 1;
    }
  }
  return { INC_OK: inc, CELLS_OK: cel };
})();
const INC_MODES = ["vertices", "edges", "faces", "cells"];
const INC_LABELS = { vertices: "Stones: vertices", edges: "Stones: edges",
                     faces: "Stones: faces", cells: "Stones: all cells" };

// Does a model entry support the given board spec? A null/absent supports
// object means "everything"; otherwise each present list is a whitelist.
function modelSupports(m, surface, mesh, incidence, scaleIdx) {
  const s = m.supports;
  if (!s) return true;
  const okIn = (list, val) => !list || list.includes(val);
  return okIn(s.surfaces, surface) && okIn(s.meshes, mesh) &&
         okIn(s.incidence, incidence) && okIn(s.scaleIdx, scaleIdx);
}

// paint palette (muted instrument hues)
const PALETTE = [0x5bb0a0, 0xc98a3d, 0x7a93c4, 0xc4707a, 0x9db06b,
                 0x9a86c8, 0xb8a04a, 0x6aa8b8];

// ---------- board construction ----------------------------------------------

function degreeHist(adj) {
  const h = {};
  for (const l of adj) h[l.length] = (h[l.length] || 0) + 1;
  return h;
}
function modalDegree(adj) {
  const h = degreeHist(adj);
  let best = 0, deg = 0;
  for (const k in h) if (h[k] > best) { best = h[k]; deg = +k; }
  return deg;
}
function minEdgeLen(pos, edges) {
  let m = Infinity;
  for (const [a, b] of edges) {
    const d = Math.hypot(pos[a][0] - pos[b][0], pos[a][1] - pos[b][1],
                         pos[a][2] - pos[b][2]);
    if (d < m) m = d;
  }
  return m;
}
function degText(adj) {
  const h = degreeHist(adj);
  return Object.keys(h).map(Number).sort((a, b) => a - b).join("\u00B7");
}

// Nearest universal-cover representative of (x2,y2) as seen from (x1,y1),
// under the given identifications. Evaluating the embedding on cover
// coordinates makes edge curves cross seams (including the Möbius flip)
// without special cases: torusPoint is periodic, mobiusPoint satisfies
// P(u + nx, v) = P(u, ny-1-v).
function coverRep(x1, y1, x2, y2, nx, ny, wrapX, wrapY, flipX, flipY) {
  const cands = [[x2, y2]];
  if (wrapX) {
    cands.push([x2 + nx, flipX ? ny - 1 - y2 : y2]);
    cands.push([x2 - nx, flipX ? ny - 1 - y2 : y2]);
  }
  if (wrapY) {
    const more = [];
    for (const [cx, cy] of cands)
      more.push([flipY ? nx - 1 - cx : cx, cy + ny],
                [flipY ? nx - 1 - cx : cx, cy - ny]);
    cands.push(...more);
  }
  let best = cands[0], bd = Infinity;
  for (const c of cands) {
    const d = (c[0] - x1) * (c[0] - x1) + (c[1] - y1) * (c[1] - y1);
    if (d < bd) { bd = d; best = c; }
  }
  return best;
}

function buildBoard(surface, meshType, scaleIdx) {
  const scale = SURFACES[surface].meshes[meshType].scales[scaleIdx];
  const B = { surface, meshType, scaleIdx, flatStones: false,
              cells: null, defects: new Set() };

  if (surface === "sphere") {
    let g, meshSym;
    if (meshType === "tri") {
      g = geodesicSphere(scale);
      meshSym = "{3,5+}(" + scale + ",0)";
      B.faces = g.faces;
    } else if (meshType === "square") {
      g = cubeSphere(scale);
      meshSym = "cube-sphere \u03BD" + scale;
      B.faces = g.faces;
    } else {
      g = goldbergSphere(scale);
      meshSym = "GP(" + scale + ",0)";
      B.faces = g.faces;
    }
    B.adj = g.adj;
    B.pos = g.positions.map(p => [p[0] * RS, p[1] * RS, p[2] * RS]);
    B.kind = "sphere";
    B.normalAt = i => {
      const p = B.pos[i], r = Math.hypot(p[0], p[1], p[2]);
      return [p[0] / r, p[1] / r, p[2] / r];
    };
    B.edgeCurve = (a, b, S) => {
      const pa = g.positions[a], pb = g.positions[b], out = [];
      for (let s = 0; s <= S; s++) {
        const p = slerp(pa, pb, s / S);
        out.push([p[0] * RS, p[1] * RS, p[2] * RS]);
      }
      return out;
    };
    const modal = modalDegree(B.adj);
    B.adj.forEach((l, i) => { if (l.length !== modal) B.defects.add(i); });
    const chiTxt = "\u03C7 2";
    B.plate = [
      { k: "plate.surface", t: "S\u00B2", edit: "surface" },
      { k: "plate.mesh", t: meshSym, edit: "mesh" },
      { k: "plate.V", t: "V " + B.adj.length, edit: "scale" },
      { k: "plate.chi", t: chiTxt },
      { k: "plate.boundary", t: "\u2202 0" },
      { k: "plate.deg", t: "deg " + degText(B.adj) },
    ];
    if (B.faces) B.cells = { faces: B.faces };
    if (B.faces) B.meshFaces = B.faces;          // incidence derivation
  }

  else if (QUOT[surface]) {
    const [nx, ny] = scale;
    const { wrapY, flipX, flipY } = QUOT[surface];
    const g = gridQuotient(nx, ny, true, wrapY, flipX, flipY, meshType);
    B.adj = g.adj; B.nx = nx; B.ny = ny;
    const { P, N } = quotPN(surface, nx, ny);
    B.pos = g.uv.map(([x, y]) => P(x, y));
    B.uv = g.uv;
    B._P = P; B._N = N; B._wrapX = true; B._wrapY = wrapY;
    B._flipX = flipX; B._flipY = flipY;
    B.kind = surface;
    B.flatStones = QUOT[surface].flat;
    B.normalAt = i => N(g.uv[i][0], g.uv[i][1]);
    B.edgeCurve = (a, b, S) => {
      const [x1, y1] = g.uv[a];
      const [x2, y2] = coverRep(x1, y1, g.uv[b][0], g.uv[b][1],
                                nx, ny, true, wrapY, flipX, flipY);
      const out = [];
      for (let s = 0; s <= S; s++) {
        const t = s / S;
        out.push(P(x1 + (x2 - x1) * t, y1 + (y2 - y1) * t));
      }
      // where an immersion cannot honor a gluing exactly (RP2's second
      // identification: no R^3 embedding exists), blend the residual to zero
      // so the drawn edge still ends on its vertex. Exact surfaces: no-op.
      const tgt = P(g.uv[b][0], g.uv[b][1]);
      const d = [tgt[0] - out[S][0], tgt[1] - out[S][1], tgt[2] - out[S][2]];
      if (d[0] || d[1] || d[2])
        for (let s = 0; s <= S; s++) {
          const t = s / S;
          out[s] = [out[s][0] + d[0] * t, out[s][1] + d[1] * t,
                    out[s][2] + d[2] * t];
        }
      return out;
    };
    if (!wrapY) {                       // open boundaries (mobius, cylinder)
      const modal = modalDegree(B.adj);
      B.adj.forEach((l, i) => { if (l.length < modal) B.defects.add(i); });
    }
    // faces: the tiling's closed 2-cells, seam identifications applied.
    // gridFaces (core.js, shared with the python package and verified
    // against it) enumerates cover cells, reduces their corners through the
    // deck transformations, and keeps only faces that genuinely close in
    // the quotient graph — so RP2's collapsed corners and the sheared
    // Möbius-honeycomb seam are handled uniformly, with no per-surface
    // special cases here.
    const gf = gridFaces(nx, ny, true, wrapY, flipX, flipY, meshType, B.adj);
    if (gf.quads.length)                       // cell-paint layer (square only)
      B.cells = { faces: gf.quads, patches: gf.quadPatches, P };
    if (gf.faces.length && gf.fullCover) {     // incidence derivation input:
      B.meshFaces = gf.faces;                  // only complete complexes feed
      B.meshFaceUV = gf.faceUV;                // edge/face/cell play (partial
    }                                          // ones would drop structure)
    const q = QUOT[surface];
    B.plate = [
      { k: "plate.surface", t: q.sym, edit: "surface" },
      { k: "plate.mesh", t: nx + "\u00D7" + ny + " " + meshType, edit: "mesh" },
      { k: "plate.V", t: "V " + B.adj.length, edit: "scale" },
      { k: "plate.chi", t: "\u03C7 " + q.chi },
      { k: "plate.boundary", t: "\u2202 " + q.bnd },
      { k: "plate.deg", t: "deg " + degText(B.adj) },
    ];
  }

  else if (surface === "plane") {          // the classical boards + variants
    const [nx, ny] = scale;
    const g = gridQuotient(nx, ny, false, false, false, false, meshType);
    B.adj = g.adj; B.nx = nx; B.ny = ny; B.uv = g.uv;
    const sp = 2.8 / (Math.max(nx, ny) - 1);
    B._sp = sp;
    // flat embedding, centered; y grows downward on the lattice, upward on
    // screen (SGF's coordinate convention keeps 'aa' at the top left)
    const P = (u, v) => [(u - (nx - 1) / 2) * sp, ((ny - 1) / 2 - v) * sp, 0];
    const N = () => [0, 0, 1];
    B._P = P; B._N = N; B._wrapX = false; B._wrapY = false; B._flipX = false;
    B.pos = g.uv.map(([x, y]) => P(x, y));
    B.kind = "plane";
    B.normalAt = () => [0, 0, 1];
    B.edgeCurve = (a, b, S) => {
      const pa = B.pos[a], pb = B.pos[b], out = [];
      for (let s = 0; s <= S; s++) {
        const t = s / S;
        out.push([pa[0] + (pb[0] - pa[0]) * t,
                  pa[1] + (pb[1] - pa[1]) * t, 0]);
      }
      return out;
    };
    // brass: boundary points (fewer liberties than the modal interior degree)
    const modal = modalDegree(B.adj);
    B.adj.forEach((l, i) => { if (l.length < modal) B.defects.add(i); });
    // faces via the shared builder: quads for square, triangle pairs for
    // tri, bricks for hex. A bounded honeycomb's rim edges bound no brick
    // (fullCover false), so plane-hex honestly stays a vertex board.
    const gf = gridFaces(nx, ny, false, false, false, false, meshType, B.adj);
    if (gf.quads.length)
      B.cells = { faces: gf.quads, patches: gf.quadPatches, P };
    if (gf.faces.length && gf.fullCover) {
      B.meshFaces = gf.faces;
      B.meshFaceUV = gf.faceUV;
    }
    B.plate = [
      { k: "plate.surface", t: "D\u00B2", edit: "surface" },
      { k: "plate.mesh", t: nx + "\u00D7" + ny + " " + meshType, edit: "mesh" },
      { k: "plate.V", t: "V " + B.adj.length, edit: "scale" },
      { k: "plate.chi", t: "\u03C7 1" },
      { k: "plate.boundary", t: "\u2202 1" },
      { k: "plate.deg", t: "deg " + degText(B.adj) },
    ];
  }

  else { // box (3D lattice)
    const n = scale;
    const g = boxLattice(n);
    const sp = 2.3 / (n - 1);
    B.adj = g.adj;
    B.pos = g.xyz.map(([x, y, z]) => [
      (x - (n - 1) / 2) * sp, (y - (n - 1) / 2) * sp, (z - (n - 1) / 2) * sp]);
    B.kind = "box";
    B.flatStones = true;
    B.normalAt = () => [0, 0, 1];
    B.edgeCurve = (a, b, S) => {
      const pa = B.pos[a], pb = B.pos[b], out = [];
      for (let s = 0; s <= S; s++) {
        const t = s / S;
        out.push([pa[0] + (pb[0] - pa[0]) * t, pa[1] + (pb[1] - pa[1]) * t,
                  pa[2] + (pb[2] - pa[2]) * t]);
      }
      return out;
    };
    B.adj.forEach((l, i) => { if (l.length === 3) B.defects.add(i); });
    B.plate = [
      { k: "plate.surface", t: "I\u00B3 \u2282 E\u00B3", edit: "surface" },
      { k: "plate.mesh", t: "cubic " + n + "\u00B3", edit: "mesh" },
      { k: "plate.V", t: "V " + B.adj.length, edit: "scale" },
      { k: "plate.deg", t: "deg " + degText(B.adj) },
    ];
  }

  B.edges = edgeList(B.adj);
  B.minEdge = minEdgeLen(B.pos, B.edges);
  return B;
}

// ---------- incidence derivation ----------------------------------------------
// Rebuild the board with a different cell type as the playable sites, via
// GeoCells: edges (line graph), faces (dual), or all cells (Hasse diagram).
// The derived object keeps the board contract — adj / pos / normalAt /
// edgeCurve / defects / plate — so the engine, renderer, picker and share
// codec are untouched. Geometry: on the sphere every site is projected
// radially; on grid quotients sites carry cover coordinates so edges and
// stones cross seams (including the Möbius flip) exactly like vertices do.

function deriveBoard(B, mode) {
  if (mode === "vertices" || !B.meshFaces) return B;
  const cx = GeoCells.fromMesh({ nVerts: B.adj.length, faces: B.meshFaces });
  const d = GeoCells.build(cx, mode);
  const nV = cx.nV, nE = cx.edges.length;

  if (B.kind === "sphere") {
    const unit = (p) => {
      const r = Math.hypot(p[0], p[1], p[2]);
      return [p[0] / r, p[1] / r, p[2] / r];
    };
    const uVert = B.pos.map(unit);
    const uEdge = cx.edges.map(([a, b]) => unit([
      uVert[a][0] + uVert[b][0], uVert[a][1] + uVert[b][1],
      uVert[a][2] + uVert[b][2]]));
    const uFace = cx.faces.map(f => {
      const c = [0, 0, 0];
      for (const v of f) for (let q = 0; q < 3; q++) c[q] += uVert[v][q];
      return unit(c);
    });
    const uSite = mode === "edges" ? uEdge : mode === "faces" ? uFace
      : uVert.concat(uEdge, uFace);
    B.pos = uSite.map(p => [p[0] * RS, p[1] * RS, p[2] * RS]);
    B.normalAt = i => uSite[i];
    B.edgeCurve = (a, b, S) => {
      const out = [];
      for (let s = 0; s <= S; s++) {
        const p = slerp(uSite[a], uSite[b], s / S);
        out.push([p[0] * RS, p[1] * RS, p[2] * RS]);
      }
      return out;
    };
  } else {                                     // torus / möbius: cover coords
    const P = B._P, N = B._N;
    const wrapX = B._wrapX, wrapY = B._wrapY, flipX = B._flipX;
    const uvVert = B.uv;
    const uvEdge = cx.edges.map(([a, b]) => {
      const [x1, y1] = uvVert[a];
      const [x2, y2] = coverRep(x1, y1, uvVert[b][0], uvVert[b][1],
                                B.nx, B.ny, wrapX, wrapY, flipX);
      return [(x1 + x2) / 2, (y1 + y2) / 2];
    });
    const uvFace = B.meshFaceUV.map(uvs => {
      let x = 0, y = 0;
      for (const [cx2, cy2] of uvs) { x += cx2; y += cy2; }
      return [x / uvs.length, y / uvs.length];
    });
    const uvSite = mode === "edges" ? uvEdge : mode === "faces" ? uvFace
      : uvVert.concat(uvEdge, uvFace);
    B.pos = uvSite.map(([u, v]) => P(u, v));
    B.uvSite = uvSite;
    B.normalAt = i => N(uvSite[i][0], uvSite[i][1]);
    B.edgeCurve = (a, b, S) => {
      const [x1, y1] = uvSite[a];
      const [x2, y2] = coverRep(x1, y1, uvSite[b][0], uvSite[b][1],
                                B.nx, B.ny, wrapX, wrapY, flipX);
      const out = [];
      for (let s = 0; s <= S; s++) {
        const t = s / S;
        out.push(P(x1 + (x2 - x1) * t, y1 + (y2 - y1) * t));
      }
      return out;
    };
  }

  B.adj = d.neighbors;
  B.dim = d.dim;
  B.siteMode = mode;
  B.inc = { mode, meta: d.meta };
  B.cells = null;                              // cell paint is a vertex-mode view
  B.edges = edgeList(B.adj);
  B.minEdge = minEdgeLen(B.pos, B.edges);

  // brass defects: degree differing from the modal degree of its dimension class
  B.defects = new Set();
  const byDim = {};
  B.adj.forEach((l, i) => {
    const k = B.dim[i];
    (byDim[k] = byDim[k] || {})[l.length] = (byDim[k][l.length] || 0) + 1;
  });
  const modalOf = {};
  for (const k in byDim) {
    let best = 0;
    for (const deg in byDim[k])
      if (byDim[k][deg] > best) { best = byDim[k][deg]; modalOf[k] = +deg; }
  }
  B.adj.forEach((l, i) => { if (l.length !== modalOf[B.dim[i]]) B.defects.add(i); });

  // plate: site mode segment + updated counts
  const vSeg = B.plate.find(s => s.k === "plate.V");
  if (vSeg) vSeg.t = "sites " + B.adj.length;
  const dSeg = B.plate.find(s => s.k === "plate.deg");
  if (dSeg) dSeg.t = "deg " + degText(B.adj);
  const mi = B.plate.findIndex(s => s.k === "plate.mesh");
  B.plate.splice(mi + 1, 0,
    { k: "plate.sites", t: "on " + mode, edit: "incidence" });
  return B;
}

// ---------- THREE scene ------------------------------------------------------

const el = document.getElementById("view");
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
el.appendChild(renderer.domElement);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 100);
camera.position.set(0, 0, 5.4);
scene.add(new THREE.AmbientLight(0xbfc8d4, 0.85));
const key1 = new THREE.DirectionalLight(0xfff2dc, 0.85);
key1.position.set(3, 4, 5);
scene.add(key1);
const key2 = new THREE.DirectionalLight(0x9db4d0, 0.35);
key2.position.set(-4, -2, -4);
scene.add(key2);
const group = new THREE.Group();
scene.add(group);

// ---------- state ------------------------------------------------------------

const state = {
  surface: "sphere", mesh: "tri", scaleIdx: 1, paint: "ink",
  incidence: "vertices",
  opponent: "human", aiColor: WHITE,
  ai: { engine: null, busy: false, seq: 0 },
  B: null, eng: null, over: false,
  showPA: false,
  meshes: { surface: null, edges: null, dots: [], stones: [], ghost: null,
            ring: null, cellMesh: null },
};

const INK = { line: 0x7e93a8, dot: 0xe8e2d4, brass: 0xc98a3d,
              faceLo: 0x1a212b, faceHi: 0x22303d };

// ---------- rendering --------------------------------------------------------

function clearGroup() {
  while (group.children.length) {
    const c = group.children.pop();
    if (c.geometry) c.geometry.dispose();
    if (c.material) (Array.isArray(c.material) ? c.material : [c.material])
      .forEach(m => m.dispose());
    group.remove(c);
  }
  state.meshes = { surface: null, edges: null, dots: [], stones: [],
                   ghost: null, ring: null, cellMesh: null };
}

function buildSurfaceMesh(B) {
  if (B.kind === "sphere") {
    const geo = new THREE.SphereGeometry(RS * 0.985, 48, 32);
    const mat = new THREE.MeshLambertMaterial({ color: INK.faceLo,
      polygonOffset: true, polygonOffsetFactor: 2, polygonOffsetUnits: 2 });
    return new THREE.Mesh(geo, mat);
  }
  if (B.kind === "plane") {
    const pad = B._sp * 0.9;
    const geo = new THREE.PlaneGeometry((B.nx - 1) * B._sp + 2 * pad,
                                        (B.ny - 1) * B._sp + 2 * pad);
    const mat = new THREE.MeshLambertMaterial({ color: INK.faceLo,
      side: THREE.DoubleSide,
      polygonOffset: true, polygonOffsetFactor: 2, polygonOffsetUnits: 2 });
    const m = new THREE.Mesh(geo, mat);
    m.position.z = -0.012;
    return m;
  }
  if (QUOT[B.kind]) {
    const nu = B.nx * 8, nv = 24;
    const vspan = B._wrapY ? B.ny : B.ny - 1;
    const P = (u, v) => B._P(u / nu * B.nx, v / nv * vspan);
    const verts = [], idx = [];
    for (let v = 0; v <= nv; v++)
      for (let u = 0; u <= nu; u++) verts.push(...P(u, v));
    for (let v = 0; v < nv; v++)
      for (let u = 0; u < nu; u++) {
        const a = v * (nu + 1) + u, b = a + 1, c = a + nu + 1, d = c + 1;
        idx.push(a, b, c, b, d, c);
      }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(verts, 3));
    geo.setIndex(idx);
    geo.computeVertexNormals();
    const mat = new THREE.MeshLambertMaterial({ color: INK.faceLo,
      side: THREE.DoubleSide,
      polygonOffset: true, polygonOffsetFactor: 2, polygonOffsetUnits: 2 });
    return new THREE.Mesh(geo, mat);
  }
  return null;  // box: no surface
}

const EDGE_SEGS = 6;
function buildEdges(B) {
  const positions = [], colors = [];
  const c0 = new THREE.Color(INK.line);
  for (const [a, b] of B.edges) {
    const pts = B.edgeCurve(a, b, EDGE_SEGS);
    for (let s = 0; s < pts.length - 1; s++) {
      positions.push(...pts[s], ...pts[s + 1]);
      colors.push(c0.r, c0.g, c0.b, c0.r, c0.g, c0.b);
    }
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geo.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
  const mat = new THREE.LineBasicMaterial({ vertexColors: true,
    transparent: true, opacity: 0.9 });
  return new THREE.LineSegments(geo, mat);
}

function paintEdges(B) {
  const attr = state.meshes.edges.geometry.getAttribute("color");
  const per = EDGE_SEGS * 2;                     // color entries per edge
  let colors;
  if (state.paint === "edge") {
    const ec = greedyEdgeColoring(B.adj);
    colors = ec.colors.map(c => new THREE.Color(PALETTE[c % PALETTE.length]));
  } else {
    colors = B.edges.map(() => new THREE.Color(INK.line));
  }
  for (let e = 0; e < B.edges.length; e++) {
    const c = colors[e];
    for (let s = 0; s < per; s++) {
      attr.setXYZ(e * per + s, c.r, c.g, c.b);
    }
  }
  attr.needsUpdate = true;
}

function buildDots(B) {
  const r = Math.max(0.012, B.minEdge * 0.085);
  const geo = new THREE.SphereGeometry(r, 8, 6);
  const brassGeo = new THREE.SphereGeometry(r * 1.7, 10, 8);
  B.pos.forEach((p, i) => {
    const brass = B.defects.has(i);
    const mat = new THREE.MeshBasicMaterial({
      color: brass ? INK.brass : INK.dot,
      transparent: true, opacity: brass ? 0.95 : 0.55 });
    const m = new THREE.Mesh(brass ? brassGeo : geo, mat);
    m.position.set(p[0], p[1], p[2]);
    group.add(m);
    state.meshes.dots.push(m);
  });
}

function paintDots(B) {
  let vc = null;
  if (state.paint === "vertex") vc = greedyVertexColoring(B.adj);
  state.meshes.dots.forEach((m, i) => {
    if (vc) {
      m.material.color.set(PALETTE[vc[i] % PALETTE.length]);
      m.material.opacity = 0.95;
    } else {
      const brass = B.defects.has(i);
      m.material.color.set(brass ? INK.brass : INK.dot);
      m.material.opacity = brass ? 0.95 : 0.55;
    }
  });
}

function buildCellMesh(B) {
  if (!B.cells) return null;
  const positions = [], colors = [];
  const push = (pts, col) => {          // fan-triangulate a convex cell
    for (let i = 1; i + 1 < pts.length; i++) {
      positions.push(...pts[0], ...pts[i], ...pts[i + 1]);
      for (let k = 0; k < 3; k++) colors.push(col.r, col.g, col.b);
    }
  };
  const fc = greedyFaceColoring(B.cells.faces);
  const lift = 1.012;
  if (B.kind === "sphere") {
    B.cells.faces.forEach((f, fid) => {
      const col = new THREE.Color(PALETTE[fc[fid] % PALETTE.length]);
      push(f.map(v => B.pos[v].map(q => q * lift)), col);
    });
  } else {                              // curved grid cells via cover coords
    const SUB = 4;
    B.cells.patches.forEach(([x, y], fid) => {
      const col = new THREE.Color(PALETTE[fc[fid] % PALETTE.length]);
      for (let i = 0; i < SUB; i++)
        for (let j = 0; j < SUB; j++) {
          const q = (a, b) => {
            const p = B.cells.P(x + a / SUB, y + b / SUB);
            const n = B._N(x + a / SUB, y + b / SUB);
            return [p[0] + n[0] * 0.012, p[1] + n[1] * 0.012,
                    p[2] + n[2] * 0.012];
          };
          push([q(i, j), q(i + 1, j), q(i + 1, j + 1), q(i, j + 1)], col);
        }
    });
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geo.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
  const mat = new THREE.MeshBasicMaterial({ vertexColors: true,
    transparent: true, opacity: 0.42, side: THREE.DoubleSide,
    depthWrite: false });
  return new THREE.Mesh(geo, mat);
}

function stoneGeometryFor(B) {
  const r = Math.min(0.46 * B.minEdge, 0.34);
  return { r, geo: new THREE.SphereGeometry(r, 20, 14) };
}

function stonePosition(B, i, r) {
  const p = B.pos[i];
  if (B.flatStones) return p;
  const n = B.normalAt(i), lift = 0.55 * r * 0.62;
  return [p[0] + n[0] * lift, p[1] + n[1] * lift, p[2] + n[2] * lift];
}

function orientStone(B, mesh, i) {
  if (B.flatStones) return;
  const n = B.normalAt(i);
  const q = new THREE.Quaternion().setFromUnitVectors(
    new THREE.Vector3(0, 1, 0), new THREE.Vector3(n[0], n[1], n[2]));
  mesh.quaternion.copy(q);
}

// ---------- game sync --------------------------------------------------------

let stoneGeo = null, stoneR = 0.1;

function newBoard(pushHash) {
  clearGroup();
  const B = deriveBoard(buildBoard(state.surface, state.mesh, state.scaleIdx),
                        state.incidence);
  state.B = B;
  state.eng = Engine(B.adj, { komi: 7.5 });
  state.over = false;
  const surf = buildSurfaceMesh(B);
  if (surf) { group.add(surf); state.meshes.surface = surf; }
  state.meshes.edges = buildEdges(B);
  group.add(state.meshes.edges);
  buildDots(B);
  const sg = stoneGeometryFor(B);
  stoneGeo = sg.geo; stoneR = sg.r;
  const gm = new THREE.MeshLambertMaterial({ color: 0xdddddd,
    transparent: true, opacity: 0.45 });
  state.meshes.ghost = new THREE.Mesh(stoneGeo, gm);
  state.meshes.ghost.visible = false;
  if (!state.B.flatStones) state.meshes.ghost.scale.set(1, 0.62, 1);
  group.add(state.meshes.ghost);
  const ringGeo = new THREE.TorusGeometry(stoneR * 0.55, stoneR * 0.09, 8, 24);
  state.meshes.ring = new THREE.Mesh(ringGeo,
    new THREE.MeshBasicMaterial({ color: INK.brass }));
  state.meshes.ring.visible = false;
  group.add(state.meshes.ring);
  state.meshes.cellMesh = buildCellMesh(B);
  if (state.meshes.cellMesh) {
    state.meshes.cellMesh.visible = state.paint === "cells";
    group.add(state.meshes.cellMesh);
  }
  paintDots(B); paintEdges(B);
  renderPlate();
  sync();
  aiRebuild();
  if (pushHash !== false) syncHash();
  scheduleAI();
}

function sync() {
  const B = state.B, eng = state.eng;
  for (const s of state.meshes.stones) {
    group.remove(s); s.material.dispose();
  }
  state.meshes.stones = [];
  const pa = state.showPA
    ? new Set([...eng.passAlive(BLACK), ...eng.passAlive(WHITE)])
    : new Set();
  const sc = state.over ? eng.score() : null;
  const owner = sc ? sc.owner : null;
  const dead = sc ? sc.dead : null;
  eng.colors.forEach((c, i) => {
    const isDead = dead && dead.has(i);
    if (c === EMPTY || isDead) {
      if (owner && owner[i]) {           // territory: dead stones now count
        const d = state.meshes.dots[i];  // for whoever surrounds them
        d.material.color.set(owner[i] === BLACK ? 0x2c343f : 0xf2ecdd);
        d.material.opacity = 1.0;
        d.scale.setScalar(1.8);
      }
      if (c === EMPTY) return;           // empty point: dot only
    }
    const mat = new THREE.MeshLambertMaterial({
      color: c === BLACK ? 0x1b2129 : 0xf0ead9 });
    if (isDead) { mat.transparent = true; mat.opacity = 0.26; }
    else if (pa.has(i)) { mat.emissive = new THREE.Color(0x1d5a4e); }
    const m = new THREE.Mesh(stoneGeo, mat);
    if (!B.flatStones) { m.scale.set(1, 0.62, 1); orientStone(B, m, i); }
    else m.scale.setScalar(0.9);
    if (isDead) m.scale.multiplyScalar(0.58);
    const p = stonePosition(B, i, stoneR);
    m.position.set(p[0], p[1], p[2]);
    group.add(m);
    state.meshes.stones.push(m);
  });
  // last-move ring
  if (eng.lastMove !== null && eng.lastMove !== undefined) {
    const i = eng.lastMove;
    const p = stonePosition(B, i, stoneR);
    const n = B.flatStones ? [0, 0, 1] : B.normalAt(i);
    state.meshes.ring.position.set(p[0] + n[0] * stoneR * 0.45,
      p[1] + n[1] * stoneR * 0.45, p[2] + n[2] * stoneR * 0.45);
    state.meshes.ring.quaternion.setFromUnitVectors(
      new THREE.Vector3(0, 0, 1), new THREE.Vector3(n[0], n[1], n[2]));
    state.meshes.ring.visible = true;
  } else state.meshes.ring.visible = false;
  renderStatus();
}

// ---------- status, plate, messages ------------------------------------------

function span(k, t, cls) {
  return '<span class="info' + (cls ? " " + cls : "") +
         '" data-info="' + k + '">' + t + "</span>";
}

function renderStatus() {
  const eng = state.eng;
  const stat = document.getElementById("status");
  if (state.over) {
    const s = eng.score();
    const nd = s.removedBlack + s.removedWhite;
    stat.innerHTML =
      span("stat.score", "game over \u2014 " + s.winner +
        (s.winner === "Draw" ? "" : " by " + Math.abs(s.margin).toFixed(1))) +
      " \u00B7 " + span("stat.score", "B " + s.black + " : W " + s.white) +
      " \u00B7 " + span("stat.komi", "komi 7.5") +
      (nd ? " \u00B7 " + span("stat.score",
        nd + " dead removed") : "");
    return;
  }
  stat.innerHTML =
    span("stat.turn", (eng.toMove === BLACK ? "Black" : "White") + " to play" +
      (aiActive() && eng.toMove === state.aiColor ? " (AI)" : "")) +
    " \u00B7 " + span("stat.captures",
      "captures B " + eng.captures[BLACK] + " / W " + eng.captures[WHITE]) +
    " \u00B7 " + span("stat.moves", "move " + eng.moves.length) +
    " \u00B7 " + span("stat.komi", "komi 7.5");
}

function renderPlate() {
  const canEdit = state.eng.moves.length === 0;
  document.getElementById("plate").innerHTML = state.B.plate.map(seg =>
    span(seg.k, seg.t, seg.edit && canEdit ? "editable" : "")).join(" \u00B7 ");
}

let msgTimer = null;
function message(t, ms) {
  const m = document.getElementById("msg");
  m.textContent = t;
  if (msgTimer) clearTimeout(msgTimer);
  if (ms !== 0) msgTimer = setTimeout(() => { m.textContent = ""; }, ms || 3200);
}

// ---------- play actions -----------------------------------------------------

const ERRTXT = {
  occupied: "that point is occupied",
  suicide: "suicide \u2014 the placed chain would have no liberties",
  superko: "positional superko \u2014 this would repeat an earlier whole-board position",
};

function tryPlay(v, byAI) {
  if (state.over) { message("the game is over \u2014 New board to start again"); return; }
  if (!byAI && aiActive() &&
      (state.ai.busy || state.eng.toMove === state.aiColor)) {
    message("the AI is to move"); return;
  }
  const r = state.eng.play(v);
  if (r.err) { message(ERRTXT[r.err] || r.err); return; }
  if (r.captured.length)
    message((state.eng.toMove === WHITE ? "Black" : "White") +
      " captures " + r.captured.length);
  sync(); renderPlate(); syncHash();
  scheduleAI();
}

function doPass(byAI) {
  if (state.over) return;
  if (!byAI && aiActive() &&
      (state.ai.busy || state.eng.toMove === state.aiColor)) {
    message("the AI is to move"); return;
  }
  const r = state.eng.pass();
  if (r.over) {
    state.over = true;
    message("two passes \u2014 dead stones removed, then area scored", 0);
  } else message((state.eng.toMove === BLACK ? "Black" : "White") + " to play after pass");
  sync(); syncHash();
  scheduleAI();
}

function doUndo() {
  cancelAI();
  if (state.eng.undo()) {
    if (aiActive() && state.eng.toMove === state.aiColor) state.eng.undo();
    state.over = false;
    // reset any territory-tinted dots
    paintDots(state.B);
    sync(); renderPlate(); syncHash();
  } else message("nothing to undo");
}

// ---------- share / correspondence -------------------------------------------

function currentCode() {
  return encodeShare(
    [state.surface, state.mesh, state.scaleIdx, state.incidence],
    state.eng.moves.map(mv => (mv[1] === null ? -1 : mv[1])));
}

function syncHash() {
  try { history.replaceState(null, "", "#g=" + currentCode()); }
  catch (e) { /* sandboxed frame: hash unavailable, share modal still works */ }
}

function loadShare(text) {
  let st;
  try { st = decodeShare(text); }
  catch (e) { message("could not read that code: " + e.message); return false; }
  const [surf, mesh, idx] = st.s;
  const inc = st.s.length > 3 ? st.s[3] : "vertices";
  if (!SURFACES[surf] || !SURFACES[surf].meshes[mesh] ||
      !SURFACES[surf].meshes[mesh].scales[idx]) {
    message("code names an unknown board spec"); return false;
  }
  if (!INC_MODES.includes(inc) ||
      (inc !== "vertices" && !INC_OK[surf + ":" + mesh])) {
    message("code names an unknown site structure"); return false;
  }
  state.surface = surf; state.mesh = mesh; state.scaleIdx = idx;
  state.incidence = inc;
  refreshSelectors(); refreshPaintOptions();
  newBoard(false);
  for (const m of st.m) {
    const r = (m < 0) ? state.eng.pass() : state.eng.play(m);
    if (r.err) {
      message("code contains an illegal move (" + r.err + ") \u2014 fresh board");
      newBoard(); return false;
    }
    if (r.over) state.over = true;
  }
  sync(); renderPlate(); syncHash();
  message("loaded shared game \u2014 " + state.eng.moves.length + " moves, " +
    (state.eng.toMove === BLACK ? "Black" : "White") + " to play", 5200);
  scheduleAI();
  return true;
}

function openShare() {
  const code = currentCode();
  let url = "";
  try {
    url = location.href.split("#")[0] + "#g=" + code;
  } catch (e) { url = ""; }
  const ta = document.getElementById("shareCode");
  ta.value = (url && url.startsWith("http")) ? url : code;
  document.getElementById("shareModal").classList.add("open");
  ta.focus(); ta.select();
}

function copyShare() {
  const ta = document.getElementById("shareCode");
  ta.select();
  const done = () => message("copied \u2014 send it; they load it, move, and send back", 4200);
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(ta.value).then(done, () => {
      document.execCommand("copy"); done();
    });
  } else { document.execCommand("copy"); done(); }
}

// ---------- SGF: download / load ----------------------------------------------
// Two dialects, chosen by what the board can honestly express:
//
//   classic SGF (FF[4])   only the classical board — plane, square mesh,
//                         stones on vertices — has the letter-pair grid
//                         coordinates SGF assumes. Files interoperate with
//                         every other Go tool.
//   gSGF-0.1 (JSON)       everything else. The board declaration is a spec
//                         reference (surface x mesh x size x incidence)
//                         instead of SZ, and moves are vertex ids. The
//                         schema matches python/gsgf.py exactly, so a file
//                         downloaded here replays with the python package
//                         (geodesics.gsgf.loads) and vice versa; meta.web
//                         additionally pins this page's own spec so a
//                         round trip through the browser is exact.
//
// Loading replays every move through the rules engine (like the share
// codec), so a corrupted or edited file fails loudly, never silently.

// vertex id -> SGF letter pair. Both use 'a' at the top-left and row-major
// order, so the mapping is direct: column letter then row letter.
function sgfCoord(v, nx) {
  return String.fromCharCode(97 + (v % nx)) +
         String.fromCharCode(97 + Math.floor(v / nx));
}

// The gSGF board declaration for the current spec: parameters that
// python's make_board (registry name "spec") — or, for non-vertex sites,
// incidence_board (registry name "incidence") — rebuilds exactly.
function gsgfBoardSpec() {
  const scale = SURFACES[state.surface].meshes[state.mesh]
    .scales[state.scaleIdx];
  let params;
  if (state.surface === "sphere") {
    // web scales are frequencies; the spec API's abstract resolution maps
    // f = r (tri, hex) and f = r + 1 (cube-sphere). frequency is also
    // passed explicitly so the size is pinned regardless of the mapping.
    params = { surface: "sphere", mesh: state.mesh,
               resolution: state.mesh === "square" ? scale - 1 : scale,
               dimension: 2, frequency: scale };
  } else if (state.surface === "box") {
    params = { surface: "box", mesh: "square", resolution: 1, dimension: 3,
               nx: scale, ny: scale, nz: scale };
  } else {
    params = { surface: state.surface, mesh: state.mesh, resolution: 1,
               dimension: 2, nx: scale[0], ny: scale[1] };
  }
  return state.incidence !== "vertices"
    ? { type: "incidence", params: { mode: state.incidence, ...params } }
    : { type: "spec", params };
}

// Serialize the current game. Returns { text, filename, kind }.
function gameToSGF() {
  const eng = state.eng;
  const stamp = new Date().toISOString().slice(0, 10);
  const classical = state.surface === "plane" && state.mesh === "square" &&
                    state.incidence === "vertices";
  if (classical) {
    const nx = state.B.nx;
    let s = "(;GM[1]FF[4]CA[UTF-8]AP[Geodesics:0.2]RU[Tromp-Taylor]" +
            "SZ[" + nx + "]KM[7.5]DT[" + stamp + "]";
    for (const [c, v] of eng.moves)
      s += ";" + (c === BLACK ? "B" : "W") +
           "[" + (v === null ? "" : sgfCoord(v, nx)) + "]";
    s += ")";
    return { text: s, kind: "SGF",
             filename: "geodesics-" + nx + "x" + nx + "-" +
                       eng.moves.length + "m.sgf" };
  }
  const doc = {
    format: "gSGF-0.1",
    board: gsgfBoardSpec(),
    rules: { allow_suicide: false, superko: "positional", komi: 7.5 },
    moves: eng.moves.map(([c, v]) => v === null
      ? { c: c === BLACK ? "B" : "W", pass: true }
      : { c: c === BLACK ? "B" : "W", v }),
    meta: {
      app: "geodesics-web-0.2", date: stamp,
      // this page's own spec, for an exact browser round trip
      web: { surface: state.surface, mesh: state.mesh,
             scaleIdx: state.scaleIdx, incidence: state.incidence },
    },
  };
  return { text: JSON.stringify(doc, null, 2), kind: "gSGF",
           filename: "geodesics-" + state.surface + "-" + state.mesh +
                     (state.incidence !== "vertices"
                       ? "-" + state.incidence : "") +
                     "-" + eng.moves.length + "m.gsgf" };
}

// Trigger a browser download of a text file (no server involved).
function downloadText(filename, text, mime) {
  const blob = new Blob([text], { type: mime || "application/octet-stream" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => { URL.revokeObjectURL(url); a.remove(); }, 0);
}

function downloadSGF() {
  const { text, filename, kind } = gameToSGF();
  downloadText(filename, text,
               kind === "SGF" ? "application/x-go-sgf" : "application/json");
  message("downloaded " + filename +
    (kind === "SGF"
      ? " \u2014 a standard SGF, readable by any Go tool"
      : " \u2014 generalized SGF (JSON); replay it here or with the python package"),
    6200);
}

// ---- classic SGF parsing ----
// A minimal FF[3]/FF[4] reader: walks the main line (the FIRST child at
// every branch point; sibling variations are skipped), collecting each
// node's properties. Property values honor backslash escapes; lowercase
// letters in old-style identifiers ("AddBlack") are ignored per the spec.
function parseClassicSGF(text) {
  const nodes = [];
  const taken = [];                        // taken[d]: a child at depth d
  let i = 0, depth = 0;
  const n = text.length;
  while (i < n) {
    const ch = text[i];
    if (ch === "(") {
      depth++;
      if (taken[depth]) {                  // a sibling variation: skip subtree
        let d = 1;
        i++;
        while (i < n && d > 0) {
          if (text[i] === "[") {           // values may contain parentheses
            i++;
            while (i < n && text[i] !== "]") { if (text[i] === "\\") i++; i++; }
          } else if (text[i] === "(") d++;
          else if (text[i] === ")") d--;
          i++;
        }
        depth--;
        continue;
      }
      taken[depth] = true;
      i++;
    } else if (ch === ")") {
      taken.length = depth + 1;            // forget only DEEPER branch marks:
      depth--;                             // taken[depth] must survive so the
      i++;                                 // next '(' here reads as a sibling
      if (depth <= 0) break;               // end of the main game tree
    } else if (ch === ";") {
      nodes.push({});
      i++;
    } else if (/[A-Za-z]/.test(ch)) {
      let id = "";
      while (i < n && /[A-Za-z]/.test(text[i])) { id += text[i]; i++; }
      id = id.replace(/[a-z]/g, "");       // FF[3] long names -> short form
      const vals = [];
      while (i < n && text[i] === "[") {
        i++;
        let v = "";
        while (i < n && text[i] !== "]") {
          if (text[i] === "\\") i++;       // escaped ] (or anything)
          v += text[i];
          i++;
        }
        i++;
        vals.push(v);
      }
      if (nodes.length && id) {
        const node = nodes[nodes.length - 1];
        (node[id] = node[id] || []).push(...vals);
      }
    } else i++;
  }
  return nodes;
}

// Load a classic SGF onto the classical board (plane / square / vertices).
function loadClassicSGF(text) {
  let nodes;
  try { nodes = parseClassicSGF(text); }
  catch (e) { message("could not parse that SGF: " + e.message); return false; }
  if (!nodes.length) { message("no SGF nodes found"); return false; }
  const root = nodes[0];
  if (root.GM && root.GM[0] !== "1") {
    message("that SGF is not a Go record (GM[" + root.GM[0] + "])");
    return false;
  }
  const szRaw = (root.SZ && root.SZ[0]) || "19";
  if (szRaw.indexOf(":") >= 0) {
    message("non-square SGF boards are not supported"); return false;
  }
  const sz = parseInt(szRaw, 10);
  const idx = SURFACES.plane.meshes.square.scales
    .findIndex(sc => sc[0] === sz);
  if (idx < 0) {
    message("SGF board size " + sz +
            " \u2014 this app offers the classical 9, 13 and 19"); return false;
  }
  if (root.AB || root.AW || (root.HA && parseInt(root.HA[0], 10) > 0)) {
    message("SGF setup / handicap stones are not supported yet");
    return false;
  }
  state.surface = "plane"; state.mesh = "square";
  state.scaleIdx = idx; state.incidence = "vertices";
  refreshSelectors(); refreshPaintOptions();
  newBoard(false);
  for (const nd of nodes) {
    const col = nd.B ? BLACK : nd.W ? WHITE : 0;
    if (!col) continue;                    // root / comment-only nodes
    if (state.eng.toMove !== col) {
      message("SGF move order violates alternation \u2014 fresh board");
      newBoard(); return false;
    }
    const cs = (nd.B || nd.W)[0] || "";
    let r;
    if (cs === "" || (sz <= 19 && cs === "tt")) r = state.eng.pass();
    else {
      const x = cs.charCodeAt(0) - 97, y = cs.charCodeAt(1) - 97;
      if (!(x >= 0 && x < sz && y >= 0 && y < sz)) {
        message("SGF coordinate '" + cs + "' out of range \u2014 fresh board");
        newBoard(); return false;
      }
      r = state.eng.play(y * sz + x);      // row-major, 'aa' top-left
    }
    if (r.err) {
      message("SGF contains an illegal move (" + (ERRTXT[r.err] || r.err) +
              ") \u2014 fresh board");
      newBoard(); return false;
    }
    if (r.over) state.over = true;
  }
  sync(); renderPlate(); syncHash();
  message("loaded SGF \u2014 " + state.eng.moves.length + " moves, " +
    (state.eng.toMove === BLACK ? "Black" : "White") + " to play", 5200);
  scheduleAI();
  return true;
}

// ---- gSGF loading ----
// Map a gSGF board declaration onto this page's spec. meta.web (present in
// files this page wrote) is exact; otherwise the spec params are matched
// against the scale tables — including the resolution -> size defaults the
// python spec API uses, so python-written files load when their size exists
// here (and fail with a pointer to the python package when it does not).
function webSpecFromGSGF(doc) {
  const w = doc.meta && doc.meta.web;
  if (w && SURFACES[w.surface] && SURFACES[w.surface].meshes[w.mesh] &&
      SURFACES[w.surface].meshes[w.mesh].scales[w.scaleIdx] !== undefined &&
      INC_MODES.includes(w.incidence || "vertices"))
    return { surface: w.surface, mesh: w.mesh, scaleIdx: w.scaleIdx,
             incidence: w.incidence || "vertices" };
  const b = doc.board || {};
  const p = b.params || {};
  let incidence = "vertices";
  if (b.type === "incidence") incidence = p.mode || "vertices";
  else if (b.type !== "spec") return null;   // explicit graphs: python-only
  const alias = { sphere: "sphere", s2: "sphere",
                  plane: "plane", disk: "plane", box: "plane",
                  cylinder: "cylinder", annulus: "cylinder",
                  torus: "torus", t2: "torus",
                  mobius: "mobius", moebius: "mobius",
                  klein: "klein", rp2: "rp2", projective: "rp2" };
  let surf = alias[String(p.surface || "").toLowerCase()];
  const dim = p.dimension || 2;
  if (dim === 3) surf = surf === "plane" ? "box" : null;  // only I^3 renders
  else if (dim !== 2) return null;
  if (!surf || !SURFACES[surf]) return null;
  const mesh = p.mesh || "square";
  const entry = SURFACES[surf].meshes[mesh];
  if (!entry) return null;
  const r = p.resolution || 1;
  let idx;
  if (surf === "sphere") {
    const f = p.frequency !== undefined ? p.frequency
      : mesh === "square" ? r + 1 : r;       // the spec API's defaults
    idx = entry.scales.indexOf(f);
  } else if (surf === "box") {
    const nxd = p.nx !== undefined ? p.nx : r + 3;
    idx = entry.scales.indexOf(nxd);
  } else {
    const nxd = p.nx !== undefined ? p.nx
      : mesh === "hex" ? 2 * r + 4 : 2 * r + 3;
    const nyd = p.ny !== undefined ? p.ny
      : mesh === "hex" ? (surf === "mobius" ? 2 * r + 3 : 2 * r + 4)
                       : 2 * r + 3;
    idx = entry.scales.findIndex(sc => sc[0] === nxd && sc[1] === nyd);
  }
  if (idx === undefined || idx < 0) return null;
  return { surface: surf, mesh, scaleIdx: idx, incidence };
}

function loadGSGF(text) {
  let doc;
  try { doc = JSON.parse(text); }
  catch (e) { message("could not parse that gSGF: " + e.message); return false; }
  if (doc.format !== "gSGF-0.1") {
    message("unsupported gSGF format: " + doc.format); return false;
  }
  const spec = webSpecFromGSGF(doc);
  if (!spec) {
    message("that gSGF names a board this page cannot build \u2014 the python " +
            "package (geodesics.gsgf) replays every gSGF file");
    return false;
  }
  if (spec.incidence !== "vertices" &&
      !INC_OK[spec.surface + ":" + spec.mesh]) {
    message("gSGF names an incidence structure unavailable on that board");
    return false;
  }
  if (doc.rules && doc.rules.komi !== undefined && doc.rules.komi !== 7.5)
    message("note: the file plays komi " + doc.rules.komi +
            " \u2014 this page scores with komi 7.5", 6000);
  state.surface = spec.surface; state.mesh = spec.mesh;
  state.scaleIdx = spec.scaleIdx; state.incidence = spec.incidence;
  refreshSelectors(); refreshPaintOptions();
  newBoard(false);
  for (const mv of doc.moves || []) {
    const col = mv.c === "B" ? BLACK : mv.c === "W" ? WHITE : 0;
    if (!col || state.eng.toMove !== col) {
      message("gSGF move order violates alternation \u2014 fresh board");
      newBoard(); return false;
    }
    const r = mv.pass ? state.eng.pass() : state.eng.play(mv.v);
    if (r.err) {
      message("gSGF contains an illegal move (" + (ERRTXT[r.err] || r.err) +
              ") \u2014 fresh board");
      newBoard(); return false;
    }
    if (r.over) state.over = true;
  }
  sync(); renderPlate(); syncHash();
  message("loaded gSGF \u2014 " + state.eng.moves.length + " moves, " +
    (state.eng.toMove === BLACK ? "Black" : "White") + " to play", 5200);
  scheduleAI();
  return true;
}

// Dispatch on the file's first character: '(' is a classic SGF game tree,
// '{' is a gSGF JSON container.
function loadSGFText(text) {
  const t = String(text).trim();
  if (!t) { message("that file is empty"); return false; }
  if (t[0] === "{") return loadGSGF(t);
  if (t[0] === "(") return loadClassicSGF(t);
  message("not an SGF or gSGF file (expected '(' or '{' at the start)");
  return false;
}


// ---------- AI opponent --------------------------------------------------------
// The engine (GeoAI) sees only the adjacency graph, the stone array and a
// legal-move mask computed here by the host rules — legality (ko, superko,
// suicide) never leaves the Engine above.

function aiActive() { return state.opponent !== "human"; }

function cancelAI() {
  state.ai.seq++;
  if (state.ai.busy) { state.ai.busy = false; message(""); }
}

function aiRebuild() {
  cancelAI();
  if (!aiActive()) { state.ai.engine = null; return; }
  const parts = state.opponent.split(":");          // "ai:<model>:<level>"
  const m = GeoAI.models.find(x => x.id === parts[1]);
  state.ai.engine = m ? m.create(state.B.adj, { level: parts[2] || "standard" })
                      : null;
}

function scheduleAI() {
  if (!aiActive() || state.over || state.ai.busy) return;
  if (!state.ai.engine) aiRebuild();
  if (!state.ai.engine) return;
  if (state.eng.toMove !== state.aiColor) return;
  state.ai.busy = true;
  message("AI is thinking\u2026", 0);
  setTimeout(aiMove, 60);                            // let the last stone paint
}

// pickMove may return a result synchronously (local engines) or a Promise
// (bridge engines). A cancellation sequence number discards stale replies
// after undo, new board, or an opponent switch; an illegal or malformed
// reply from a misbehaving engine degrades to a pass, never to a bad board.
function aiMove() {
  if (!aiActive() || state.over || state.eng.toMove !== state.aiColor) {
    state.ai.busy = false; message(""); return;
  }
  const eng = state.eng, n = state.B.adj.length;
  const mask = new Uint8Array(n);
  for (let v = 0; v < n; v++)
    if (eng.colors[v] === EMPTY && !eng.trySim(v, state.aiColor).err) mask[v] = 1;
  const seq = state.ai.seq;
  const apply = (r) => {
    if (seq !== state.ai.seq) return;               // canceled meanwhile
    state.ai.busy = false;
    message("");
    if (state.over || state.eng.toMove !== state.aiColor) return;
    if (!r || typeof r.move !== "number" ||
        (r.move >= 0 && !mask[r.move])) {
      doPass(true);
      if (!state.over) message("AI passes (no usable move returned)");
      return;
    }
    if (r.move < 0) {
      doPass(true);
      if (!state.over)
        message("AI passes" + (r.reason ? " (" + r.reason + ")" : ""));
    } else tryPlay(r.move, true);
  };
  const fail = (e) => {
    if (seq !== state.ai.seq) return;
    state.ai.busy = false;
    message("AI unavailable (" + ((e && e.message) || e) + ") \u2014 your move",
            6000);
  };
  let r;
  try { r = state.ai.engine.pickMove(eng.colors, state.aiColor,
                                     { legalMask: mask }); }
  catch (e) { fail(e); return; }
  if (r && typeof r.then === "function") r.then(apply, fail);
  else apply(r);
}

// ---------- bridge: local engines over a WebSocket -----------------------------
// Run `python3 bridge/serve.py` beside the app; every bot it advertises
// appears in the Opponent menu (and vanishes when the socket drops). Requests
// are stateless: full stones, legal mask, move history and adjacency go out,
// a single vertex index (or -1 for pass) comes back. Legality stays with the
// host Engine regardless of what the remote answers. See bridge/README.md.

// Resolution order: an explicit page global wins (self-hosted deployments),
// then a bridge=<ws-url> link parameter (?bridge= or #bridge= — see
// urlParams below; this makes "play against my server's engines" a
// shareable link), then the local-development default. Only ws(s) URLs are
// accepted from links.
const BRIDGE_URL = (() => {
  if (typeof window !== "undefined" && window.GEO_BRIDGE_URL)
    return window.GEO_BRIDGE_URL;
  const p = urlParams();
  if (p.bridge && /^wss?:\/\//i.test(p.bridge)) return p.bridge;
  return "ws://127.0.0.1:8765";
})();
const bridge = { ws: null, next: 1, pending: new Map(), models: [], timer: null };

function bridgeRemoveModels() {
  if (!bridge.models.length) return;
  GeoAI.models = GeoAI.models.filter(m => !m.remote);
  bridge.models = [];
  refreshOpponentOptions();          // falls back to human if selection vanished
}

function bridgeEngine(modelId, level) {
  return {
    pickMove(stones, color, opts) {
      if (!bridge.ws || bridge.ws.readyState !== 1)
        return Promise.reject(new Error("bridge offline"));
      const id = bridge.next++;
      const B = state.B;
      const req = {
        type: "genmove", id, model: modelId, level: level || "standard",
        spec: { surface: state.surface, mesh: state.mesh,
                scaleIdx: state.scaleIdx, incidence: state.incidence,
                nx: B.nx || 0, ny: B.ny || 0 },
        board: {
          n: B.adj.length,
          neighbors: B.adj,
          stones: Array.from(stones),
          toMove: color,
          legalMask: Array.from(opts && opts.legalMask || []),
          moves: state.eng.moves.map(mv => [mv[0], mv[1] === null ? -1 : mv[1]]),
        },
      };
      return new Promise((resolve, reject) => {
        bridge.pending.set(id, { resolve, reject });
        try { bridge.ws.send(JSON.stringify(req)); }
        catch (e) { bridge.pending.delete(id); reject(e); return; }
        setTimeout(() => {
          if (bridge.pending.has(id)) {
            bridge.pending.delete(id);
            reject(new Error("bridge timeout"));
          }
        }, 120000);
      });
    },
  };
}

function bridgeConnect() {
  if (typeof WebSocket === "undefined") return;
  let ws;
  try { ws = new WebSocket(BRIDGE_URL); } catch (e) { return bridgeRetry(); }
  bridge.ws = ws;
  ws.onopen = () => ws.send(JSON.stringify({ type: "hello", app: "geodesics" }));
  ws.onmessage = (ev) => {
    let m;
    try { m = JSON.parse(ev.data); } catch (e) { return; }
    if (m.type === "models") {
      bridgeRemoveModels();
      for (const spec of m.models || []) {
        const entry = {
          id: "rx-" + spec.id,
          name: spec.name || spec.id,
          levels: (spec.levels && spec.levels.length) ? spec.levels : ["standard"],
          supports: spec.supports || null,
          remote: true,
          create: (nb, opts) => bridgeEngine(spec.id, opts && opts.level),
        };
        GeoAI.models.push(entry);
        bridge.models.push(entry);
      }
      refreshOpponentOptions();
      if (bridge.models.length)
        message("bridge connected \u2014 " + bridge.models.length + " engine" +
          (bridge.models.length > 1 ? "s" : "") + " available", 3800);
    } else if (m.type === "move" && bridge.pending.has(m.id)) {
      const p = bridge.pending.get(m.id);
      bridge.pending.delete(m.id);
      p.resolve({ move: m.move, reason: m.info || "bridge" });
    } else if (m.type === "error" && bridge.pending.has(m.id)) {
      const p = bridge.pending.get(m.id);
      bridge.pending.delete(m.id);
      p.reject(new Error(m.message || "bridge error"));
    }
  };
  ws.onclose = () => {
    for (const p of bridge.pending.values())
      p.reject(new Error("bridge closed"));
    bridge.pending.clear();
    bridge.ws = null;
    bridgeRemoveModels();
    bridgeRetry();
  };
  ws.onerror = () => { try { ws.close(); } catch (e) { /* noop */ } };
}

function bridgeRetry() {
  if (bridge.timer) return;
  bridge.timer = setTimeout(() => { bridge.timer = null; bridgeConnect(); }, 4000);
}

// ---------- explanations (hover \u24D8, right-click / long-press) ----------------

function fmtHist(adj) {
  const h = degreeHist(adj);
  return Object.keys(h).map(Number).sort((a, b) => a - b)
    .map(k => "deg " + k + " \u00D7 " + h[k]).join(", ");
}

const EXPLAIN = {
  "plate.surface": () => {
    const s = state.surface;
    if (s === "sphere") return { t: "S\u00B2 \u2014 the 2-sphere",
      b: "A closed, orientable surface: Euler characteristic \u03C7 = 2, no boundary (\u2202 = 0). Every direction wraps around, so there is no first line to crawl on and no corner in which to live cheaply \u2014 all territory must be enclosed in the open. Click this readout before the first move to cycle surfaces." };
    if (s === "torus") return { t: "T\u00B2 \u2014 the torus",
      b: "The quotient of a rectangle with opposite sides glued, no flips. Closed, orientable, genus 1, \u03C7 = 0, \u2202 = 0. The board is vertex-transitive: every point is equivalent, so there are no corners, no edges, no hoshi \u2014 opening theory must start from pure symmetry. Chains and ladders wrap around both ways." };
    if (s === "cylinder") return { t: "Cylinder \u2014 an annulus",
      b: "One periodic direction, two boundary circles: \u03C7 = 0, orientable. Play wraps around the tube but meets a real edge top and bottom \u2014 corner-like safety exists only along the rims. The halfway house between the plane and the torus." };
    if (s === "klein") return { t: "K\u00B2 \u2014 the Klein bottle",
      b: "Both directions close up, but one gluing carries a flip: \u03C7 = 0, non-orientable, no boundary at all. Every vertex has full degree \u2014 there is nowhere safe. Drawn as the figure-8 immersion, which necessarily passes through itself: K\u00B2 does not embed in three-space. Walk a chain around the flipped direction and it returns mirror-imaged \u2014 the reflection holonomy the rules never notice but strategy must." };
    if (s === "rp2") return { t: "\u211DP\u00B2 \u2014 the real projective plane",
      b: "The square with BOTH edge pairs glued antipodally: \u03C7 = 1, non-orientable, no boundary. The simplest non-orientable closed surface \u2014 the sphere with antipodes identified. Shown as Steiner's Roman surface; since \u211DP\u00B2 admits no embedding in \u211D\u00B3, one gluing unavoidably appears as a self-intersection seam. Two independent orientation-reversing loop classes make this the strangest board in the set." };
    if (s === "mobius") return { t: "M\u00B2 \u2014 the M\u00F6bius band",
      b: "Glue the left and right edges of a rectangle with a flip: the result is one-sided and non-orientable, \u03C7 = 0, with exactly one boundary circle (\u2202 = 1) of twice the apparent length. A chain crossing the seam comes back mirrored \u2014 what looks like two rims is a single connected edge. The brass rim dots mark the boundary points (fewer liberties)." };
    if (s === "plane") return { t: "D\u00B2 \u2014 the disk (the classical board)",
      b: "A square region of the plane, no gluing: \u03C7 = 1, one boundary circle (\u2202 = 1). This is the board Go was born on, and its topology is the tamest here \u2014 contractible, so all the strategic texture comes from the boundary: corners are cheapest to enclose (two sides come free), then edges, then the open center. Play it against any closed surface above to feel how much of classical opening theory is really a theory of \u2202." };
    return { t: "I\u00B3 \u2282 E\u00B3 \u2014 a 3-dimensional box",
      b: "Go on a solid cubic lattice. Interior points have six neighbors \u2014 capturing a lone stone in open space takes six moves \u2014 while the 8 corners (brass) have only three liberties. Nothing in the rules changes: the engine only ever sees the adjacency graph, whatever its dimension." };
  },
  "plate.mesh": () => {
    const s = state.surface, m = state.mesh;
    if (s === "sphere" && m === "tri") return { t: "{3,5+}(f,0) \u2014 geodesic polyhedron",
      b: "Class-I icosahedral geodesic: each triangle of an icosahedron is subdivided at frequency f and projected to the sphere. Almost every vertex has degree 6, but Euler's formula forces exactly 12 vertices of degree 5 \u2014 the brass points. Discrete Gauss\u2013Bonnet: \u03A3(6 \u2212 deg v) = 6\u03C7 = 12. Click to cycle mesh types." };
    if (s === "sphere" && m === "square") return { t: "cube-sphere \u2014 quadrangulated S\u00B2",
      b: "Each face of a cube is subdivided \u03BD\u00D7\u03BD and projected outward. Almost everywhere the board is degree 4, like a classical goban \u2014 but the 8 cube corners (brass) have degree 3. The quad version of Gauss\u2013Bonnet: \u03A3(4 \u2212 deg v) = 4\u03C7 = 8, all concentrated at those corners." };
    if (s === "sphere" && m === "hex") return { t: "GP(f,0) \u2014 Goldberg polyhedron",
      b: "The dual of the geodesic triangulation: every vertex has degree 3, and the curvature moves into the faces \u2014 exactly 12 pentagons among the hexagons, the football / C\u2086\u2080 fullerene pattern. Three liberties per point makes every stone fragile: the whole board plays like the first line." };
    if (m === "tri") return { t: "triangular lattice \u2014 degree 6",
      b: "The square grid plus one diagonal per cell: every interior vertex has degree 6. Because \u03C7 = 0 here, no defects are forced \u2014 on the torus the lattice is perfectly 6-regular. Six liberties per point makes chains thick and hard to kill." + (s === "mobius" ? " Across the M\u00F6bius seam the diagonal's handedness reverses \u2014 the lattice is chiral, the surface is not \u2014 but the adjacency graph remains perfectly well-defined, which is all the rules need." : "") };
    if (m === "hex") return { t: "honeycomb lattice \u2014 degree 3",
      b: "A hexagonal tiling (built as a brick wall: vertical bonds on alternating columns). Every interior vertex has degree 3, so every stone starts with three liberties \u2014 sharp, fragile, first-line Go everywhere. Wrapping a honeycomb imposes parity conditions: even side lengths across plain seams, and odd height across the M\u00F6bius flip \u2014 the Klein bottle refuses it outright." };
    if (s === "box") return { t: "cubic lattice",
      b: "\u2124\u00B3 restricted to an n\u00D7n\u00D7n box: the three-dimensional analogue of the square grid. Degree 6 in the interior, 5 on faces, 4 on edges, 3 at corners." };
    if (s === "plane") return { t: "square grid \u2014 the classical goban",
      b: "A patch of \u2124\u00B2 with no gluing at all: 4 liberties inside, 3 on the sides, 2 at the corners \u2014 the liberty gradient that makes corners the cheapest territory and drives all classical opening theory. 9\u00D79, 13\u00D713 and 19\u00D719 are the traditional teaching, club and tournament sizes; the brass rim marks the boundary points. Every other board in this app is an answer to the question: what happens to Go when this edge is glued away, twisted, or curved?" };
    return { t: "square grid \u2014 degree 4",
      b: "The classical goban lattice, glued according to the chosen surface. Four liberties per interior point." };
  },
  "plate.V": () => ({ t: "V \u2014 vertices (playable points)",
    b: "This board has V = " + state.B.adj.length + " points and E = " +
       state.B.edges.length + " connections. The Scale selector changes the mesh resolution \u2014 or click this readout before the first move to cycle it. Area scoring is a census of exactly these V points: stone, territory, or neutral." }),
  "plate.chi": () => {
    const B = state.B;
    const M = B.inc ? B.inc.meta : null;
    let b;
    if (B.kind === "sphere") {
      const V = M ? M.nV : B.adj.length;
      const E = M ? M.nE : B.edges.length;
      const F = M ? M.nF : (B.faces ? B.faces.length : (B.adj.length / 2 + 2));
      b = "\u03C7 = V \u2212 E + F = " + V + " \u2212 " + E +
          " + " + F + " = 2. The Euler characteristic is a topological invariant \u2014 refine the mesh however you like, it never changes. \u03C7 = 2 is what forces the defects: a perfectly regular degree-6 (or 4, or 3) mesh can only exist at \u03C7 = 0." +
          (M ? " The formula is evaluated on the underlying mesh; the sites you are playing on are its " + B.siteMode + ", a derived graph drawn on the same surface \u2014 so \u03C7 is untouched." : "");
    } else if (B.kind === "plane") {
      const V = M ? M.nV : B.adj.length;
      const E = M ? M.nE : B.edges.length;
      const F = M ? M.nF : B.meshFaces.length;
      b = "\u03C7 = V \u2212 E + F = " + V + " \u2212 " + E + " + " + F +
          " = 1. The disk is contractible \u2014 topologically trivial \u2014 so unlike the sphere it forces no curvature defects, and unlike the flat quotients it isn't closed. What it has instead is a boundary, and on this board the boundary is the whole story: every strategic asymmetry of classical Go (corner, side, center) is a distance-to-\u2202 effect, not a \u03C7 effect.";
    } else if (B.kind === "rp2") {
      b = "\u03C7 = 1: like the disk \u2014 but closed. The projective plane is the sphere with antipodes identified, so its Euler characteristic is half the sphere's." +
          (M ? " Check it on this very mesh: V \u2212 E + F = " + M.nV + " \u2212 " + M.nE + " + " + M.nF + " = " + (M.nV - M.nE + M.nF) + " (the doubly-flipped corner identifications collapse just enough edges to land on 1)." : "") +
          " Odd \u03C7 on a closed surface is only possible without orientability \u2014 this board is the proof.";
    } else {
      b = "\u03C7 = 0: the torus, cylinder, M\u00F6bius band and Klein bottle are the flat surfaces." +
          (M ? " Check it on this very mesh: V \u2212 E + F = " + M.nV + " \u2212 " + M.nE + " + " + M.nF + " = " + (M.nV - M.nE + M.nF) + "." : "") +
          " Zero Euler characteristic is exactly the condition under which perfectly regular lattices close up with no defects \u2014 which is why this board has no forced brass points (only the boundary, if any).";
    }
    return { t: "\u03C7 \u2014 Euler characteristic", b };
  },
  "plate.boundary": () => ({ t: "\u2202 \u2014 boundary circles",
    b: "The number of boundary circles of the surface. A classical 19\u00D719 board is a disk (\u2202 = 1) and its edge dominates strategy: corners first, then sides, then center. Closed surfaces (\u2202 = 0) abolish the edge entirely. The M\u00F6bius band keeps exactly one boundary circle \u2014 a single circle of double length that visits what looks like both rims." }),
  "plate.deg": () => ({ t: "degree \u2014 liberties per point",
    b: "Vertex degree = liberties of a lone stone there = the local branching factor. This board: " + fmtHist(state.B.adj) + ". Degree is the strongest strategy knob after topology: deg-3 boards are razor-sharp (eyes are cheap, chains die fast), deg-4 is classical, deg-6 favors thick unkillable shapes." }),
  "plate.sites": () => {
    const m = state.B.siteMode, M = state.B.inc.meta;
    const t = { edges: "sites: the mesh's edges",
                faces: "sites: the mesh's faces",
                cells: "sites: all cells" }[m];
    const b = m === "edges"
      ? "Stones sit on the " + M.nE + " edges of the mesh; two sites are adjacent when the edges share a vertex. This is the <b>line graph</b> L(G) \u2014 the same move that turns the arcs of a topological graph into classifiable nodes. Liberties now count incident edges, so the feel of the game shifts even though the rules are word-for-word identical."
      : m === "faces"
      ? "Stones sit on the " + M.nF + " faces; two sites are adjacent when the faces share an edge. This is the <b>dual graph</b> \u2014 exactly the construction that turns the geodesic triangulation into the Goldberg board, now available on every surface here. Triangulated meshes give a deg-3 dual (sharp, fragile Go); quad meshes give deg-4."
      : "Stones sit on every cell \u2014 " + M.nV + " vertices, " + M.nE + " edges and " + M.nF + " faces at once \u2014 adjacent by <b>incidence</b>: a vertex touches the edges it ends, an edge touches the faces it bounds. This is the Hasse diagram of the face poset: a cross-dimensional board where a chain can climb from a corner through an edge into a face. Degrees differ by dimension, so brass marks the irregulars within each class.";
    return { t, b };
  },
  "sel.incidence": () => ({ t: "stones on \u2014 incidence structures",
    b: "Which cells of the mesh are the playable sites. <b>Vertices</b>: the 1-skeleton \u2014 canonical Go. <b>Edges</b>: sites are mesh edges, adjacent when they share an endpoint \u2014 the line graph L(G). <b>Faces</b>: sites are faces, adjacent across shared edges \u2014 the dual graph (the geodesic sphere's dual is the Goldberg board, so this generalizes that construction to every surface). <b>All cells</b>: vertices, edges and faces together, adjacent by incidence \u2014 the Hasse diagram of the face poset, cross-dimensional Go. The rules never change: liberties, capture, superko and scoring only ever read the adjacency graph, whichever graph that is." }),
  "sel.opponent": () => ({ t: "opponent \u2014 hierarchical GNN",
    b: "Play against a fixed-weight hierarchical graph neural network. It reads the position as fields on the board graph: exact tactical features (liberties, capture, atari, eyes), message passing over 1- and 2-ring neighborhoods, and a <b>multi-persistence hierarchy</b> \u2014 the ascending/descending basins of its influence function, cancelled by topological persistence into nested coarser partitions, with messages combined jointly across all levels (after Leventhal, Gyulassy, Pascucci &amp; Heimann, NeurIPS 2022). A 1-ply lookahead guides the final choice; strength sets exploration noise, candidate width and lookahead. Because it only ever sees the adjacency graph, the same network plays every surface, mesh and incidence structure above." }),
  "sel.aicolor": () => ({ t: "AI color",
    b: "Which side the network plays. Give it Black and it opens; give it White and you do. Change it any time \u2014 the AI simply takes over that color's next turn." }),
  "stat.turn": () => ({ t: "to play",
    b: "Black and White alternate, Black first; a turn is a stone on an empty point, or a pass. After a placement, opponent chains left with no liberties are removed first, then the rule checks your own chain (self-capture is forbidden here). Finally, positional superko: a move may never recreate any earlier whole-board position \u2014 checked by hashing every position ever seen, which on wrap-around boards matters far more often than on the classical grid." }),
  "stat.captures": () => ({ t: "capturing",
    b: "A chain is a maximal group of same-colored stones connected along board edges. Its liberties are the empty points adjacent to it \u2014 adjacency in the board graph, so a liberty can sit across a seam or around the back of the sphere. Fill a chain's last liberty and the whole chain is removed. This counter totals stones captured by each player; captures matter for the position, not the score (area scoring)." }),
  "stat.moves": () => ({ t: "move counter",
    b: "Total moves played, including passes. Two consecutive passes end the game and trigger scoring. The full move list is what the Share code carries." }),
  "stat.komi": () => ({ t: "komi 7.5",
    b: "Compensation added to White's score for moving second. The half point guarantees no draws." }),
  "stat.score": () => ({ t: "Tromp\u2013Taylor area scoring, dead stones removed",
    b: "At two passes, stones that are not unconditionally alive and are sealed inside an opponent's pass-alive enclosure are removed as dead (translucent), turning into that opponent's territory. The board is then scored by area: your remaining stones + empty regions bordering only your color (+ komi for White). Removal is provably safe \u2014 in seki or unsettled shapes nothing is taken, giving plain Tromp\u2013Taylor \u2014 and, being defined purely on the graph via Benson's theorem, it survives every topology unchanged." }),
  "opt.passalive": () => ({ t: "pass-alive (Benson 1976)",
    b: "Stones glow jade when they are unconditionally alive: the opponent cannot capture them even if the owner passes forever. Computed as a greatest fixpoint \u2014 repeatedly discard chains with fewer than two vital enclosed regions, and regions bordered by discarded chains, until stable. Benson's theorem is stated purely in graph terms, so it holds verbatim on spheres, tori, M\u00F6bius bands and 3D lattices." }),
  "opt.paint": () => ({ t: "paint \u2014 incidence colorings",
    b: "Color the board's incidence structure. <b>Vertices</b>: a proper graph coloring \u2014 adjacent points always differ; the number of colors needed is the chromatic number (a plain grid needs 2; add diagonals or odd wraps and it grows). <b>Edges</b>: a proper edge coloring \u2014 edges meeting at a vertex differ (Vizing: \u0394 or \u0394+1 colors suffice). <b>Cells</b>: faces colored so faces sharing an edge differ \u2014 watch the seams on quotient boards, where a checkerboard can fail to close up. The brass dots are the mesh's curvature defects." }),
  "act.pass": () => ({ t: "pass",
    b: "Decline to place a stone. Two consecutive passes end the game; dead stones (not pass-alive, sealed inside an opponent's pass-alive enclosure) are then removed and the position is scored by area. Genuinely unsettled groups are never auto-removed, so if a boundary is still open, play it out rather than passing." }),
  "act.undo": () => ({ t: "undo",
    b: "Rewinds one move (including the superko history, so a retracted position becomes playable again)." }),
  "act.new": () => ({ t: "new board",
    b: "Rebuilds the board from the current Surface \u00D7 Mesh \u00D7 Scale spec and starts a fresh game." }),
  "act.models": () => ({ t: "models \u2014 load trained networks",
    b: "Add opponents by loading a <b>model card</b>: one JSON file carrying a trained network's weights plus its identity (name, strength levels, which boards it supports). Cards come from the training scripts (<code>--export-model</code>) and load from a file, a URL, or automatically from a <code>#model=&lt;url&gt;</code> link \u2014 so sharing an opponent is sharing a link, on desktop or phone. Only <b>weights</b> are ever loaded, never code: the card names one of this page's built-in runtimes \u2014 pure JavaScript for zero-style graph nets, or a sandboxed in-browser Python interpreter (Pyodide) that runs the actual hatz.py for HATZ, downloaded lazily on its first move and cached. Every reply is still validated by the host rules, so a bad card can at worst pass." }),
  "act.share": () => ({ t: "share \u2014 correspondence play",
    b: "Produces a code (and URL) encoding this exact game: the board spec plus every move. Send it to your opponent; they load it, play a move, and send the new code back \u2014 remote Go on any topology, no server involved. Codes are validated on load by replaying every move through the rules engine, so a corrupted code fails loudly instead of silently. The same sheet downloads and loads <b>SGF</b> files: standard SGF on the classical board, generalized SGF (gSGF, the python package's JSON dialect) on every other topology." }),
  "act.sgfdown": () => ({ t: "download SGF",
    b: "Saves the current game as a file. On the classical board (plane, square mesh, stones on vertices) this is a standard <b>SGF</b> (FF[4]) that any Go tool opens. Every other board is beyond SGF's letter-pair grid coordinates, so those games download as <b>gSGF</b> \u2014 the generalized Smart Game Format: a JSON container whose board declaration is the spec (surface \u00D7 mesh \u00D7 size \u00D7 incidence) and whose moves are vertex ids. gSGF files replay here and in the python package (geodesics.gsgf) interchangeably." }),
  "act.sgfload": () => ({ t: "load SGF",
    b: "Opens a game file: a standard SGF (loaded onto the classical 9/13/19 board; setup and handicap stones are not supported yet) or a gSGF written by this page or the python package. Every move is replayed through the rules engine on load, so files that violate the rules \u2014 or name boards this page cannot build \u2014 fail with an explanation rather than a corrupted position." }),
  "sel.surface": () => ({ t: "surface",
    b: "The underlying manifold \u2014 which points exist and how the world wraps. Right-click the plate segments above the board for the mathematics of the current choice." }),
  "sel.mesh": () => ({ t: "mesh",
    b: "The tiling whose vertices are the playable points: triangular (degree 6), square (degree 4), or hexagonal (degree 3). On the sphere these become the geodesic polyhedron, the cube-sphere, and the Goldberg polyhedron \u2014 each with the defects Euler's formula demands. Some combinations are impossible (honeycombs refuse certain gluings); those options are disabled." }),
  "sel.scale": () => ({ t: "scale",
    b: "Mesh resolution \u2014 the abstract size knob. Each step raises the subdivision frequency or grid size; the option labels show the resulting vertex count." }),
};

// tooltip + panel machinery
const tip = document.getElementById("tip");
let tipFor = null;
document.addEventListener("pointerover", e => {
  const t = e.target.closest("[data-info]");
  if (!t) { tip.classList.remove("show"); tipFor = null; return; }
  tipFor = t;
  const ex = EXPLAIN[t.dataset.info];
  tip.textContent = "\u24D8 " + (ex ? ex().t : "") + " \u2014 right-click to explain";
  tip.classList.add("show");
});
document.addEventListener("pointermove", e => {
  if (!tipFor) return;
  tip.style.left = Math.min(e.clientX + 14, innerWidth - 240) + "px";
  tip.style.top = (e.clientY + 18) + "px";
});
document.addEventListener("pointerout", e => {
  if (e.target.closest && e.target.closest("[data-info]") === tipFor) {
    tip.classList.remove("show"); tipFor = null;
  }
});

function openPanel(key) {
  const ex = EXPLAIN[key];
  if (!ex) return;
  const { t, b } = ex();
  document.getElementById("panelTitle").textContent = t;
  document.getElementById("panelBody").innerHTML = b;
  document.getElementById("panel").classList.add("open");
}
document.addEventListener("contextmenu", e => {
  const t = e.target.closest("[data-info]");
  if (t) { e.preventDefault(); openPanel(t.dataset.info); }
});
let lpTimer = null;
document.addEventListener("touchstart", e => {
  const t = e.target.closest("[data-info]");
  if (!t) return;
  lpTimer = setTimeout(() => { openPanel(t.dataset.info); lpTimer = null; }, 520);
}, { passive: true });
document.addEventListener("touchend", () => { if (lpTimer) clearTimeout(lpTimer); });
document.addEventListener("touchmove", () => { if (lpTimer) clearTimeout(lpTimer); });
document.getElementById("panelClose").addEventListener("click",
  () => document.getElementById("panel").classList.remove("open"));

// plate editing: click a readout before the first move to cycle it
document.getElementById("plate").addEventListener("click", e => {
  const t = e.target.closest("[data-info]");
  if (!t) return;
  const seg = state.B.plate.find(s => s.k === t.dataset.info);
  if (!seg || !seg.edit) return;
  if (state.eng.moves.length) {
    message("board readouts are editable only before the first move \u2014 right-click to learn what this one means");
    return;
  }
  if (seg.edit === "surface") {
    const keys = Object.keys(SURFACES);
    state.surface = keys[(keys.indexOf(state.surface) + 1) % keys.length];
    if (!SURFACES[state.surface].meshes[state.mesh])
      state.mesh = Object.keys(SURFACES[state.surface].meshes)[0];
  } else if (seg.edit === "mesh") {
    const keys = Object.keys(SURFACES[state.surface].meshes);
    state.mesh = keys[(keys.indexOf(state.mesh) + 1) % keys.length];
  } else if (seg.edit === "incidence") {
    const modes = INC_OK[state.surface + ":" + state.mesh] ? INC_MODES : ["vertices"];
    state.incidence = modes[(modes.indexOf(state.incidence) + 1) % modes.length];
  } else {
    state.scaleIdx = (state.scaleIdx + 1) %
      SURFACES[state.surface].meshes[state.mesh].scales.length;
  }
  refreshSelectors(); refreshPaintOptions(); newBoard();
});

// ---------- input: rotate, zoom, pick ----------------------------------------

const ray = new THREE.Raycaster();
let downXY = null, dragging = false;

function pickVertex(cx, cy) {
  const rect = renderer.domElement.getBoundingClientRect();
  const nd = new THREE.Vector2(((cx - rect.left) / rect.width) * 2 - 1,
                               -((cy - rect.top) / rect.height) * 2 + 1);
  ray.setFromCamera(nd, camera);
  const inv = new THREE.Matrix4().copy(group.matrixWorld).invert();
  const o = ray.ray.origin.clone().applyMatrix4(inv);
  const d = ray.ray.direction.clone().transformDirection(inv).normalize();
  const thresh = 0.5 * state.B.minEdge;
  let best = -1, bestT = Infinity;
  state.B.pos.forEach((p, i) => {
    const vx = p[0] - o.x, vy = p[1] - o.y, vz = p[2] - o.z;
    const t = vx * d.x + vy * d.y + vz * d.z;
    if (t <= 0) return;
    const px = vx - t * d.x, py = vy - t * d.y, pz = vz - t * d.z;
    if (Math.hypot(px, py, pz) < thresh && t < bestT) { bestT = t; best = i; }
  });
  return best;
}

el.addEventListener("pointerdown", e => {
  downXY = [e.clientX, e.clientY]; dragging = false;
  el.setPointerCapture(e.pointerId);
});
el.addEventListener("pointermove", e => {
  if (downXY) {
    const dx = e.clientX - downXY[0], dy = e.clientY - downXY[1];
    if (Math.hypot(dx, dy) > 4) dragging = true;
    if (dragging) {
      group.rotateOnWorldAxis(new THREE.Vector3(0, 1, 0), dx * 0.0055);
      group.rotateOnWorldAxis(new THREE.Vector3(1, 0, 0), dy * 0.0055);
      downXY = [e.clientX, e.clientY];
    }
    state.meshes.ghost.visible = false;
    return;
  }
  const v = pickVertex(e.clientX, e.clientY);
  const g = state.meshes.ghost;
  const aiTurn = aiActive() &&
    (state.ai.busy || state.eng.toMove === state.aiColor);
  if (v >= 0 && state.eng.colors[v] === EMPTY && !state.over && !aiTurn) {
    const p = stonePosition(state.B, v, stoneR);
    g.position.set(p[0], p[1], p[2]);
    if (!state.B.flatStones) orientStone(state.B, g, v);
    g.material.color.set(state.eng.toMove === BLACK ? 0x2a323d : 0xf0ead9);
    g.visible = true;
  } else g.visible = false;
});
el.addEventListener("pointerup", e => {
  const wasDrag = dragging || !downXY ||
    Math.hypot(e.clientX - downXY[0], e.clientY - downXY[1]) > 7;
  downXY = null; dragging = false;
  if (wasDrag) return;
  const v = pickVertex(e.clientX, e.clientY);
  if (v >= 0) tryPlay(v);
});
el.addEventListener("pointerleave", () => { state.meshes.ghost.visible = false; });
el.addEventListener("wheel", e => {
  e.preventDefault();
  camera.position.z = Math.min(11, Math.max(2.4,
    camera.position.z + e.deltaY * 0.004));
}, { passive: false });

// ---------- selectors ---------------------------------------------------------

function specCount(surface, mesh, idx) {
  const sc = SURFACES[surface].meshes[mesh].scales[idx];
  if (surface === "sphere")
    return mesh === "tri" ? 10 * sc * sc + 2
         : mesh === "square" ? 6 * sc * sc + 2 : 20 * sc * sc;
  if (surface === "box") return sc * sc * sc;
  return sc[0] * sc[1];
}

function refreshSelectors() {
  const sSel = document.getElementById("surface");
  const mSel = document.getElementById("mesh");
  const cSel = document.getElementById("scale");
  sSel.innerHTML = Object.keys(SURFACES).map(k =>
    '<option value="' + k + '"' + (k === state.surface ? " selected" : "") +
    ">" + SURFACES[k].label + "</option>").join("");
  const meshes = Object.keys(SURFACES[state.surface].meshes);
  if (!meshes.includes(state.mesh)) state.mesh = meshes[0];
  mSel.innerHTML = ["tri", "square", "hex"].map(k => {
    const ok = meshes.includes(k);
    return '<option value="' + k + '"' + (k === state.mesh ? " selected" : "") +
      (ok ? "" : " disabled") + ">" + MESH_LABELS[k] +
      (ok ? "" : " \u2014 n/a") + "</option>";
  }).join("");
  const nScales = SURFACES[state.surface].meshes[state.mesh].scales.length;
  if (state.scaleIdx >= nScales) state.scaleIdx = nScales - 1;
  cSel.innerHTML = SURFACES[state.surface].meshes[state.mesh].scales.map(
    (sc, i) => '<option value="' + i + '"' +
      (i === state.scaleIdx ? " selected" : "") + ">" + SCALE_LABELS[i] +
      " \u00B7 V " + specCount(state.surface, state.mesh, i) + "</option>"
  ).join("");
  refreshIncidenceOptions();
  refreshOpponentOptions();
}

function refreshIncidenceOptions() {
  const ok = !!INC_OK[state.surface + ":" + state.mesh];
  if (!ok) state.incidence = "vertices";
  document.getElementById("incidence").innerHTML = INC_MODES.map(m => {
    const avail = m === "vertices" || ok;
    return '<option value="' + m + '"' +
      (m === state.incidence ? " selected" : "") +
      (avail ? "" : " disabled") + ">" + INC_LABELS[m] +
      (avail ? "" : " \u2014 n/a") + "</option>";
  }).join("");
}

function refreshOpponentOptions() {
  const o = document.getElementById("opponent");
  const many = GeoAI.models.length > 1;
  const opts = ['<option value="human">Opponent: human</option>'];
  let currentOk = state.opponent === "human";
  for (const m of GeoAI.models) {
    const ok = modelSupports(m, state.surface, state.mesh, state.incidence,
                             state.scaleIdx);
    for (const lv of m.levels) {
      const val = "ai:" + m.id + ":" + lv;
      if (val === state.opponent && ok) currentOk = true;
      opts.push('<option value="' + val + '"' + (ok ? "" : " disabled") +
        (val === state.opponent && ok ? " selected" : "") + ">AI" +
        (many ? " \u00B7 " + m.name : "") + " \u00B7 " + lv +
        (ok ? "" : " \u2014 n/a") + "</option>");
    }
  }
  if (!currentOk && state.opponent !== "human") {
    state.opponent = "human";
    cancelAI(); state.ai.engine = null;
  }
  o.innerHTML = opts.join("");
  o.value = state.opponent;
  const c = document.getElementById("aiColor");
  c.innerHTML =
    '<option value="2">AI plays White</option>' +
    '<option value="1">AI plays Black</option>';
  c.value = String(state.aiColor);
  c.hidden = !aiActive();
}

function refreshPaintOptions() {
  const p = document.getElementById("paint");
  const cellsOk = !!CELLS_OK[state.surface + ":" + state.mesh] &&
                  state.incidence === "vertices";
  if (!cellsOk && state.paint === "cells") state.paint = "ink";
  p.innerHTML = [
    ["ink", "Paint: ink"], ["vertex", "Paint: vertices"],
    ["edge", "Paint: edges"], ["cells", "Paint: cells" + (cellsOk ? "" : " \u2014 n/a")],
  ].map(([k, l]) => '<option value="' + k + '"' +
    (k === state.paint ? " selected" : "") +
    (k === "cells" && !cellsOk ? " disabled" : "") + ">" + l + "</option>").join("");
}

document.getElementById("surface").addEventListener("change", e => {
  state.surface = e.target.value;
  refreshSelectors(); refreshPaintOptions(); newBoard();
});
document.getElementById("mesh").addEventListener("change", e => {
  state.mesh = e.target.value;
  refreshSelectors(); refreshPaintOptions(); newBoard();
});
document.getElementById("scale").addEventListener("change", e => {
  state.scaleIdx = +e.target.value;
  refreshSelectors(); newBoard();
});
document.getElementById("incidence").addEventListener("change", e => {
  state.incidence = e.target.value;
  refreshSelectors(); refreshPaintOptions(); newBoard();
});
document.getElementById("opponent").addEventListener("change", e => {
  state.opponent = e.target.value;
  document.getElementById("aiColor").hidden = !aiActive();
  aiRebuild(); scheduleAI();
});
document.getElementById("aiColor").addEventListener("change", e => {
  state.aiColor = +e.target.value;
  scheduleAI();
});
document.getElementById("paint").addEventListener("change", e => {
  state.paint = e.target.value;
  paintDots(state.B); paintEdges(state.B);
  if (state.meshes.cellMesh)
    state.meshes.cellMesh.visible = state.paint === "cells";
});
document.getElementById("passBtn").addEventListener("click", () => {
  doPass(); renderPlate();
});
document.getElementById("undoBtn").addEventListener("click", doUndo);
document.getElementById("newBtn").addEventListener("click", () => newBoard());
document.getElementById("shareBtn").addEventListener("click", openShare);
document.getElementById("shareCopy").addEventListener("click", copyShare);
document.getElementById("shareLoad").addEventListener("click", () => {
  const t = document.getElementById("shareIn").value.trim();
  if (!t) return;
  document.getElementById("shareModal").classList.remove("open");
  // the load box is polyglot: a pasted SGF game tree or gSGF JSON works
  // exactly like a share code (all three replay through the rules engine)
  if (t[0] === "(" || t[0] === "{") loadSGFText(t);
  else loadShare(t);
});
// SGF download: serialize the current game (classic SGF on the classical
// board, gSGF everywhere else) and hand it to the browser as a file.
document.getElementById("sgfDown").addEventListener("click", downloadSGF);
// SGF load: the visible button proxies a hidden <input type=file>.
document.getElementById("sgfLoadBtn").addEventListener("click",
  () => document.getElementById("sgfFile").click());
document.getElementById("sgfFile").addEventListener("change", e => {
  const f = e.target.files && e.target.files[0];
  if (!f) return;
  const reader = new FileReader();
  reader.onload = () => {
    document.getElementById("shareModal").classList.remove("open");
    loadSGFText(String(reader.result));
  };
  reader.onerror = () => message("could not read " + f.name);
  reader.readAsText(f);
  e.target.value = "";           // allow re-picking the same file later
});
document.getElementById("shareClose").addEventListener("click",
  () => document.getElementById("shareModal").classList.remove("open"));
document.getElementById("shareModal").addEventListener("click", e => {
  if (e.target.id === "shareModal")
    document.getElementById("shareModal").classList.remove("open");
});
document.getElementById("paToggle").addEventListener("change", e => {
  state.showPA = e.target.checked; sync();
});

// ---------- loadable models (GeoModels) ----------------------------------------
// Trained networks arrive as portable JSON "model cards" (see loader.js):
// picked as a file, fetched from a pasted URL, or auto-fetched from a
// #model=<url> link. The loader validates the card and installs it into
// GeoAI.models under one of the page's built-in runtimes — pure-JS for
// zero-style graph nets, a lazily-booted Pyodide worker running the real
// hatz.py for HATZ — so only weights are ever loaded, never code. The
// hooks object is how a runtime asks the app about the current board
// without loader.js ever reaching into app state: HATZ's orientation
// cocycle needs the surface name and the grid shape to place the seam.

const MODEL_HOOKS = {
  boardInfo: () => {
    const B = state.B || {};
    // On derived (edges/faces/cells) boards training builds the Bundle
    // from the derived graph with NO surface name and NO faces — the
    // orientation cocycle is the trivial gauge there (see cells.derive /
    // train_hatz.py). Reporting the same here keeps browser play
    // bit-identical to training. On vertex boards we send everything we
    // have: the mesh's 2-cells (verified face-for-face against python)
    // and the lattice/embedding coordinates, so the worker's Bundle is
    // the training Bundle. Boards whose face complex is deliberately
    // withheld (partial covers: an open honeycomb rim) fall back to the
    // faceless name-based cocycle, same as the bridge bots.
    const vertexMode = state.incidence === "vertices";
    return {
      surface: vertexMode ? state.surface : null,
      nx: B.nx || 0, ny: B.ny || 0,
      faces: (vertexMode && B.meshFaces) || null,
      coords: vertexMode ? (B.uv || B.pos || null) : null,
    };
  },
};

// Install a parsed card, refresh the Opponent menu, and report. Returns
// the entry (or null after showing the error) — shared by all three entry
// points (file, URL box, #model= link).
function installModelCard(card, sourceLabel) {
  let entry;
  try { entry = GeoModels.install(card, GeoAI, MODEL_HOOKS); }
  catch (e) { message("could not load model: " + e.message, 7000); return null; }
  refreshOpponentOptions();
  renderModelsList();
  message("loaded model \u201C" + entry.name + "\u201D (" + entry.arch +
    " runtime" + (entry.arch === "hatz"
      ? "; python interpreter downloads on its first move" : "") + ")" +
    (sourceLabel ? " from " + sourceLabel : ""), 7000);
  return entry;
}

// The Models sheet's list of user-loaded models, with per-model Remove.
// Built-in and bridge models are managed elsewhere and are not listed.
function renderModelsList() {
  const el2 = document.getElementById("modelsList");
  const loaded = GeoModels.list(GeoAI);
  if (!loaded.length) {
    el2.innerHTML = '<div class="mrow"><span class="mname" style="color:var(--ink)">' +
      "no loaded models yet</span></div>";
    return;
  }
  el2.innerHTML = loaded.map(m =>
    '<div class="mrow"><span class="mname">' + m.name + "</span>" +
    '<span class="march">' + m.arch + " \u00B7 " + m.levels.join("/") +
    '</span><button data-rm="' + m.id + '">remove</button></div>').join("");
  el2.querySelectorAll("button[data-rm]").forEach(b =>
    b.addEventListener("click", () => {
      GeoModels.remove(GeoAI, b.dataset.rm);
      refreshOpponentOptions();      // falls back to human if it was selected
      renderModelsList();
    }));
}

function openModels() {
  renderModelsList();
  document.getElementById("modelsModal").classList.add("open");
}

document.getElementById("modelsBtn").addEventListener("click", openModels);
document.getElementById("modelsClose").addEventListener("click",
  () => document.getElementById("modelsModal").classList.remove("open"));
document.getElementById("modelsModal").addEventListener("click", e => {
  if (e.target.id === "modelsModal")
    document.getElementById("modelsModal").classList.remove("open");
});
// file picker path
document.getElementById("modelFileBtn").addEventListener("click",
  () => document.getElementById("modelFile").click());
document.getElementById("modelFile").addEventListener("change", e => {
  const f = e.target.files && e.target.files[0];
  if (!f) return;
  const reader = new FileReader();
  reader.onload = () => {
    try { installModelCard(GeoModels.fromText(String(reader.result)), f.name); }
    catch (err) { message("could not load model: " + err.message, 7000); }
  };
  reader.onerror = () => message("could not read " + f.name);
  reader.readAsText(f);
  e.target.value = "";               // allow re-picking the same file later
});
// URL path (also used by the #model= boot link below)
function loadModelURL(url) {
  message("fetching model\u2026", 0);
  GeoModels.fromURL(url).then(
    card => installModelCard(card, new URL(url).hostname),
    err => message("could not load model: " + err.message, 7000));
}
document.getElementById("modelURLBtn").addEventListener("click", () => {
  const u = document.getElementById("modelURL").value.trim();
  if (u) loadModelURL(u);
});

// ---------- URL parameters ------------------------------------------------------
// The fragment (and, for the bridge, also the query string) is a tiny
// parameter bag: #g=<code> restores a shared game, #model=<url> loads a
// model card, and bridge=<ws-url> points the bridge client somewhere other
// than localhost — which turns "play my server's engines" into a link:
//   https://…/geodesics.html?bridge=wss://bots.example.org#model=https://…
// Everything is optional and order-free; g= keeps its legacy first slot so
// existing share URLs continue to work unchanged.
function urlParams() {
  const out = {};
  try {
    const add = (s) => {
      for (const kv of s.split("&")) {
        const q = kv.indexOf("=");
        if (q > 0) out[kv.slice(0, q)] = decodeURIComponent(kv.slice(q + 1));
      }
    };
    if (location.search && location.search.length > 1)
      add(location.search.slice(1));
    if (location.hash && location.hash.length > 1)
      add(location.hash.slice(1));
  } catch (e) { /* sandboxed frame: no location */ }
  return out;
}



function resize() {
  const w = el.clientWidth, h = el.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
addEventListener("resize", resize);

refreshSelectors();
refreshPaintOptions();
newBoard(false);
resize();
try {
  const boot = urlParams();
  if (boot.g) loadShare(boot.g);       // a shared game (legacy #g=... links)
  else syncHash();
  if (boot.model) loadModelURL(boot.model);   // an auto-loading model link
} catch (e) { /* no hash in sandbox */ }
group.rotation.x = 0.35; group.rotation.y = -0.5;
(function loop() { requestAnimationFrame(loop); renderer.render(scene, camera); })();
try { bridgeConnect(); }              // optional: never allowed to break boot
catch (e) { console.warn("bridge unavailable:", e); }
