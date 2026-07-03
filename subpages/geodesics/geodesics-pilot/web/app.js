// Geodesics web app v0.2 — the board design space, playable.
// Depends on core.js (topologies, engine, colorings, share codec) and THREE.

/* eslint-disable no-undef */

// ---------- configuration: the design space exposed to the player -----------

const RS = 1.55;                        // sphere embedding radius
const TOR = { R: 1.12, r: 0.56 };       // torus embedding
const MOB = { R: 1.30, w: 0.50 };       // Möbius embedding

const SURFACES = {
  sphere: {
    label: "Sphere S\u00B2",
    meshes: {
      tri:    { scales: [2, 3, 4] },        // geodesic frequency
      square: { scales: [3, 4, 5] },        // cube-sphere frequency
      hex:    { scales: [2, 3, 4] },        // Goldberg frequency
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
};
const MESH_LABELS = { tri: "Triangular \u00B7 deg 6",
                      square: "Square \u00B7 deg 4",
                      hex: "Hexagonal \u00B7 deg 3" };
const SCALE_LABELS = ["I", "II", "III"];
const CELLS_OK = { "sphere:tri": 1, "sphere:square": 1,
                   "torus:square": 1, "mobius:square": 1 };

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
function coverRep(x1, y1, x2, y2, nx, ny, wrapX, wrapY, flipX) {
  const cands = [[x2, y2]];
  if (wrapX) {
    cands.push([x2 + nx, flipX ? ny - 1 - y2 : y2]);
    cands.push([x2 - nx, flipX ? ny - 1 - y2 : y2]);
  }
  if (wrapY) {
    const more = [];
    for (const [cx, cy] of cands) more.push([cx, cy + ny], [cx, cy - ny]);
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
  }

  else if (surface === "torus" || surface === "mobius") {
    const [nx, ny] = scale;
    const wrapY = surface === "torus", flipX = surface === "mobius";
    const g = gridQuotient(nx, ny, true, wrapY, flipX, false, meshType);
    B.adj = g.adj; B.nx = nx; B.ny = ny;
    const P = surface === "torus"
      ? (u, v) => torusPoint(u, v, nx, ny, TOR.R, TOR.r)
      : (u, v) => mobiusPoint(u, v, nx, ny, MOB.R, MOB.w);
    const N = surface === "torus"
      ? (u, v) => torusNormal(u, v, nx, ny)
      : (u, v) => mobiusNormal(u, v, nx, ny, MOB.R, MOB.w);
    B.pos = g.uv.map(([x, y]) => P(x, y));
    B.kind = surface;
    B.flatStones = surface === "mobius";
    B.normalAt = i => N(g.uv[i][0], g.uv[i][1]);
    B.edgeCurve = (a, b, S) => {
      const [x1, y1] = g.uv[a];
      const [x2, y2] = coverRep(x1, y1, g.uv[b][0], g.uv[b][1],
                                nx, ny, true, wrapY, flipX);
      const out = [];
      for (let s = 0; s <= S; s++) {
        const t = s / S;
        out.push(P(x1 + (x2 - x1) * t, y1 + (y2 - y1) * t));
      }
      return out;
    };
    if (surface === "mobius") {
      const modal = modalDegree(B.adj);
      B.adj.forEach((l, i) => { if (l.length < modal) B.defects.add(i); });
    }
    if (meshType === "square") {
      // grid cells as faces (corner vertex ids, with seam identifications)
      const vid = (x, y) => y * nx + x;
      const red = (x, y) => {
        if (x >= nx) { x -= nx; if (flipX) y = ny - 1 - y; }
        if (y >= ny) { if (!wrapY) return null; y -= ny; }
        else if (y < 0) { if (!wrapY) return null; y += ny; }
        return [x, y];
      };
      const faces = [], patches = [];
      const ymax = wrapY ? ny : ny - 1;
      for (let y = 0; y < ymax; y++)
        for (let x = 0; x < nx; x++) {
          const c = [[x, y], [x + 1, y], [x + 1, y + 1], [x, y + 1]].map(
            ([cx, cy]) => red(cx, cy));
          if (c.some(q => q === null)) continue;
          faces.push(c.map(([cx, cy]) => vid(cx, cy)));
          patches.push([x, y]);            // cover-coordinate cell corner
        }
      B.cells = { faces, patches, P };
    }
    const surfSym = surface === "torus" ? "T\u00B2" : "M\u00B2";
    B.plate = [
      { k: "plate.surface", t: surfSym, edit: "surface" },
      { k: "plate.mesh", t: nx + "\u00D7" + ny + " " + meshType, edit: "mesh" },
      { k: "plate.V", t: "V " + B.adj.length, edit: "scale" },
      { k: "plate.chi", t: "\u03C7 0" },
      { k: "plate.boundary", t: "\u2202 " + (surface === "torus" ? 0 : 1) },
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
  if (B.kind === "torus" || B.kind === "mobius") {
    const nu = B.nx * 8, nv = 24;
    const P = B.kind === "torus"
      ? (u, v) => torusPoint(u / nu * B.nx, v / nv * (B.ny - (B.kind === "torus" ? 0 : 1)), B.nx, B.ny, TOR.R, TOR.r)
      : (u, v) => mobiusPoint(u / nu * B.nx, v / nv * (B.ny - 1), B.nx, B.ny, MOB.R, MOB.w);
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
            const n = B.kind === "torus"
              ? torusNormal(x + a / SUB, y + b / SUB, B.nx, B.ny)
              : mobiusNormal(x + a / SUB, y + b / SUB, B.nx, B.ny, MOB.R, MOB.w);
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
  const B = buildBoard(state.surface, state.mesh, state.scaleIdx);
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
  if (pushHash !== false) syncHash();
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
  const owner = state.over ? eng.score().owner : null;
  eng.colors.forEach((c, i) => {
    if (c === EMPTY) {
      if (owner && owner[i]) {
        const d = state.meshes.dots[i];
        d.material.color.set(owner[i] === BLACK ? 0x2c343f : 0xf2ecdd);
        d.material.opacity = 1.0;
        d.scale.setScalar(1.8);
      }
      return;
    }
    const mat = new THREE.MeshLambertMaterial({
      color: c === BLACK ? 0x1b2129 : 0xf0ead9 });
    if (pa.has(i)) { mat.emissive = new THREE.Color(0x1d5a4e); }
    const m = new THREE.Mesh(stoneGeo, mat);
    if (!B.flatStones) { m.scale.set(1, 0.62, 1); orientStone(B, m, i); }
    else m.scale.setScalar(0.9);
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
    stat.innerHTML =
      span("stat.score", "game over \u2014 " + s.winner +
        (s.winner === "Draw" ? "" : " by " + Math.abs(s.margin).toFixed(1))) +
      " \u00B7 " + span("stat.score", "B " + s.black + " : W " + s.white) +
      " \u00B7 " + span("stat.komi", "komi 7.5");
    return;
  }
  stat.innerHTML =
    span("stat.turn", (eng.toMove === BLACK ? "Black" : "White") + " to play") +
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

function tryPlay(v) {
  if (state.over) { message("the game is over \u2014 New board to start again"); return; }
  const r = state.eng.play(v);
  if (r.err) { message(ERRTXT[r.err] || r.err); return; }
  if (r.captured.length)
    message((state.eng.toMove === WHITE ? "Black" : "White") +
      " captures " + r.captured.length);
  sync(); renderPlate(); syncHash();
}

function doPass() {
  if (state.over) return;
  const r = state.eng.pass();
  if (r.over) {
    state.over = true;
    message("two passes \u2014 territory scored (Tromp\u2013Taylor)", 0);
  } else message((state.eng.toMove === BLACK ? "Black" : "White") + " to play after pass");
  sync(); syncHash();
}

function doUndo() {
  if (state.eng.undo()) {
    state.over = false;
    // reset any territory-tinted dots
    paintDots(state.B);
    sync(); renderPlate(); syncHash();
  } else message("nothing to undo");
}

// ---------- share / correspondence -------------------------------------------

function currentCode() {
  return encodeShare([state.surface, state.mesh, state.scaleIdx],
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
  if (!SURFACES[surf] || !SURFACES[surf].meshes[mesh] ||
      !SURFACES[surf].meshes[mesh].scales[idx]) {
    message("code names an unknown board spec"); return false;
  }
  state.surface = surf; state.mesh = mesh; state.scaleIdx = idx;
  refreshSelectors();
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
    if (s === "mobius") return { t: "M\u00B2 \u2014 the M\u00F6bius band",
      b: "Glue the left and right edges of a rectangle with a flip: the result is one-sided and non-orientable, \u03C7 = 0, with exactly one boundary circle (\u2202 = 1) of twice the apparent length. A chain crossing the seam comes back mirrored \u2014 what looks like two rims is a single connected edge. The brass rim dots mark the boundary points (fewer liberties)." };
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
    return { t: "square grid \u2014 degree 4",
      b: "The classical goban lattice, glued according to the chosen surface. Four liberties per interior point." };
  },
  "plate.V": () => ({ t: "V \u2014 vertices (playable points)",
    b: "This board has V = " + state.B.adj.length + " points and E = " +
       state.B.edges.length + " connections. The Scale selector changes the mesh resolution \u2014 or click this readout before the first move to cycle it. Area scoring is a census of exactly these V points: stone, territory, or neutral." }),
  "plate.chi": () => {
    const B = state.B;
    let b;
    if (B.kind === "sphere") {
      const F = B.faces ? B.faces.length : (B.adj.length / 2 + 2);
      b = "\u03C7 = V \u2212 E + F = " + B.adj.length + " \u2212 " + B.edges.length +
          " + " + F + " = 2. The Euler characteristic is a topological invariant \u2014 refine the mesh however you like, it never changes. \u03C7 = 2 is what forces the defects: a perfectly regular degree-6 (or 4, or 3) mesh can only exist at \u03C7 = 0.";
    } else {
      b = "\u03C7 = 0: the torus, M\u00F6bius band and Klein bottle are the flat surfaces. Zero Euler characteristic is exactly the condition under which perfectly regular lattices close up with no defects \u2014 which is why this board has no forced brass points (only the boundary, if any).";
    }
    return { t: "\u03C7 \u2014 Euler characteristic", b };
  },
  "plate.boundary": () => ({ t: "\u2202 \u2014 boundary circles",
    b: "The number of boundary circles of the surface. A classical 19\u00D719 board is a disk (\u2202 = 1) and its edge dominates strategy: corners first, then sides, then center. Closed surfaces (\u2202 = 0) abolish the edge entirely. The M\u00F6bius band keeps exactly one boundary circle \u2014 a single circle of double length that visits what looks like both rims." }),
  "plate.deg": () => ({ t: "degree \u2014 liberties per point",
    b: "Vertex degree = liberties of a lone stone there = the local branching factor. This board: " + fmtHist(state.B.adj) + ". Degree is the strongest strategy knob after topology: deg-3 boards are razor-sharp (eyes are cheap, chains die fast), deg-4 is classical, deg-6 favors thick unkillable shapes." }),
  "stat.turn": () => ({ t: "to play",
    b: "Black and White alternate, Black first; a turn is a stone on an empty point, or a pass. After a placement, opponent chains left with no liberties are removed first, then the rule checks your own chain (self-capture is forbidden here). Finally, positional superko: a move may never recreate any earlier whole-board position \u2014 checked by hashing every position ever seen, which on wrap-around boards matters far more often than on the classical grid." }),
  "stat.captures": () => ({ t: "capturing",
    b: "A chain is a maximal group of same-colored stones connected along board edges. Its liberties are the empty points adjacent to it \u2014 adjacency in the board graph, so a liberty can sit across a seam or around the back of the sphere. Fill a chain's last liberty and the whole chain is removed. This counter totals stones captured by each player; captures matter for the position, not the score (area scoring)." }),
  "stat.moves": () => ({ t: "move counter",
    b: "Total moves played, including passes. Two consecutive passes end the game and trigger scoring. The full move list is what the Share code carries." }),
  "stat.komi": () => ({ t: "komi 7.5",
    b: "Compensation added to White's score for moving second. The half point guarantees no draws." }),
  "stat.score": () => ({ t: "Tromp\u2013Taylor area scoring",
    b: "Score = your stones on the board + empty regions whose border touches only your color (+ komi for White). Empty regions touching both colors count for no one. Enlarged dots show territory ownership. This definition needs nothing but the graph \u2014 no notion of 'inside' \u2014 which is why it survives every topology unchanged." }),
  "opt.passalive": () => ({ t: "pass-alive (Benson 1976)",
    b: "Stones glow jade when they are unconditionally alive: the opponent cannot capture them even if the owner passes forever. Computed as a greatest fixpoint \u2014 repeatedly discard chains with fewer than two vital enclosed regions, and regions bordered by discarded chains, until stable. Benson's theorem is stated purely in graph terms, so it holds verbatim on spheres, tori, M\u00F6bius bands and 3D lattices." }),
  "opt.paint": () => ({ t: "paint \u2014 incidence colorings",
    b: "Color the board's incidence structure. <b>Vertices</b>: a proper graph coloring \u2014 adjacent points always differ; the number of colors needed is the chromatic number (a plain grid needs 2; add diagonals or odd wraps and it grows). <b>Edges</b>: a proper edge coloring \u2014 edges meeting at a vertex differ (Vizing: \u0394 or \u0394+1 colors suffice). <b>Cells</b>: faces colored so faces sharing an edge differ \u2014 watch the seams on quotient boards, where a checkerboard can fail to close up. The brass dots are the mesh's curvature defects." }),
  "act.pass": () => ({ t: "pass",
    b: "Decline to place a stone. Two consecutive passes end the game and the position is scored as it stands (Tromp\u2013Taylor: dead stones are not removed by agreement \u2014 capture them before passing)." }),
  "act.undo": () => ({ t: "undo",
    b: "Rewinds one move (including the superko history, so a retracted position becomes playable again)." }),
  "act.new": () => ({ t: "new board",
    b: "Rebuilds the board from the current Surface \u00D7 Mesh \u00D7 Scale spec and starts a fresh game." }),
  "act.share": () => ({ t: "share \u2014 correspondence play",
    b: "Produces a code (and URL) encoding this exact game: the board spec plus every move. Send it to your opponent; they load it, play a move, and send the new code back \u2014 remote Go on any topology, no server involved. Codes are validated on load by replaying every move through the rules engine, so a corrupted code fails loudly instead of silently." }),
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
  if (v >= 0 && state.eng.colors[v] === EMPTY && !state.over) {
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
}

function refreshPaintOptions() {
  const p = document.getElementById("paint");
  const cellsOk = !!CELLS_OK[state.surface + ":" + state.mesh];
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
  const t = document.getElementById("shareIn").value;
  if (t.trim()) {
    document.getElementById("shareModal").classList.remove("open");
    loadShare(t);
  }
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

// ---------- boot ---------------------------------------------------------------

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
  if (location.hash && location.hash.startsWith("#g=")) loadShare(location.hash);
  else syncHash();
} catch (e) { /* no hash in sandbox */ }
group.rotation.x = 0.35; group.rotation.y = -0.5;
(function loop() { requestAnimationFrame(loop); renderer.render(scene, camera); })();
