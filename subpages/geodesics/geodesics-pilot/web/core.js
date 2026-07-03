/* Geodesics web core — rules + topology, framework-free (testable in node). */
"use strict";

// ---------- topology: grid quotients (with mesh types) ----------------------
function gridQuotient(nx, ny, wrapX, wrapY, flipX, flipY, mesh) {
  mesh = mesh || "square";
  const vid = (x, y) => y * nx + x;
  const key = (a, b) => (a < b ? a + ":" + b : b + ":" + a);
  const eset = new Set();
  const reduce = (x, y) => {
    if (x >= nx) { if (!wrapX) return null; x -= nx; if (flipX) y = ny - 1 - y; }
    if (y >= ny) { if (!wrapY) return null; y -= ny; if (flipY) x = nx - 1 - x; }
    else if (y < 0) { if (!wrapY) return null; y += ny; if (flipY) x = nx - 1 - x; }
    return [x, y];
  };
  const add = (a, t) => {
    if (t) { const b = vid(t[0], t[1]); if (a !== b) eset.add(key(a, b)); }
  };
  for (let y = 0; y < ny; y++) {
    for (let x = 0; x < nx; x++) {
      const v = vid(x, y);
      add(v, reduce(x + 1, y));
      if (mesh !== "hex" || (x + y) % 2 === 0) add(v, reduce(x, y + 1));
      if (mesh === "tri") add(v, reduce(x + 1, y + 1));
    }
  }
  const adj = Array.from({ length: nx * ny }, () => []);
  for (const k of eset) {
    const [a, b] = k.split(":").map(Number);
    adj[a].push(b); adj[b].push(a);
  }
  const uv = [];
  for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) uv.push([x, y]);
  return { adj, uv, nx, ny };
}

// ---------- topology: 3D box lattice ----------------------------------------
function boxLattice(n) {
  const vid = (x, y, z) => (z * n + y) * n + x;
  const adj = Array.from({ length: n * n * n }, () => []);
  const xyz = [];
  for (let z = 0; z < n; z++)
    for (let y = 0; y < n; y++)
      for (let x = 0; x < n; x++) {
        xyz.push([x, y, z]);
        const v = vid(x, y, z);
        if (x + 1 < n) { adj[v].push(vid(x + 1, y, z)); adj[vid(x + 1, y, z)].push(v); }
        if (y + 1 < n) { adj[v].push(vid(x, y + 1, z)); adj[vid(x, y + 1, z)].push(v); }
        if (z + 1 < n) { adj[v].push(vid(x, y, z + 1)); adj[vid(x, y, z + 1)].push(v); }
      }
  return { adj, xyz, n };
}

// ---------- topology: geodesic sphere {3,5+}_(f,0) -------------------------
function geodesicSphere(f) {
  const T = (1 + Math.sqrt(5)) / 2;
  const norm = (p) => {
    const r = Math.hypot(p[0], p[1], p[2]);
    return [p[0] / r, p[1] / r, p[2] / r];
  };
  const BV = [
    [-1, T, 0], [1, T, 0], [-1, -T, 0], [1, -T, 0],
    [0, -1, T], [0, 1, T], [0, -1, -T], [0, 1, -T],
    [T, 0, -1], [T, 0, 1], [-T, 0, -1], [-T, 0, 1],
  ].map(norm);
  const BF = [
    [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
    [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
    [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
    [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
  ];
  const verts = [], index = new Map();
  function keyPos(fid, face, i, j, k) {
    const [a, b, c] = face;
    if (j === 0 && k === 0) return [["v", a], BV[a]];
    if (i === 0 && k === 0) return [["v", b], BV[b]];
    if (i === 0 && j === 0) return [["v", c], BV[c]];
    const edge = (p, q, wq) => {
      if (p > q) { const t = p; p = q; q = t; wq = f - wq; }
      const pos = [0, 1, 2].map(d => (f - wq) * BV[p][d] + wq * BV[q][d]);
      return [["e", p, q, wq], norm(pos)];
    };
    if (k === 0) return edge(a, b, j);
    if (j === 0) return edge(a, c, k);
    if (i === 0) return edge(b, c, k);
    const pos = [0, 1, 2].map(d => i * BV[a][d] + j * BV[b][d] + k * BV[c][d]);
    return [["f", fid, i, j], norm(pos)];
  }
  function vidOf(fid, face, i, j, k) {
    const [key, pos] = keyPos(fid, face, i, j, k);
    const sk = key.join(",");
    if (!index.has(sk)) { index.set(sk, verts.length); verts.push(pos); }
    return index.get(sk);
  }
  const faces = [];
  BF.forEach((face, fid) => {
    const grid = new Map();
    for (let i = 0; i <= f; i++)
      for (let j = 0; j <= f - i; j++)
        grid.set(i + "," + j, vidOf(fid, face, i, j, f - i - j));
    for (let i = 0; i < f; i++)
      for (let j = 0; j < f - i; j++) {
        faces.push([grid.get(i + "," + j), grid.get((i + 1) + "," + j),
                    grid.get(i + "," + (j + 1))]);
        if (i + j <= f - 2)
          faces.push([grid.get((i + 1) + "," + j),
                      grid.get((i + 1) + "," + (j + 1)),
                      grid.get(i + "," + (j + 1))]);
      }
  });
  const nb = Array.from({ length: verts.length }, () => new Set());
  for (const [a, b, c] of faces) {
    nb[a].add(b).add(c); nb[b].add(a).add(c); nb[c].add(a).add(b);
  }
  return { adj: nb.map(s => [...s].sort((x, y) => x - y)),
           positions: verts, faces };
}

// ---------- Goldberg sphere: graph-level dual of the geodesic ----------------
// Dual board only needs positions + adjacency: vertices at triangle centroids
// (projected), adjacency across shared edges. Face polygons aren't required.
function goldbergSphere(f) {
  const g = geodesicSphere(f);
  const norm = (p) => {
    const r = Math.hypot(p[0], p[1], p[2]); return [p[0] / r, p[1] / r, p[2] / r];
  };
  const positions = g.faces.map(([a, b, c]) => norm([
    (g.positions[a][0] + g.positions[b][0] + g.positions[c][0]) / 3,
    (g.positions[a][1] + g.positions[b][1] + g.positions[c][1]) / 3,
    (g.positions[a][2] + g.positions[b][2] + g.positions[c][2]) / 3,
  ]));
  const byEdge = new Map();
  const adj = Array.from({ length: g.faces.length }, () => []);
  g.faces.forEach((face, fid) => {
    for (let i = 0; i < 3; i++) {
      const a = face[i], b = face[(i + 1) % 3];
      const k = a < b ? a + ":" + b : b + ":" + a;
      if (byEdge.has(k)) {
        const other = byEdge.get(k);
        adj[fid].push(other); adj[other].push(fid);
      } else byEdge.set(k, fid);
    }
  });
  return { adj: adj.map(l => l.sort((x, y) => x - y)), positions };
}

// ---------- cube-sphere: f x f quad subdivision of the cube ------------------
function cubeSphere(f) {
  const norm = (p) => {
    const r = Math.hypot(p[0], p[1], p[2]); return [p[0] / r, p[1] / r, p[2] / r];
  };
  const BV = [];
  for (const x of [-1, 1]) for (const y of [-1, 1]) for (const z of [-1, 1])
    BV.push([x, y, z]);
  const BF = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4],
              [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]];
  const verts = [], index = new Map();
  function keyOf(fid, face, i, j) {
    const [a, b, c, d] = face;
    if (i === 0 && j === 0) return "v," + a;
    if (i === f && j === 0) return "v," + b;
    if (i === f && j === f) return "v," + c;
    if (i === 0 && j === f) return "v," + d;
    const edge = (p, q, w) => {
      if (p > q) { const t = p; p = q; q = t; w = f - w; }
      return "e," + p + "," + q + "," + w;
    };
    if (j === 0) return edge(a, b, i);
    if (i === f) return edge(b, c, j);
    if (j === f) return edge(d, c, i);
    if (i === 0) return edge(a, d, j);
    return "f," + fid + "," + i + "," + j;
  }
  function vidOf(fid, face, i, j) {
    const k = keyOf(fid, face, i, j);
    if (!index.has(k)) {
      const [a, b, c, d] = face.map(v => BV[v]);
      const s = i / f, t = j / f;
      index.set(k, verts.length);
      verts.push(norm([0, 1, 2].map(q =>
        (1 - s) * (1 - t) * a[q] + s * (1 - t) * b[q]
        + s * t * c[q] + (1 - s) * t * d[q])));
    }
    return index.get(k);
  }
  const eset = new Set();
  const ek = (a, b) => (a < b ? a + ":" + b : b + ":" + a);
  const faces = [];
  BF.forEach((face, fid) => {
    const grid = new Map();
    for (let i = 0; i <= f; i++)
      for (let j = 0; j <= f; j++)
        grid.set(i + "," + j, vidOf(fid, face, i, j));
    for (let i = 0; i < f; i++)
      for (let j = 0; j < f; j++) {
        const a = grid.get(i + "," + j), b = grid.get((i + 1) + "," + j),
              c = grid.get((i + 1) + "," + (j + 1)), d = grid.get(i + "," + (j + 1));
        faces.push([a, b, c, d]);
        eset.add(ek(a, b)); eset.add(ek(b, c));
        eset.add(ek(c, d)); eset.add(ek(d, a));
      }
  });
  const adj = Array.from({ length: verts.length }, () => []);
  for (const k of eset) {
    const [a, b] = k.split(":").map(Number);
    adj[a].push(b); adj[b].push(a);
  }
  return { adj: adj.map(l => l.sort((x, y) => x - y)), positions: verts, faces };
}

// ---------- rule engine (Tromp-Taylor on a graph) ---------------------------
const EMPTY = 0, BLACK = 1, WHITE = 2;
const OTHER = [0, 2, 1];

function chainAt(colors, adj, v) {
  const color = colors[v], stones = new Set([v]), libs = new Set(), st = [v];
  while (st.length) {
    const u = st.pop();
    for (const w of adj[u]) {
      const c = colors[w];
      if (c === EMPTY) libs.add(w);
      else if (c === color && !stones.has(w)) { stones.add(w); st.push(w); }
    }
  }
  return { stones, libs };
}

function Engine(adj, opts) {
  opts = opts || {};
  const n = adj.length;
  const self = {
    adj, n,
    allowSuicide: !!opts.allowSuicide,
    komi: opts.komi || 0,
    colors: new Array(n).fill(EMPTY),
    toMove: BLACK,
    captures: [0, 0, 0],
    passes: 0,
    lastMove: null,
    moves: [],
    posHist: [],
    snaps: [],
  };
  const key = (cs) => cs.join("");
  self.posHist.push(key(self.colors));

  self.trySim = function (v, color) {
    if (self.colors[v] !== EMPTY) return { err: "occupied" };
    const nc = self.colors.slice();
    nc[v] = color;
    const opp = OTHER[color];
    let captured = [];
    const seen = new Set();
    for (const w of adj[v]) {
      if (nc[w] === opp && !seen.has(w)) {
        const { stones, libs } = chainAt(nc, adj, w);
        for (const s of stones) seen.add(s);
        if (libs.size === 0) {
          for (const s of stones) { nc[s] = EMPTY; captured.push(s); }
        }
      }
    }
    let suicided = [];
    const own = chainAt(nc, adj, v);
    if (own.libs.size === 0) {
      if (!self.allowSuicide) return { err: "suicide" };
      for (const s of own.stones) { nc[s] = EMPTY; suicided.push(s); }
    }
    const k = key(nc);
    if (self.posHist.includes(k)) return { err: "superko" };
    return { nc, captured, suicided, k };
  };

  self.play = function (v) {
    const r = self.trySim(v, self.toMove);
    if (r.err) return r;
    self.snaps.push({
      colors: self.colors, toMove: self.toMove,
      captures: self.captures.slice(), passes: self.passes,
      lastMove: self.lastMove,
    });
    self.captures[self.toMove] += r.captured.length;
    if (r.suicided.length) self.captures[OTHER[self.toMove]] += r.suicided.length;
    self.colors = r.nc;
    self.moves.push([self.toMove, v]);
    self.lastMove = v;
    self.toMove = OTHER[self.toMove];
    self.passes = 0;
    self.posHist.push(r.k);
    return { ok: true, captured: r.captured, suicided: r.suicided };
  };

  self.pass = function () {
    self.snaps.push({
      colors: self.colors, toMove: self.toMove,
      captures: self.captures.slice(), passes: self.passes,
      lastMove: self.lastMove,
    });
    self.moves.push([self.toMove, null]);
    self.toMove = OTHER[self.toMove];
    self.passes += 1;
    self.lastMove = null;
    self.posHist.push(self.posHist[self.posHist.length - 1]);
    return { ok: true, over: self.passes >= 2 };
  };

  self.undo = function () {
    if (!self.snaps.length) return false;
    const s = self.snaps.pop();
    self.colors = s.colors; self.toMove = s.toMove;
    self.captures = s.captures; self.passes = s.passes;
    self.lastMove = s.lastMove;
    self.moves.pop(); self.posHist.pop();
    return true;
  };

  self.gameOver = () => self.passes >= 2;

  self.score = function () {
    const terr = [0, 0, 0];
    const stones = [0, 0, 0];
    for (const c of self.colors) stones[c]++;
    const seen = new Set();
    const owner = new Array(n).fill(0);       // territory owner per empty vertex
    for (let v = 0; v < n; v++) {
      if (self.colors[v] !== EMPTY || seen.has(v)) continue;
      const comp = [v], border = new Set(), st = [v];
      seen.add(v);
      while (st.length) {
        const u = st.pop();
        for (const w of adj[u]) {
          const c = self.colors[w];
          if (c === EMPTY) {
            if (!seen.has(w)) { seen.add(w); comp.push(w); st.push(w); }
          } else border.add(c);
        }
      }
      let o = 0;
      if (border.size === 1) o = [...border][0];
      if (o) { terr[o] += comp.length; for (const u of comp) owner[u] = o; }
    }
    const b = stones[BLACK] + terr[BLACK];
    const w = stones[WHITE] + terr[WHITE] + self.komi;
    return { black: b, white: w, margin: b - w, owner,
             winner: b > w ? "Black" : w > b ? "White" : "Draw" };
  };

  self.passAlive = function (color) {
    const cid = new Array(n).fill(-1);
    const chains = [];
    for (let v = 0; v < n; v++) {
      if (self.colors[v] === color && cid[v] === -1) {
        const comp = [v], st = [v];
        cid[v] = chains.length;
        while (st.length) {
          const u = st.pop();
          for (const w of adj[u])
            if (self.colors[w] === color && cid[w] === -1) {
              cid[w] = chains.length; comp.push(w); st.push(w);
            }
        }
        chains.push(comp);
      }
    }
    if (!chains.length) return new Set();
    const rid = new Array(n).fill(-1);
    const regions = [];
    for (let v = 0; v < n; v++) {
      if (self.colors[v] !== color && rid[v] === -1) {
        const comp = [v], st = [v], border = new Set();
        rid[v] = regions.length;
        while (st.length) {
          const u = st.pop();
          for (const w of adj[u]) {
            if (self.colors[w] !== color) {
              if (rid[w] === -1) { rid[w] = regions.length; comp.push(w); st.push(w); }
            } else border.add(cid[w]);
          }
        }
        const empties = comp.filter(u => self.colors[u] === EMPTY);
        regions.push({ empties, border });
      }
    }
    const vital = regions.map(r => {
      const vs = new Set();
      for (const c of r.border) {
        let ok = true;
        for (const e of r.empties) {
          if (!adj[e].some(w => cid[w] === c)) { ok = false; break; }
        }
        if (ok) vs.add(c);
      }
      return vs;
    });
    const liveC = new Set(chains.keys());
    const liveR = new Set(regions.keys());
    let changed = true;
    while (changed) {
      changed = false;
      for (const c of [...liveC]) {
        let count = 0;
        for (const r of liveR) if (vital[r].has(c)) count++;
        if (count < 2) { liveC.delete(c); changed = true; }
      }
      for (const r of [...liveR]) {
        for (const b of regions[r].border)
          if (!liveC.has(b)) { liveR.delete(r); changed = true; break; }
      }
    }
    const out = new Set();
    for (const c of liveC) for (const s of chains[c]) out.add(s);
    return out;
  };

  return self;
}

// ---------- embeddings (pure math; THREE-free) ------------------------------
function torusPoint(u, v, nx, ny, R, r) {
  const th = 2 * Math.PI * u / nx, ph = 2 * Math.PI * v / ny;
  const q = R + r * Math.cos(ph);
  return [q * Math.cos(th), r * Math.sin(ph), q * Math.sin(th)];
}
function torusNormal(u, v, nx, ny) {
  const th = 2 * Math.PI * u / nx, ph = 2 * Math.PI * v / ny;
  return [Math.cos(ph) * Math.cos(th), Math.sin(ph), Math.cos(ph) * Math.sin(th)];
}
function mobiusPoint(u, v, nx, ny, R, w) {
  const th = 2 * Math.PI * u / nx;
  const t = w * (2 * v / (ny - 1) - 1);
  const q = R + t * Math.cos(th / 2);
  return [q * Math.cos(th), t * Math.sin(th / 2), q * Math.sin(th)];
}
function mobiusNormal(u, v, nx, ny, R, w) {
  const e = 1e-4;
  const p0 = mobiusPoint(u, v, nx, ny, R, w);
  const pu = mobiusPoint(u + e, v, nx, ny, R, w);
  const pv = mobiusPoint(u, v + e, nx, ny, R, w);
  const du = [pu[0] - p0[0], pu[1] - p0[1], pu[2] - p0[2]];
  const dv = [pv[0] - p0[0], pv[1] - p0[1], pv[2] - p0[2]];
  const c = [du[1] * dv[2] - du[2] * dv[1],
             du[2] * dv[0] - du[0] * dv[2],
             du[0] * dv[1] - du[1] * dv[0]];
  const r = Math.hypot(c[0], c[1], c[2]) || 1;
  return [c[0] / r, c[1] / r, c[2] / r];
}
function slerp(a, b, t) {
  let dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  dot = Math.min(1, Math.max(-1, dot));
  const th = Math.acos(dot);
  if (th < 1e-6) return a.slice();
  const s = Math.sin(th);
  const ka = Math.sin((1 - t) * th) / s, kb = Math.sin(t * th) / s;
  return [ka * a[0] + kb * b[0], ka * a[1] + kb * b[1], ka * a[2] + kb * b[2]];
}

// ---------- incidence colorings ----------------------------------------------
// Proper colorings of the board's incidence structure. Greedy over a fixed
// order: at most Delta+1 vertex colors (Brooks-adjacent) and 2*Delta-1 edge
// colors (Vizing gives Delta or Delta+1; greedy is close enough for paint).
function edgeList(adjacency) {
  const out = [];
  for (let a = 0; a < adjacency.length; a++)
    for (const b of adjacency[a]) if (a < b) out.push([a, b]);
  return out;
}

function greedyVertexColoring(adjacency) {
  const n = adjacency.length, color = new Array(n).fill(-1);
  for (let v = 0; v < n; v++) {
    const used = new Set();
    for (const w of adjacency[v]) if (color[w] >= 0) used.add(color[w]);
    let c = 0;
    while (used.has(c)) c++;
    color[v] = c;
  }
  return color;
}

function greedyEdgeColoring(adjacency) {
  const edges = edgeList(adjacency);
  const atVertex = adjacency.map(() => []);
  const colors = new Array(edges.length).fill(-1);
  edges.forEach(([a, b], i) => {
    const used = new Set(atVertex[a].concat(atVertex[b]));
    let c = 0;
    while (used.has(c)) c++;
    colors[i] = c;
    atVertex[a].push(c); atVertex[b].push(c);
  });
  return { edges, colors };
}

// Greedy coloring of faces so faces sharing an edge differ (for cell paint).
function greedyFaceColoring(faces) {
  const byEdge = new Map();
  const nbrs = faces.map(() => []);
  faces.forEach((f, fid) => {
    for (let i = 0; i < f.length; i++) {
      const a = f[i], b = f[(i + 1) % f.length];
      const k = a < b ? a + ":" + b : b + ":" + a;
      if (byEdge.has(k)) {
        const o = byEdge.get(k);
        nbrs[fid].push(o); nbrs[o].push(fid);
      } else byEdge.set(k, fid);
    }
  });
  const color = new Array(faces.length).fill(-1);
  for (let f = 0; f < faces.length; f++) {
    const used = new Set();
    for (const g of nbrs[f]) if (color[g] >= 0) used.add(color[g]);
    let c = 0;
    while (used.has(c)) c++;
    color[f] = c;
  }
  return color;
}

// ---------- share codec: game state <-> base64url string ---------------------
// A shared game is {v: 1, s: [surface, mesh, scaleIdx], m: moves}, where each
// move is a vertex index or -1 for a pass. Replaying through the engine
// validates every move, so a tampered code fails loudly rather than silently.
const _b64e = (typeof btoa !== "undefined") ? btoa
  : (s) => Buffer.from(s, "binary").toString("base64");
const _b64d = (typeof atob !== "undefined") ? atob
  : (s) => Buffer.from(s, "base64").toString("binary");

function encodeShare(spec, moves) {
  const json = JSON.stringify({ v: 1, s: spec, m: moves });
  const bytes = unescape(encodeURIComponent(json));
  return _b64e(bytes).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function decodeShare(code) {
  code = code.trim();
  const h = code.indexOf("#g=");
  if (h >= 0) code = code.slice(h + 3);          // accept a full shared URL
  code = code.replace(/-/g, "+").replace(/_/g, "/");
  while (code.length % 4) code += "=";
  const json = decodeURIComponent(escape(_b64d(code)));
  const st = JSON.parse(json);
  if (st.v !== 1 || !Array.isArray(st.s) || !Array.isArray(st.m))
    throw new Error("not a Geodesics game code");
  return st;
}

if (typeof module !== "undefined") {
  module.exports = { gridQuotient, boxLattice, geodesicSphere, goldbergSphere,
                     cubeSphere, Engine, chainAt,
                     torusPoint, torusNormal, mobiusPoint, mobiusNormal,
                     slerp, edgeList, greedyVertexColoring, greedyEdgeColoring,
                     greedyFaceColoring, encodeShare, decodeShare,
                     EMPTY, BLACK, WHITE };
}
