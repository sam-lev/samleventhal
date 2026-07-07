/*! GeoCells v0.1 — incidence relationships for Geodesics boards.
 *
 * Given a polygonal mesh (vertex count + face vertex-lists, optional
 * positions), builds the play graph for a chosen site type:
 *
 *   vertices  stones on vertices, adjacency along edges — canonical Go
 *   edges     stones on edges. Default adjacency is the LINE GRAPH (two
 *             edges connect when they share a vertex) — the same
 *             representation as the priors graph of Leventhal, Gyulassy,
 *             Pascucci & Heimann (NeurIPS 2022), where the arcs of a
 *             topological graph become the classified nodes.
 *             opts.edgeAdjacency = 'medial' instead connects edges that
 *             are consecutive around a face (4-regular on closed
 *             polyhedra).
 *   faces     stones on faces, adjacency across shared edges — the dual
 *   cells     stones on ALL cells (vertices ∪ edges ∪ faces) with Hasse
 *             incidence adjacency: v—e when v is an endpoint of e, e—f
 *             when e bounds f. Cross-dimensional Go.
 *
 * Output is board-shaped — { mode, n, neighbors, positions?, dim, meta } —
 * so everything downstream (rules, Zobrist, share codec, the AI engine,
 * rendering by site positions) is unchanged. Quotient identifications
 * (Möbius, Klein, RP²) are already baked into vertex indices, so the same
 * pipeline applies; doubled edges arising from identifications are merged.
 * Meshes without faces (1-D boards) support only 'vertices'.
 *
 * No dependencies. Browser: window.GeoCells. Node: module.exports.
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.GeoCells = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  const MODES = ["vertices", "edges", "faces", "cells"];

  /**
   * Build the incidence complex from a mesh.
   *   mesh: { faces: [[v0, v1, ...], ...], nVerts?, positions?: [[x,y(,z)], ...] }
   * Edges are derived from face boundaries and deduplicated by vertex pair.
   */
  function fromMesh(mesh) {
    const faces = (mesh.faces || []).map((f) => f.slice());
    let nV = mesh.nVerts != null ? mesh.nVerts : mesh.positions ? mesh.positions.length : 0;
    if (!nV) for (const f of faces) for (const v of f) nV = Math.max(nV, v + 1);
    const eIndex = new Map();
    const edges = [];
    const faceEdges = faces.map(() => []);
    const key = (a, b) => (a < b ? a + "," + b : b + "," + a);
    faces.forEach((f, fi) => {
      for (let i = 0; i < f.length; i++) {
        const a = f[i];
        const b = f[(i + 1) % f.length];
        if (a === b) continue;
        const k = key(a, b);
        let ei = eIndex.get(k);
        if (ei === undefined) {
          ei = edges.length;
          eIndex.set(k, ei);
          edges.push([Math.min(a, b), Math.max(a, b)]);
        }
        faceEdges[fi].push(ei);
      }
    });
    const vertEdges = Array.from({ length: nV }, () => []);
    edges.forEach(([a, b], ei) => {
      vertEdges[a].push(ei);
      vertEdges[b].push(ei);
    });
    const edgeFaces = edges.map(() => []);
    faceEdges.forEach((fe, fi) => fe.forEach((ei) => edgeFaces[ei].push(fi)));
    return { nV, faces, edges, faceEdges, vertEdges, edgeFaces, positions: mesh.positions || null };
  }

  const uniq = (arr) => Array.from(new Set(arr));
  const meta = (cx) => ({ nV: cx.nV, nE: cx.edges.length, nF: cx.faces.length });

  function midpoint(pos, a, b) {
    const d = pos[a].length;
    const c = new Array(d);
    for (let i = 0; i < d; i++) c[i] = (pos[a][i] + pos[b][i]) / 2;
    return c;
  }
  function centroid(pos, f) {
    const d = pos[f[0]].length;
    const c = new Array(d).fill(0);
    for (let j = 0; j < f.length; j++) for (let i = 0; i < d; i++) c[i] += pos[f[j]][i];
    return c.map((x) => x / f.length);
  }

  /** Build the play graph for one incidence mode. */
  function build(cx, mode, opts) {
    opts = opts || {};
    const pos = cx.positions;

    if (mode === "vertices") {
      const nb = Array.from({ length: cx.nV }, () => []);
      cx.edges.forEach(([a, b]) => {
        nb[a].push(b);
        nb[b].push(a);
      });
      return {
        mode,
        n: cx.nV,
        neighbors: nb.map(uniq),
        positions: pos ? pos.map((p) => p.slice()) : null,
        dim: new Int8Array(cx.nV),
        meta: meta(cx),
      };
    }

    if (mode === "edges") {
      const nE = cx.edges.length;
      const nb = Array.from({ length: nE }, () => new Set());
      if (opts.edgeAdjacency === "medial") {
        cx.faceEdges.forEach((fe) => {
          const L = fe.length;
          for (let i = 0; i < L; i++) {
            const a = fe[i];
            const b = fe[(i + 1) % L];
            if (a !== b) {
              nb[a].add(b);
              nb[b].add(a);
            }
          }
        });
      } else {
        // line graph: edges sharing a vertex
        cx.vertEdges.forEach((ve) => {
          for (let i = 0; i < ve.length; i++)
            for (let j = i + 1; j < ve.length; j++) {
              nb[ve[i]].add(ve[j]);
              nb[ve[j]].add(ve[i]);
            }
        });
      }
      return {
        mode,
        n: nE,
        neighbors: nb.map((s) => Array.from(s)),
        positions: pos ? cx.edges.map(([a, b]) => midpoint(pos, a, b)) : null,
        dim: new Int8Array(nE).fill(1),
        meta: meta(cx),
      };
    }

    if (mode === "faces") {
      const nF = cx.faces.length;
      const nb = Array.from({ length: nF }, () => new Set());
      cx.edgeFaces.forEach((fs) => {
        for (let i = 0; i < fs.length; i++)
          for (let j = i + 1; j < fs.length; j++)
            if (fs[i] !== fs[j]) {
              nb[fs[i]].add(fs[j]);
              nb[fs[j]].add(fs[i]);
            }
      });
      return {
        mode,
        n: nF,
        neighbors: nb.map((s) => Array.from(s)),
        positions: pos ? cx.faces.map((f) => centroid(pos, f)) : null,
        dim: new Int8Array(nF).fill(2),
        meta: meta(cx),
      };
    }

    if (mode === "cells") {
      const nV = cx.nV;
      const nE = cx.edges.length;
      const nF = cx.faces.length;
      const n = nV + nE + nF;
      const nb = Array.from({ length: n }, () => []);
      cx.edges.forEach(([a, b], ei) => {
        const e = nV + ei;
        nb[a].push(e);
        nb[b].push(e);
        nb[e].push(a, b);
      });
      cx.faceEdges.forEach((fe, fi) => {
        const F = nV + nE + fi;
        uniq(fe).forEach((ei) => {
          nb[nV + ei].push(F);
          nb[F].push(nV + ei);
        });
      });
      const positions = pos
        ? pos
            .map((p) => p.slice())
            .concat(cx.edges.map(([a, b]) => midpoint(pos, a, b)))
            .concat(cx.faces.map((f) => centroid(pos, f)))
        : null;
      const dim = new Int8Array(n);
      for (let i = 0; i < nE; i++) dim[nV + i] = 1;
      for (let i = 0; i < nF; i++) dim[nV + nE + i] = 2;
      return { mode, n, neighbors: nb.map(uniq), positions, dim, meta: meta(cx) };
    }

    throw new Error("GeoCells: unknown mode '" + mode + "' (expected " + MODES.join(" | ") + ")");
  }

  return { version: "0.1.0", MODES, fromMesh, build };
});
