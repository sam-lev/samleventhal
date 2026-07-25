"""cells.py — incidence structures: play Go on the edges, faces, or all
cells of a mesh, not just its vertices.

This is the Python twin of the web app's GeoCells (web/cells.js), kept
behavior-identical so that browser games and training runs agree on every
derived graph. Given a Board that carries a face list, `derive` rebuilds
the play graph for a chosen *site* type:

    vertices  stones on vertices, adjacency along edges — canonical Go
              (the identity: the input Board is returned unchanged)
    edges     stones on the mesh's edges. Two sites are adjacent when the
              edges share a vertex — the LINE GRAPH L(G), the same
              representation as the priors graph of Leventhal, Gyulassy,
              Pascucci & Heimann (NeurIPS 2022), where the arcs of a
              topological graph become the classified nodes.
    faces     stones on faces, adjacency across shared edges — the DUAL
              graph. (The geodesic sphere's dual is the Goldberg board;
              this generalizes that construction to every surface.)
    cells     stones on ALL cells (vertices ∪ edges ∪ faces), adjacent by
              Hasse incidence: v—e when v is an endpoint of e, e—f when e
              bounds f. Cross-dimensional Go.

The rules never change: liberties, capture, superko and scoring only ever
read the adjacency graph, whichever graph that is. Quotient identifications
(Möbius, Klein, RP²) are already baked into vertex indices upstream, so the
same pipeline applies unchanged; doubled edges arising from identifications
are merged.

'vertices' and 'edges' are available on every board — the line graph needs
only the adjacency graph. 'faces' and 'cells' need a complete face list
(every edge bounding at least one face); boards without one (1-D boards,
3-D lattices, hex meshes on flipped quotients — see topology._grid_faces —
and bounded honeycombs with an open rim) reject those modes with a message
naming the board.

Derived Boards keep the Board contract (adj / coords / meta) so the rule
engine, gSGF, and the training stack are untouched. When the base board was
built from the spec API, the derived board is registered under the name
'incidence' with params {mode, **spec}, so gSGF files of derived-board
games reconstruct exactly via topology.build("incidence", ...).
"""

from __future__ import annotations

from .board import Board

MODES = ("vertices", "edges", "faces", "cells")


# ---------------------------------------------------------------------------
# The incidence complex
# ---------------------------------------------------------------------------

class Complex:
    """The full V–E–F incidence data of a polygonal mesh.

    Edges are derived from face boundaries and deduplicated by (unordered)
    vertex pair. Fields:

        n_v         number of vertices
        faces       face vertex-cycles (lists of vertex ids)
        edges       [(a, b), ...] with a < b, one per undirected edge
        face_edges  face id  -> list of incident edge ids (in cycle order)
        vert_edges  vertex id -> list of incident edge ids
        edge_faces  edge id   -> list of faces it bounds (1 on a boundary,
                                 2 in the interior of a surface)
    """

    def __init__(self, board: Board):
        if not board.faces:
            raise ValueError(
                f"board '{board.name}' carries no face list: only "
                f"'vertices' incidence is available on it")
        self.n_v = board.n
        self.faces = [list(f) for f in board.faces]
        e_index: dict[tuple[int, int], int] = {}
        self.edges: list[tuple[int, int]] = []
        self.face_edges: list[list[int]] = []
        for f in self.faces:
            fe = []
            for k in range(len(f)):
                a, b = f[k], f[(k + 1) % len(f)]
                if a == b:                       # collapsed corner: skip
                    continue
                key = (a, b) if a < b else (b, a)
                ei = e_index.get(key)
                if ei is None:
                    ei = len(self.edges)
                    e_index[key] = ei
                    self.edges.append(key)
                fe.append(ei)
            self.face_edges.append(fe)
        # NOTE: edges of the board graph that bound *no* face (possible on
        # partial complexes) are intentionally absent here: incidence play
        # is defined on the mesh's cells, and topology attaches face lists
        # only when the complex is complete — so in practice the edge sets
        # coincide (asserted by derive()).
        self.vert_edges: list[list[int]] = [[] for _ in range(self.n_v)]
        for ei, (a, b) in enumerate(self.edges):
            self.vert_edges[a].append(ei)
            self.vert_edges[b].append(ei)
        self.edge_faces: list[list[int]] = [[] for _ in self.edges]
        for fi, fe in enumerate(self.face_edges):
            for ei in fe:
                if fi not in self.edge_faces[ei]:
                    self.edge_faces[ei].append(fi)

    def counts(self) -> dict:
        return {"nV": self.n_v, "nE": len(self.edges), "nF": len(self.faces)}


# ---------------------------------------------------------------------------
# Site positions (for rendering / geometric features; rules never use them)
# ---------------------------------------------------------------------------

def _midpoint(coords, a, b):
    return tuple((pa + pb) / 2 for pa, pb in zip(coords[a], coords[b]))


def _centroid(coords, f):
    d = len(coords[f[0]])
    c = [0.0] * d
    for v in f:
        for i in range(d):
            c[i] += coords[v][i]
    return tuple(x / len(f) for x in c)


# ---------------------------------------------------------------------------
# The derivation
# ---------------------------------------------------------------------------

def derive(board: Board, mode: str = "vertices") -> Board:
    """Rebuild `board` with `mode` cells as the playable sites.

    Availability:

      * vertices — always (the identity derivation).
      * edges    — always: the line graph is defined by the adjacency graph
                   alone, so it exists on every board, faces or not.
      * faces / cells — only when the board carries a *complete* face list
                   (every graph edge bounds at least one face). A partial
                   complex — e.g. the open rim of a bounded honeycomb —
                   would silently drop playable structure, so it is
                   rejected with an explanatory error instead.

    Returns a fresh Board whose adjacency is the derived graph; coords are
    carried over (vertex positions / edge midpoints / face centroids) when
    the base board has them. meta records the base counts and the per-site
    dimension list (0 = vertex, 1 = edge, 2 = face) — the analogue of
    GeoCells' `dim` array, used e.g. for per-class defect statistics.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, not '{mode}'")
    if mode == "vertices":
        return board                              # the identity derivation

    coords = board.coords
    if mode == "edges":
        # LINE GRAPH: needs only the adjacency graph. Sites are the board's
        # undirected edges; two sites are adjacent when they share a vertex.
        #
        # Site ORDERING matters for interop: a gSGF move is a site index, so
        # this must match the web app's GeoCells numbering exactly. GeoCells
        # discovers edges by walking the face boundaries in face order;
        # whenever the board carries a complete face complex we reproduce
        # that enumeration (Complex walks faces identically). Boards without
        # one (3-D lattices, open honeycombs) fall back to the sorted edge
        # list — those boards are python-only for non-vertex play anyway,
        # so the two orderings never meet in a file.
        edges = None
        if board.faces:
            cx = Complex(board)
            if set(cx.edges) == set(board.edges()):
                edges = cx.edges                  # web-identical enumeration
        if edges is None:
            edges = board.edges()
        n = len(edges)
        vert_edges: list[list[int]] = [[] for _ in range(board.n)]
        for ei, (a, b) in enumerate(edges):
            vert_edges[a].append(ei)
            vert_edges[b].append(ei)
        nb: list[set[int]] = [set() for _ in range(n)]
        for ve in vert_edges:
            for i in range(len(ve)):
                for j in range(i + 1, len(ve)):
                    nb[ve[i]].add(ve[j])
                    nb[ve[j]].add(ve[i])
        pos = ([_midpoint(coords, a, b) for a, b in edges]
               if coords else None)
        dim = [1] * n
        counts = {"nV": board.n, "nE": n,
                  "nF": len(board.faces) if board.faces else 0}
    else:
        # FACES / CELLS: need the full incidence complex.
        cx = Complex(board)
        counts = cx.counts()
        cover = set(cx.edges)
        missing = [e for e in board.edges() if e not in cover]
        if missing:
            raise ValueError(
                f"board '{board.name}': the face complex covers only part of "
                f"the edge set ({len(missing)} uncovered edges, e.g. an open "
                f"honeycomb rim) — '{mode}' incidence is unavailable here")
        if mode == "faces":
            # DUAL GRAPH: two faces are adjacent when they share an edge
            n = len(cx.faces)
            nb = [set() for _ in range(n)]
            for fs in cx.edge_faces:
                for i in range(len(fs)):
                    for j in range(i + 1, len(fs)):
                        if fs[i] != fs[j]:
                            nb[fs[i]].add(fs[j])
                            nb[fs[j]].add(fs[i])
            pos = ([_centroid(coords, f) for f in cx.faces]
                   if coords else None)
            dim = [2] * n
        else:
            # ALL CELLS: the Hasse diagram of the face poset
            n_v, n_e, n_f = cx.n_v, len(cx.edges), len(cx.faces)
            n = n_v + n_e + n_f
            nb = [set() for _ in range(n)]
            for ei, (a, b) in enumerate(cx.edges):    # v — e incidences
                e = n_v + ei
                nb[a].add(e)
                nb[b].add(e)
                nb[e].add(a)
                nb[e].add(b)
            for fi, fe in enumerate(cx.face_edges):   # e — f incidences
                f = n_v + n_e + fi
                for ei in set(fe):
                    nb[n_v + ei].add(f)
                    nb[f].add(n_v + ei)
            pos = None
            if coords:
                pos = ([tuple(p) for p in coords]
                       + [_midpoint(coords, a, b) for a, b in cx.edges]
                       + [_centroid(coords, f) for f in cx.faces])
            dim = [0] * n_v + [1] * n_e + [2] * n_f

    # name/params: reproducible through topology.build("incidence", ...)
    # whenever the base board came from the spec API; otherwise the base
    # constructor reference is embedded for provenance only.
    if board.name == "spec":
        params = {"mode": mode, **board.params}
    else:
        params = {"mode": mode,
                  "base": {"type": board.name, "params": board.params}}
    out = Board(
        name="incidence",
        params=params,
        adj=[tuple(sorted(s)) for s in nb],
        coords=pos,
        faces=None,                # derived graphs carry no 2-cells of their own
        meta={**counts, "mode": mode, "dim": dim,
              "base_name": board.name, "base_meta": dict(board.meta)},
    )
    out.validate()
    return out
