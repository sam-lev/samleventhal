"""Layer 2 (extended) — meshes as manipulable objects.

A Mesh is an oriented closed 2-manifold: vertices with 3D coordinates and
faces given as consistently wound vertex cycles (counter-clockwise seen from
outside). This is the player-facing board-design layer: seed polyhedra can
be transformed by Conway operators and subdivision before being flattened
into a rule-engine Board.

Implemented operators (composable, applied right-to-left like Conway
notation):

    d  dual        vertices <-> faces
    k  kis         raise a pyramid on every face
    a  ambo        rectify: vertices at edge midpoints
    t  truncate    cut every vertex           (identity: t = dkd)
    j  join        rhombic faces per edge     (identity: j = da)

Classical combinations come free:  e = aa (expand), o = jj (ortho),
b = ta (bevel), m = kj (meet), n = kd (needle).  Examples against the
canonical catalogue (dmccooey.com/polyhedra):

    t(I)  = truncated icosahedron        60 v / 90 e / 32 f   (soccer ball)
    a(C)  = cuboctahedron                12 / 24 / 14
    aa(D) = rhombicosidodecahedron       60 / 120 / 62
    ta(D) = great rhombicosidodecahedron 120 / 180 / 62

Subdivision surfaces:

    geodesic_subdivide(m, f)   Class-I {3,q+}_(f,0) on any all-triangle mesh
    quad_subdivide(m, f)       f x f quad grids on any all-quad mesh (cube
                               -> "cube-sphere")

The rotation system (cyclic order of edges around each vertex) is derived
from face orientation — the combinatorial-map machinery the design report
calls for, in its orientable special case.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .board import Board

Vec = tuple[float, float, float]


def _add(a: Vec, b: Vec) -> Vec:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scale(a: Vec, s: float) -> Vec:
    return (a[0] * s, a[1] * s, a[2] * s)


def _norm(a: Vec) -> Vec:
    r = math.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2)
    return (a[0] / r, a[1] / r, a[2] / r)


def _cross(a: Vec, b: Vec) -> Vec:
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _dot(a: Vec, b: Vec) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@dataclass
class Mesh:
    verts: list[Vec]
    faces: list[tuple[int, ...]]
    name: str = "mesh"
    history: list[str] = field(default_factory=list)

    @property
    def v(self) -> int:
        return len(self.verts)

    def edges(self) -> list[tuple[int, int]]:
        es = set()
        for f in self.faces:
            for i in range(len(f)):
                a, b = f[i], f[(i + 1) % len(f)]
                es.add((a, b) if a < b else (b, a))
        return sorted(es)

    @property
    def e(self) -> int:
        return len(self.edges())

    @property
    def f(self) -> int:
        return len(self.faces)

    def euler(self) -> int:
        return self.v - self.e + self.f

    def face_sizes(self) -> dict[int, int]:
        out: dict[int, int] = {}
        for f in self.faces:
            out[len(f)] = out.get(len(f), 0) + 1
        return out

    def vertex_degrees(self) -> dict[int, int]:
        deg = [0] * self.v
        for a, b in self.edges():
            deg[a] += 1
            deg[b] += 1
        out: dict[int, int] = {}
        for d in deg:
            out[d] = out.get(d, 0) + 1
        return out


# ---------------------------------------------------------------------------
# validation & rotation system
# ---------------------------------------------------------------------------

def check_closed_oriented(m: Mesh) -> None:
    """Every directed edge appears exactly once, and with its reverse:
    the definition of a closed, consistently oriented 2-manifold mesh."""
    seen: set[tuple[int, int]] = set()
    for f in m.faces:
        for i in range(len(f)):
            de = (f[i], f[(i + 1) % len(f)])
            if de in seen:
                raise ValueError(f"directed edge {de} repeated: bad orientation")
            seen.add(de)
    for a, b in seen:
        if (b, a) not in seen:
            raise ValueError(f"edge {(a, b)} has no opposite: mesh not closed")


def _directed_edge_faces(m: Mesh) -> dict[tuple[int, int], int]:
    de: dict[tuple[int, int], int] = {}
    for fid, f in enumerate(m.faces):
        for i in range(len(f)):
            de[(f[i], f[(i + 1) % len(f)])] = fid
    return de


def vertex_rotations(m: Mesh):
    """For each vertex: (neighbor cycle, incident-face cycle), in the
    rotational order induced by the mesh orientation."""
    de = _directed_edge_faces(m)
    nxt: dict[tuple[int, int], int] = {}     # (u,v) -> vertex after v in that face
    for fid, f in enumerate(m.faces):
        k = len(f)
        for i in range(k):
            nxt[(f[i], f[(i + 1) % k])] = f[(i + 2) % k]
    incoming: dict[int, int] = {}
    for (u, v) in de:
        incoming.setdefault(v, u)
    nbr_cycles, face_cycles = [], []
    for v in range(m.v):
        u0 = incoming[v]
        u, nbrs, fcs = u0, [], []
        while True:
            fcs.append(de[(u, v)])
            w = nxt[(u, v)]
            nbrs.append(w)
            u = w
            if u == u0:
                break
        nbr_cycles.append(nbrs)
        face_cycles.append(fcs)
    return nbr_cycles, face_cycles


def _face_centroid(m: Mesh, f: tuple[int, ...]) -> Vec:
    c = (0.0, 0.0, 0.0)
    for v in f:
        c = _add(c, m.verts[v])
    return _scale(c, 1.0 / len(f))


# ---------------------------------------------------------------------------
# Conway operators
# ---------------------------------------------------------------------------

def dual(m: Mesh) -> Mesh:
    """d — vertices become faces and vice versa."""
    _, face_cycles = vertex_rotations(m)
    verts = [_face_centroid(m, f) for f in m.faces]
    faces = [tuple(reversed(fc)) for fc in face_cycles]
    out = Mesh(verts, faces, name=f"d{m.name}", history=m.history + ["d"])
    check_closed_oriented(out)
    return out


def kis(m: Mesh) -> Mesh:
    """k — erect a pyramid on every face (all faces become triangles)."""
    verts = list(m.verts)
    faces: list[tuple[int, ...]] = []
    for f in m.faces:
        c = len(verts)
        verts.append(_face_centroid(m, f))
        k = len(f)
        for i in range(k):
            faces.append((f[i], f[(i + 1) % k], c))
    out = Mesh(verts, faces, name=f"k{m.name}", history=m.history + ["k"])
    check_closed_oriented(out)
    return out


def ambo(m: Mesh) -> Mesh:
    """a — rectification: new vertices at edge midpoints."""
    mid: dict[frozenset[int], int] = {}
    verts: list[Vec] = []

    def midv(a: int, b: int) -> int:
        key = frozenset((a, b))
        if key not in mid:
            mid[key] = len(verts)
            verts.append(_scale(_add(m.verts[a], m.verts[b]), 0.5))
        return mid[key]

    faces: list[tuple[int, ...]] = []
    for f in m.faces:                                   # shrunk original faces
        k = len(f)
        faces.append(tuple(midv(f[i], f[(i + 1) % k]) for i in range(k)))
    nbr_cycles, _ = vertex_rotations(m)                 # vertex figures
    for v, nbrs in enumerate(nbr_cycles):
        faces.append(tuple(reversed([midv(v, u) for u in nbrs])))
    out = Mesh(verts, faces, name=f"a{m.name}", history=m.history + ["a"])
    check_closed_oriented(out)
    return out


def truncate(m: Mesh) -> Mesh:
    """t = dkd — cut every vertex."""
    out = dual(kis(dual(m)))
    out.name = f"t{m.name}"
    out.history = m.history + ["t"]
    return out


def join(m: Mesh) -> Mesh:
    """j = da — one rhombus per original edge."""
    out = dual(ambo(m))
    out.name = f"j{m.name}"
    out.history = m.history + ["j"]
    return out


_OPS = {"d": dual, "k": kis, "a": ambo, "t": truncate, "j": join}


def conway(ops: str, m: Mesh) -> Mesh:
    """Apply a Conway operator string right-to-left, e.g. conway('ta', D)."""
    for ch in reversed(ops):
        if ch not in _OPS:
            raise KeyError(f"unknown Conway operator '{ch}' (have {sorted(_OPS)})")
        m = _OPS[ch](m)
    return m


def project_to_sphere(m: Mesh, radius: float = 1.0) -> Mesh:
    out = Mesh([_scale(_norm(p), radius) for p in m.verts], list(m.faces),
               name=m.name, history=m.history + ["project"])
    return out


# ---------------------------------------------------------------------------
# subdivision (generalized from the icosahedral special case)
# ---------------------------------------------------------------------------

def geodesic_subdivide(m: Mesh, f: int, project: bool = True) -> Mesh:
    """Class-I frequency-f subdivision of an all-triangle mesh.

    Vertex identification uses exact combinatorial keys (never float
    comparison): base vertices, edge-fraction points, and face-lattice
    points dedupe deterministically across shared faces.
    """
    if f < 1:
        raise ValueError("frequency must be >= 1")
    if any(len(face) != 3 for face in m.faces):
        raise ValueError("geodesic subdivision needs an all-triangle mesh")
    base = m.verts
    verts: list[Vec] = []
    index: dict[tuple, int] = {}

    def key_pos(fid, face, i, j, k):
        a, b, c = face
        if j == 0 and k == 0:
            return ("v", a), base[a]
        if i == 0 and k == 0:
            return ("v", b), base[b]
        if i == 0 and j == 0:
            return ("v", c), base[c]

        def edge(p, q, wq):
            if p > q:
                p, q, wq = q, p, f - wq
            pos = tuple((f - wq) * base[p][d] + wq * base[q][d] for d in range(3))
            return ("e", p, q, wq), _scale(pos, 1.0 / f)
        if k == 0:
            return edge(a, b, j)
        if j == 0:
            return edge(a, c, k)
        if i == 0:
            return edge(b, c, k)
        pos = tuple(i * base[a][d] + j * base[b][d] + k * base[c][d]
                    for d in range(3))
        return ("f", fid, i, j), _scale(pos, 1.0 / f)

    def vid(fid, face, i, j, k):
        key, pos = key_pos(fid, face, i, j, k)
        if key not in index:
            index[key] = len(verts)
            verts.append(pos)
        return index[key]

    faces_out: list[tuple[int, ...]] = []
    for fid, face in enumerate(m.faces):
        grid = {}
        for i in range(f + 1):
            for j in range(f + 1 - i):
                grid[(i, j)] = vid(fid, face, i, j, f - i - j)
        for i in range(f):
            for j in range(f - i):
                faces_out.append((grid[(i, j)], grid[(i + 1, j)],
                                  grid[(i, j + 1)]))
                if i + j <= f - 2:
                    faces_out.append((grid[(i + 1, j)], grid[(i + 1, j + 1)],
                                      grid[(i, j + 1)]))
    out = Mesh(verts, faces_out, name=f"{m.name}~{f}",
               history=m.history + [f"geodesic({f})"])
    check_closed_oriented(out)
    return project_to_sphere(out) if project else out


def quad_subdivide(m: Mesh, f: int, project: bool = True) -> Mesh:
    """f x f subdivision of an all-quad mesh (cube -> cube-sphere)."""
    if f < 1:
        raise ValueError("frequency must be >= 1")
    if any(len(face) != 4 for face in m.faces):
        raise ValueError("quad subdivision needs an all-quad mesh")
    base = m.verts
    verts: list[Vec] = []
    index: dict[tuple, int] = {}

    def bilinear(face, i, j):
        a, b, c, d = (base[v] for v in face)
        s, t = i / f, j / f
        return tuple((1 - s) * (1 - t) * a[q] + s * (1 - t) * b[q]
                     + s * t * c[q] + (1 - s) * t * d[q] for q in range(3))

    def key_of(fid, face, i, j):
        a, b, c, d = face
        corners = {(0, 0): a, (f, 0): b, (f, f): c, (0, f): d}
        if (i, j) in corners:
            return ("v", corners[(i, j)])

        def edge(p, q, w):
            if p > q:
                p, q, w = q, p, f - w
            return ("e", p, q, w)
        if j == 0:
            return edge(a, b, i)
        if i == f:
            return edge(b, c, j)
        if j == f:
            return edge(d, c, i)
        if i == 0:
            return edge(a, d, j)
        return ("f", fid, i, j)

    def vid(fid, face, i, j):
        key = key_of(fid, face, i, j)
        if key not in index:
            index[key] = len(verts)
            verts.append(bilinear(face, i, j))
        return index[key]

    faces_out: list[tuple[int, ...]] = []
    for fid, face in enumerate(m.faces):
        grid = {(i, j): vid(fid, face, i, j)
                for i in range(f + 1) for j in range(f + 1)}
        for i in range(f):
            for j in range(f):
                faces_out.append((grid[(i, j)], grid[(i + 1, j)],
                                  grid[(i + 1, j + 1)], grid[(i, j + 1)]))
    out = Mesh(verts, faces_out, name=f"{m.name}#{f}",
               history=m.history + [f"quad({f})"])
    check_closed_oriented(out)
    return project_to_sphere(out) if project else out


# ---------------------------------------------------------------------------
# seed polyhedra
# ---------------------------------------------------------------------------

def _orient_convex(verts: list[Vec], faces: list[tuple[int, ...]],
                   name: str) -> Mesh:
    """Fix each face's winding so its Newell normal points away from the
    origin (valid for convex, origin-centered seeds), then validate."""
    fixed = []
    for f in faces:
        n = (0.0, 0.0, 0.0)
        for i in range(len(f)):
            n = _add(n, _cross(verts[f[i]], verts[f[(i + 1) % len(f)]]))
        c = _face_centroid(Mesh(verts, [f]), f)
        fixed.append(tuple(reversed(f)) if _dot(n, c) < 0 else tuple(f))
    m = Mesh(verts, fixed, name=name)
    check_closed_oriented(m)
    return m


def tetrahedron() -> Mesh:
    v = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
    f = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    return _orient_convex([_norm(p) for p in v], f, "T")


def cube() -> Mesh:
    v = [(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
    f = [(0, 1, 3, 2), (4, 5, 7, 6),      # x = -1, x = +1
         (0, 1, 5, 4), (2, 3, 7, 6),      # y = -1, y = +1
         (0, 2, 6, 4), (1, 3, 7, 5)]      # z = -1, z = +1
    return _orient_convex([_norm(p) for p in v], f, "C")


def octahedron() -> Mesh:
    v = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    f = [(0, 2, 4), (0, 4, 3), (0, 3, 5), (0, 5, 2),
         (1, 2, 4), (1, 4, 3), (1, 3, 5), (1, 5, 2)]
    return _orient_convex(v, f, "O")


def icosahedron() -> Mesh:
    t = (1 + math.sqrt(5)) / 2
    v = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
         (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
         (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
         (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
         (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
         (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    return _orient_convex([_norm(p) for p in v], f, "I")


def dodecahedron() -> Mesh:
    m = dual(icosahedron())
    m = project_to_sphere(m)
    m.name, m.history = "D", []
    return m


SEEDS = {
    "T": tetrahedron, "tetrahedron": tetrahedron,
    "C": cube, "cube": cube,
    "O": octahedron, "octahedron": octahedron,
    "D": dodecahedron, "dodecahedron": dodecahedron,
    "I": icosahedron, "icosahedron": icosahedron,
}


def seed(name: str) -> Mesh:
    if name not in SEEDS:
        raise KeyError(f"unknown seed '{name}' (have T, C, O, D, I)")
    return SEEDS[name]()


# ---------------------------------------------------------------------------
# mesh -> Board
# ---------------------------------------------------------------------------

def board_from_mesh(m: Mesh, name: str | None = None,
                    params: dict | None = None) -> Board:
    """Flatten a mesh into a rule-engine Board (stones on vertices)."""
    nbrs: list[set[int]] = [set() for _ in range(m.v)]
    for a, b in m.edges():
        nbrs[a].add(b)
        nbrs[b].add(a)
    board = Board(
        name=name or "mesh",
        params=params or {},
        adj=[tuple(sorted(s)) for s in nbrs],
        coords=list(m.verts),
        faces=list(m.faces),
        meta={"mesh_name": m.name, "history": list(m.history)},
    )
    board.validate()
    return board
