"""Layers 1 + 2 — topology primitives, discretizations, and the board spec.

The player-facing entry point is `make_board`, which assembles a Board from
four orthogonal choices:

    surface     which manifold        sphere | plane | cylinder | torus |
                                      mobius | klein | rp2
    mesh        which tiling          tri (degree-6 vertices) |
                                      square (degree-4) | hex (degree-3)
    resolution  abstract scale        1, 2, 3, ...  (see table below)
    dimension   1, 2, or 3            1: path/cycle; 2: surfaces;
                                      3: cubic lattices (box, T^3)

Stones sit on the *vertices* of the named tiling, so the mesh type sets the
local branching factor — the single most strategy-shaping knob after the
topology itself: degree 3 boards make liberties scarce everywhere (the whole
board plays like the first line), degree 6 boards make chains hard to kill.

On the sphere the three mesh types become the three classical families of
"almost-regular" tilings, each with its Euler-mandated defects:

    tri     geodesic polyhedron {3,5+}_(r,0)   12 vertices of degree 5
    square  cube-sphere (quad subdivision)      8 vertices of degree 3
    hex     Goldberg polyhedron GP(r,0)        12 pentagonal faces

Beyond the spec API, `conway_board` exposes seed polyhedra (T, C, O, D, I)
transformed by Conway operator strings (see geodesics.mesh) — e.g.
conway_board('I', 't') is Go on the soccer ball, conway_board('D', 'aa')
on the rhombicosidodecahedron.

Approximate vertex counts by resolution r:

    dim 2 grid, tri/square    (2r+3)^2            25, 49, 81, ... 361 @ r=8
    dim 2 grid, hex           ~(2r+4)^2           36, 64, 100, ...
    sphere tri                10r^2+2             12, 42, 92, 162, ...
    sphere square             6(r+1)^2+2          26, 56, 98, ...
    sphere hex                20r^2               20, 80, 180, 320, ...
    dim 3                     (r+3)^3             64, 125, 216, ...
    dim 1                     4r+5                9, 13, 17, ...
"""

from __future__ import annotations

import math

from .board import Board
from . import mesh as _mesh

# ---------------------------------------------------------------------------
# Family 1: grid quotients (fundamental polygons), with mesh types
# ---------------------------------------------------------------------------

MESH_TYPES = ("tri", "square", "hex")


def _grid_faces(nx: int, ny: int, wrap_x: bool, wrap_y: bool,
                flip_x: bool, flip_y: bool, mesh: str,
                adj: list) -> list | None:
    """Face list of a grid quotient — the 2-cells of the tiling, with seam
    identifications applied.

    Faces are enumerated on the universal cover (one quad / two triangles /
    one brick-wall hexagon per unit cell) and each corner is reduced back
    into the fundamental domain by the same deck transformations the edge
    construction uses. A candidate face is kept only if it *closes* in the
    quotient graph:

      * every reduced corner exists (open boundaries drop the outer ring),
      * all corner ids are distinct (identifications can collapse corners —
        e.g. at the doubly-flipped RP² corners),
      * every boundary edge of the face is an actual edge of the quotient
        (on a Möbius honeycomb the flip shears the bond parity at the seam,
        so seam hexagons genuinely do not close).

    Faces are deduplicated by vertex set (a small fundamental domain can
    reach the same 2-cell from two cover cells).

    Returns None — "this board carries no face data" — for hex meshes on
    flipped quotients: there the honeycomb's seam cells cannot close, and a
    *partial* face complex would be worse than none, because downstream
    consumers (HATZ's orientation cocycle, face-incidence play) would see a
    complex that never crosses the seam and silently conclude the surface
    is orientable. Absent faces keep the honest fallbacks in charge.
    """
    if mesh == "hex" and (flip_x or flip_y):
        return None

    def vid(x: int, y: int) -> int:
        return y * nx + x

    def red(x: int, y: int):
        """reduce_pt, restated for face corners (same deck transformations)."""
        if x >= nx:
            if not wrap_x:
                return None
            x -= nx
            if flip_x:
                y = ny - 1 - y
        if y >= ny:
            if not wrap_y:
                return None
            y -= ny
            if flip_y:
                x = nx - 1 - x
        elif y < 0:
            if not wrap_y:
                return None
            y += ny
            if flip_y:
                x = nx - 1 - x
        return x, y

    # the quotient's edge set, for the closure check
    ek = set()
    for a, nbrs in enumerate(adj):
        for b in nbrs:
            ek.add((a, b) if a < b else (b, a))

    def closed(ids: list[int]) -> bool:
        return all(
            ((ids[k], ids[(k + 1) % len(ids)]) if ids[k] < ids[(k + 1) % len(ids)]
             else (ids[(k + 1) % len(ids)], ids[k])) in ek
            for k in range(len(ids)))

    # cover-cell templates: corner offsets, one list per face of the cell
    if mesh == "square":
        templates = [[(0, 0), (1, 0), (1, 1), (0, 1)]]
    elif mesh == "tri":
        templates = [[(0, 0), (1, 0), (1, 1)], [(0, 0), (1, 1), (0, 1)]]
    else:  # hex: one brick per cell whose left wall parity matches the row
        templates = [[(0, 0), (1, 0), (2, 0), (2, 1), (1, 1), (0, 1)]]

    faces: list[tuple[int, ...]] = []
    seen: set[frozenset[int]] = set()
    ymax = ny if wrap_y else ny - 1
    for y in range(ymax):
        xs = range(y % 2, nx, 2) if mesh == "hex" else range(nx)
        for x in xs:
            for corners in templates:
                pts = [red(x + dx, y + dy) for dx, dy in corners]
                if any(p is None for p in pts):
                    continue                      # ran off an open boundary
                ids = [vid(*p) for p in pts]
                key = frozenset(ids)
                if len(key) != len(ids) or key in seen:
                    continue                      # collapsed or duplicate
                if not closed(ids):
                    continue                      # a boundary edge is absent
                seen.add(key)
                faces.append(tuple(ids))
    return faces or None


def _grid_quotient(name: str, nx: int, ny: int, wrap_x: bool, wrap_y: bool,
                   flip_x: bool, flip_y: bool, mesh: str = "square") -> Board:
    """Grid with optional side identifications and a choice of tiling.

    wrap_x glues column nx-1 to column 0; flip_x reverses the y index across
    that seam (orientation-reversing). Mesh types:

      square  the plain grid (degree 4)
      tri     grid plus one diagonal per cell (degree 6); across flipped
              seams the lattice chirality reverses — the graph stays valid,
              which is all the rules require
      hex     brick-wall honeycomb (degree 3): vertical edges only where
              (x + y) is even; wrapping imposes parity conditions
    """
    if mesh not in MESH_TYPES:
        raise ValueError(f"mesh must be one of {MESH_TYPES}")
    if nx < 3 or ny < 3:
        raise ValueError("grid quotients need nx >= 3 and ny >= 3")
    if mesh == "hex":
        if (flip_x and wrap_y) or flip_y:
            raise ValueError(
                f"hex mesh on '{name}' is obstructed: the brick-wall parity "
                "cannot be made consistent across these identifications "
                "(pilot limitation — Klein bottle / RP2 need a two-site basis)")
        if wrap_x and nx % 2:
            raise ValueError("hex mesh with an x-wrap needs even nx")
        if wrap_y and ny % 2:
            raise ValueError("hex mesh with a y-wrap needs even ny")
        if flip_x and ny % 2 == 0:
            raise ValueError("hex mesh on a Möbius band needs odd ny")

    def vid(x: int, y: int) -> int:
        return y * nx + x

    def reduce_pt(x: int, y: int):
        """Map a universal-cover point one cell out of range back into the
        fundamental domain, applying seam identifications. An x-seam flip
        can push y to -1, which must then fold back through the
        y-identification (deck transformations compose)."""
        if x >= nx:
            if not wrap_x:
                return None
            x -= nx
            if flip_x:
                y = ny - 1 - y
        if y >= ny:
            if not wrap_y:
                return None
            y -= ny
            if flip_y:
                x = nx - 1 - x
        elif y < 0:
            if not wrap_y:
                return None
            y += ny
            if flip_y:
                x = nx - 1 - x
        return x, y

    edges: set[frozenset[int]] = set()

    def add(a, b) -> None:
        if b is not None:
            va, vb = vid(*a), vid(*b)
            if va != vb:
                edges.add(frozenset((va, vb)))

    for y in range(ny):
        for x in range(nx):
            add((x, y), reduce_pt(x + 1, y))                    # horizontal
            if mesh != "hex" or (x + y) % 2 == 0:               # vertical
                add((x, y), reduce_pt(x, y + 1))
            if mesh == "tri":                                   # diagonal
                add((x, y), reduce_pt(x + 1, y + 1))

    adj: list[list[int]] = [[] for _ in range(nx * ny)]
    for e in edges:
        a, b = tuple(e)
        adj[a].append(b)
        adj[b].append(a)

    coords = [(float(x), float(y)) for y in range(ny) for x in range(nx)]
    labels = [f"{x},{y}" for y in range(ny) for x in range(nx)]
    adj_t = [tuple(sorted(nbrs)) for nbrs in adj]
    board = Board(
        name=name,
        params={"nx": nx, "ny": ny, "mesh": mesh},
        adj=adj_t,
        coords=coords,
        # the tiling's 2-cells (None where the honeycomb cannot close):
        # enables Euler-characteristic checks, face/cell incidence play,
        # and face-based orientation cocycles on quotient boards
        faces=_grid_faces(nx, ny, wrap_x, wrap_y, flip_x, flip_y, mesh, adj_t),
        labels=labels,
        meta={"wrap_x": wrap_x, "wrap_y": wrap_y,
              "flip_x": flip_x, "flip_y": flip_y},
    )
    board.validate()
    return board


def plane(nx: int, ny: int, mesh: str = "square") -> Board:
    """The classical goban topology: a bounded disk."""
    return _grid_quotient("plane", nx, ny, False, False, False, False, mesh)


def cylinder(nx: int, ny: int, mesh: str = "square") -> Board:
    """Annulus / cylinder: wraps in x. Two boundary circles."""
    return _grid_quotient("cylinder", nx, ny, True, False, False, False, mesh)


def torus(nx: int, ny: int, mesh: str = "square") -> Board:
    """Torus: wraps both ways. No boundary, no corners; genus 1."""
    return _grid_quotient("torus", nx, ny, True, True, False, False, mesh)


def mobius(nx: int, ny: int, mesh: str = "square") -> Board:
    """Möbius band: x-wrap with an orientation-reversing flip.
    Non-orientable; a single boundary circle of length 2*nx."""
    return _grid_quotient("mobius", nx, ny, True, False, True, False, mesh)


def klein(nx: int, ny: int, mesh: str = "square") -> Board:
    """Klein bottle: one flipped seam plus a plain one. Closed,
    non-orientable, Euler characteristic 0. (square / tri meshes)"""
    return _grid_quotient("klein", nx, ny, True, True, True, False, mesh)


def rp2(nx: int, ny: int, mesh: str = "square") -> Board:
    """Real projective plane: antipodal square gluing (both seams flipped).
    Grid corners collapse doubled identifications to degree 3.
    (square / tri meshes)"""
    return _grid_quotient("rp2", nx, ny, True, True, True, True, mesh)


# ---------------------------------------------------------------------------
# Family 2: sphere meshes — tri / square / hex
# ---------------------------------------------------------------------------

def sphere_geodesic(frequency: int) -> Board:
    """Triangular sphere: Class-I geodesic icosahedron {3,5+}_(v,0).
    V = 10v^2+2, E = 30v^2, F = 20v^2; 12 degree-5 defects."""
    m = _mesh.geodesic_subdivide(_mesh.icosahedron(), frequency)
    b = _mesh.board_from_mesh(m, "sphere_geodesic", {"frequency": frequency})
    b.meta["schlafli"] = f"{{3,5+}}_({frequency},0)"
    return b


def sphere_quad(frequency: int) -> Board:
    """Square sphere: the cube-sphere (each cube face subdivided v x v and
    projected). V = 6v^2+2, E = 12v^2, F = 6v^2; 8 degree-3 defects."""
    m = _mesh.quad_subdivide(_mesh.cube(), frequency)
    return _mesh.board_from_mesh(m, "sphere_quad", {"frequency": frequency})


def sphere_goldberg(frequency: int) -> Board:
    """Hexagonal sphere: Goldberg polyhedron GP(v,0), the dual geodesic.
    V = 20v^2 (all degree 3), E = 30v^2, F = 10v^2+2 with exactly 12
    pentagonal faces among hexagons."""
    g = _mesh.geodesic_subdivide(_mesh.icosahedron(), frequency)
    m = _mesh.project_to_sphere(_mesh.dual(g))
    return _mesh.board_from_mesh(m, "sphere_goldberg", {"frequency": frequency})


# ---------------------------------------------------------------------------
# Family 3: dimension 1 and 3 lattices
# ---------------------------------------------------------------------------

def path(n: int) -> Board:
    """Dimension-1 'plane': Go on a line segment."""
    if n < 3:
        raise ValueError("path needs n >= 3")
    adj = [tuple(w for w in (v - 1, v + 1) if 0 <= w < n) for v in range(n)]
    return Board("path", {"n": n}, adj,
                 coords=[(float(i), 0.0) for i in range(n)])


def cycle(n: int) -> Board:
    """Dimension-1 'torus': Go on a circle."""
    if n < 3:
        raise ValueError("cycle needs n >= 3")
    adj = [((v - 1) % n, (v + 1) % n) for v in range(n)]
    coords = [(math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n))
              for i in range(n)]
    return Board("cycle", {"n": n}, adj, coords=coords)


def lattice3(nx: int, ny: int, nz: int,
             wrap: tuple[bool, bool, bool] = (False, False, False),
             name: str = "box3") -> Board:
    """Dimension-3 cubic lattice (6-adjacency). wrap=(True,)*3 is the
    3-torus T^3: closed, homogeneous, every vertex degree 6."""
    if min(nx, ny, nz) < 3:
        raise ValueError("3D lattices need every side >= 3")

    def vid(x, y, z):
        return (z * ny + y) * nx + x

    edges: set[frozenset[int]] = set()
    dims = (nx, ny, nz)
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                p = [x, y, z]
                for axis in range(3):
                    q = p.copy()
                    q[axis] += 1
                    if q[axis] == dims[axis]:
                        if not wrap[axis]:
                            continue
                        q[axis] = 0
                    a, b = vid(*p), vid(*q)
                    if a != b:
                        edges.add(frozenset((a, b)))
    adj: list[list[int]] = [[] for _ in range(nx * ny * nz)]
    for e in edges:
        a, b = tuple(e)
        adj[a].append(b)
        adj[b].append(a)
    coords = [(float(x), float(y), float(z))
              for z in range(nz) for y in range(ny) for x in range(nx)]
    board = Board(name, {"nx": nx, "ny": ny, "nz": nz, "wrap": list(wrap)},
                  [tuple(sorted(nbrs)) for nbrs in adj], coords=coords)
    board.validate()
    return board


def box3(nx: int, ny: int, nz: int) -> Board:
    return lattice3(nx, ny, nz, (False, False, False), "box3")


def torus3(nx: int, ny: int, nz: int) -> Board:
    return lattice3(nx, ny, nz, (True, True, True), "torus3")


# ---------------------------------------------------------------------------
# Family 4: Conway polyhedron boards (cf. dmccooey.com/polyhedra)
# ---------------------------------------------------------------------------

def conway_board(seed: str = "I", ops: str = "") -> Board:
    """Go on a polyhedron built from a seed (T, C, O, D, I) transformed by a
    Conway operator string, applied right-to-left. Examples:

        conway_board('I', 't')    truncated icosahedron (soccer ball, 60 pts)
        conway_board('D', 'aa')   rhombicosidodecahedron (60 pts, degree 4)
        conway_board('D', 'ta')   great rhombicosidodecahedron (120 pts)
        conway_board('C', 'k')    tetrakis-hexahedron-oid (14 pts)
    """
    m = _mesh.seed(seed)
    if ops:
        m = _mesh.conway(ops, m)
    m = _mesh.project_to_sphere(m)
    return _mesh.board_from_mesh(m, "conway", {"seed": seed, "ops": ops})


# ---------------------------------------------------------------------------
# The unified board spec
# ---------------------------------------------------------------------------

_SURFACE_ALIASES = {
    "sphere": "sphere", "s2": "sphere",
    "plane": "plane", "disk": "plane", "box": "plane",
    "cylinder": "cylinder", "annulus": "cylinder",
    "torus": "torus", "t2": "torus", "circle": "torus",
    "mobius": "mobius", "moebius": "mobius",
    "klein": "klein", "rp2": "rp2", "projective": "rp2",
}


def make_board(surface: str = "sphere", mesh: str = "tri",
               resolution: int = 3, dimension: int = 2, **overrides) -> Board:
    """Assemble a Board from an abstract spec.

    surface     sphere | plane | cylinder | torus | mobius | klein | rp2
                (dim 1: plane -> path, torus/circle -> cycle;
                 dim 3: plane/box -> box lattice, torus -> T^3,
                        cylinder -> x-periodic slab)
    mesh        tri | square | hex  (vertex degrees 6 / 4 / 3)
    resolution  abstract scale, r >= 1 (see module docstring for counts)
    dimension   1 | 2 | 3
    overrides   pass nx/ny/nz/frequency/n to pin exact sizes

    The returned Board's name is 'spec' and its params echo this call, so
    gSGF files reproduce spec boards exactly.
    """
    if resolution < 1:
        raise ValueError("resolution must be >= 1")
    surf = _SURFACE_ALIASES.get(surface.lower())
    if surf is None:
        raise ValueError(f"unknown surface '{surface}'")
    if mesh not in MESH_TYPES:
        raise ValueError(f"mesh must be one of {MESH_TYPES}")
    r = resolution
    spec_params = {"surface": surface, "mesh": mesh,
                   "resolution": resolution, "dimension": dimension,
                   **overrides}

    if dimension == 1:
        n = overrides.get("n", 4 * r + 5)
        base = cycle(n) if surf in ("torus", "sphere") else path(n)
    elif dimension == 3:
        if mesh != "square":
            raise ValueError("dimension 3 supports mesh='square' (cubic "
                             "lattice) in the pilot; FCC/A3 lattices are "
                             "an M2 roadmap item")
        n = r + 3
        nx = overrides.get("nx", n)
        ny = overrides.get("ny", n)
        nz = overrides.get("nz", n)
        wraps = {"plane": (False, False, False),
                 "cylinder": (True, False, False),
                 "torus": (True, True, True)}
        if surf not in wraps:
            raise ValueError(f"dimension 3 supports plane/box, cylinder, "
                             f"torus — not '{surface}'")
        base = lattice3(nx, ny, nz, wraps[surf],
                        "torus3" if surf == "torus" else
                        "slab3" if surf == "cylinder" else "box3")
    elif dimension == 2:
        if surf == "sphere":
            if mesh == "tri":
                base = sphere_geodesic(overrides.get("frequency", r))
            elif mesh == "square":
                base = sphere_quad(overrides.get("frequency", r + 1))
            else:
                base = sphere_goldberg(overrides.get("frequency", r))
        else:
            if mesh == "hex":
                nx_d = 2 * r + 4
                ny_d = 2 * r + 3 if surf == "mobius" else 2 * r + 4
            else:
                nx_d = ny_d = 2 * r + 3
            nx = overrides.get("nx", nx_d)
            ny = overrides.get("ny", ny_d)
            base = {"plane": plane, "cylinder": cylinder, "torus": torus,
                    "mobius": mobius, "klein": klein, "rp2": rp2}[surf](
                        nx, ny, mesh)
    else:
        raise ValueError("dimension must be 1, 2, or 3")

    return Board(name="spec", params=spec_params, adj=base.adj,
                 coords=base.coords, faces=base.faces, labels=base.labels,
                 meta={**base.meta, "resolved": base.name,
                       "resolved_params": base.params})


# ---------------------------------------------------------------------------
# Incidence-derived boards (stones on edges / faces / all cells)
# ---------------------------------------------------------------------------

def incidence_board(mode: str = "vertices", **spec) -> Board:
    """A spec board replayed onto a different cell type (see geodesics.cells).

    mode      vertices | edges | faces | cells
    **spec    forwarded verbatim to make_board (surface, mesh, resolution,
              dimension, and any nx/ny/frequency overrides)

    'edges' plays on the line graph L(G), 'faces' on the dual graph, and
    'cells' on the Hasse diagram of the face poset. Registered as
    'incidence' so gSGF files of derived-board games reconstruct exactly.
    """
    from . import cells as _cells             # local import: cells needs Board
    return _cells.derive(make_board(**spec), mode)


# ---------------------------------------------------------------------------
# Registry (gSGF reconstruction and the demo CLI)
# ---------------------------------------------------------------------------

REGISTRY = {
    "plane": plane,
    "cylinder": cylinder,
    "torus": torus,
    "mobius": mobius,
    "klein": klein,
    "rp2": rp2,
    "sphere_geodesic": sphere_geodesic,
    "sphere_quad": sphere_quad,
    "sphere_goldberg": sphere_goldberg,
    "path": path,
    "cycle": cycle,
    "box3": box3,
    "torus3": torus3,
    "conway": conway_board,
    "spec": make_board,
    "incidence": incidence_board,
}


def build(name: str, **params) -> Board:
    if name == "explicit":
        board = Board(name="explicit", params={},
                      adj=[tuple(nbrs) for nbrs in params["adj"]],
                      coords=params.get("coords"))
        board.validate()
        return board
    if name not in REGISTRY:
        raise KeyError(f"unknown topology '{name}'; known: {sorted(REGISTRY)}")
    return REGISTRY[name](**params)
