"""Layer 3 — the adjacency graph.

A Board is a finite simple graph: vertices are playable points, edges are
adjacency (the generalization of the goban's lines). Everything above this
layer (rule engine, sessions) is topology-agnostic; everything below it
(topology constructors, meshes) exists only to *produce* a Board.

Optional geometric data (coords, faces) is carried for rendering and for
validating topological invariants (Euler characteristic), but the rules
never consult it.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field


@dataclass
class Board:
    name: str                                  # constructor name, e.g. "torus"
    params: dict                               # constructor parameters (for gSGF round-trip)
    adj: list[tuple[int, ...]]                 # adjacency lists, vertex id -> neighbor ids
    coords: list[tuple[float, ...]] | None = None   # optional embedding (2D or 3D)
    faces: list[tuple[int, ...]] | None = None      # optional face list (mesh boards)
    labels: list[str] | None = None            # optional human-readable vertex labels
    meta: dict = field(default_factory=dict)

    # ---- basic accessors -------------------------------------------------

    @property
    def n(self) -> int:
        return len(self.adj)

    def edges(self) -> list[tuple[int, int]]:
        es = set()
        for v, nbrs in enumerate(self.adj):
            for w in nbrs:
                es.add((v, w) if v < w else (w, v))
        return sorted(es)

    def degree_histogram(self) -> dict[int, int]:
        return dict(Counter(len(nbrs) for nbrs in self.adj))

    def euler_characteristic(self) -> int | None:
        """V - E + F when a face list is available (mesh boards only)."""
        if self.faces is None:
            return None
        return self.n - len(self.edges()) + len(self.faces)

    # ---- convenience for grid-based boards -------------------------------

    def vertex_index(self, x: int, y: int, z: int | None = None) -> int:
        """Grid boards index vertices as y * nx + x (2D) or
        (z * ny + y) * nx + x (3D lattices)."""
        nx = self.params.get("nx")
        if nx is None:
            raise ValueError(f"board '{self.name}' is not grid-based")
        if self.params.get("nz") is not None:
            if z is None:
                raise ValueError(f"board '{self.name}' is 3D: pass x, y, z")
            return (z * self.params["ny"] + y) * nx + x
        return y * nx + x

    def vertex_xy(self, v: int) -> tuple[int, int]:
        nx = self.params.get("nx")
        if nx is None:
            raise ValueError(f"board '{self.name}' is not grid-based")
        return v % nx, v // nx

    # ---- integrity -------------------------------------------------------

    def validate(self) -> None:
        for v, nbrs in enumerate(self.adj):
            assert len(set(nbrs)) == len(nbrs), f"duplicate edge at vertex {v}"
            assert v not in nbrs, f"self-loop at vertex {v}"
            for w in nbrs:
                assert 0 <= w < self.n, f"edge to nonexistent vertex {w}"
                assert v in self.adj[w], f"asymmetric edge {v}->{w}"
