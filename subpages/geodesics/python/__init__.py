"""Geodesics — Go, topologically abstracted.

Pilot reference implementation of the five-layer architecture:
Topology -> Mesh -> Adjacency Graph -> Rule Engine -> Session.
"""

from .board import Board
from .topology import (plane, cylinder, torus, mobius, klein, rp2,
                       sphere_geodesic, sphere_quad, sphere_goldberg,
                       path, cycle, box3, torus3, lattice3,
                       conway_board, make_board, build, REGISTRY, MESH_TYPES)
from .engine import (Game, RuleConfig, IllegalMove, chain_at,
                     EMPTY, BLACK, WHITE)
from .benson import pass_alive
from . import gsgf
from . import mesh

__version__ = "0.2.0"
__all__ = [
    "Board", "plane", "cylinder", "torus", "mobius", "klein", "rp2",
    "sphere_geodesic", "sphere_quad", "sphere_goldberg",
    "path", "cycle", "box3", "torus3", "lattice3",
    "conway_board", "make_board", "build", "REGISTRY", "MESH_TYPES",
    "Game", "RuleConfig", "IllegalMove", "chain_at",
    "EMPTY", "BLACK", "WHITE", "pass_alive", "gsgf", "mesh",
]
