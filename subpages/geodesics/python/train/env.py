"""env.py — the self-play environment, which is simply the geodesics package.

`make_board()` constructs any surface x mesh x resolution; `Game` enforces
capture, ko/superko and Tromp-Taylor scoring. Nothing else is needed to
train: an "agent" here is any callable (game, color) -> vertex or -1, and
`play_game` runs two of them headlessly to completion.

SPECS names the lowest-complexity variant of every topology (plus the
classical 9x9); training scripts take these keys on the command line.
"""

from __future__ import annotations

import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_DIR = os.path.dirname(_HERE)                   # the geodesics package dir

if "geodesics" not in sys.modules:                  # register it whatever the
    _spec = importlib.util.spec_from_file_location( # folder happens to be named
        "geodesics", os.path.join(_PKG_DIR, "__init__.py"),
        submodule_search_locations=[_PKG_DIR])
    _pkg = importlib.util.module_from_spec(_spec)
    sys.modules["geodesics"] = _pkg
    _spec.loader.exec_module(_pkg)

from geodesics.engine import BLACK, WHITE, Game, RuleConfig   # noqa: E402
from geodesics.topology import make_board                     # noqa: E402

# name -> (surface, mesh, resolution): the smallest sensible board of each
# topology, i.e. what "train on the lowest complexity variants" means here.
SPECS = {
    "sphere":   ("sphere",   "tri",    2),   # geodesic f2, 42 vertices
    "sphere-q": ("sphere",   "square", 1),   # cube-sphere, 26
    "sphere-h": ("sphere",   "hex",    1),   # Goldberg GP(1), 20
    "plane":    ("plane",    "square", 1),   # 5x5 disk, 25
    "plane9":   ("plane",    "square", 3),   # the classical 9x9
    "cylinder": ("cylinder", "square", 1),
    "torus":    ("torus",    "square", 1),
    "torus-h":  ("torus",    "hex",    1),
    "mobius":   ("mobius",   "square", 1),
    "klein":    ("klein",    "square", 1),
    "rp2":      ("rp2",      "square", 1),
    # harder tier: bigger, higher-genus-adjacent, non-orientable at scale, 3D
    "sphere2":  ("sphere",   "tri",    3),   # geodesic f3, 92
    "torus2":   ("torus",    "square", 2),   # 7x7, 49
    "torus-h2": ("torus",    "hex",    2),
    "klein2":   ("klein",    "square", 2),
    "rp22":     ("rp2",      "square", 2),
    "mobius-h": ("mobius",   "hex",    1),
    "mobius2":  ("mobius",   "square", 2),
}

# 3D lattices need the dimension override
SPECS_3D = {
    "box3":   ("box",   "square", 1),
    "torus3": ("torus", "square", 1),
}
SPECS.update(SPECS_3D)

HARDER = ["sphere2", "torus2", "klein2", "rp22", "box3", "torus3"]

LOWEST = ["sphere", "plane", "cylinder", "torus", "mobius", "klein", "rp2"]


def board_for(key: str):
    surface, mesh, r = SPECS[key]
    dim = 3 if key in SPECS_3D else 2
    return make_board(surface=surface, mesh=mesh, resolution=r, dimension=dim)


def new_game(key: str, komi: float = 7.5) -> Game:
    return Game(board_for(key), RuleConfig(komi=komi))


def adjacency(board):
    return [list(nbrs) for nbrs in board.adj]


def play_game(game: Game, agent_black, agent_white, max_moves=None,
              on_move=None):
    """Run a full game. Agents return a vertex index or -1 to pass; any
    illegal suggestion is converted to a pass (the engine stays authoritative).
    Returns (winner, moves) with winner in {'B', 'W', 'draw'}."""
    n = game.board.n
    cap = max_moves or 4 * n
    agents = {BLACK: agent_black, WHITE: agent_white}
    moves = 0
    while not game.game_over and moves < cap:
        color = game.to_move
        v = agents[color](game, color)
        if v is None or v < 0 or not game.is_legal(v):
            game.play_pass()
        else:
            game.play(v)
        if on_move:
            on_move(game)
        moves += 1
    return game.score()["winner"], moves


def random_agent(rng):
    def agent(game, color):
        legal = game.legal_moves()
        return rng.choice(legal) if legal else -1
    return agent
