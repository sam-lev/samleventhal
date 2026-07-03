"""gSGF 0.1 — generalized Smart Game Format.

Classic SGF assumes a square grid and letter-pair coordinates; gSGF replaces
the board declaration with either a topology-constructor reference (name +
parameters, reproducible via the registry) or an explicit adjacency graph,
and records moves as vertex ids. JSON container for the pilot.
"""

from __future__ import annotations

import json
from dataclasses import asdict

from . import topology
from .engine import Game, RuleConfig, NAME_COLOR

FORMAT = "gSGF-0.1"


def dumps(game: Game, meta: dict | None = None) -> str:
    if game.board.name == "explicit":
        board_spec = {"type": "explicit",
                      "adjacency": [list(nbrs) for nbrs in game.board.adj],
                      "coords": game.board.coords}
    else:
        board_spec = {"type": game.board.name, "params": game.board.params}
    doc = {
        "format": FORMAT,
        "board": board_spec,
        "rules": asdict(game.rules),
        "moves": [{"c": c, "v": v} if v is not None else {"c": c, "pass": True}
                  for c, v in game.moves],
        "meta": meta or {},
    }
    return json.dumps(doc, indent=2)


def loads(text: str) -> Game:
    doc = json.loads(text)
    if doc.get("format") != FORMAT:
        raise ValueError(f"unsupported format: {doc.get('format')!r}")
    spec = doc["board"]
    if spec["type"] == "explicit":
        board = topology.build("explicit", adj=spec["adjacency"],
                               coords=spec.get("coords"))
    else:
        board = topology.build(spec["type"], **spec["params"])
    game = Game(board, RuleConfig(**doc.get("rules", {})))
    for mv in doc["moves"]:
        expected = NAME_COLOR[mv["c"]]
        if game.to_move != expected:
            raise ValueError("move sequence violates alternation")
        if mv.get("pass"):
            game.play_pass()
        else:
            game.play(mv["v"])       # replay validates legality
    return game
