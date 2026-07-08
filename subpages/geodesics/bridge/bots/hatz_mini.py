"""bots/hatz_mini.py — serves the Holonomy-Aware Topological AlphaZero
checkpoint (python/train/train_hatz.py; override path with HATZ_CKPT).

The request carries only the adjacency graph, so grid coordinates are
synthesized from nx/ny when present — that is enough for the orientation
cocycle to place the Möbius/Klein/RP² seam correctly on remote boards.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TRAIN = os.path.join(ROOT, "python", "train")
if TRAIN not in sys.path:
    sys.path.insert(0, TRAIN)

CKPT = os.environ.get("HATZ_CKPT",
                      os.path.join(TRAIN, "checkpoints", "hatz.npz"))
STRONG_SIMS = 64


def _make_bundle(req):
    from env import Game  # noqa: F401  (registers the geodesics package)
    from geodesics.board import Board
    from hatz import Bundle
    spec, bd = req.get("spec", {}), req["board"]
    nx, ny = spec.get("nx", 0), spec.get("ny", 0)
    coords = tuple((v % nx, v // nx) for v in range(bd["n"])) \
        if nx and ny else None
    board = Board(name=spec.get("surface", "remote"), params={},
                  adj=tuple(tuple(a) for a in bd["neighbors"]),
                  coords=coords, faces=None)
    return board, Bundle(board, spec.get("surface"))


class HatzBot:
    id = "hatz"
    name = "HATZ"
    levels = ["casual", "standard", "strong"]

    def __init__(self):
        self.net = None
        self.supports = {"incidence": ["vertices"]}

    def available(self):
        if not os.path.isfile(CKPT):
            return False
        try:
            from hatz import HATZ
            self.net = HATZ.load(CKPT)
            self.supports = {
                "surfaces": self.net.meta.get("surfaces"),
                "meshes": self.net.meta.get("meshes"),
                "incidence": ["vertices"],
            }
            return True
        except Exception as e:                        # pragma: no cover
            print(f"  hatz: failed to load {CKPT}: {e}")
            return False

    def genmove(self, req):
        bd = req["board"]
        n = bd["n"]
        stones = list(bd["stones"])
        to_move = bd["toMove"]
        legal = bd.get("legalMask") or [1] * n
        mask = np.zeros(n + 1)
        mask[n] = 1
        for v in range(n):
            if stones[v] == 0 and legal[v]:
                mask[v] = 1
        board, bundle = _make_bundle(req)
        level = req.get("level", "standard")

        if level == "strong":
            from env import Game, RuleConfig
            from mcts import MCTS
            game = Game(board, RuleConfig(komi=7.5))
            game.set_position({v: c for v, c in enumerate(stones) if c},
                              to_move)
            tree = MCTS(self.net, board, bundle, seed=len(stones))
            counts = tree.run(game, STRONG_SIMS, root_noise=False)
            counts *= mask
            if counts.sum() <= 0:
                return -1, f"{self.name} passes"
            a = int(np.argmax(counts))
            return (-1 if a == n else a), f"{self.name} ({STRONG_SIMS} sims)"

        probs, value = self.net.policy_value(bundle, stones, to_move, mask)
        probs = probs * mask
        if probs.sum() <= 0:
            return -1, f"{self.name} passes"
        if level == "casual":
            probs = probs / probs.sum()
            a = int(np.random.default_rng().choice(n + 1, p=probs))
        else:
            a = int(np.argmax(probs))
        tag = " \u00b7 non-orientable seam active" if bundle.nonorientable \
            else ""
        return (-1 if a == n else a), \
            f"{self.name} policy, value {value:+.2f}{tag}"


BOT = HatzBot()
