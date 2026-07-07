"""bots/zero_mini.py — serves the zero-style self-play checkpoint.

Hidden until python/train/train_zero.py has produced
python/train/checkpoints/zero_mini.npz (override with ZERO_CKPT); then it
advertises exactly the surfaces and meshes it was trained on. Levels:
casual samples the raw policy, standard plays its argmax, strong runs a
small PUCT search on the request's own adjacency graph.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TRAIN = os.path.join(ROOT, "python", "train")
if TRAIN not in sys.path:
    sys.path.insert(0, TRAIN)

CKPT = os.environ.get(
    "ZERO_CKPT", os.path.join(TRAIN, "checkpoints", "zero_mini.npz"))

STRONG_SIMS = 96


class ZeroMiniBot:
    id = "zero-mini"
    name = "Zero mini"
    levels = ["casual", "standard", "strong"]

    def __init__(self):
        self.net = None
        self.supports = {"incidence": ["vertices"]}

    def available(self):
        if not os.path.isfile(CKPT):
            return False
        try:
            from zeronet import ZeroNet
            self.net = ZeroNet.load(CKPT)
            self.supports = {
                "surfaces": self.net.meta.get("surfaces"),
                "meshes": self.net.meta.get("meshes"),
                "incidence": ["vertices"],
            }
            return True
        except Exception as e:                       # pragma: no cover
            print(f"  zero-mini: failed to load {CKPT}: {e}")
            return False

    def genmove(self, req):
        from zeronet import features, mean_matrix
        board_msg = req["board"]
        n = board_msg["n"]
        adj = [tuple(x) for x in board_msg["neighbors"]]
        stones = list(board_msg["stones"])
        to_move = board_msg["toMove"]
        legal = board_msg.get("legalMask") or [1] * n
        mask = np.zeros(n + 1)
        mask[n] = 1
        for v in range(n):
            if stones[v] == 0 and legal[v]:
                mask[v] = 1
        level = req.get("level", "standard")

        if level == "strong":
            from env import Game, RuleConfig     # registers the package
            from geodesics.board import Board
            from mcts import MCTS
            board = Board(name="remote", params={}, adj=adj)
            game = Game(board, RuleConfig(komi=7.5))
            game.set_position({v: c for v, c in enumerate(stones) if c},
                              to_move)
            tree = MCTS(self.net, board, mean_matrix(adj), seed=len(stones))
            counts = tree.run(game, STRONG_SIMS, root_noise=False)
            counts *= mask                       # host legality is final
            if counts.sum() <= 0:
                return -1, f"{self.name} passes"
            a = int(np.argmax(counts))
            return (-1 if a == n else a), \
                f"{self.name} ({STRONG_SIMS} sims)"

        X = features(adj, stones, to_move)
        probs, value, _ = self.net.forward(mean_matrix(adj), X, mask)
        probs = probs * mask
        if probs.sum() <= 0:
            return -1, f"{self.name} passes"
        if level == "casual":
            probs = probs / probs.sum()
            a = int(np.random.default_rng().choice(n + 1, p=probs))
        else:
            a = int(np.argmax(probs))
        info = f"{self.name} policy, value {value:+.2f}"
        return (-1 if a == n else a), info


BOT = ZeroMiniBot()
