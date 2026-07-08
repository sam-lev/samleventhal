"""mcts.py — compact PUCT search for zero-style self-play.

States are (stones, to_move, passes); legality inside the tree comes from a
scratch Game re-seated with set_position (basic ko + suicide — the root uses
the true game's legality, superko included). Two passes end the game and the
terminal value is the Tromp-Taylor result from the mover's perspective.
"""

from __future__ import annotations

import math

import numpy as np

from env import Game, RuleConfig
from zeronet import features


class MCTS:
    def __init__(self, net, board, A, komi=7.5, c_puct=1.5, seed=0):
        self.net = net
        self.board = board
        self.A = A
        self.n = board.n
        self.scratch = Game(board, RuleConfig(komi=komi))
        self.c = c_puct
        self.rng = np.random.default_rng(seed)
        self.nodes = {}

    # ---- node bookkeeping ---------------------------------------------------

    def _seat(self, stones, to_move):
        self.scratch.set_position(
            {v: c for v, c in enumerate(stones) if c}, to_move)

    def _expand(self, key):
        stones, to_move, passes = key
        if passes >= 2:
            node = {"terminal": self._terminal_value(stones, to_move)}
            self.nodes[key] = node
            return node
        self._seat(stones, to_move)
        legal = self.scratch.legal_moves()
        mask = np.zeros(self.n + 1)
        mask[self.n] = 1                       # pass is always available
        for v in legal:
            mask[v] = 1
        X = features(self.board.adj, stones, to_move)
        probs, value, _ = self.net.forward(self.A, X, mask)
        node = {"P": probs, "mask": mask, "N": np.zeros(self.n + 1),
                "W": np.zeros(self.n + 1), "value": value}
        self.nodes[key] = node
        return node

    def _terminal_value(self, stones, to_move):
        self._seat(stones, to_move)
        sc = self.scratch.score()
        if sc["winner"] == "draw":
            return 0.0
        me = "B" if to_move == 1 else "W"
        return 1.0 if sc["winner"] == me else -1.0

    def _step(self, key, action):
        stones, to_move, passes = key
        if action == self.n:                   # pass
            return stones, 3 - to_move, passes + 1
        self._seat(stones, to_move)
        self.scratch.play(action)
        return tuple(self.scratch.colors), 3 - to_move, 0

    # ---- search ----------------------------------------------------------------

    def run(self, game, sims, root_noise=True):
        """Search from the live game's position. The root action set is the
        true legal set (superko included); deeper nodes use the scratch
        engine. Returns visit counts over n+1 actions (pass last)."""
        root_key = (tuple(game.colors), game.to_move, game.passes)
        root = self.nodes.get(root_key) or self._expand(root_key)
        if "terminal" in root:
            counts = np.zeros(self.n + 1)
            counts[self.n] = 1
            return counts
        mask = np.zeros(self.n + 1)
        mask[self.n] = 1
        for v in game.legal_moves():
            mask[v] = 1
        root["mask"] = mask
        P = root["P"] * mask
        if root_noise and mask.sum() > 1:
            idx = np.where(mask > 0)[0]
            noise = self.rng.dirichlet([0.3] * len(idx))
            P = P.copy()
            P[idx] = 0.75 * P[idx] + 0.25 * noise
        root["P"] = P / max(P.sum(), 1e-12)

        for _ in range(sims):
            node, key, path = root, root_key, []
            seen = {root_key}                  # positional-superko guard:
            while True:                        # a repeated state in one descent
                if len(path) > 4 * self.n:     # is a ko cycle; score it a draw
                    value = 0.0
                    break
                total = node["N"].sum()
                U = self.c * node["P"] * math.sqrt(total + 1) / (1 + node["N"])
                Q = np.where(node["N"] > 0,
                             node["W"] / np.maximum(node["N"], 1), 0.0)
                score = np.where(node["mask"] > 0, Q + U, -1e18)
                a = int(np.argmax(score))
                path.append((node, a))
                key = self._step(key, a)
                if key in seen:
                    value = 0.0
                    break
                seen.add(key)
                child = self.nodes.get(key)
                if child is None:
                    child = self._expand(key)
                    value = child["terminal"] if "terminal" in child                         else child["value"]
                    break
                if "terminal" in child:
                    value = child["terminal"]
                    break
                node = child
            # `value` is from the perspective of the player to move at the
            # leaf; flip once per edge walking back to the root
            for nd, a in reversed(path):
                value = -value
                nd["N"][a] += 1
                nd["W"][a] += value
        return root["N"].copy()


def policy_target(counts, temperature=1.0):
    c = counts.astype(float)
    if temperature <= 1e-3:
        out = np.zeros_like(c)
        out[int(np.argmax(c))] = 1.0
        return out
    c = c ** (1.0 / temperature)
    return c / max(c.sum(), 1e-12)
