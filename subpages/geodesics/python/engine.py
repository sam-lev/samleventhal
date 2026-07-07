"""Layer 4 — the rule engine.

Tromp–Taylor logical rules of Go, stated over an arbitrary finite simple
graph. The engine never inspects geometry: the Board's adjacency lists are
the complete rule-relevant structure, which is what makes the whole system
topology-agnostic.

Rule kernel (Tromp–Taylor, graph form):
  * A chain is a connected monochromatic set of vertices; its liberties are
    the empty vertices adjacent to it.
  * Playing at empty vertex v: place the stone; remove adjacent opponent
    chains with no liberties; then, if v's own chain has no liberties, the
    move is suicide (illegal by default; legal-and-self-removing if the
    ruleset allows it, as Tromp-Taylor does).
  * Superko: a stone move may not recreate a previous whole-board position
    (positional, the default) or previous position-with-same-player-to-move
    (situational). Detected via Zobrist hashing generalized to arbitrary
    vertex sets: one random 64-bit key per (vertex, color).
  * Two consecutive passes end the game. Area scoring: stones plus empty
    regions bordering only one color.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .board import Board

EMPTY, BLACK, WHITE = 0, 1, 2
OTHER = {BLACK: WHITE, WHITE: BLACK}
COLOR_NAME = {BLACK: "B", WHITE: "W"}
NAME_COLOR = {"B": BLACK, "W": WHITE}


class IllegalMove(Exception):
    def __init__(self, reason: str, vertex: int | None = None):
        super().__init__(f"illegal move at {vertex}: {reason}")
        self.reason = reason
        self.vertex = vertex


@dataclass
class RuleConfig:
    allow_suicide: bool = False          # Tromp-Taylor permits it; off by default
    superko: str = "positional"          # 'positional' | 'situational' | 'simple' | 'none'
    komi: float = 0.0


def chain_at(colors: list[int], board: Board, v: int):
    """The chain containing v and its liberty set (flood fill)."""
    color = colors[v]
    stones = {v}
    libs: set[int] = set()
    stack = [v]
    while stack:
        u = stack.pop()
        for w in board.adj[u]:
            cw = colors[w]
            if cw == EMPTY:
                libs.add(w)
            elif cw == color and w not in stones:
                stones.add(w)
                stack.append(w)
    return stones, libs


class Game:
    """A game session on an arbitrary Board (Layer 5 entry point)."""

    def __init__(self, board: Board, rules: RuleConfig | None = None, seed: int = 0):
        self.board = board
        self.rules = rules or RuleConfig()
        rng = random.Random((seed, board.name, board.n).__repr__())
        # Zobrist keys generalized to an arbitrary vertex set
        self._z = [[0, rng.getrandbits(64), rng.getrandbits(64)]
                   for _ in range(board.n)]
        self.colors: list[int] = [EMPTY] * board.n
        self.to_move: int = BLACK
        self.captures = {BLACK: 0, WHITE: 0}
        self.passes = 0
        self.moves: list[tuple[str, int | None]] = []
        self._hash_history: list[int] = [self._hash(self.colors)]
        self._situ_history: list[tuple[int, int]] = [(self._hash_history[0], self.to_move)]
        self._snapshots: list[tuple] = []

    # ---- position hashing --------------------------------------------------

    def _hash(self, colors: list[int]) -> int:
        h = 0
        for v, c in enumerate(colors):
            if c != EMPTY:
                h ^= self._z[v][c]
        return h

    # ---- setup (testing / problems) -----------------------------------------

    def set_position(self, stones: dict[int, int], to_move: int = BLACK) -> None:
        """Install an arbitrary position and rebase the ko history on it."""
        self.colors = [EMPTY] * self.board.n
        for v, c in stones.items():
            self.colors[v] = c
        self.to_move = to_move
        self.passes = 0
        self.moves = []
        self._snapshots = []
        h = self._hash(self.colors)
        self._hash_history = [h]
        self._situ_history = [(h, to_move)]

    # ---- move simulation -----------------------------------------------------

    def _simulate(self, v: int, color: int):
        if not (0 <= v < self.board.n):
            raise IllegalMove("no such vertex", v)
        if self.colors[v] != EMPTY:
            raise IllegalMove("occupied", v)
        new = self.colors[:]
        new[v] = color
        opp = OTHER[color]
        captured: list[int] = []
        seen: set[int] = set()
        for nb in self.board.adj[v]:
            if new[nb] == opp and nb not in seen:
                stones, libs = chain_at(new, self.board, nb)
                seen |= stones
                if not libs:
                    captured.extend(stones)
                    for s in stones:
                        new[s] = EMPTY
        suicided: list[int] = []
        own_stones, own_libs = chain_at(new, self.board, v)
        if not own_libs:
            # captures always open a liberty adjacent to v, so reaching here
            # means nothing was captured
            if not self.rules.allow_suicide:
                raise IllegalMove("suicide", v)
            suicided = list(own_stones)
            for s in own_stones:
                new[s] = EMPTY

        h = self._hash(new)
        mode = self.rules.superko
        if mode == "positional" and h in self._hash_history:
            raise IllegalMove("positional superko", v)
        if mode == "situational" and (h, OTHER[color]) in self._situ_history:
            raise IllegalMove("situational superko", v)
        if mode == "simple" and len(self._hash_history) >= 2 \
                and h == self._hash_history[-2]:
            raise IllegalMove("ko", v)
        return new, captured, suicided, h

    def is_legal(self, v: int) -> bool:
        try:
            self._simulate(v, self.to_move)
            return True
        except IllegalMove:
            return False

    def legal_moves(self) -> list[int]:
        return [v for v in range(self.board.n)
                if self.colors[v] == EMPTY and self.is_legal(v)]

    # ---- play ---------------------------------------------------------------

    def _push_snapshot(self) -> None:
        self._snapshots.append((
            self.colors[:], self.to_move, dict(self.captures), self.passes,
        ))

    def play(self, v: int) -> dict:
        color = self.to_move
        new, captured, suicided, h = self._simulate(v, color)
        self._push_snapshot()
        self.colors = new
        self.captures[color] += len(captured)
        if suicided:
            self.captures[OTHER[color]] += len(suicided)
        self.passes = 0
        self.moves.append((COLOR_NAME[color], v))
        self.to_move = OTHER[color]
        self._hash_history.append(h)
        self._situ_history.append((h, self.to_move))
        return {"vertex": v, "color": COLOR_NAME[color],
                "captured": captured, "suicided": suicided}

    def play_pass(self) -> dict:
        color = self.to_move
        self._push_snapshot()
        self.passes += 1
        self.moves.append((COLOR_NAME[color], None))
        self.to_move = OTHER[color]
        h = self._hash_history[-1]
        self._hash_history.append(h)
        self._situ_history.append((h, self.to_move))
        return {"vertex": None, "color": COLOR_NAME[color], "captured": []}

    def undo(self) -> None:
        if not self._snapshots:
            raise ValueError("nothing to undo")
        self.colors, self.to_move, self.captures, self.passes = self._snapshots.pop()
        self.moves.pop()
        self._hash_history.pop()
        self._situ_history.pop()

    @property
    def game_over(self) -> bool:
        return self.passes >= 2

    # ---- scoring (Tromp-Taylor area) -----------------------------------------

    def score(self) -> dict:
        terr = {BLACK: 0, WHITE: 0}
        stones = {BLACK: self.colors.count(BLACK), WHITE: self.colors.count(WHITE)}
        seen: set[int] = set()
        for v in range(self.board.n):
            if self.colors[v] != EMPTY or v in seen:
                continue
            comp = {v}
            stack = [v]
            border: set[int] = set()
            while stack:
                u = stack.pop()
                for w in self.board.adj[u]:
                    c = self.colors[w]
                    if c == EMPTY:
                        if w not in comp:
                            comp.add(w)
                            stack.append(w)
                    else:
                        border.add(c)
            seen |= comp
            if border == {BLACK}:
                terr[BLACK] += len(comp)
            elif border == {WHITE}:
                terr[WHITE] += len(comp)
        b = stones[BLACK] + terr[BLACK]
        w = stones[WHITE] + terr[WHITE] + self.rules.komi
        margin = b - w
        winner = "B" if margin > 0 else "W" if margin < 0 else "draw"
        return {"black": b, "white": w, "margin": margin, "winner": winner,
                "territory": {"B": terr[BLACK], "W": terr[WHITE]},
                "stones": {"B": stones[BLACK], "W": stones[WHITE]}}

    # ---- diagnostics ----------------------------------------------------------

    def ascii(self) -> str:
        """ASCII rendering for grid-based boards ('.', 'X'=black, 'O'=white)."""
        nx = self.board.params.get("nx")
        ny = self.board.params.get("ny")
        if nx is None:
            raise ValueError("ascii() only supports grid-based boards")
        glyph = {EMPTY: ".", BLACK: "X", WHITE: "O"}
        rows = []
        for y in range(ny - 1, -1, -1):
            rows.append(" ".join(glyph[self.colors[y * nx + x]] for x in range(nx)))
        return "\n".join(rows)
