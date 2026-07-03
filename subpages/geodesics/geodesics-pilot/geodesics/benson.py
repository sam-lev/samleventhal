"""Benson's algorithm (1976) for unconditional life, generalized to graphs.

"Two eyes make a living body" made rigorous: a set of chains is pass-alive
(cannot be captured even if the owner passes forever) iff every chain in the
set has at least two *vital* enclosed regions whose bordering chains all lie
in the set. Benson's original proof is purely combinatorial and transfers to
arbitrary graphs unchanged — nothing in it uses the square grid.

Definitions on a graph, for color X:
  * chain     — connected component of X-stones.
  * region    — connected component of non-X vertices (empty or opponent).
                By maximality, every vertex adjacent to a region from
                outside is an X stone, i.e. regions are X-enclosed.
  * vital     — region R is vital to bordering chain C iff every EMPTY
                vertex of R is a liberty of C.
  * pass-alive — the greatest fixpoint of: keep chains with >= 2 vital
                regions; keep regions all of whose borders are kept chains.
"""

from __future__ import annotations

from .board import Board
from .engine import EMPTY


def pass_alive(board: Board, colors: list[int], color: int) -> set[int]:
    """Vertices of `color` chains that are unconditionally alive."""
    n = board.n

    # --- chains of `color` -------------------------------------------------
    cid = [-1] * n
    chains: list[set[int]] = []
    for v in range(n):
        if colors[v] == color and cid[v] == -1:
            comp = {v}
            stack = [v]
            cid[v] = len(chains)
            while stack:
                u = stack.pop()
                for w in board.adj[u]:
                    if colors[w] == color and cid[w] == -1:
                        cid[w] = len(chains)
                        comp.add(w)
                        stack.append(w)
            chains.append(comp)
    if not chains:
        return set()

    # --- enclosed regions: components of non-X vertices ---------------------
    rid = [-1] * n
    regions: list[dict] = []
    for v in range(n):
        if colors[v] != color and rid[v] == -1:
            comp = {v}
            stack = [v]
            rid[v] = len(regions)
            border: set[int] = set()
            while stack:
                u = stack.pop()
                for w in board.adj[u]:
                    if colors[w] != color:
                        if rid[w] == -1:
                            rid[w] = len(regions)
                            comp.add(w)
                            stack.append(w)
                    else:
                        border.add(cid[w])
            empties = [u for u in comp if colors[u] == EMPTY]
            regions.append({"verts": comp, "empties": empties, "border": border})

    # --- vitality: every empty vertex of R adjacent to chain C ---------------
    vital: list[set[int]] = []
    for reg in regions:
        vs = set()
        for c in reg["border"]:
            if all(any(cid[w] == c for w in board.adj[e]) for e in reg["empties"]):
                vs.add(c)
        vital.append(vs)

    # --- Benson fixpoint ------------------------------------------------------
    live_chains = set(range(len(chains)))
    live_regions = set(range(len(regions)))
    changed = True
    while changed:
        changed = False
        for c in list(live_chains):
            if sum(1 for r in live_regions if c in vital[r]) < 2:
                live_chains.discard(c)
                changed = True
        for r in list(live_regions):
            if any(b not in live_chains for b in regions[r]["border"]):
                live_regions.discard(r)
                changed = True

    out: set[int] = set()
    for c in live_chains:
        out |= chains[c]
    return out
