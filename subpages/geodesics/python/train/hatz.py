"""hatz.py — Holonomy-Aware Topological AlphaZero, v1 (numpy, no deps).

A first draft of the Stage-2 research architecture, scoped to what can be
implemented honestly and lightly:

1. **Z2 / reflection gauge structure.** Full O(2)-steerability needs tangent
   frames; what non-orientability *forces* (Weiler et al. 2021) is the
   reflection part, realized combinatorially as an orientation cocycle
   eps: E -> {+1,-1}. eps is computed from face cycles by dual-BFS when faces
   exist, and analytically for the quotient grids (the flipped gluing is the
   seam: x-wrap edges on Mobius/Klein, both wraps on RP2). Cycle holonomy
   (product of eps around a loop) is gauge-invariant; the seam itself is a
   representative of w1.

2. **Sheaf-style transport.** Messages along vertex edges are passed through
   restriction maps conditioned on the cocycle: T(u->v) = W0 + eps(u,v) W1,
   with W0, W1 *shared globally* (transfer across boards) rather than
   per-edge (Neural Sheaf Diffusion's per-edge maps don't transfer). Crossing
   the orientation seam therefore applies a genuinely different learned map.
   Because eps enters linearly, the whole layer reduces to two fixed
   aggregation matrices A and A_eps — plain matmuls, hand-differentiable.

3. **Cell ranks.** Vertex and edge features always; face features when the
   board provides face cycles (currently spheres). Messages pass over the
   incidence structure (V<->E<->F), a light CW-network.

4. **Morse–Smale pooling.** The influence field's ascending/descending basin
   partition (reusing hgnn.basin_hierarchy — the same code that powers the
   browser engine) gives a position-dependent, parameter-independent pooling
   matrix: pool to regions, message-pass on the region graph, unpool, add
   residually. No gradients flow through the segmentation itself, so
   backprop stays exact.

5. **Automorphism equivariance** is inherited exactly: weight-shared message
   passing commutes with every graph automorphism, and all inputs are
   Aut-invariant by construction (no positional encodings). Verified in
   tests via WL-orbit constancy of the empty-board policy.

Also: PH-lite tactical inputs (group size/liberties = localized H0 data),
curvature (angle defect) where coords+faces allow, a KataGo-style global
pooling bias, and an ownership auxiliary head.

Known v1 gaps, stated plainly: no continuous SO(2) part (isotropic
aggregation), eps is used in a fixed canonical gauge per board (holonomy
features are gauge-invariant; the learned transport is gauge-covariant only
up to reparameterization), and PH features are local proxies rather than
full persistence diagrams.
"""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hgnn import Kernel, basin_hierarchy  # noqa: E402


def _basins_le(f, N1):
    """basin_hierarchy variant whose label extraction merges persistence
    <= p (not < p): equal-persistence families — which arise exactly on
    symmetric positions of a learned field — merge simultaneously, keeping
    the partition automorphism-equivariant instead of index-tie-broken."""
    h = basin_hierarchy(f, N1)
    n = len(f)

    def higher(a, b):
        return f[a] > f[b] or (f[a] == f[b] and a > b)

    # re-run the union-find to expose pers/merged_into with <= semantics
    order = sorted(range(n), key=lambda v: (-float(f[v]), -v))
    parent, cmax = [-1] * n, [-1] * n
    pers = [float("inf")] * n
    merged_into = [-1] * n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for v in order:
        parent[v] = v
        cmax[v] = v
        for u in N1[v]:
            if parent[u] == -1:
                continue
            ru, rv = find(u), find(v)
            if ru == rv:
                continue
            live, die = ru, rv
            if higher(cmax[die], cmax[live]):
                live, die = rv, ru
            pers[cmax[die]] = float(f[cmax[die]]) - float(f[v])
            merged_into[cmax[die]] = cmax[live]
            parent[die] = live

    def labels_at(p):
        lab = [0] * n
        for v in range(n):
            m = h["basin"][v]
            while merged_into[m] != -1 and pers[m] <= p:
                m = merged_into[m]
            lab[v] = m
        return lab

    return {"finitePers": h["finitePers"], "labelsAt": labels_at}

NONORIENTABLE = {"mobius", "klein", "rp2"}


# ---------- geometry: cocycle, curvature, incidences, orbits -------------------

def _grid_axes(coords):
    xs = sorted({round(c[0], 6) for c in coords})
    ys = sorted({round(c[1], 6) for c in coords})
    return xs, ys


def orientation_cocycle(board, surface=None):
    """eps[u][v] in {+1,-1}; -1 on a representative seam of w1."""
    n = board.n
    eps = {v: {} for v in range(n)}
    for v in range(n):
        for u in board.adj[v]:
            eps[v][u] = 1
    name = (surface or getattr(board, "name", "") or "").lower()
    faces = getattr(board, "faces", None)
    if faces:
        # dual BFS: orient faces coherently; conflicted shared edges = seam
        edge_faces = {}
        for fi, f in enumerate(faces):
            for k in range(len(f)):
                e = frozenset((f[k], f[(k + 1) % len(f)]))
                edge_faces.setdefault(e, []).append(fi)
        sign = [0] * len(faces)
        for start in range(len(faces)):
            if sign[start]:
                continue
            sign[start] = 1
            stack = [start]
            while stack:
                fi = stack.pop()
                f = faces[fi]
                dirs = {(f[k], f[(k + 1) % len(f)]) for k in range(len(f))}
                for e, fs in edge_faces.items():
                    if fi not in fs or len(fs) != 2:
                        continue
                    gi = fs[0] if fs[1] == fi else fs[1]
                    g = faces[gi]
                    gdirs = {(g[k], g[(k + 1) % len(g)]) for k in range(len(g))}
                    a, b = tuple(e)
                    same_dir = ((a, b) in dirs) == ((a, b) in gdirs)
                    want = -sign[fi] if same_dir else sign[fi]
                    if sign[gi] == 0:
                        sign[gi] = want
                        stack.append(gi)
                    elif sign[gi] != want:
                        eps[a][b] = eps[b][a] = -1
        return eps
    coords = getattr(board, "coords", None)
    if name in NONORIENTABLE and coords is not None and len(coords[0]) >= 2:
        xs, ys = _grid_axes(coords)
        span_x, span_y = xs[-1] - xs[0], ys[-1] - ys[0]
        for v in range(n):
            for u in board.adj[v]:
                dx = abs(coords[u][0] - coords[v][0])
                dy = abs(coords[u][1] - coords[v][1])
                wrap_x = span_x > 0 and dx > span_x * 0.6
                wrap_y = span_y > 0 and dy > span_y * 0.6
                if (wrap_x and name in ("mobius", "klein", "rp2")) or \
                   (wrap_y and name == "rp2"):
                    eps[v][u] = -1
    return eps


def angle_defect(board):
    """Discrete curvature 2*pi - sum(incident face angles); 0 without data."""
    n = board.n
    out = np.zeros(n)
    faces = getattr(board, "faces", None)
    coords = getattr(board, "coords", None)
    if not faces or coords is None or len(coords[0]) < 3:
        return out
    P = np.asarray(coords, dtype=float)
    total = np.zeros(n)
    for f in faces:
        m = len(f)
        for k in range(m):
            v, a, b = f[k], f[(k - 1) % m], f[(k + 1) % m]
            e1, e2 = P[a] - P[v], P[b] - P[v]
            c = np.dot(e1, e2) / max(np.linalg.norm(e1) * np.linalg.norm(e2),
                                     1e-12)
            total[v] += math.acos(max(-1.0, min(1.0, c)))
    for v in range(n):
        if total[v] > 0:
            out[v] = 2 * math.pi - total[v]
    return out


def wl_orbits(adj, rounds=8):
    """Orbit partition upper bound via WL color refinement (exact on these
    highly symmetric boards in practice; used for equivariance checks)."""
    n = len(adj)
    color = [len(adj[v]) for v in range(n)]
    for _ in range(rounds):
        sig = [(color[v], tuple(sorted(color[u] for u in adj[v])))
               for v in range(n)]
        uniq = {s: i for i, s in enumerate(sorted(set(sig)))}
        new = [uniq[s] for s in sig]
        if new == color:
            break
        color = new
    return color


def _mean_matrix(adj_lists, n, m, signs=None):
    A = np.zeros((n, m))
    for i, lst in enumerate(adj_lists):
        if lst:
            w = 1.0 / len(lst)
            for j in lst:
                A[i, j] = w * (signs(i, j) if signs else 1.0)
    return A


class Bundle:
    """All static per-board structure, precomputed once."""

    def __init__(self, board, surface=None):
        self.board = board
        self.n = n = board.n
        self.adj = [list(a) for a in board.adj]
        self.kernel = Kernel(self.adj)
        eps = orientation_cocycle(board, surface)
        self.eps = eps
        self.A = _mean_matrix(self.adj, n, n)
        self.Asum = np.zeros((n, n))
        for v in range(n):
            for u in self.adj[v]:
                self.Asum[v, u] = 1.0
        self.Ae = _mean_matrix(self.adj, n, n,
                               signs=lambda v, u: eps[v][u])
        self.nonorientable = any(eps[v][u] < 0
                                 for v in range(n) for u in self.adj[v])
        # edge rank
        eset, self.edges = set(), []
        for v in range(n):
            for u in self.adj[v]:
                if (u, v) not in eset and (v, u) not in eset:
                    eset.add((v, u))
                    self.edges.append((v, u))
        self.ne = len(self.edges)
        v_edges = [[] for _ in range(n)]
        for ei, (a, b) in enumerate(self.edges):
            v_edges[a].append(ei)
            v_edges[b].append(ei)
        self.M_ev = _mean_matrix(v_edges, n, self.ne)      # vertex <- edges
        self.M_ve = _mean_matrix([[a, b] for a, b in self.edges],
                                 self.ne, n)               # edge <- vertices
        # face rank (when the board carries face cycles)
        faces = getattr(board, "faces", None) or []
        self.faces = [list(f) for f in faces]
        self.nf = len(self.faces)
        if self.nf:
            eidx = {}
            for ei, (a, b) in enumerate(self.edges):
                eidx[(a, b)] = eidx[(b, a)] = ei
            f_edges = [[eidx[(f[k], f[(k + 1) % len(f)])]
                        for k in range(len(f))] for f in self.faces]
            e_faces = [[] for _ in range(self.ne)]
            for fi, lst in enumerate(f_edges):
                for ei in lst:
                    e_faces[ei].append(fi)
            self.M_fe = _mean_matrix(e_faces, self.ne, self.nf)  # edge <- faces
            self.M_ef = _mean_matrix(f_edges, self.nf, self.ne)  # face <- edges
        self.curv = angle_defect(board)
        self.deg = np.array([len(a) for a in self.adj], dtype=float)
        self.orbits = wl_orbits(self.adj)
        chi = n - self.ne + self.nf
        self.gfeat = np.array([1.0 if self.nonorientable else 0.0,
                               chi / max(n, 1)])

    def with_gauge(self, rng):
        """Gauge-transform the cocycle: eps'(u,v) = g(u) eps(u,v) g(v) for a
        random vertex sign field g. Holonomy (and hence non-orientability) is
        invariant; the seam moves. Used as training augmentation so the net
        learns gauge invariance the architecture doesn't yet guarantee."""
        import copy as _copy
        g = rng.choice([-1, 1], size=self.n)
        out = _copy.copy(self)
        out.eps = {v: {u: int(g[v] * self.eps[v][u] * g[u])
                       for u in self.adj[v]} for v in range(self.n)}
        out.Ae = _mean_matrix(self.adj, self.n, self.n,
                              signs=lambda v, u: out.eps[v][u])
        return out

    # ---- per-position structures -------------------------------------------

    def msc_pool(self, stones, to_move):
        """Morse–Smale basin pooling of the diffused influence field:
        S (n x R) one-hot assignment, P (R x n) mean-pool, Ar region mean."""
        n = self.n
        f = np.zeros(n)
        for v in range(n):
            if stones[v]:
                f[v] = 1.0 if stones[v] == to_move else -1.0
        for _ in range(2):
            f = 0.55 * f + 0.45 * (self.A @ f)
        asc = basin_hierarchy(f.astype(np.float32), self.adj)
        desc = basin_hierarchy((-f).astype(np.float32), self.adj)
        pers = sorted(asc["finitePers"] + desc["finitePers"])
        p = pers[int(0.6 * len(pers))] if pers else 0
        lab = [(asc["labelsAt"](p)[v], desc["labelsAt"](p)[v])
               for v in range(n)]
        ids = {}
        rid = [ids.setdefault(lab[v], len(ids)) for v in range(n)]
        R = len(ids)
        S = np.zeros((n, R))
        for v in range(n):
            S[v, rid[v]] = 1.0
        P = S.T / np.maximum(S.sum(axis=0), 1.0)[:, None]
        radj = [set() for _ in range(R)]
        for v in range(n):
            for u in self.adj[v]:
                if rid[u] != rid[v]:
                    radj[rid[v]].add(rid[u])
        Ar = _mean_matrix([sorted(s) for s in radj], R, R)
        return S, P, Ar

    def features(self, stones, to_move):
        """Aut-invariant input features per rank."""
        n = self.n
        A = self.kernel.analyze(stones)
        Xv = np.zeros((n, 10))
        md = max(self.deg.max(), 1.0)
        for v in range(n):
            c = stones[v]
            Xv[v, 0] = 1.0 if c == to_move else 0.0
            Xv[v, 1] = 1.0 if c and c != to_move else 0.0
            Xv[v, 2] = 1.0 if c == 0 else 0.0
            Xv[v, 3] = self.deg[v] / md
            Xv[v, 4] = self.curv[v]
            if c:
                g = A["groups"][A["gid"][v]]
                own = c == to_move
                Xv[v, 5 if own else 7] = min(g["size"], 10) / 10.0
                Xv[v, 6 if own else 8] = min(len(g["libs"]), 6) / 6.0
            Xv[v, 9] = 1.0
        Xe = np.zeros((self.ne, 5))
        for ei, (a, b) in enumerate(self.edges):
            Xe[ei, 0] = 1.0 if self.eps[a][b] < 0 else 0.0     # seam flag
            ca, cb = stones[a], stones[b]
            Xe[ei, 1] = 1.0 if (ca and ca == cb) else 0.0
            Xe[ei, 2] = 1.0 if (ca and cb and ca != cb) else 0.0
            Xe[ei, 3] = 1.0 if (ca == 0 and cb == 0) else 0.0
            Xe[ei, 4] = 1.0
        Xf = None
        if self.nf:
            Xf = np.zeros((self.nf, 5))
            for fi, f in enumerate(self.faces):
                cs = [stones[v] for v in f]
                Xf[fi, 0] = 1.0 if all(c == 0 for c in cs) else 0.0
                ring = [c for c in cs if c]
                Xf[fi, 1] = 1.0 if ring and all(c == to_move for c in ring) \
                    else 0.0
                Xf[fi, 2] = 1.0 if ring and all(c not in (0, to_move)
                                                for c in ring) else 0.0
                Xf[fi, 3] = len(f) / 8.0
                Xf[fi, 4] = 1.0
        return Xv, Xe, Xf


# ---------- the network ---------------------------------------------------------

FV, FE, FF, FG = 10, 5, 5, 2


def cayley(T):
    """Orthogonal map from a raw square matrix: W = (I-A)(I+A)^{-1},
    A = T - T^T skew. Returns (W, A, Q) with Q = (I+A)^{-1} cached for the
    hand-derived backward pass."""
    A = T - T.T
    Q = np.linalg.inv(np.eye(T.shape[0]) + A)
    return (np.eye(T.shape[0]) - A) @ Q, A, Q


def cayley_backward(G, A, Q):
    """dL/dT given dL/dW for W = (I-A)(I+A)^{-1}."""
    dA = -G @ Q.T - Q.T @ (np.eye(A.shape[0]) - A).T @ G @ Q.T
    return dA - dA.T


class HATZ:
    """v2: outcome-supervised filtration levels with per-node attention,
    ownership-driven Morse-Smale pooling, and Cayley-orthogonal
    eps-conditioned transport. See module docstring."""

    LEVELS = (0.75, 0.5, None)          # co-ownership thresholds; None = full

    def __init__(self, hidden=32, layers=3, seed=0):
        rng = np.random.default_rng(seed)
        D = hidden

        def init(a, b):
            return rng.standard_normal((a, b)) * np.sqrt(2.0 / a)

        p = {"Wv_in": init(FV, D), "We_in": init(FE, D), "Wf_in": init(FF, D),
             "Wg_in": init(FG, D),
             "wp": rng.standard_normal(D) * 0.05, "bp": np.zeros(1),
             "wq": rng.standard_normal(D) * 0.05, "bq": np.zeros(1),
             "wv": rng.standard_normal(D) * 0.05, "bv": np.zeros(1),
             "wo": rng.standard_normal(D) * 0.05, "bo": np.zeros(1),
             "Wp": init(D, D), "Wr": init(D, D), "Wu": init(D, D),
             # GIN-eps co-ownership edge filter head
             "Wgin": init(D, D), "bgin": np.zeros(D),
             "eps_gin": np.zeros(1),
             "Wf1": init(2 * D, D), "bf1": np.zeros(D),
             "wf2": rng.standard_normal(D) * 0.05, "bf2": np.zeros(1)}
        for l in range(layers):
            p[f"T0{l}"] = rng.standard_normal((D, D)) * 0.05   # Cayley raw
            p[f"T1{l}"] = rng.standard_normal((D, D)) * 0.05
            for name in ("Ws", "Wev", "Wgl", "Wes", "Wve", "Wfe",
                         "Wfs", "Wef"):
                p[f"{name}{l}"] = init(D, D)
            p[f"a{l}"] = rng.standard_normal((len(self.LEVELS), D)) * 0.05
            p[f"bv{l}"] = np.zeros(D)
            p[f"be{l}"] = np.zeros(D)
            p[f"bf{l}"] = np.zeros(D)
        self.p = p
        self.hidden, self.layers = D, layers
        self.meta = {}
        self._adam = None

    # ---- per-position structure builders ------------------------------------

    @staticmethod
    def _level_mats(bundle, keep):
        """Row-normalized plain and eps-signed aggregation over the kept edge
        subset (a boolean per undirected edge). Structure only: no gradient."""
        n = bundle.n
        A = np.zeros((n, n))
        Ae = np.zeros((n, n))
        deg = np.zeros(n)
        for ei, (u, v) in enumerate(bundle.edges):
            if keep[ei]:
                deg[u] += 1
                deg[v] += 1
        for ei, (u, v) in enumerate(bundle.edges):
            if keep[ei]:
                e = bundle.eps[u][v]
                A[u, v] += 1.0 / deg[u]
                A[v, u] += 1.0 / deg[v]
                Ae[u, v] += e / deg[u]
                Ae[v, u] += e / deg[v]
        return A, Ae

    def _msc_pool(self, bundle, f):
        """Filtration pooling of the mid-network ownership field: vertices
        are banded by interlevel sets of f ({f >= t}, {|f| < t}, {f <= -t}
        at the median-|f| threshold) and regions are the connected
        components of each band — Black-leaning territory, contested
        frontier, White-leaning territory. Components map to components
        under any symmetry of f, so the partition is exactly
        automorphism-equivariant; basin partitions are not, because equal
        chiral plateaus force index tie-breaks. Structure only: the
        gradient path is the field's own supervision."""
        n = bundle.n
        fq = np.round(np.asarray(f, float), 6)
        t = np.round(float(np.quantile(np.abs(fq), 0.5)), 6)
        band = np.where(fq >= max(t, 1e-6), 1,
                        np.where(fq <= -max(t, 1e-6), -1, 0))
        rid = [-1] * n
        R = 0
        for v in range(n):
            if rid[v] != -1:
                continue
            rid[v] = R
            stack = [v]
            while stack:
                x = stack.pop()
                for u in bundle.adj[x]:
                    if band[u] == band[v] and rid[u] == -1:
                        rid[u] = R
                        stack.append(u)
            R += 1
        S = np.zeros((n, R))
        for v in range(n):
            S[v, rid[v]] = 1.0
        P = S.T / np.maximum(S.sum(axis=0), 1.0)[:, None]
        radj = [set() for _ in range(R)]
        for v in range(n):
            for u in bundle.adj[v]:
                if rid[u] != rid[v]:
                    radj[rid[v]].add(rid[u])
        Ar = _mean_matrix([sorted(x) for x in radj], R, R)
        return S, P, Ar

    # ---- forward -------------------------------------------------------------

    def forward(self, bundle, stones, to_move, mask, frozen=None):
        """frozen: optional {"mats": ..., "msc": (S, P, Ar)} to reuse
        position structure — used by the gradient check (structure is
        stop-grad, so finite differences must not cross partition flips)
        and available for tree reuse."""
        p, D, L = self.p, self.hidden, self.layers
        Xv, Xe, Xf = bundle.features(stones, to_move)
        n = bundle.n
        C = {"b": bundle, "Xv": Xv, "Xe": Xe, "Xf": Xf,
             "mask": np.asarray(mask, float)}
        g_in = np.tile(bundle.gfeat, (1, 1))
        Hv = np.maximum(Xv @ p["Wv_in"] + (g_in @ p["Wg_in"]), 0)
        He = np.maximum(Xe @ p["We_in"], 0)
        Hf = np.maximum(Xf @ p["Wf_in"], 0) if Xf is not None else None
        C["Hv0"], C["He0"], C["Hf0"] = Hv, He, Hf

        # GIN-eps co-ownership edge filter (Leventhal 2025; Hofer et al. 2020)
        Zg = ((1 + p["eps_gin"][0]) * Hv + bundle.Asum @ Hv) @ p["Wgin"]             + p["bgin"]
        Gh = np.maximum(Zg, 0)
        eu = np.array([e[0] for e in bundle.edges])
        ev = np.array([e[1] for e in bundle.edges])
        xsum = Gh[eu] + Gh[ev]
        xdif = Gh[eu] - Gh[ev]
        Xef = np.concatenate([xsum, np.abs(xdif)], axis=1)
        Zf1 = Xef @ p["Wf1"] + p["bf1"]
        Hf1 = np.maximum(Zf1, 0)
        tlog = Hf1 @ p["wf2"] + p["bf2"][0]
        phi = 1.0 / (1.0 + np.exp(-tlog))
        C.update(Zg=Zg, Gh=Gh, eu=eu, ev=ev, xdif=xdif, Zf1=Zf1, Hf1=Hf1,
                 tlog=tlog, phi=phi)

        # filtration levels over the filter values (structure: stop-grad)
        mats = []
        for th in self.LEVELS:
            keep = np.ones(len(bundle.edges), bool) if th is None                 else (phi >= th)
            mats.append(self._level_mats(bundle, keep))
        C["mats"] = mats

        C["Ls"] = []
        C["msc"] = None
        for l in range(L):
            W0, A0c, Q0 = cayley(p[f"T0{l}"])
            W1, A1c, Q1 = cayley(p[f"T1{l}"])
            gmean = Hv.mean(axis=0, keepdims=True)
            ms, ss, als = [], [], None
            for (Al, Ael) in mats:
                ms.append((Al @ Hv) @ W0 + (Ael @ Hv) @ W1)
            a = p[f"a{l}"]
            S_att = np.stack([m @ a[i] for i, m in enumerate(ms)], axis=1)
            S_att = S_att - S_att.max(axis=1, keepdims=True)
            expS = np.exp(S_att)
            alpha = expS / expS.sum(axis=1, keepdims=True)      # (n, P)
            M = sum(alpha[:, i:i + 1] * ms[i] for i in range(len(ms)))
            EH = bundle.M_ev @ He
            Zv = (Hv @ p[f"Ws{l}"] + M + EH @ p[f"Wev{l}"]
                  + gmean @ p[f"Wgl{l}"] + p[f"bv{l}"])
            VH = bundle.M_ve @ Hv
            Ze = He @ p[f"Wes{l}"] + VH @ p[f"Wve{l}"] + p[f"be{l}"]
            FHe = EHf = Zf = None
            if Hf is not None:
                FHe = bundle.M_fe @ Hf
                Ze = Ze + FHe @ p[f"Wfe{l}"]
                EHf = bundle.M_ef @ He
                Zf = Hf @ p[f"Wfs{l}"] + EHf @ p[f"Wef{l}"] + p[f"bf{l}"]
            C["Ls"].append({"Hv": Hv, "He": He, "Hf": Hf, "ms": ms,
                            "alpha": alpha, "M": M, "EH": EH, "VH": VH,
                            "gmean": gmean, "Zv": Zv, "Ze": Ze, "FHe": FHe,
                            "EHf": EHf, "Zf": Zf,
                            "W0": W0, "A0c": A0c, "Q0": Q0,
                            "W1": W1, "A1c": A1c, "Q1": Q1})
            Hv, He = np.maximum(Zv, 0), np.maximum(Ze, 0)
            Hf = np.maximum(Zf, 0) if Zf is not None else None
            if l == 0:
                # ownership at mid-depth: supervised, and the Morse function
                # whose basins define the pooling regions
                uo = Hv @ p["wo"] + p["bo"][0]
                own_mid = np.tanh(uo)
                if frozen is not None:
                    S, Pp, Ar = frozen["msc"]
                else:
                    S, Pp, Ar = self._msc_pool(bundle, own_mid)
                PH = Pp @ Hv
                Zr = PH @ p["Wp"] + (Ar @ PH) @ p["Wr"]
                Rh = np.maximum(Zr, 0)
                Hv = Hv + (S @ Rh) @ p["Wu"]
                C["msc"] = {"S": S, "P": Pp, "Ar": Ar, "PH": PH, "Zr": Zr,
                            "Rh": Rh, "own_mid": own_mid, "uo": uo}

        C["Hv"], C["He"], C["Hf"] = Hv, He, Hf
        gv = Hv.mean(axis=0)
        logits = np.concatenate([Hv @ p["wp"] + p["bp"][0],
                                 [gv @ p["wq"] + p["bq"][0]]])
        z = logits + np.where(C["mask"] > 0, 0.0, -1e9)
        z = z - z.max()
        e = np.exp(z)
        probs = e / e.sum()
        u = gv @ p["wv"] + p["bv"][0]
        value = float(np.tanh(u))
        own = np.tanh(Hv @ p["wo"] + p["bo"][0])
        C.update(gv=gv, probs=probs, u=u, value=value, own=own)
        return probs, value, own, C

    def policy_value(self, bundle, stones, to_move, mask):
        probs, value, _, _ = self.forward(bundle, stones, to_move, mask)
        return probs, value

    # ---- backward ------------------------------------------------------------

    def backward(self, C, pi, z_t, own_t, grads, eown_t=None,
                 vw=1.0, ow=0.5, mw=0.25, fw=0.25):
        p, b = self.p, C["b"]
        n = b.n
        probs, value, own, Hv, gv = (C["probs"], C["value"], C["own"],
                                     C["Hv"], C["gv"])
        loss = -float(np.sum(pi * np.log(probs + 1e-12)))             + vw * (z_t - value) ** 2
        dlog = probs - pi
        dHv = np.outer(dlog[:n], p["wp"])
        grads["wp"] += Hv.T @ dlog[:n]
        grads["bp"][0] += dlog[:n].sum()
        dgv = dlog[n] * p["wq"]
        grads["wq"] += gv * dlog[n]
        grads["bq"][0] += dlog[n]
        du = vw * 2 * (value - z_t) * (1 - value * value)
        dgv = dgv + du * p["wv"]
        grads["wv"] += gv * du
        grads["bv"][0] += du
        if own_t is not None:
            do = ow * 2 * (own - own_t) / n * (1 - own * own)
            loss += ow * float(np.mean((own - own_t) ** 2))
            dHv = dHv + np.outer(do, p["wo"])
            grads["wo"] += Hv.T @ do
            grads["bo"][0] += do.sum()
        dHv = dHv + dgv[None, :] / n

        # edge co-ownership filter supervision (gradient reaches the filter
        # head here; the level masks themselves are stop-grad structure)
        dGh = np.zeros_like(C["Gh"])
        if eown_t is not None and len(C["phi"]):
            ne = len(C["phi"])
            loss += fw * float(np.mean(
                -eown_t * np.log(C["phi"] + 1e-12)
                - (1 - eown_t) * np.log(1 - C["phi"] + 1e-12)))
            dt = fw * (C["phi"] - eown_t) / ne
            grads["wf2"] += C["Hf1"].T @ dt
            grads["bf2"][0] += dt.sum()
            dHf1 = np.outer(dt, p["wf2"]) * (C["Zf1"] > 0)
            grads["Wf1"] += np.concatenate(
                [C["Gh"][C["eu"]] + C["Gh"][C["ev"]],
                 np.abs(C["xdif"])], axis=1).T @ dHf1
            grads["bf1"] += dHf1.sum(axis=0)
            dXef = dHf1 @ p["Wf1"].T
            D = self.hidden
            dsum, ddif = dXef[:, :D], dXef[:, D:] * np.sign(C["xdif"])
            np.add.at(dGh, C["eu"], dsum + ddif)
            np.add.at(dGh, C["ev"], dsum - ddif)

        L = self.layers
        dHe = np.zeros_like(C["He"])
        dHf = np.zeros_like(C["Hf"]) if C["Hf"] is not None else None
        for l in reversed(range(L)):
            if l == 0 and C["msc"] is not None:
                m = C["msc"]
                grads["Wu"] += m["Rh"].T @ (m["S"].T @ dHv)
                dRh = (m["S"].T @ dHv) @ p["Wu"].T
                dZr = dRh * (m["Zr"] > 0)
                grads["Wp"] += m["PH"].T @ dZr
                grads["Wr"] += (m["Ar"] @ m["PH"]).T @ dZr
                dPH = dZr @ p["Wp"].T + m["Ar"].T @ (dZr @ p["Wr"].T)
                dHv = dHv + m["P"].T @ dPH
                if own_t is not None:                 # mid-ownership term
                    om = m["own_mid"]
                    dm = mw * 2 * (om - own_t) / n * (1 - om * om)
                    loss += mw * float(np.mean((om - own_t) ** 2))
                    dHv = dHv + np.outer(dm, p["wo"])
                    HvA = np.maximum(C["Ls"][0]["Zv"], 0)
                    grads["wo"] += HvA.T @ dm
                    grads["bo"][0] += dm.sum()
            Lc = C["Ls"][l]
            dZv = dHv * (Lc["Zv"] > 0)
            dZe = dHe * (Lc["Ze"] > 0)
            grads[f"Ws{l}"] += Lc["Hv"].T @ dZv
            grads[f"Wev{l}"] += Lc["EH"].T @ dZv
            grads[f"Wgl{l}"] += Lc["gmean"].T @ dZv.sum(axis=0, keepdims=True)
            grads[f"bv{l}"] += dZv.sum(axis=0)
            grads[f"Wes{l}"] += Lc["He"].T @ dZe
            grads[f"Wve{l}"] += Lc["VH"].T @ dZe
            grads[f"be{l}"] += dZe.sum(axis=0)

            # level attention backward: M = sum_i alpha_i * m_i
            alpha, ms, M = Lc["alpha"], Lc["ms"], Lc["M"]
            a = p[f"a{l}"]
            GM = dZv
            gm_dot_M = (GM * M).sum(axis=1)
            dW0 = np.zeros_like(Lc["W0"])
            dW1 = np.zeros_like(Lc["W1"])
            dHv_new = (dZv @ p[f"Ws{l}"].T
                       + b.M_ve.T @ (dZe @ p[f"Wve{l}"].T)
                       + np.ones((n, 1)) @ ((dZv @ p[f"Wgl{l}"].T)
                                            .sum(axis=0, keepdims=True)) / n)
            for i, (Al, Ael) in enumerate(C["mats"]):
                ds = alpha[:, i] * ((GM * ms[i]).sum(axis=1) - gm_dot_M)
                dm = alpha[:, i:i + 1] * GM + np.outer(ds, a[i])
                grads[f"a{l}"][i] += ms[i].T @ ds
                AH = Al @ Lc["Hv"]
                AeH = Ael @ Lc["Hv"]
                dW0 += AH.T @ dm
                dW1 += AeH.T @ dm
                dHv_new += Al.T @ (dm @ Lc["W0"].T)                     + Ael.T @ (dm @ Lc["W1"].T)
            grads[f"T0{l}"] += cayley_backward(dW0, Lc["A0c"], Lc["Q0"])
            grads[f"T1{l}"] += cayley_backward(dW1, Lc["A1c"], Lc["Q1"])

            dHe_new = dZe @ p[f"Wes{l}"].T + b.M_ev.T @ (dZv @ p[f"Wev{l}"].T)
            if Lc["Hf"] is not None:
                dZf = dHf * (Lc["Zf"] > 0)
                grads[f"Wfe{l}"] += Lc["FHe"].T @ dZe
                grads[f"Wfs{l}"] += Lc["Hf"].T @ dZf
                grads[f"Wef{l}"] += Lc["EHf"].T @ dZf
                grads[f"bf{l}"] += dZf.sum(axis=0)
                dHf = dZf @ p[f"Wfs{l}"].T + b.M_fe.T @ (dZe @ p[f"Wfe{l}"].T)
                dHe_new = dHe_new + b.M_ef.T @ (dZf @ p[f"Wef{l}"].T)
            dHv, dHe = dHv_new, dHe_new

        # GIN filter head backward into the input embedding
        if np.any(dGh):
            dZg = dGh * (C["Zg"] > 0)
            pre = (1 + p["eps_gin"][0]) * C["Hv0"] + b.Asum @ C["Hv0"]
            grads["Wgin"] += pre.T @ dZg
            grads["bgin"] += dZg.sum(axis=0)
            dpre = dZg @ p["Wgin"].T
            grads["eps_gin"][0] += float((dpre * C["Hv0"]).sum())
            dHv = dHv + (1 + p["eps_gin"][0]) * dpre + b.Asum.T @ dpre

        dZ = dHv * (C["Hv0"] > 0)
        grads["Wv_in"] += C["Xv"].T @ dZ
        grads["Wg_in"] += np.tile(b.gfeat, (1, 1)).T @ dZ.sum(
            axis=0, keepdims=True)
        dZe0 = dHe * (C["He0"] > 0)
        grads["We_in"] += C["Xe"].T @ dZe0
        if C["Hf0"] is not None:
            dZf0 = dHf * (C["Hf0"] > 0)
            grads["Wf_in"] += C["Xf"].T @ dZf0
        return loss

    def zero_grads(self):
        return {k: np.zeros_like(v) for k, v in self.p.items()}

    def adam_step(self, grads, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, wd=1e-5):
        if self._adam is None:
            self._adam = {"t": 0,
                          "m": {k: np.zeros_like(v) for k, v in self.p.items()},
                          "v": {k: np.zeros_like(v) for k, v in self.p.items()}}
        a = self._adam
        a["t"] += 1
        for k in self.p:
            gk = grads[k] + wd * self.p[k]
            a["m"][k] = b1 * a["m"][k] + (1 - b1) * gk
            a["v"][k] = b2 * a["v"][k] + (1 - b2) * gk * gk
            mh = a["m"][k] / (1 - b1 ** a["t"])
            vh = a["v"][k] / (1 - b2 ** a["t"])
            self.p[k] -= lr * mh / (np.sqrt(vh) + eps)

    def save(self, path):
        np.savez(path, __meta__=json.dumps(
            {"hidden": self.hidden, "layers": self.layers, **self.meta}),
            **self.p)

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=False)
        meta = json.loads(str(d["__meta__"]))
        net = cls(hidden=meta["hidden"], layers=meta["layers"])
        for k in net.p:
            net.p[k] = d[k]
        net.meta = {k: v for k, v in meta.items()
                    if k not in ("hidden", "layers")}
        return net


def co_ownership_targets(bundle, own_black):
    """Per-edge target: 1 if the endpoints end up owned by the same player,
    0 if by opposite players, 0.5 if either is neutral. Perspective-invariant."""
    t = np.full(len(bundle.edges), 0.5)
    for ei, (u, v) in enumerate(bundle.edges):
        s = own_black[u] * own_black[v]
        if s > 0:
            t[ei] = 1.0
        elif s < 0:
            t[ei] = 0.0
    return t


# ---------- ownership targets ----------------------------------------------------

def final_ownership(board, colors):
    """Tromp-Taylor ownership of the final position, per vertex, in {-1,0,1}
    from BLACK's perspective."""
    n = board.n
    own = np.zeros(n)
    for v in range(n):
        if colors[v]:
            own[v] = 1.0 if colors[v] == 1 else -1.0
    seen = [False] * n
    for v in range(n):
        if colors[v] or seen[v]:
            continue
        region, border, stack = [], set(), [v]
        seen[v] = True
        while stack:
            x = stack.pop()
            region.append(x)
            for u in board.adj[x]:
                if colors[u]:
                    border.add(colors[u])
                elif not seen[u]:
                    seen[u] = True
                    stack.append(u)
        if border == {1}:
            for x in region:
                own[x] = 1.0
        elif border == {2}:
            for x in region:
                own[x] = -1.0
    return own
