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


class HATZ:
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
             "Wp": init(D, D), "Wr": init(D, D), "Wu": init(D, D)}
        for l in range(layers):
            for name in ("Ws", "W0", "W1", "Wev", "Wgl",
                         "Wes", "Wve", "Wfe", "Wfs", "Wef"):
                p[f"{name}{l}"] = init(D, D)
            p[f"bv{l}"] = np.zeros(D)
            p[f"be{l}"] = np.zeros(D)
            p[f"bf{l}"] = np.zeros(D)
        self.p = p
        self.hidden, self.layers = D, layers
        self.meta = {}
        self._adam = None

    # ---- forward ---------------------------------------------------------------

    def forward(self, bundle, stones, to_move, mask):
        p, D, L = self.p, self.hidden, self.layers
        Xv, Xe, Xf = bundle.features(stones, to_move)
        S, P, Ar = bundle.msc_pool(stones, to_move)
        n = bundle.n
        C = {"b": bundle, "Xv": Xv, "Xe": Xe, "Xf": Xf,
             "S": S, "P": P, "Ar": Ar, "mask": np.asarray(mask, float)}
        g_in = np.tile(bundle.gfeat, (1, 1))
        Hv = np.maximum(Xv @ p["Wv_in"] + (g_in @ p["Wg_in"]), 0)
        He = np.maximum(Xe @ p["We_in"], 0)
        Hf = np.maximum(Xf @ p["Wf_in"], 0) if Xf is not None else None
        C["Hv0"], C["He0"], C["Hf0"] = Hv, He, Hf
        C["Ls"] = []
        for l in range(L):
            gmean = Hv.mean(axis=0, keepdims=True)
            AH, AeH = bundle.A @ Hv, bundle.Ae @ Hv
            EH = bundle.M_ev @ He
            Zv = (Hv @ p[f"Ws{l}"] + AH @ p[f"W0{l}"] + AeH @ p[f"W1{l}"]
                  + EH @ p[f"Wev{l}"] + gmean @ p[f"Wgl{l}"] + p[f"bv{l}"])
            VH = bundle.M_ve @ Hv
            Ze = He @ p[f"Wes{l}"] + VH @ p[f"Wve{l}"] + p[f"be{l}"]
            FH = None
            if Hf is not None:
                FHe = bundle.M_fe @ Hf
                Ze = Ze + FHe @ p[f"Wfe{l}"]
                EHf = bundle.M_ef @ He
                Zf = Hf @ p[f"Wfs{l}"] + EHf @ p[f"Wef{l}"] + p[f"bf{l}"]
                FH = np.maximum(Zf, 0)
            Hv2, He2 = np.maximum(Zv, 0), np.maximum(Ze, 0)
            C["Ls"].append({"Hv": Hv, "He": He, "Hf": Hf, "AH": AH,
                            "AeH": AeH, "EH": EH, "VH": VH, "gmean": gmean,
                            "Zv": Zv, "Ze": Ze,
                            "FHe": FHe if Hf is not None else None,
                            "EHf": EHf if Hf is not None else None,
                            "Zf": Zf if Hf is not None else None})
            Hv, He, Hf = Hv2, He2, FH
            if l == 0:                                   # Morse–Smale block
                PH = P @ Hv
                Zr = PH @ p["Wp"] + (Ar @ PH) @ p["Wr"]
                Rh = np.maximum(Zr, 0)
                Hv = Hv + (S @ Rh) @ p["Wu"]
                C["msc"] = {"PH": PH, "Zr": Zr, "Rh": Rh, "Hv_pre": Hv}
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

    # ---- backward --------------------------------------------------------------

    def backward(self, C, pi, z_t, own_t, grads, vw=1.0, ow=0.5):
        p, b = self.p, C["b"]
        n = b.n
        probs, value, own, Hv, gv = (C["probs"], C["value"], C["own"],
                                     C["Hv"], C["gv"])
        loss = -float(np.sum(pi * np.log(probs + 1e-12))) \
            + vw * (z_t - value) ** 2
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

        L = self.layers
        dHe = np.zeros_like(C["He"])
        dHf = np.zeros_like(C["Hf"]) if C["Hf"] is not None else None
        for l in reversed(range(L)):
            if l == 0:                                   # MSC block backward
                m = C["msc"]
                dRhWu = dHv.copy()
                grads["Wu"] += m["Rh"].T @ (C["S"].T @ dHv)
                dRh = (C["S"].T @ dRhWu) @ p["Wu"].T
                dZr = dRh * (m["Zr"] > 0)
                grads["Wp"] += m["PH"].T @ dZr
                grads["Wr"] += (C["Ar"] @ m["PH"]).T @ dZr
                dPH = dZr @ p["Wp"].T + C["Ar"].T @ (dZr @ p["Wr"].T)
                dHv = dHv + C["P"].T @ dPH
            Lc = C["Ls"][l]
            dZv = dHv * (Lc["Zv"] > 0)
            dZe = dHe * (Lc["Ze"] > 0)
            grads[f"Ws{l}"] += Lc["Hv"].T @ dZv
            grads[f"W0{l}"] += Lc["AH"].T @ dZv
            grads[f"W1{l}"] += Lc["AeH"].T @ dZv
            grads[f"Wev{l}"] += Lc["EH"].T @ dZv
            grads[f"Wgl{l}"] += Lc["gmean"].T @ dZv.sum(axis=0, keepdims=True)
            grads[f"bv{l}"] += dZv.sum(axis=0)
            grads[f"Wes{l}"] += Lc["He"].T @ dZe
            grads[f"Wve{l}"] += Lc["VH"].T @ dZe
            grads[f"be{l}"] += dZe.sum(axis=0)
            dHv_new = (dZv @ p[f"Ws{l}"].T + b.A.T @ (dZv @ p[f"W0{l}"].T)
                       + b.Ae.T @ (dZv @ p[f"W1{l}"].T)
                       + b.M_ve.T @ (dZe @ p[f"Wve{l}"].T)
                       + np.ones((n, 1)) @ ((dZv @ p[f"Wgl{l}"].T)
                                            .sum(axis=0, keepdims=True)) / n)
            dHe_new = dZe @ p[f"Wes{l}"].T + b.M_ev.T @ (dZv @ p[f"Wev{l}"].T)
            if Lc["Hf"] is not None:
                dZf = dHf * (Lc["Zf"] > 0)
                grads[f"Wfe{l}"] += Lc["FHe"].T @ dZe
                grads[f"Wfs{l}"] += Lc["Hf"].T @ dZf
                grads[f"Wef{l}"] += Lc["EHf"].T @ dZf
                grads[f"bf{l}"] += dZf.sum(axis=0)
                dHf_new = dZf @ p[f"Wfs{l}"].T + b.M_fe.T @ (dZe @ p[f"Wfe{l}"].T)
                dHe_new = dHe_new + b.M_ef.T @ (dZf @ p[f"Wef{l}"].T)
                dHf = dHf_new
            dHv, dHe = dHv_new, dHe_new
        dZ = dHv * (C["Hv0"] > 0)
        grads["Wv_in"] += C["Xv"].T @ dZ
        grads["Wg_in"] += np.tile(C["b"].gfeat, (1, 1)).T @ dZ.sum(
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
