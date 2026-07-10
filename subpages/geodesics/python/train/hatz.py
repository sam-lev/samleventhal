"""hatz.py — Holonomy-Aware Topological AlphaZero, v3 (numpy, no deps).

v1 established the Z2/reflection gauge structure (orientation cocycle eps
from face cycles or quotient gluings), sheaf-style eps-conditioned transport
T(u->v) = W0 + eps(u,v) W1 with globally shared maps, cell-rank V-E-F
message passing, Morse-Smale pooling, and exact automorphism equivariance of
the weight-shared core. v2 added Cayley-orthogonal transport, an
outcome-supervised GIN-eps co-ownership edge filtration with fixed
thresholds and per-node attention over the resulting levels, and
ownership-driven interlevel-set pooling (component pooling of the mid-depth
ownership field, which stays equivariant where basin tie-breaking does not).

v3, this file:

1. **Persistence-pair injection.** The H0 persistence pairs of the
   phi-filtration (phi = the learned co-ownership potential per edge) are
   computed by a descending Kruskal sweep and injected additively into the
   vertex and edge embeddings: every vertex carries the (birth, death,
   lifetime, essential) of the superlevel component it was born into, every
   edge carries its pairing role (tree edge that killed a class / cycle edge
   that births an H1 class) and the lifetime it terminated. The hierarchy
   therefore carries not just level membership but the topological lifetime
   of every connection. Pairing and values are structure (stop-grad); phi's
   gradient path is its own co-ownership supervision.

2. **Learned quantile thresholds (Hofer-style differentiable filtration).**
   Instead of fixed thresholds (0.75, 0.5, full), level l keeps edges with
   phi >= p_l where p_l is the *interpolated empirical quantile* of the
   current phi distribution at a learned fraction sigmoid(qraw_l). Hard
   keeps and hard-count row normalization are structure (stop-grad); a soft
   gate sigma((phi_e - p_l)/tau) multiplies each kept edge, carrying
   gradient to phi_e, to p_l, and through the interpolated quantile to the
   two bracketing order statistics of phi and to qraw. As heterophily rises
   through a game the levels slide to track the actual contact structure:
   early game the phi distribution is tight and the levels cluster near the
   full graph; in a midgame contact fight phi spreads and the levels
   separate to isolate the fight. No relaxation of the combinatorics is
   needed — exactly the split Hofer et al. (2020) exploit.

3. **Successive-training curriculum.** forward() takes levels_active: the
   attention mixes only the first levels_active levels (most homophilous
   first). Early self-play iterations train on the cleanest co-ownership
   level (separable chains, where a stalled policy can actually learn),
   then heterophilous levels are annealed in (train_hatz.py --curriculum).

4. **Cayley-geometric transport (between copresheaf and gauge).** The
   copresheaf restriction maps are constrained orthogonal by the Cayley
   parameterization, and the transport is *initialized at the mesh's actual
   parallel transporter*: messages are rotated by the discrete Levi-Civita
   angle theta_{u<-v} (2x2 blocks on channel pairs; exact on boards with 3D
   coords + faces, identity on flat boards) and then passed through the
   Cayley map W0, whose raw parameter starts at 0 so W0 = I. At
   initialization the layer is exactly the gauge convolution's geometric
   transporter; self-play deforms W0 away from I while orthogonality — the
   gauge conv's key virtue — is guaranteed for every step. On non-orientable
   boards the sin channel is zeroed exactly: w1 obstructs a globally
   consistent rotation sign, and the seam crossing is carried by the
   reflection-aware eps path (W1) instead. Frame choice is a gauge; like
   the eps gauge it is randomized by with_gauge() during training so
   invariance is learned rather than assumed.

5. **Morse-flow directed propagation.** The NeurIPS'22 segmentation stops
   being only a pooling operator and becomes the routing of message
   passing. Let f be the mid-depth ownership field (already computed,
   already supervised). Its discrete gradient orients edges; the ascending
   basins of f (steepest-ascent union-find, <=-persistence merge for
   equivariant plateau handling) partition the board. Every aggregation at
   layers l >= 1 is split into three learned channels by an edge's role
   relative to the flow: ascending (lower -> higher f inside a basin, W_up),
   descending (W_dn), and lateral/separatrix (edges between different
   basins — the ridges where ascending flows diverge — plus in-basin
   plateau ties, W_lat):

       m_v = W_up  . mean over up-neighbors
           + W_dn  . mean over down-neighbors
           + W_lat . mean over cross-basin neighbors.

   Isotropic MP averages all three; this distinguishes them, and the
   separatrix channel makes contested boundaries — where Go fights live —
   first-class carriers of information rather than edges averaged away.
   A node near a contested boundary receives ascending messages from its
   own basin distinctly from lateral messages across the ridge, so it can
   represent "on the Black side of a fight with White pressing from that
   direction", which mean aggregation cannot express. Basin labels are
   structure (stop-grad); f's gradient path is its own supervision.

Known gaps, stated plainly: transport frames (like the canonical eps
gauge) break *exact* orbit equivariance on curved boards — it is trained
toward invariance by frame-gauge augmentation, and remains exact on flat
boards where theta = 0; PH features are H0 of the phi-filtration rather
than full multiparameter persistence.
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


def transport_angles(board, edges, nonorientable):
    """Discrete Levi-Civita parallel transport angle theta_{a<-b} per stored
    edge (a, b): a tangent vector expressed in b's frame is rotated into a's
    frame along the edge, preserving its angle to the shared edge direction:

        theta_{a<-b} = ang_a(b) + pi - ang_b(a)

    where ang_v(u) is the angle of the projected direction (P[u]-P[v]) in
    v's tangent frame. Returns (cos, sin) arrays in the canonical (a, b)
    direction — cos is symmetric, sin antisymmetric (theta_{b<-a} =
    -theta_{a<-b}). Requires 3D coords + faces (spheres, seed polyhedra);
    on flat boards this degenerates to theta = 0 and the layer reduces
    exactly to v2. On non-orientable boards the sin channel is zeroed
    exactly: w1 obstructs a globally consistent rotation sign (the same
    obstruction eps represents), so the seam is carried by the
    reflection-aware eps path instead. Frame choice is a gauge; it is
    randomized by Bundle.with_gauge during training."""
    ne = len(edges)
    ecos, esin = np.ones(ne), np.zeros(ne)
    coords = getattr(board, "coords", None)
    faces = getattr(board, "faces", None)
    if nonorientable or not faces or coords is None or len(coords[0]) < 3:
        return ecos, esin
    P = np.asarray(coords, float)
    n = board.n
    ctr = P.mean(axis=0)
    N = np.zeros((n, 3))
    for fc in faces:
        fn = np.cross(P[fc[1]] - P[fc[0]], P[fc[2]] - P[fc[0]])
        nm = np.linalg.norm(fn)
        if nm < 1e-12:
            continue
        fn = fn / nm
        if np.dot(fn, P[list(fc)].mean(axis=0) - ctr) < 0:
            fn = -fn                       # align outward (convex-ish seeds)
        for v in fc:
            N[v] += fn
    e1 = np.zeros((n, 3))
    e2 = np.zeros((n, 3))
    for v in range(n):
        nm = np.linalg.norm(N[v])
        nv = N[v] / nm if nm > 1e-12 else np.array([0.0, 0.0, 1.0])
        N[v] = nv
        d = None
        for u in board.adj[v]:
            t = P[u] - P[v]
            t = t - np.dot(t, nv) * nv
            if np.linalg.norm(t) > 1e-9:
                d = t
                break
        if d is None:
            d = np.array([1.0, 0.0, 0.0])
            d = d - np.dot(d, nv) * nv
        e1[v] = d / np.linalg.norm(d)
        e2[v] = np.cross(nv, e1[v])

    def ang(v, u):
        d = P[u] - P[v]
        return math.atan2(float(np.dot(d, e2[v])), float(np.dot(d, e1[v])))

    for ei, (a, b) in enumerate(edges):
        th = ang(a, b) + math.pi - ang(b, a)
        ecos[ei], esin[ei] = math.cos(th), math.sin(th)
    return ecos, esin


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
        self.eu = np.array([e[0] for e in self.edges], dtype=int)
        self.ev = np.array([e[1] for e in self.edges], dtype=int)
        self.eeps = np.array([eps[a][b] for a, b in self.edges], dtype=float)
        # geometric parallel transport per edge (canonical direction)
        self.ecos, self.esin = transport_angles(board, self.edges,
                                                self.nonorientable)
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
        """Gauge-transform the structure: eps'(u,v) = g(u) eps(u,v) g(v) for
        a random vertex sign field g (holonomy/w1 invariant, seam moves),
        and — on boards with nontrivial transport — rotate every tangent
        frame by a random angle beta_v, so theta'_{a<-b} = theta + beta_a -
        beta_b (holonomy around every cycle invariant, per-edge angles
        move). Used as training augmentation so the net learns the gauge
        invariance the architecture doesn't yet guarantee."""
        import copy as _copy
        g = rng.choice([-1, 1], size=self.n)
        out = _copy.copy(self)
        out.eps = {v: {u: int(g[v] * self.eps[v][u] * g[u])
                       for u in self.adj[v]} for v in range(self.n)}
        out.Ae = _mean_matrix(self.adj, self.n, self.n,
                              signs=lambda v, u: out.eps[v][u])
        out.eeps = np.array([out.eps[a][b] for a, b in self.edges],
                            dtype=float)
        if np.any(self.esin != 0):
            beta = rng.uniform(0, 2 * np.pi, size=self.n)
            db = beta[self.eu] - beta[self.ev]
            out.ecos = self.ecos * np.cos(db) - self.esin * np.sin(db)
            out.esin = self.esin * np.cos(db) + self.ecos * np.sin(db)
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
FPV, FPE = 4, 3          # persistence features per vertex / per edge
GATE_TAU = 0.1           # soft-gate temperature of the level filtration


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


def rotagg(Ac, As, H):
    """Aggregate with per-edge 2x2 rotation blocks on channel pairs:
    out_even = Ac @ H_even - As @ H_odd, out_odd = As @ H_even + Ac @ H_odd.
    When theta = 0 (Ac plain aggregation, As = 0) this is exactly A @ H. An
    odd leftover channel is transported trivially (Ac only)."""
    D = H.shape[1]
    De = D - (D % 2)
    out = np.empty_like(H)
    He, Ho = H[:, 0:De:2], H[:, 1:De:2]
    out[:, 0:De:2] = Ac @ He - As @ Ho
    out[:, 1:De:2] = As @ He + Ac @ Ho
    if D % 2:
        out[:, -1] = Ac @ H[:, -1]
    return out


def rotagg_T(Ac, As, G):
    """Gradient w.r.t. H of rotagg(Ac, As, H), given dL/d(out) = G."""
    D = G.shape[1]
    De = D - (D % 2)
    out = np.empty_like(G)
    Ge, Go = G[:, 0:De:2], G[:, 1:De:2]
    out[:, 0:De:2] = Ac.T @ Ge + As.T @ Go
    out[:, 1:De:2] = -As.T @ Ge + Ac.T @ Go
    if D % 2:
        out[:, -1] = Ac.T @ G[:, -1]
    return out


def h0_pairs(bundle, phi):
    """H0 persistence pairs of the superlevel filtration of the learned
    co-ownership potential phi on edges (structure: stop-grad; phi's
    gradient path is its co-ownership supervision).

    Descending Kruskal sweep: vertices enter with their first (highest-phi)
    incident edge; a component's class is born at its first internal edge
    and dies (elder rule) when absorbed by an older component. Equal-birth
    merges kill both sides simultaneously — the pair values coincide, so tie
    handling stays automorphism-equivariant. Zero-persistence singleton
    absorptions are diagonal points and are discarded.

    Returns PersV (n x 4): [birth, death, lifetime, essential] of the class
    each vertex was born into (death of the never-absorbed class is the
    global phi minimum, flagged essential); and PersE (ne x 3):
    [tree flag, cycle flag (an H1 birth at phi_e), lifetime of the class
    this edge killed]."""
    n, ne = bundle.n, bundle.ne
    PersV = np.zeros((n, FPV))
    PersE = np.zeros((ne, FPE))
    if ne == 0:
        PersV[:, 3] = 1.0
        return PersV, PersE
    # structure is stop-grad, so quantize: symmetric positions produce phi
    # values equal only to floating-point noise, and exact-tie batching must
    # see them as equal for the pairing to stay equivariant
    phif = np.round(np.asarray(phi, float), 9)
    order = np.argsort(-phif, kind="stable")
    parent = list(range(n))
    members = [[v] for v in range(n)]
    birth = [None] * n          # per-root: phi of the component's first edge
    vbirth = [None] * n
    vdeath = [None] * n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    # equal-phi edges are processed as one simultaneous batch: tie handling
    # (which arises exactly on symmetric positions of a learned field) then
    # depends only on values and sets, never on edge indices, keeping the
    # pairing automorphism-equivariant instead of index-tie-broken
    i = 0
    while i < ne:
        f = float(phif[order[i]])
        batch = []
        while i < ne and float(phif[order[i]]) == f:
            batch.append(int(order[i]))
            i += 1
        for ei in batch:                            # vertices entering at f
            for w in bundle.edges[ei]:
                if vbirth[w] is None:
                    vbirth[w] = f
        pre = {}                                    # tree/cycle w.r.t. the
        touched = set()                             # pre-batch state
        for ei in batch:
            a, b = bundle.edges[ei]
            ra, rb = find(a), find(b)
            pre[ei] = (ra, rb)
            touched.update((ra, rb))
            PersE[ei, 1 if ra == rb else 0] = 1.0
        members_pre = {r: list(members[r]) for r in touched}
        for ei in batch:                            # union the batch
            ra, rb = find(bundle.edges[ei][0]), find(bundle.edges[ei][1])
            if ra == rb:
                continue
            if len(members[ra]) < len(members[rb]):
                ra, rb = rb, ra
            parent[rb] = ra
            members[ra].extend(members[rb])
            members[rb] = []
        groups = {}
        for r in touched:
            groups.setdefault(find(r), []).append(r)
        glife = {}
        for g, roots in groups.items():
            births = [birth[r] for r in roots if birth[r] is not None]
            if not births:                          # all singletons: born now
                birth[g] = f
                glife[g] = 0.0
                continue
            mx = max(births)
            elders = [r for r in roots if birth[r] == mx]
            # elder rule over the merged super-group; a tied elder pair
            # kills both (values coincide, so ties stay equivariant);
            # singleton absorptions are zero-persistence diagonal points
            dying = [r for r in roots if birth[r] is not None
                     and (birth[r] < mx or len(elders) >= 2)]
            for r in dying:
                for w in members_pre[r]:
                    if vdeath[w] is None:
                        vdeath[w] = f
            birth[g] = mx
            glife[g] = max((birth[r] - f for r in dying), default=0.0)
        for ei in batch:
            if PersE[ei, 0]:                        # tree edges carry the
                PersE[ei, 2] = glife[find(bundle.edges[ei][0])]   # kill
    pmin = float(phif.min())
    for v in range(n):
        bv = vbirth[v] if vbirth[v] is not None else pmin
        if vdeath[v] is None:
            PersV[v] = (bv, pmin, bv - pmin, 1.0)
        else:
            PersV[v] = (bv, vdeath[v], bv - vdeath[v], 0.0)
    return PersV, PersE


class HATZ:
    """v3: persistence-pair injection, learned quantile filtration levels
    with Hofer-style soft gates, curriculum masking over levels,
    Cayley-geometric orthogonal transport, and Morse-flow directed
    propagation. See module docstring."""

    K = 5            # filtration levels: two learned quantiles + full graph
    
    def __init__(self, hidden=32, layers=5, seed=0):
        rng = np.random.default_rng(seed)
        D = hidden

        def init(a, b):
            return rng.standard_normal((a, b)) * np.sqrt(2.0 / a)

        initial_quantiles = np.linspace(0.9, 0.3, self.K - 1)

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
             "wf2": rng.standard_normal(D) * 0.05, "bf2": np.zeros(1),
             # persistence-pair injection (pairs are stop-grad structure)
             "Wpv": rng.standard_normal((FPV, D)) * 0.05,
             "Wpe": rng.standard_normal((FPE, D)) * 0.05,
             # learned quantile fractions; sigmoid -> (0.75, 0.5) at init,
             # matching v2's fixed thresholds in spirit but sliding with
             # the position's phi distribution
             "qraw": np.log(initial_quantiles / (1.0 - initial_quantiles)),
             }
             
        for l in range(layers):
            # Cayley raw at 0 => W0 = I: the transport *starts* at the
            # mesh's geometric parallel transporter R(theta) and self-play
            # deforms it, orthogonality guaranteed throughout
            p[f"T0{l}"] = np.zeros((D, D))
            p[f"T1{l}"] = rng.standard_normal((D, D)) * 0.05
            for name in ("Ws", "Wev", "Wgl", "Wes", "Wve", "Wfe",
                         "Wfs", "Wef", "Wup", "Wdn", "Wlat"):
                p[f"{name}{l}"] = init(D, D)
            p[f"a{l}"] = rng.standard_normal((self.K, D)) * 0.05
            p[f"bv{l}"] = np.zeros(D)
            p[f"be{l}"] = np.zeros(D)
            p[f"bf{l}"] = np.zeros(D)
        self.p = p
        self.hidden, self.layers = D, layers
        self.meta = {}
        self._adam = None
        self.levels_active = None        # curriculum default: all levels

    # ---- per-position structure builders ------------------------------------

    @staticmethod
    def _gated_mats(bundle, ki, gate):
        """Aggregation matrices over the kept edge subset ki (indices into
        bundle.edges): plain-cos (Ac), sin (As, antisymmetric — the
        geometric rotation), and eps-signed (Ae), all row-normalized by the
        *hard* kept-degree count (structure) and multiplied by the *soft*
        gate (differentiable)."""
        n = bundle.n
        u, v = bundle.eu[ki], bundle.ev[ki]
        deg = np.zeros(n)
        np.add.at(deg, u, 1.0)
        np.add.at(deg, v, 1.0)
        dn = np.maximum(deg, 1.0)
        Ac = np.zeros((n, n))
        As = np.zeros((n, n))
        Ae = np.zeros((n, n))
        c, s, e = bundle.ecos[ki], bundle.esin[ki], bundle.eeps[ki]
        np.add.at(Ac, (u, v), gate * c / dn[u])
        np.add.at(Ac, (v, u), gate * c / dn[v])
        np.add.at(As, (u, v), gate * s / dn[u])
        np.add.at(As, (v, u), -gate * s / dn[v])
        np.add.at(Ae, (u, v), gate * e / dn[u])
        np.add.at(Ae, (v, u), gate * e / dn[v])
        return Ac, As, Ae, dn

    def _flow_mats(self, bundle, f):
        """Morse-flow routing matrices from the mid-depth ownership field f:
        ascending basins via steepest-ascent union-find (<=-persistence
        merges keep plateau ties equivariant), merged at the 60th-percentile
        persistence. Per vertex: mean over up-neighbors (same basin, higher
        f), down-neighbors (same basin, lower f), and lateral neighbors
        (different basin — a separatrix crossing — or in-basin plateau
        ties). Structure only: f's gradient path is its own supervision."""
        n = bundle.n
        fq = np.round(np.asarray(f, float), 6)
        # ascending-manifold labels by *reachability* on the
        # plateau-contracted graph: basin interiors match steepest ascent,
        # but boundary vertices carry the set of (plateau) maxima their
        # ascending flows can reach — value/set-defined throughout, so
        # exactly Aut-equivariant where raw steepest-ascent pointers would
        # index-tie-break on chiral plateaus. A vertex reaching >= 2 maxima
        # sits on a separatrix, and its edges route laterally — exactly the
        # Morse-Smale cell-boundary reading of "where ascending flows
        # diverge". Persistence simplification of the routing field is
        # deliberately omitted: equal-persistence family merging cannot be
        # made index-free (the same chiral-plateau obstruction, one level
        # up), and over-segmentation only shifts edges into the lateral
        # channel, which is exactly where contested structure belongs.
        sup = [-1] * n
        S = 0
        for v in range(n):
            if sup[v] != -1:
                continue
            sup[v] = S
            stack = [v]
            while stack:
                x = stack.pop()
                for u in bundle.adj[x]:
                    if fq[u] == fq[v] and sup[u] == -1:
                        sup[u] = S
                        stack.append(u)
            S += 1
        sval = [0.0] * S
        snbr = [set() for _ in range(S)]
        for v in range(n):
            sval[sup[v]] = fq[v]
            for u in bundle.adj[v]:
                if sup[u] != sup[v]:
                    snbr[sup[v]].add(sup[u])
        reach = [frozenset()] * S
        for s in sorted(range(S), key=lambda t: -sval[t]):
            higher = [t for t in snbr[s] if sval[t] > sval[s]]
            if not higher:                 # a (plateau) maximum seeds itself
                reach[s] = frozenset([s])
            else:
                r = set()
                for t in higher:
                    r |= reach[t]
                reach[s] = frozenset(r)
        lab = [reach[sup[v]] for v in range(n)]
        up = [[] for _ in range(n)]
        dn = [[] for _ in range(n)]
        lat = [[] for _ in range(n)]
        for v in range(n):
            for u in bundle.adj[v]:
                if lab[u] != lab[v]:
                    lat[v].append(u)       # separatrix crossing
                elif fq[u] > fq[v]:
                    up[v].append(u)
                elif fq[u] < fq[v]:
                    dn[v].append(u)
                else:
                    lat[v].append(u)       # in-basin plateau: boundary-like
        return (_mean_matrix(up, n, n), _mean_matrix(dn, n, n),
                _mean_matrix(lat, n, n))

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

    def freeze_struct(self, C):
        """Snapshot every stop-grad structural choice of a forward pass, so
        the gradient check (and tree reuse) can re-run the differentiable
        parts without crossing partition/keep/pairing flips."""
        return {"struct": {"qord": C["qord"],
                           "qi": [(lv.get("i0"), lv.get("i1"))
                                  for lv in C["levels"]],
                           "keeps": [lv["ki"] for lv in C["levels"]]},
                "msc": (C["msc"]["S"], C["msc"]["P"], C["msc"]["Ar"]),
                "flow": C["flow"],
                "pers": (C["PersV"], C["PersE"])}

    # ---- forward -------------------------------------------------------------

    def forward(self, bundle, stones, to_move, mask, frozen=None,
                levels_active=None):
        """frozen: optional output of freeze_struct() to reuse position
        structure (keeps, quantile order statistics, basins, msc partition,
        persistence pairing). levels_active: curriculum masking — attention
        mixes only the first levels_active levels (most homophilous first);
        None uses self.levels_active, which defaults to all."""
        p, D, L, K = self.p, self.hidden, self.layers, self.K
        if levels_active is None:
            levels_active = self.levels_active
        act = K if levels_active is None else max(1, min(K,
                                                         int(levels_active)))
        Xv, Xe, Xf = bundle.features(stones, to_move)
        n, ne = bundle.n, bundle.ne
        C = {"b": bundle, "Xv": Xv, "Xe": Xe, "Xf": Xf,
             "mask": np.asarray(mask, float), "act": act}
        g_in = np.tile(bundle.gfeat, (1, 1))
        Hv0 = np.maximum(Xv @ p["Wv_in"] + (g_in @ p["Wg_in"]), 0)
        He0 = np.maximum(Xe @ p["We_in"], 0)
        Hf = np.maximum(Xf @ p["Wf_in"], 0) if Xf is not None else None
        C["Hv0"], C["He0"], C["Hf0"] = Hv0, He0, Hf

        # GIN-eps co-ownership edge filter on the pre-injection embedding
        # (Leventhal 2025; Hofer et al. 2020)
        Zg = ((1 + p["eps_gin"][0]) * Hv0 + bundle.Asum @ Hv0) @ p["Wgin"] \
            + p["bgin"]
        Gh = np.maximum(Zg, 0)
        eu, ev = bundle.eu, bundle.ev
        xsum = Gh[eu] + Gh[ev]
        xdif = Gh[eu] - Gh[ev]
        Xef = np.concatenate([xsum, np.abs(xdif)], axis=1)
        Zf1 = Xef @ p["Wf1"] + p["bf1"]
        Hf1 = np.maximum(Zf1, 0)
        tlog = Hf1 @ p["wf2"] + p["bf2"][0]
        phi = 1.0 / (1.0 + np.exp(-tlog))
        C.update(Zg=Zg, Gh=Gh, eu=eu, ev=ev, xdif=xdif, Zf1=Zf1, Hf1=Hf1,
                 tlog=tlog, phi=phi)

        # persistence pairs of the phi-filtration (structure: stop-grad),
        # injected additively so the hierarchy carries every connection's
        # topological lifetime, not just its level membership
        if frozen is not None and "pers" in frozen:
            PersV, PersE = frozen["pers"]
        else:
            PersV, PersE = h0_pairs(bundle, phi)
        C["PersV"], C["PersE"] = PersV, PersE
        Hv = Hv0 + PersV @ p["Wpv"]
        He = He0 + PersE @ p["Wpe"]

        # learned quantile thresholds (Hofer-style differentiable
        # filtration): hard keeps are structure (stop-grad, hard-count row
        # normalization); a soft gate sigma((phi - p_l)/tau) multiplies each
        # kept edge, carrying gradient to phi, to the thresholds p_l, and
        # through the interpolated empirical quantile to the learned
        # fractions qraw — so the levels slide to track the position's
        # contact structure.
        st = frozen["struct"] if frozen is not None and "struct" in frozen \
            else None
        # structural decisions (ordering, keeps) use quantized phi so exact
        # ties on symmetric positions stay ties; gates and thresholds stay
        # live on the unquantized values
        phir = np.round(phi, 9)
        qord = st["qord"] if st is not None else (
            np.argsort(phir, kind="stable") if ne else np.array([], int))
        levels, pl = [], []
        for i in range(act):
            if i == K - 1 or ne < 2:
                ki = st["keeps"][i] if st is not None else np.arange(ne)
                lv = {"ki": ki, "gate": np.ones(len(ki)), "p": None}
            else:
                q = 1.0 / (1.0 + math.exp(-p["qraw"][i]))
                pos = q * (ne - 1)
                if st is not None:
                    i0, i1 = st["qi"][i]
                    ki = st["keeps"][i]
                else:
                    i0 = min(int(pos), ne - 2)
                    i1 = i0 + 1
                    ki = None
                w = pos - i0
                j0, j1 = qord[i0], qord[i1]
                pv = (1 - w) * phi[j0] + w * phi[j1]
                if ki is None:
                    pvq = np.round((1 - w) * phir[j0] + w * phir[j1], 9)
                    ki = np.where(phir >= pvq)[0]
                gate = 1.0 / (1.0 + np.exp(-(phi[ki] - pv) / GATE_TAU))
                lv = {"ki": ki, "gate": gate, "p": float(pv), "i0": i0,
                      "i1": i1, "w": w, "q": q}
                pl.append(float(pv))
            Ac, As, Ae, dn = self._gated_mats(bundle, lv["ki"], lv["gate"])
            lv.update(Ac=Ac, As=As, Ae=Ae, dn=dn)
            levels.append(lv)
        C["levels"], C["pl"], C["qord"] = levels, pl, qord

        C["Ls"] = []
        C["msc"] = None
        C["flow"] = frozen["flow"] if frozen is not None and "flow" in frozen \
            else None
        for l in range(L):
            W0, A0c, Q0 = cayley(p[f"T0{l}"])
            W1, A1c, Q1 = cayley(p[f"T1{l}"])
            gmean = Hv.mean(axis=0, keepdims=True)
            ms, Rots, AeHs = [], [], []
            for lv in levels:
                Rot = rotagg(lv["Ac"], lv["As"], Hv)
                AeH = lv["Ae"] @ Hv
                ms.append(Rot @ W0 + AeH @ W1)
                Rots.append(Rot)
                AeHs.append(AeH)
            a = p[f"a{l}"]
            S_att = np.stack([m @ a[i] for i, m in enumerate(ms)], axis=1)
            S_att = S_att - S_att.max(axis=1, keepdims=True)
            expS = np.exp(S_att)
            alpha = expS / expS.sum(axis=1, keepdims=True)      # (n, act)
            M = sum(alpha[:, i:i + 1] * ms[i] for i in range(len(ms)))
            EH = bundle.M_ev @ He
            Zv = (Hv @ p[f"Ws{l}"] + M + EH @ p[f"Wev{l}"]
                  + gmean @ p[f"Wgl{l}"] + p[f"bv{l}"])
            UH = DH = XH = None
            if l >= 1 and C["flow"] is not None:
                # Morse-flow directed propagation: route messages along the
                # ownership field's ascending/descending manifolds, with the
                # separatrix channel carrying contested boundaries
                Aup, Adn, Alat = C["flow"]
                UH, DH, XH = Aup @ Hv, Adn @ Hv, Alat @ Hv
                Zv = (Zv + UH @ p[f"Wup{l}"] + DH @ p[f"Wdn{l}"]
                      + XH @ p[f"Wlat{l}"])
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
                            "EHf": EHf, "Zf": Zf, "Rots": Rots, "AeHs": AeHs,
                            "UH": UH, "DH": DH, "XH": XH,
                            "W0": W0, "A0c": A0c, "Q0": Q0,
                            "W1": W1, "A1c": A1c, "Q1": Q1})
            Hv, He = np.maximum(Zv, 0), np.maximum(Ze, 0)
            Hf = np.maximum(Zf, 0) if Zf is not None else None
            if l == 0:
                # ownership at mid-depth: supervised, the Morse function
                # whose interlevel components define the pooling regions,
                # and the flow field that routes propagation above
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
                if C["flow"] is None:
                    C["flow"] = self._flow_mats(bundle, own_mid)

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

    @staticmethod
    def _gate_grad(bundle, lv, Hv, dRot, G1):
        """dL/d(gate_e) for every kept edge of a level, given dL/dRot and
        G1 = dm @ W1.T at one layer. Vectorized over the kept edges: the
        rotation blocks R(c, s) into u and R(c, -s) into v, each scaled by
        1/deg, plus the eps path."""
        ki = lv["ki"]
        u, v = bundle.eu[ki], bundle.ev[ki]
        c, s, e = bundle.ecos[ki], bundle.esin[ki], bundle.eeps[ki]
        dn = lv["dn"]
        D = Hv.shape[1]
        De = D - (D % 2)
        Hve, Hvo = Hv[:, 0:De:2], Hv[:, 1:De:2]
        dRe, dRo = dRot[:, 0:De:2], dRot[:, 1:De:2]
        Rvb_e = c[:, None] * Hve[v] - s[:, None] * Hvo[v]
        Rvb_o = s[:, None] * Hve[v] + c[:, None] * Hvo[v]
        dot_a = (dRe[u] * Rvb_e).sum(1) + (dRo[u] * Rvb_o).sum(1)
        Rua_e = c[:, None] * Hve[u] + s[:, None] * Hvo[u]
        Rua_o = -s[:, None] * Hve[u] + c[:, None] * Hvo[u]
        dot_b = (dRe[v] * Rua_e).sum(1) + (dRo[v] * Rua_o).sum(1)
        if D % 2:
            dot_a += c * dRot[u, -1] * Hv[v, -1]
            dot_b += c * dRot[v, -1] * Hv[u, -1]
        return (dot_a / dn[u] + dot_b / dn[v]
                + e * ((G1[u] * Hv[v]).sum(1) / dn[u]
                       + (G1[v] * Hv[u]).sum(1) / dn[v]))

    def backward(self, C, pi, z_t, own_t, grads, eown_t=None,
                 vw=1.0, ow=0.5, mw=0.25, fw=0.25):
        p, b = self.p, C["b"]
        n, ne = b.n, b.ne
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

        # gate/quantile gradients into phi accumulate across the layer loop;
        # the edge-filter head chain runs after it, combining them with the
        # co-ownership supervision term
        dphi = np.zeros(ne)
        dp_acc = np.zeros(max(self.K - 1, 1))
        levels = C["levels"]

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
            dW0 = np.zeros((self.hidden, self.hidden))
            dW1 = np.zeros((self.hidden, self.hidden))
            dHv_new = (dZv @ p[f"Ws{l}"].T
                       + b.M_ve.T @ (dZe @ p[f"Wve{l}"].T)
                       + np.ones((n, 1)) @ ((dZv @ p[f"Wgl{l}"].T)
                                            .sum(axis=0, keepdims=True)) / n)
            if l >= 1 and Lc["UH"] is not None:
                Aup, Adn, Alat = C["flow"]
                grads[f"Wup{l}"] += Lc["UH"].T @ dZv
                grads[f"Wdn{l}"] += Lc["DH"].T @ dZv
                grads[f"Wlat{l}"] += Lc["XH"].T @ dZv
                dHv_new += (Aup.T @ (dZv @ p[f"Wup{l}"].T)
                            + Adn.T @ (dZv @ p[f"Wdn{l}"].T)
                            + Alat.T @ (dZv @ p[f"Wlat{l}"].T))
            for i, lv in enumerate(levels):
                ds = alpha[:, i] * ((GM * ms[i]).sum(axis=1) - gm_dot_M)
                dm = alpha[:, i:i + 1] * GM + np.outer(ds, a[i])
                grads[f"a{l}"][i] += ms[i].T @ ds
                dW0 += Lc["Rots"][i].T @ dm
                dW1 += Lc["AeHs"][i].T @ dm
                dRot = dm @ Lc["W0"].T
                G1 = dm @ Lc["W1"].T
                dHv_new += rotagg_T(lv["Ac"], lv["As"], dRot) \
                    + lv["Ae"].T @ G1
                if lv["p"] is not None and len(lv["ki"]):
                    dg = self._gate_grad(b, lv, Lc["Hv"], dRot, G1)
                    contrib = dg * lv["gate"] * (1 - lv["gate"]) / GATE_TAU
                    dphi[lv["ki"]] += contrib
                    dp_acc[i] -= contrib.sum()
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

        # learned quantile thresholds backward:
        # p_i = (1 - w) phi[j0] + w phi[j1], w = sigmoid(qraw_i)(ne-1) - i0
        for i, lv in enumerate(levels):
            if lv.get("p") is None or "q" not in lv or dp_acc[i] == 0:
                continue
            j0, j1 = C["qord"][lv["i0"]], C["qord"][lv["i1"]]
            dphi[j0] += dp_acc[i] * (1 - lv["w"])
            dphi[j1] += dp_acc[i] * lv["w"]
            q = lv["q"]
            grads["qraw"][i] += (dp_acc[i] * (C["phi"][j1] - C["phi"][j0])
                                 * (ne - 1) * q * (1 - q))

        # edge co-ownership filter head: the supervision term (gradient
        # reaches the filter head here) combines with the differentiable
        # filtration gradients; keeps/pairings themselves are stop-grad
        # structure
        dGh = np.zeros_like(C["Gh"])
        dtlog = np.zeros(ne)
        if eown_t is not None and ne:
            loss += fw * float(np.mean(
                -eown_t * np.log(C["phi"] + 1e-12)
                - (1 - eown_t) * np.log(1 - C["phi"] + 1e-12)))
            dtlog += fw * (C["phi"] - eown_t) / ne
        dtlog += dphi * C["phi"] * (1 - C["phi"])
        if ne and np.any(dtlog):
            grads["wf2"] += C["Hf1"].T @ dtlog
            grads["bf2"][0] += dtlog.sum()
            dHf1 = np.outer(dtlog, p["wf2"]) * (C["Zf1"] > 0)
            grads["Wf1"] += np.concatenate(
                [C["Gh"][C["eu"]] + C["Gh"][C["ev"]],
                 np.abs(C["xdif"])], axis=1).T @ dHf1
            grads["bf1"] += dHf1.sum(axis=0)
            dXef = dHf1 @ p["Wf1"].T
            D = self.hidden
            dsum, ddif = dXef[:, :D], dXef[:, D:] * np.sign(C["xdif"])
            np.add.at(dGh, C["eu"], dsum + ddif)
            np.add.at(dGh, C["ev"], dsum - ddif)

        # persistence injection weights (pairs are stop-grad structure)
        grads["Wpv"] += C["PersV"].T @ dHv
        grads["Wpe"] += C["PersE"].T @ dHe

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
        missing = [k for k in net.p if k not in d.files]
        for k in net.p:
            if k in d.files:
                net.p[k] = d[k]
        if missing:
            print(f"hatz.load: {path} predates v3; keeping fresh init for "
                  f"{len(missing)} new parameter(s): {sorted(missing)[:6]}"
                  + ("..." if len(missing) > 6 else ""))
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
