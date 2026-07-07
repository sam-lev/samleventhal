"""hgnn.py — Python port of the browser engine (web/ai.js), op-for-op.

Same fixed-weight hierarchical graph network: higher-order (1- and 2-ring)
message passing joined across a multi-persistence Morse-Smale hierarchy of
the influence field (Leventhal, Gyulassy, Pascucci & Heimann, NeurIPS 2022),
a matching-coarsened structural pyramid, exact tactical features, and 1-ply
lookahead. Arrays are float32 to match the browser's Float32Array rounding,
so the two implementations agree move-for-move (verified in test_train.py by
running the actual ai.js under node).

The weight schema is identical to engine.getWeights()/setWeights() in the
app, which is the whole point: train here, export the JSON, drop it in
web/models/, and the browser plays it natively.
"""

from __future__ import annotations

import copy
import json
import math

import numpy as np

F32 = np.float32

DEFAULT_WEIGHTS = {
    "diffusion": {
        "blocks": [
            {"steps": 3, "self": 0.5, "n1": 0.36, "n2": 0.14},
            {"steps": 2, "self": 0.55, "n1": 0.33, "n2": 0.12},
        ],
    },
    "pyramid": {"maxLevels": 4, "minSize": 24, "steps": 3, "self": 0.5,
                "n1": 0.5, "mix": [1.0, 0.6, 0.38, 0.24]},
    "msc": {
        "enabled": True,
        "minVerts": 8,
        "minBasins": 3,
        "quantiles": [0, 0.6, 0.9],
        "mix": [0.4, 0.28, 0.18],
        "regionSmooth": 0.3,
        "smooth": {"steps": 2, "self": 0.5, "n1": 0.36, "n2": 0.14},
    },
    "sigma": 0.35,
    "head": {
        "capture": 6.0, "escape": 5.0, "atari": 1.2, "libs": 1.2,
        "frontier": 1.6, "grad": 1.1, "mscBoundary": 0.9, "expand": 0.8,
        "lowDeg": 0.35, "selfAtari": 5.0, "passThresh": 0.35,
        "valueGain": 2.2, "replyFear": 1.4, "lookBias": 0.15,
    },
    "select": {
        "casual":   {"topk": 0,  "temp": 1.1,  "noise": 0.3},
        "standard": {"topk": 8,  "temp": 0.45, "noise": 0.1},
        "strong":   {"topk": 14, "temp": 0.15, "noise": 0.03},
    },
}


def deep_merge(dst: dict, src: dict) -> dict:
    for k, v in src.items():
        if isinstance(v, dict) and not isinstance(v, list):
            if not isinstance(dst.get(k), dict):
                dst[k] = {}
            deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst


def mulberry32(seed: int):
    """Exact port of the browser PRNG so seeded play matches."""
    a = seed & 0xFFFFFFFF

    def rnd():
        nonlocal a
        a = (a + 0x6D2B79F5) & 0xFFFFFFFF
        t = a
        t = ((t ^ (t >> 15)) * ((t | 1) & 0xFFFFFFFF)) & 0xFFFFFFFF
        t = (t ^ (t + (((t ^ (t >> 7)) * ((t | 61) & 0xFFFFFFFF)) & 0xFFFFFFFF))) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296.0

    return rnd


# ---------- graph structure ---------------------------------------------------

def clean_adjacency(neighbors):
    n = len(neighbors)
    out = []
    for v in range(n):
        seen, lst = set(), []
        for u in (neighbors[v] or []):
            u = int(u)
            if u != v and 0 <= u < n and u not in seen:
                seen.add(u)
                lst.append(u)
        out.append(lst)
    return out


def build_rings(N1):
    n = len(N1)
    N2 = []
    mark = [-1] * n
    for v in range(n):
        mark[v] = v
        for u in N1[v]:
            mark[u] = v
        ring = []
        for u in N1[v]:
            for w in N1[u]:
                if mark[w] != v:
                    mark[w] = v
                    ring.append(w)
        N2.append(ring)
    return N2


def coarsen(N1):
    n = len(N1)
    assign = [-1] * n
    c = 0
    for v in range(n):
        if assign[v] != -1:
            continue
        mate = -1
        for u in N1[v]:
            if assign[u] == -1:
                mate = u
                break
        assign[v] = c
        if mate != -1:
            assign[mate] = c
        c += 1
    # insertion order matters: the next level's greedy matching reads
    # "first unmatched neighbor", exactly like the JS Set iteration order
    lists = [[] for _ in range(c)]
    members = [set() for _ in range(c)]
    for v in range(n):
        for u in N1[v]:
            a, b = assign[v], assign[u]
            if a != b and b not in members[a]:
                members[a].add(b)
                lists[a].append(b)
    return assign, lists, c


def build_pyramid(N1, max_levels, min_size):
    levels = [{"N1": N1, "n": len(N1)}]
    while len(levels) < max_levels and levels[-1]["n"] > min_size:
        assign, cN1, size = coarsen(levels[-1]["N1"])
        if size >= levels[-1]["n"]:
            break
        levels[-1]["assign"] = assign
        levels.append({"N1": cN1, "n": size})
    map_to = [None]
    prev = None
    for l in range(1, len(levels)):
        a = levels[l - 1]["assign"]
        m = [a[prev[v]] if prev else a[v] for v in range(levels[0]["n"])]
        map_to.append(m)
        prev = m
    return levels, map_to


# ---------- multi-persistence Morse-Smale hierarchy ----------------------------

def basin_hierarchy(f, N1):
    n = len(f)

    def higher(a, b):
        return f[a] > f[b] or (f[a] == f[b] and a > b)

    up = [-1] * n
    for v in range(n):
        best = v
        for u in N1[v]:
            if higher(u, best):
                best = u
        if best != v:
            up[v] = best
    basin = [-1] * n
    for v in range(n):
        x, path = v, []
        while basin[x] == -1 and up[x] != -1:
            path.append(x)
            x = up[x]
        m = x if basin[x] == -1 else basin[x]
        basin[x] = m
        for p in path:
            basin[p] = m
    order = sorted(range(n), key=lambda v: (-float(f[v]), -v))
    parent = [-1] * n
    cmax = [-1] * n
    pers = [math.inf] * n
    merged_into = [-1] * n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    maxima = []
    for v in order:
        parent[v] = v
        cmax[v] = v
        if up[v] == -1:
            maxima.append(v)
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
    finite_pers = sorted(pers[m] for m in maxima if pers[m] < math.inf)

    def labels_at(p):
        lab = [0] * n
        for v in range(n):
            m = basin[v]
            while merged_into[m] != -1 and pers[m] < p:
                m = merged_into[m]
            lab[v] = m
        return lab

    return {"basin": basin, "up": up, "maxima": maxima,
            "finitePers": finite_pers, "labelsAt": labels_at}


# ---------- message passing -----------------------------------------------------

def _mean_op(N1, n):
    """Row-normalized sparse neighbor-mean as a dense f64 matrix (boards are
    small); zero rows for isolated vertices, matching the JS mean-of-empty=0."""
    A = np.zeros((n, n), dtype=np.float64)
    for v in range(n):
        if N1[v]:
            w = 1.0 / len(N1[v])
            for u in N1[v]:
                A[v, u] = w
    return A


def diffuse(h, M1, M2, steps, w_self, w1, w2):
    a = np.asarray(h, dtype=F32).copy()
    for _ in range(steps):
        b = w_self * a.astype(np.float64) + w1 * (M1 @ a.astype(np.float64))
        if M2 is not None and w2 != 0:
            b = b + w2 * (M2 @ a.astype(np.float64))
        a = b.astype(F32)
    return a


# ---------- rules kernel ---------------------------------------------------------

class Kernel:
    def __init__(self, N1):
        self.N1 = N1
        self.n = len(N1)

    def dead_group(self, s, start, color):
        st, seen, g = [start], {start}, []
        while st:
            x = st.pop()
            g.append(x)
            for u in self.N1[x]:
                if s[u] == 0:
                    return None
                if s[u] == color and u not in seen:
                    seen.add(u)
                    st.append(u)
        return g

    def apply_move(self, stones, v, color):
        if stones[v] != 0:
            return None
        s = list(stones)
        s[v] = color
        opp = 3 - color
        captured = []
        for u in self.N1[v]:
            if s[u] == opp:
                g = self.dead_group(s, u, opp)
                if g:
                    for x in g:
                        s[x] = 0
                    captured.extend(g)
        if self.dead_group(s, v, color):
            return None
        return s, captured

    def analyze(self, stones):
        n = self.n
        gid = [-1] * n
        groups = []
        for v in range(n):
            if stones[v] == 0 or gid[v] != -1:
                continue
            color = stones[v]
            gi = len(groups)
            g = {"color": color, "size": 0, "libs": set()}
            groups.append(g)
            st = [v]
            gid[v] = gi
            while st:
                x = st.pop()
                g["size"] += 1
                for u in self.N1[x]:
                    if stones[u] == 0:
                        g["libs"].add(u)
                    elif stones[u] == color and gid[u] == -1:
                        gid[u] = gi
                        st.append(u)
        return {"gid": gid, "groups": groups}


# ---------- the engine ------------------------------------------------------------

class Engine:
    def __init__(self, neighbors, level="standard", seed=0x9E3779B9,
                 weights=None, select=None):
        self.N1 = clean_adjacency(neighbors)
        self.n = len(self.N1)
        self.N2 = build_rings(self.N1)
        self.weights = deep_merge(copy.deepcopy(DEFAULT_WEIGHTS), weights or {})
        self.levels, self.map_to = build_pyramid(
            self.N1, self.weights["pyramid"]["maxLevels"],
            self.weights["pyramid"]["minSize"])
        self.kernel = Kernel(self.N1)
        self.rng = mulberry32(seed)
        self.level = level
        self.select_override = select
        self.deg = [len(a) for a in self.N1]
        self.median_deg = sorted(self.deg)[self.n >> 1] if self.n else 0
        self.M1 = _mean_op(self.N1, self.n)
        self.M2 = _mean_op(self.N2, self.n)
        self.MP = [None] + [_mean_op(l["N1"], l["n"]) for l in self.levels[1:]]

    # -- weights ------------------------------------------------------------

    def get_weights(self):
        return copy.deepcopy(self.weights)

    def set_weights(self, w):
        deep_merge(self.weights, w or {})

    def _select_cfg(self):
        base = dict(self.weights["select"].get(
            self.level, self.weights["select"]["standard"]))
        if self.select_override:
            base.update(self.select_override)
        return base

    # -- MSC hierarchy --------------------------------------------------------

    def _make_regions(self, labels, f):
        id_of = {}
        rid = [0] * self.n
        for v in range(self.n):
            if labels[v] not in id_of:
                id_of[labels[v]] = len(id_of)
            rid[v] = id_of[labels[v]]
        R = len(id_of)
        adj_l = [[] for _ in range(R)]
        adj_m = [set() for _ in range(R)]
        for v in range(self.n):
            for u in self.N1[v]:
                a, b = rid[v], rid[u]
                if a != b and b not in adj_m[a]:
                    adj_m[a].add(b)
                    adj_l[a].append(b)
        peak = np.zeros(R, dtype=F32)
        for max_v, i in id_of.items():
            peak[i] = F32(f[max_v])
        return {"R": R, "rid": rid, "adj": adj_l, "peak": peak}

    def _build_msc(self, f0):
        C = self.weights["msc"]
        if not C["enabled"] or self.n < C["minVerts"]:
            return None
        asc = basin_hierarchy(f0, self.N1)
        neg = (-f0).astype(F32)
        desc = basin_hierarchy(neg, self.N1)
        if len(asc["maxima"]) + len(desc["maxima"]) < C["minBasins"]:
            return None
        pers_all = sorted(asc["finitePers"] + desc["finitePers"])

        def q(x):
            if not pers_all:
                return math.inf
            return pers_all[min(len(pers_all) - 1, int(x * len(pers_all)))]

        levels = []
        for qu in C["quantiles"]:
            p = 0 if qu <= 0 else q(qu)
            levels.append({
                "p": p,
                "asc": self._make_regions(asc["labelsAt"](p), f0),
                "desc": self._make_regions(desc["labelsAt"](p), neg),
            })
        L0 = levels[0]
        boundary = np.zeros(self.n, dtype=F32)
        for v in range(self.n):
            mine = float(L0["asc"]["peak"][L0["asc"]["rid"][v]])
            for u in self.N1[v]:
                r = L0["asc"]["rid"][u]
                if r != L0["asc"]["rid"][v] and \
                        (mine * float(L0["asc"]["peak"][r]) <= 0
                         or abs(float(L0["asc"]["peak"][r])) < 0.02):
                    boundary[v] = 1
                    break
        return {"levels": levels, "boundary": boundary}

    def _region_message(self, h, reg, smooth):
        R = reg["R"]
        sm = np.zeros(R, dtype=np.float64)
        cnt = np.zeros(R, dtype=np.float64)
        for v in range(self.n):
            sm[reg["rid"][v]] += float(h[v])
            cnt[reg["rid"][v]] += 1
        m = np.where(cnt > 0, sm / np.maximum(cnt, 1), 0.0).astype(F32)
        m2 = np.zeros(R, dtype=F32)
        for i in range(R):
            a = reg["adj"][i]
            if a:
                s = sum(float(m[j]) for j in a)
                m2[i] = F32((1 - smooth) * float(m[i]) + smooth * s / len(a))
            else:
                m2[i] = m[i]
        out = np.zeros(self.n, dtype=F32)
        for v in range(self.n):
            out[v] = m2[reg["rid"][v]]
        return out

    # -- feature network ------------------------------------------------------

    def _stone_field(self, stones, color, A):
        f = np.zeros(self.n, dtype=F32)
        for v in range(self.n):
            c = stones[v]
            if not c:
                continue
            g = A["groups"][A["gid"][v]]
            health = min(len(g["libs"]), 3) / 3
            f[v] = F32((1 if c == color else -1) * (0.4 + 0.6 * health))
        return f

    def _compute_fields(self, stones, color, A):
        w = self.weights
        base = self._stone_field(stones, color, A)
        S = w["msc"]["smooth"]
        f0 = diffuse(base, self.M1, self.M2, S["steps"], S["self"],
                     S["n1"], S["n2"])
        msc = self._build_msc(f0)

        fine = base
        for b in w["diffusion"]["blocks"]:
            fine = diffuse(fine, self.M1, self.M2, b["steps"], b["self"],
                           b["n1"], b["n2"])
            if msc:
                mixes = w["msc"]["mix"]
                for li, L in enumerate(msc["levels"]):
                    mw = mixes[min(li, len(mixes) - 1)]
                    if not mw:
                        continue
                    ma = self._region_message(fine, L["asc"],
                                              w["msc"]["regionSmooth"])
                    md = self._region_message(fine, L["desc"],
                                              w["msc"]["regionSmooth"])
                    fine = (fine.astype(np.float64)
                            + mw * 0.5 * (ma.astype(np.float64)
                                          + md.astype(np.float64))).astype(F32)
            fine = np.tanh(fine.astype(F32)).astype(F32)

        P = w["pyramid"]
        M = (P["mix"][0] * fine.astype(np.float64)).astype(F32)
        for l in range(1, len(self.levels)):
            mixw = P["mix"][min(l, len(P["mix"]) - 1)]
            mp = self.map_to[l]
            cn = self.levels[l]["n"]
            sm = np.zeros(cn, dtype=np.float64)
            cnt = np.zeros(cn, dtype=np.float64)
            for v in range(self.n):
                sm[mp[v]] += float(base[v])
                cnt[mp[v]] += 1
            cf = np.where(cnt > 0, sm / np.maximum(cnt, 1), 0.0).astype(F32)
            cf = diffuse(cf, self.MP[l], None, P["steps"], P["self"],
                         P["n1"], 0)
            cf = np.tanh(cf).astype(F32)
            M = (M.astype(np.float64)
                 + mixw * cf.astype(np.float64)[mp]).astype(F32)
        M = np.tanh(M).astype(F32)

        inv2s2 = 1 / (2 * w["sigma"] * w["sigma"])
        frontier = np.exp(-(M.astype(np.float64) ** 2) * inv2s2).astype(F32)
        grad = np.zeros(self.n, dtype=F32)
        for v in range(self.n):
            nb = self.N1[v]
            if nb:
                grad[v] = F32(sum(abs(float(M[v]) - float(M[u]))
                                  for u in nb) / len(nb))
        V = float(M.astype(np.float64).sum()) / max(1, self.n)
        return {"M": M, "frontier": frontier, "grad": grad, "V": V,
                "msc": msc}

    def _candidate_features(self, stones, color, A):
        feats = [None] * self.n
        for v in range(self.n):
            if stones[v] != 0:
                continue
            nb = self.N1[v]
            empty_n1 = 0
            own_g, opp_g = set(), set()
            for u in nb:
                c = stones[u]
                if c == 0:
                    empty_n1 += 1
                elif c == color:
                    own_g.add(A["gid"][u])
                else:
                    opp_g.add(A["gid"][u])
            captures = atari_threat = 0
            for gi in opp_g:
                g = A["groups"][gi]
                if len(g["libs"]) == 1:
                    captures += g["size"]
                elif len(g["libs"]) == 2:
                    atari_threat += g["size"]
            lib_set = {u for u in nb if stones[u] == 0}
            merged_size = 1
            own_in_atari = 0
            min_own_libs = math.inf
            for gi in own_g:
                g = A["groups"][gi]
                merged_size += g["size"]
                if len(g["libs"]) == 1:
                    own_in_atari += g["size"]
                min_own_libs = min(min_own_libs, len(g["libs"]))
                lib_set |= g["libs"]
            lib_set.discard(v)
            libs_after = len(lib_set)
            if captures > 0:
                libs_after = max(libs_after, min(captures, 2))
            if captures == 0 and libs_after == 0:
                continue
            self_atari = captures == 0 and libs_after == 1
            escape = own_in_atari if (own_in_atari > 0
                                      and (libs_after >= 2 or captures > 0)) \
                else 0
            eye_fill = (empty_n1 == 0 and not opp_g and captures == 0
                        and own_g and min_own_libs >= 2)
            e2 = sum(1 for u in self.N2[v] if stones[u] == 0)
            expansion = (empty_n1 + 0.5 * e2) / max(
                1, len(nb) + 0.5 * len(self.N2[v]))
            feats[v] = {"captures": captures, "atariThreat": atari_threat,
                        "libsAfter": libs_after, "selfAtari": self_atari,
                        "escape": escape, "eyeFill": eye_fill,
                        "expansion": expansion, "mergedSize": merged_size}
        return feats

    def _head_score(self, f, fields, v):
        W = self.weights["head"]
        s = 0.0
        s += W["capture"] * min(f["captures"], 10)
        s += W["escape"] * min(f["escape"], 10)
        s += W["atari"] * min(f["atariThreat"], 6)
        s += W["libs"] * min(f["libsAfter"], 6) / 6
        s += W["frontier"] * float(fields["frontier"][v])
        s += W["grad"] * min(float(fields["grad"][v]) * 4, 1)
        if fields["msc"]:
            s += W["mscBoundary"] * float(fields["msc"]["boundary"][v])
        s += W["expand"] * f["expansion"]
        s -= W["lowDeg"] * max(0, self.median_deg - self.deg[v])
        if f["selfAtari"]:
            s -= W["selfAtari"] * (1 + f["mergedSize"] / 4)
        return s

    def _quick_value(self, stones, color):
        A = self.kernel.analyze(stones)
        base = self._stone_field(stones, color, A)
        f = diffuse(base, self.M1, None, 2, 0.55, 0.45, 0)
        V = float(np.tanh(1.8 * f.astype(np.float64)).sum()) / max(1, self.n)
        if len(self.levels) > 1:
            mp = self.map_to[1]
            cn = self.levels[1]["n"]
            sm = np.zeros(cn, dtype=np.float64)
            cnt = np.zeros(cn, dtype=np.float64)
            for v in range(self.n):
                sm[mp[v]] += float(base[v])
                cnt[mp[v]] += 1
            cf = np.where(cnt > 0, sm / np.maximum(cnt, 1), 0.0).astype(F32)
            cf = diffuse(cf, self.MP[1], None, 3, 0.5, 0.5, 0)
            CV = float(np.tanh(2.2 * cf.astype(np.float64)).sum()) / cn
            V = 0.5 * V + 0.5 * CV
        return V

    def _reply_threat(self, stones, color):
        A = self.kernel.analyze(stones)
        t = sum(g["size"] for g in A["groups"]
                if g["color"] == color and len(g["libs"]) == 1)
        return min(t, 8) / 8

    def _evaluate_policy(self, stones, color, is_legal, noise):
        A = self.kernel.analyze(stones)
        fields = self._compute_fields(stones, color, A)
        feats = self._candidate_features(stones, color, A)
        lst = []
        saw_legal = False
        for v in range(self.n):
            if stones[v] != 0:
                continue
            if is_legal and not is_legal(v):
                continue
            f = feats[v]
            if not f:
                continue
            saw_legal = True
            s = -1e9 if f["eyeFill"] else \
                self._head_score(f, fields, v) + noise * (self.rng() - 0.5) * 2
            lst.append({"v": v, "s": s, "f": f})
        lst.sort(key=lambda c: -c["s"])
        return fields, lst, saw_legal

    def _softmax_sample(self, cands, temp):
        if len(cands) == 1 or temp <= 1e-4:
            return cands[0]["v"]
        m = cands[0]["s"]
        ps = [math.exp((c["s"] - m) / temp) for c in cands]
        r = self.rng() * sum(ps)
        for c, p in zip(cands, ps):
            r -= p
            if r <= 0:
                return c["v"]
        return cands[0]["v"]

    # -- public API -----------------------------------------------------------

    def pick_move(self, stones, color, legal_mask=None):
        stones = [int(x) for x in stones]
        is_legal = None
        if legal_mask is not None:
            if callable(legal_mask):
                is_legal = legal_mask
            else:
                is_legal = lambda v: bool(legal_mask[v])  # noqa: E731
        sel = self._select_cfg()
        fields, lst, saw_legal = self._evaluate_policy(
            stones, color, is_legal, sel["noise"])
        cands = [c for c in lst if c["s"] > -1e8]
        if not cands:
            return {"move": -1, "value": fields["V"],
                    "reason": "only-eye-fills" if saw_legal
                    else "no-legal-moves"}
        if cands[0]["s"] < self.weights["head"]["passThresh"] \
                and fields["V"] > 0.12:
            return {"move": -1, "value": fields["V"],
                    "reason": "nothing-gains"}
        ranked = cands
        if sel["topk"] > 0 and self.n > 2:
            K = min(sel["topk"], len(cands))
            base_q = self._quick_value(stones, color)
            evals = []
            for c in cands[:K]:
                sim = self.kernel.apply_move(stones, c["v"], color)
                if not sim:
                    continue
                dq = self._quick_value(sim[0], color) - base_q
                threat = self._reply_threat(sim[0], color)
                W = self.weights["head"]
                evals.append({"v": c["v"],
                              "s": W["valueGain"] * dq
                              - W["replyFear"] * threat
                              + W["lookBias"] * c["s"]})
            if evals:
                evals.sort(key=lambda e: -e["s"])
                ranked = evals
        move = self._softmax_sample(ranked, sel["temp"])
        return {"move": move, "value": fields["V"],
                "candidates": [{"v": c["v"], "score": c["s"]}
                               for c in ranked[:5]]}

    def evaluate(self, stones, color):
        stones = [int(x) for x in stones]
        A = self.kernel.analyze(stones)
        return self._compute_fields(stones, color, A)


# ---------- trainable-parameter vectorization -----------------------------------

# paths into the weights dict that self-play optimization may move
TRAINABLE = (
    [("head", k) for k in ("capture", "escape", "atari", "libs", "frontier",
                           "grad", "mscBoundary", "expand", "lowDeg",
                           "selfAtari", "passThresh", "valueGain",
                           "replyFear", "lookBias")]
    + [("sigma",)]
    + [("diffusion", "blocks", i, k)
       for i in (0, 1) for k in ("self", "n1", "n2")]
    + [("msc", "mix", i) for i in (0, 1, 2)]
    + [("msc", "regionSmooth")]
    + [("msc", "smooth", k) for k in ("self", "n1", "n2")]
    + [("pyramid", "mix", i) for i in (0, 1, 2, 3)]
    + [("pyramid", "self"), ("pyramid", "n1")]
)

_LOWER = {("sigma",): 0.05, ("head", "passThresh"): 0.01,
          ("msc", "regionSmooth"): 0.0}


def _get(w, path):
    x = w
    for p in path:
        x = x[p]
    return x


def _set(w, path, val):
    x = w
    for p in path[:-1]:
        x = x[p]
    x[path[-1]] = val


def flatten_weights(w):
    return np.array([float(_get(w, p)) for p in TRAINABLE], dtype=np.float64)


def unflatten_weights(vec, base=None):
    w = copy.deepcopy(base or DEFAULT_WEIGHTS)
    for p, val in zip(TRAINABLE, vec):
        _set(w, p, max(float(val), _LOWER.get(p, -1e9)))
    return w


def weights_json(vec_or_w):
    w = vec_or_w if isinstance(vec_or_w, dict) else unflatten_weights(vec_or_w)
    return json.dumps(w, indent=2)
