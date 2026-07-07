"""zeronet.py — a small policy+value graph network in plain numpy.

Architecture (graph-agnostic, so one net plays every surface/mesh):
  X[n,F]  per-vertex features: own stone, opponent stone, empty,
          degree / max degree, bias 1
  H       relu(X W0 + b0), then L rounds of
          H <- relu(H Ws_l + (A H) Wn_l + b_l)   (A row-normalized adjacency)
  policy  per-vertex logit H wp + bp, plus a pass logit from the mean
  value   tanh of a linear readout of the mean embedding

Loss = cross-entropy(pi, softmax(masked logits)) + (z - v)^2.
Backprop is written by hand (verified against numerical gradients in
test_train.py), so training needs nothing beyond numpy.
"""

from __future__ import annotations

import json

import numpy as np


def features(adj, stones, to_move):
    n = len(adj)
    X = np.zeros((n, 5))
    deg = np.array([len(a) for a in adj], dtype=float)
    md = deg.max() if n else 1.0
    for v in range(n):
        c = stones[v]
        X[v, 0] = 1.0 if c == to_move else 0.0
        X[v, 1] = 1.0 if c and c != to_move else 0.0
        X[v, 2] = 1.0 if c == 0 else 0.0
        X[v, 3] = deg[v] / max(md, 1.0)
        X[v, 4] = 1.0
    return X


def mean_matrix(adj):
    n = len(adj)
    A = np.zeros((n, n))
    for v in range(n):
        if adj[v]:
            A[v, list(adj[v])] = 1.0 / len(adj[v])
    return A


class ZeroNet:
    F = 5

    def __init__(self, hidden=32, layers=3, seed=0):
        rng = np.random.default_rng(seed)
        D = hidden

        def init(a, b):
            return rng.standard_normal((a, b)) * np.sqrt(2.0 / a)

        self.p = {"W0": init(self.F, D), "b0": np.zeros(D),
                  "wp": rng.standard_normal(D) * 0.05, "bp": np.zeros(1),
                  "wq": rng.standard_normal(D) * 0.05, "bq": np.zeros(1),
                  "wv": rng.standard_normal(D) * 0.05, "bv": np.zeros(1)}
        for l in range(layers):
            self.p[f"Ws{l}"] = init(D, D)
            self.p[f"Wn{l}"] = init(D, D)
            self.p[f"b{l+1}"] = np.zeros(D)
        self.layers = layers
        self.hidden = D
        self.meta = {}
        self._adam = None

    # ---- forward -----------------------------------------------------------

    def forward(self, A, X, mask):
        """mask: length n+1 (pass last), 1 where playable. Returns
        (p, v, cache): p over n+1, v scalar."""
        p = self.p
        cache = {"A": A, "X": X, "Hs": [], "Zs": []}
        Z = X @ p["W0"] + p["b0"]
        H = np.maximum(Z, 0)
        cache["Z0"], cache["H0"] = Z, H
        for l in range(self.layers):
            AH = A @ H
            Z = H @ p[f"Ws{l}"] + AH @ p[f"Wn{l}"] + p[f"b{l+1}"]
            Hn = np.maximum(Z, 0)
            cache["Zs"].append(Z)
            cache["Hs"].append((H, AH))
            H = Hn
        g = H.mean(axis=0)
        node_logits = H @ p["wp"] + p["bp"][0]
        pass_logit = g @ p["wq"] + p["bq"][0]
        logits = np.concatenate([node_logits, [pass_logit]])
        neg = np.where(np.asarray(mask) > 0, 0.0, -1e9)
        z = logits + neg
        z = z - z.max()
        e = np.exp(z)
        probs = e / e.sum()
        u = g @ p["wv"] + p["bv"][0]
        v = np.tanh(u)
        cache.update(H=H, g=g, probs=probs, u=u, v=v)
        return probs, float(v), cache

    # ---- backward -----------------------------------------------------------

    def backward(self, cache, pi, z_target, grads, value_weight=1.0):
        """Accumulate parameter gradients for one sample into `grads`.
        Loss = -pi . log p + value_weight (z - v)^2. Returns the loss."""
        p = self.p
        probs, v, u = cache["probs"], cache["v"], cache["u"]
        H, g, A = cache["H"], cache["g"], cache["A"]
        n = H.shape[0]
        loss = -float(np.sum(pi * np.log(probs + 1e-12))) \
            + value_weight * (z_target - v) ** 2

        dlogits = probs - pi                       # masked entries: 0 - 0
        dnode, dpass = dlogits[:n], dlogits[n]
        du = value_weight * 2 * (v - z_target) * (1 - v * v)

        grads["wp"] += H.T @ dnode
        grads["bp"][0] += dnode.sum()
        grads["wq"] += g * dpass
        grads["bq"][0] += dpass
        grads["wv"] += g * du
        grads["bv"][0] += du

        dg = dpass * p["wq"] + du * p["wv"]
        dH = np.outer(dnode, p["wp"]) + dg[None, :] / n

        for l in reversed(range(self.layers)):
            Z = cache["Zs"][l]
            Hprev, AH = cache["Hs"][l]
            dZ = dH * (Z > 0)
            grads[f"Ws{l}"] += Hprev.T @ dZ
            grads[f"Wn{l}"] += AH.T @ dZ
            grads[f"b{l+1}"] += dZ.sum(axis=0)
            dH = dZ @ p[f"Ws{l}"].T + A.T @ (dZ @ p[f"Wn{l}"].T)
        dZ0 = dH * (cache["Z0"] > 0)
        grads["W0"] += cache["X"].T @ dZ0
        grads["b0"] += dZ0.sum(axis=0)
        return loss

    def zero_grads(self):
        return {k: np.zeros_like(v) for k, v in self.p.items()}

    # ---- optimizer ------------------------------------------------------------

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

    # ---- persistence ------------------------------------------------------------

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
