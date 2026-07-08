#!/usr/bin/env python3
"""HATZ v1 tests:  python3 python/train/test_hatz.py
Covers the orientation cocycle across every board type, full numerical
gradient checks through the seam transport / Morse-Smale pooling / ownership
head / V-E-F coupling, the exact-equivariance claims (orbit constancy on
orientable boards, seam-preserving automorphisms on Mobius), gauge-transform
invariants, ownership targets, a one-iteration training run, and the bridge
bot on a remote Mobius board.
"""

import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(os.path.dirname(HERE))

from env import SPECS, board_for, new_game                      # noqa: E402
from hatz import HATZ, Bundle, final_ownership                  # noqa: E402
from mcts import MCTS                                           # noqa: E402

passed = failed = 0


def ok(cond, name):
    global passed, failed
    print(("PASS  " if cond else "FAIL  ") + name)
    passed, failed = passed + bool(cond), failed + (not cond)


# ---- orientation cocycle -----------------------------------------------------

expect = {"sphere": False, "plane": False, "cylinder": False, "torus": False,
          "mobius": True, "klein": True, "rp2": True, "box3": False,
          "sphere-h": False, "torus-h": False, "mobius-h": True}
bundles = {k: Bundle(board_for(k), SPECS[k][0]) for k in expect}
ok(all(bundles[k].nonorientable == w for k, w in expect.items()),
   f"cocycle: w1 detection matches known orientability on {len(expect)} "
   "board types (incl. hex Mobius and 3D)")

# ---- gradient checks ----------------------------------------------------------


def gradcheck(bundle):
    n = bundle.n
    net = HATZ(hidden=6, layers=2, seed=3)
    rng = np.random.default_rng(0)
    stones = [int(rng.integers(0, 3)) for _ in range(n)]
    mask = np.ones(n + 1)
    for v in range(n):
        if stones[v]:
            mask[v] = 0
    pi = rng.random(n + 1) * mask
    pi /= pi.sum()
    z_t, own_t = 0.4, rng.uniform(-1, 1, n)
    eown_t = (rng.random(len(bundle.edges)) > 0.5).astype(float)
    _, _, _, C0 = net.forward(bundle, stones, 1, mask)
    frozen = {"mats": C0["mats"],
              "msc": (C0["msc"]["S"], C0["msc"]["P"], C0["msc"]["Ar"])}
    grads = net.zero_grads()
    _, _, _, C = net.forward(bundle, stones, 1, mask, frozen=frozen)
    net.backward(C, pi, z_t, own_t, grads, eown_t=eown_t)

    def loss_at():
        _, _, _, c = net.forward(bundle, stones, 1, mask, frozen=frozen)
        p_, v, o = c["probs"], c["value"], c["own"]
        om, phi = c["msc"]["own_mid"], c["phi"]
        return (-float(np.sum(pi * np.log(p_ + 1e-12))) + (z_t - v) ** 2
                + 0.5 * float(np.mean((o - own_t) ** 2))
                + 0.25 * float(np.mean((om - own_t) ** 2))
                + 0.25 * float(np.mean(
                    -eown_t * np.log(phi + 1e-12)
                    - (1 - eown_t) * np.log(1 - phi + 1e-12))))

    h, worst = 1e-5, 0.0
    for k, w in net.p.items():
        it = np.nditer(w, flags=["multi_index"])
        for _ in it:
            i = it.multi_index
            old = w[i]
            w[i] = old + h
            lp = loss_at()
            w[i] = old - h
            lm = loss_at()
            w[i] = old
            num = (lp - lm) / (2 * h)
            rel = abs(num - grads[k][i]) / max(1e-6,
                                               abs(num) + abs(grads[k][i]))
            worst = max(worst, rel)
    return worst


w1 = gradcheck(bundles["mobius"])
ok(w1 < 1e-4, "gradcheck mobius (Cayley transport + filtration attention + "
   f"GIN filter + ownership-driven MSC): worst rel err {w1:.1e}")
from hatz import cayley                                          # noqa: E402
_netO = HATZ(hidden=8, layers=2, seed=1)
_W, _, _ = cayley(_netO.p["T00"])
ok(float(np.abs(_W.T @ _W - np.eye(8)).max()) < 1e-12,
   "Cayley eps-transport maps are orthogonal to machine precision")
from geodesics.board import Board                               # noqa: E402
tetra = Board(name="tetra", params={},
              adj=((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)),
              coords=((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)),
              faces=((0, 1, 2), (0, 3, 1), (1, 3, 2), (0, 2, 3)))
bt = Bundle(tetra)
w2 = gradcheck(bt)
ok(w2 < 1e-4 and abs(bt.curv[0] - np.pi) < 1e-9,
   "gradcheck tetrahedron (V-E-F + all v2 paths; Gauss-Bonnet defect = pi): "
   f"worst rel err {w2:.1e}")

# ---- equivariance claims -------------------------------------------------------

net = HATZ(hidden=16, layers=2, seed=2)
claims = []
for key in ("sphere", "torus"):
    b = bundles[key]
    probs, _, _, _ = net.forward(b, [0] * b.n, 1, np.ones(b.n + 1))
    lg = np.log(probs[:b.n] + 1e-15)
    orbs = {}
    for v in range(b.n):
        orbs.setdefault(b.orbits[v], []).append(lg[v])
    claims.append(max(max(g) - min(g) for g in orbs.values()) < 1e-9)
ok(all(claims), "equivariance: empty-board policy exactly constant on WL "
   "orbits of orientable boards (sphere: 2 orbits, torus: 1)")

b = bundles["mobius"]
probs, _, _, _ = net.forward(b, [0] * 25, 1, np.ones(26))
lg = np.log(probs[:25] + 1e-15)
perm = [(4 - (v // 5)) * 5 + (v % 5) for v in range(25)]
ok(float(np.abs(lg - lg[perm]).max()) < 1e-12,
   "equivariance: exact under the seam-preserving Mobius reflection "
   "(the gauge-dependent part is trained away by gauge augmentation)")

g = b.with_gauge(np.random.default_rng(7))
holo_ok = g.nonorientable and any(g.eps[v][u] != b.eps[v][u]
                                  for v in range(b.n) for u in b.adj[v])
ok(holo_ok, "gauge transform: seam moves but holonomy class (w1) is invariant")

# ---- ownership targets ---------------------------------------------------------

gp = new_game("plane")
for mv in [12, 0]:
    gp.play(mv)
own = final_ownership(gp.board, gp.colors)
ok(own[12] == 1.0 and own[0] == -1.0 and own[13] == 0.0,
   "ownership: stones owned by their color, contested empties neutral")

# ---- MCTS integration ----------------------------------------------------------

gk = new_game("klein")
tree = MCTS(HATZ(hidden=8, layers=2, seed=1), gk.board, bundles["klein"],
            seed=2)
counts = tree.run(gk, 24)
a = int(np.argmax(counts))
ok(counts.sum() == 24 and (a == gk.board.n or gk.is_legal(a)),
   "mcts: searches with the HATZ evaluator on a Klein bottle; argmax legal")

# ---- trainer end-to-end --------------------------------------------------------

ck = "/tmp/hatz_smoke.npz"
if os.path.isfile(ck):
    os.remove(ck)
r = subprocess.run(
    [sys.executable, os.path.join(HERE, "train_hatz.py"),
     "--specs", "mobius,klein", "--iters", "1", "--games-per-iter", "2",
     "--sims", "10", "--steps-per-iter", "6", "--batch", "8",
     "--eval-every", "1", "--seed", "5", "--checkpoint", ck],
    capture_output=True, text=True)
tr_ok = r.returncode == 0 and os.path.isfile(ck)
if tr_ok:
    net2 = HATZ.load(ck)
    tr_ok = net2.meta.get("arch") == "hatz" \
        and net2.meta.get("specs") == ["mobius", "klein"]
else:
    print(r.stdout[-500:], r.stderr[-500:])
ok(tr_ok, "train_hatz: one self-play iteration on Mobius+Klein trains, "
   "evaluates, and checkpoints with arch metadata")

# ---- bridge bot ----------------------------------------------------------------

os.environ["HATZ_CKPT"] = ck
sys.path.insert(0, os.path.join(ROOT, "bridge", "bots"))
import importlib                                                # noqa: E402
hm = importlib.import_module("hatz_mini")
importlib.reload(hm)
bot = hm.BOT
mb = board_for("mobius")
req = {"level": "standard",
       "spec": {"surface": "mobius", "mesh": "square", "nx": 5, "ny": 5},
       "board": {"n": 25, "neighbors": [list(a) for a in mb.adj],
                 "stones": [0] * 25, "toMove": 1,
                 "legalMask": [1] * 25, "moves": []}}
bot_ok = bot.available() and bot.supports["surfaces"] == ["klein", "mobius"]
seam_seen = False
for lvl in ("casual", "standard", "strong"):
    req["level"] = lvl
    mv, info = bot.genmove(req)
    bot_ok = bot_ok and (mv == -1 or 0 <= mv < 25)
    seam_seen = seam_seen or "seam active" in info
ok(bot_ok and seam_seen, "bridge bot: reconstructs the Mobius seam from the "
   "request, plays legally at all levels, reports non-orientability")

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
