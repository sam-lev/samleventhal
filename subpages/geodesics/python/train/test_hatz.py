#!/usr/bin/env python3
"""HATZ v3 tests:  python3 python/train/test_hatz.py
Covers the orientation cocycle across every board type, full numerical
gradient checks through every v3 path (Cayley-geometric transport with
live sin channels, Hofer-style gated quantile levels, persistence
injection, Morse-flow routing, V-E-F coupling), the exact-equivariance
claims on flat boards (transport frames on curved boards are a gauge,
trained by augmentation), gauge-transform invariants, persistence-pair and
quantile-level sanity, curriculum masking, ownership targets, one-iteration
training runs (plain and --curriculum), and the bridge bot on a remote
Mobius board.
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


def gradcheck(bundle, hidden=6):
    n = bundle.n
    net = HATZ(hidden=hidden, layers=2, seed=3)
    rng = np.random.default_rng(0)
    # jitter every measure-zero degeneracy of the differentiable filtration:
    # the interpolated-quantile grid points (qraw), the geometric-identity
    # Cayley point (T0 = 0), and the GIN eps = 0 point, where complete
    # graphs make xdif sit exactly on the abs() kink
    net.p["qraw"] += rng.uniform(0.011, 0.017, size=net.p["qraw"].shape)
    net.p["eps_gin"][0] = 0.07
    for l in range(net.layers):
        net.p[f"T0{l}"] += rng.standard_normal((hidden, hidden)) * 0.01
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
    frozen = net.freeze_struct(C0)
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
ok(w1 < 1e-4, "gradcheck mobius (Cayley transport + gated quantile levels + "
   "persistence injection + Morse-flow routing + GIN filter + MSC): "
   f"worst rel err {w1:.1e}")
from hatz import cayley                                          # noqa: E402
_netO = HATZ(hidden=8, layers=2, seed=1)
_T = _netO.p["T00"] + np.random.default_rng(1).standard_normal((8, 8)) * 0.3
_W, _, _ = cayley(_T)
ok(float(np.abs(_W.T @ _W - np.eye(8)).max()) < 1e-12
   and float(np.abs(cayley(_netO.p["T00"])[0] - np.eye(8)).max()) < 1e-15,
   "Cayley transport maps are orthogonal to machine precision, and the raw "
   "init at 0 gives W0 = I exactly (the geometric transporter)")
from geodesics.board import Board                               # noqa: E402
tetra = Board(name="tetra", params={},
              adj=((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)),
              coords=((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)),
              faces=((0, 1, 2), (0, 3, 1), (1, 3, 2), (0, 2, 3)))
bt = Bundle(tetra)
w2 = gradcheck(bt)
ok(w2 < 1e-4 and abs(bt.curv[0] - np.pi) < 1e-9,
   "gradcheck tetrahedron (V-E-F + live sin transport; Gauss-Bonnet defect "
   f"= pi): worst rel err {w2:.1e}")
w2b = gradcheck(bt, hidden=5)
ok(w2b < 1e-4, "gradcheck tetrahedron with odd hidden width (the unpaired "
   f"transport channel): worst rel err {w2b:.1e}")
w2c = gradcheck(bundles["sphere"])
ok(w2c < 1e-4, "gradcheck geodesic sphere (dense transport angles through "
   f"the gated levels): worst rel err {w2c:.1e}")

# ---- equivariance claims -------------------------------------------------------

net = HATZ(hidden=16, layers=2, seed=2)
claims = []
for key in ("plane", "torus"):
    # exactness holds where the geometric transport is trivial (theta = 0);
    # on curved boards the frame choice is a gauge, randomized in training
    # (with_gauge) exactly like the eps gauge, so invariance is learned
    b = bundles[key]
    probs, _, _, _ = net.forward(b, [0] * b.n, 1, np.ones(b.n + 1))
    lg = np.log(probs[:b.n] + 1e-15)
    orbs = {}
    for v in range(b.n):
        orbs.setdefault(b.orbits[v], []).append(lg[v])
    claims.append(max(max(g) - min(g) for g in orbs.values()) < 1e-9)
ok(all(claims), "equivariance: empty-board policy exactly constant on WL "
   "orbits of flat boards (plane, torus), where transport is trivial")

# ---- geometric transport --------------------------------------------------------

bs = bundles["sphere"]
unit = float(np.abs(bs.ecos ** 2 + bs.esin ** 2 - 1).max())
flat = all(float(np.abs(bundles[k].esin).max()) == 0.0
           for k in ("plane", "torus", "mobius", "klein", "rp2"))
ok(unit < 1e-12 and float(np.abs(bs.esin).max()) > 0.1 and flat,
   "transport: per-edge parallel transport is an exact rotation on the "
   "sphere (cos^2+sin^2 = 1, sin live); flat and non-orientable boards "
   "reduce to theta = 0 (the w1 obstruction zeroes sin)")
gs = bs.with_gauge(np.random.default_rng(3))
ok(float(np.abs(gs.ecos ** 2 + gs.esin ** 2 - 1).max()) < 1e-12
   and float(np.abs(gs.esin - bs.esin).max()) > 1e-3,
   "transport: frame gauge augmentation moves per-edge angles but keeps "
   "them exact rotations")

# ---- persistence pairs ----------------------------------------------------------

from hatz import h0_pairs                                        # noqa: E402
_rngP = np.random.default_rng(11)
_phi = _rngP.uniform(0.05, 0.95, bundles["torus"].ne)
PV, PE = h0_pairs(bundles["torus"], _phi)
_chk = (PV.shape == (25, 4) and PE.shape == (bundles["torus"].ne, 3)
        and np.all(PV[:, 2] >= -1e-12)                  # lifetimes >= 0
        and np.all(PV[:, 0] >= PV[:, 1] - 1e-12)        # birth >= death
        and int(PV[:, 3].sum()) >= 1                    # essential class
        and int(PE[:, 0].sum()) == 24                   # spanning tree: n-1
        and int(PE[:, 1].sum()) == bundles["torus"].ne - 24)   # cycles: H1
ok(_chk, "persistence: H0 pairs of the phi-filtration — nonnegative "
   "lifetimes, birth >= death, one essential class, and the tree/cycle "
   "split matches n-1 / (ne-n+1)")

# ---- learned quantile levels ----------------------------------------------------

_netQ = HATZ(hidden=8, layers=2, seed=6)
_bq = bundles["klein"]
_stQ = [int(x) for x in np.random.default_rng(9).integers(0, 3, _bq.n)]
_, _, _, CQ = _netQ.forward(_bq, _stQ, 1, np.ones(_bq.n + 1))
lv0, lv1 = CQ["levels"][0], CQ["levels"][1]
_chk = (len(CQ["pl"]) == 2 and CQ["pl"][0] >= CQ["pl"][1] - 1e-12
        and len(lv0["ki"]) <= len(lv1["ki"])
        and len(CQ["levels"][2]["ki"]) == _bq.ne
        and np.all(lv0["gate"] >= 0.5 - 1e-12))
ok(_chk, "levels: learned quantile thresholds are ordered (most homophilous "
   "first), keeps nest into the full graph, and soft gates sit above 1/2 "
   "on kept edges")

# ---- Morse-flow routing ---------------------------------------------------------

Aup, Adn, Alat = CQ["flow"]
_deg = np.array([len(a) for a in _bq.adj], float)
_ind = (Aup > 0).astype(int) + (Adn > 0) + (Alat > 0)
_adjm = np.zeros((_bq.n, _bq.n), int)
for _v in range(_bq.n):
    for _u in _bq.adj[_v]:
        _adjm[_v, _u] = 1
_chk = np.array_equal(_ind, _adjm)             # each directed edge in
#                                                exactly one channel
rows_ok = all(abs(m[v].sum() - 1) < 1e-9 or m[v].sum() == 0
              for m in (Aup, Adn, Alat) for v in range(_bq.n))
ok(_chk and rows_ok, "morse flow: up/down/separatrix channels partition "
   "every directed edge exactly once, each row mean-normalized")

# ---- curriculum masking ---------------------------------------------------------

_p1, _, _, C1 = _netQ.forward(_bq, _stQ, 1, np.ones(_bq.n + 1),
                              levels_active=1)
_p3, _, _, C3 = _netQ.forward(_bq, _stQ, 1, np.ones(_bq.n + 1),
                              levels_active=3)
ok(len(C1["levels"]) == 1 and len(C3["levels"]) == 3
   and float(np.abs(_p1 - _p3).max()) > 1e-9,
   "curriculum: levels_active masks the attention to the most homophilous "
   "prefix and changes the policy")

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

# ---- curriculum resume ---------------------------------------------------------

ck_small = "/tmp/hatz_small.npz"
ck_cur = "/tmp/hatz_cur.npz"
for f in (ck_small, ck_cur):
    if os.path.isfile(f):
        os.remove(f)
r1 = subprocess.run(
    [sys.executable, os.path.join(HERE, "train_hatz.py"),
     "--specs", "mobius", "--iters", "1", "--games-per-iter", "2",
     "--sims", "8", "--steps-per-iter", "4", "--batch", "6",
     "--eval-every", "1", "--seed", "5", "--checkpoint", ck_small],
    capture_output=True, text=True)
r2 = subprocess.run(
    [sys.executable, os.path.join(HERE, "train_hatz.py"),
     "--specs", "mobius2", "--iters", "1", "--games-per-iter", "2",
     "--sims", "8", "--steps-per-iter", "4", "--batch", "6",
     "--eval-every", "1", "--seed", "6", "--resume", ck_small,
     "--checkpoint", ck_cur],
    capture_output=True, text=True)
resume_ok = (r1.returncode == 0 and r2.returncode == 0
             and os.path.isfile(ck_cur) and "resumed from" in r2.stdout)
if resume_ok:
    net_cur = HATZ.load(ck_cur)
    # supports must now cover BOTH the small and the medium board's surface
    resume_ok = net_cur.meta.get("specs") == ["mobius", "mobius2"]
else:
    print(r2.stdout[-400:], r2.stderr[-400:])
ok(resume_ok, "curriculum: --resume warm-starts from a small-board checkpoint "
   "onto a medium board and unions spec provenance")

# weights actually carried over (not re-initialized): a resumed 0-step run
# leaves parameters identical to the checkpoint
ck_a, ck_b = "/tmp/hatz_ra.npz", "/tmp/hatz_rb.npz"
subprocess.run(
    [sys.executable, os.path.join(HERE, "train_hatz.py"), "--specs", "mobius",
     "--iters", "1", "--games-per-iter", "2", "--sims", "8",
     "--steps-per-iter", "0", "--eval-every", "1", "--seed", "5",
     "--checkpoint", ck_a], capture_output=True, text=True)
subprocess.run(
    [sys.executable, os.path.join(HERE, "train_hatz.py"), "--specs", "mobius",
     "--iters", "1", "--games-per-iter", "1", "--sims", "8",
     "--steps-per-iter", "0", "--eval-every", "1", "--seed", "9",
     "--resume", ck_a, "--checkpoint", ck_b],
    capture_output=True, text=True)
na, nb = HATZ.load(ck_a), HATZ.load(ck_b)
identical = all(np.allclose(na.p[k], nb.p[k]) for k in na.p)
ok(identical, "curriculum: resumed weights are the checkpoint's (0 training "
   "steps leaves every parameter unchanged)")

# level curriculum: --curriculum trains on the homophilous prefix and
# persists its place in the schedule via iters_done
ck_lv = "/tmp/hatz_lvl.npz"
if os.path.isfile(ck_lv):
    os.remove(ck_lv)
r3 = subprocess.run(
    [sys.executable, os.path.join(HERE, "train_hatz.py"),
     "--specs", "mobius", "--iters", "3", "--games-per-iter", "1",
     "--sims", "8", "--steps-per-iter", "2", "--batch", "4",
     "--eval-every", "3", "--seed", "5", "--curriculum", "1",
     "--anneal-every", "2", "--checkpoint", ck_lv],
    capture_output=True, text=True)
lvl_ok = (r3.returncode == 0 and "levels 1/3" in r3.stdout
          and "levels 2/3" in r3.stdout)
if lvl_ok:
    lvl_ok = HATZ.load(ck_lv).meta.get("iters_done") == 3
else:
    print(r3.stdout[-400:], r3.stderr[-400:])
ok(lvl_ok, "level curriculum: --curriculum anneals levels 1 -> 2 across "
   "iterations and records iters_done for resumed schedules")

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
