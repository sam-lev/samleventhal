#!/usr/bin/env python3
"""Training-scaffold tests. Run from anywhere:
    python3 python/train/test_train.py
Covers: environment termination on every lowest-complexity topology, a full
numerical gradient check of the numpy net, move-for-move agreement between
hgnn.py and the browser's ai.js (under node), a two-generation CEM run whose
export loads back into ai.js, a one-iteration zero training run whose
checkpoint the bridge bot serves, and MCTS sanity.
"""

import json
import os
import random
import subprocess
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(os.path.dirname(HERE))

from env import LOWEST, adjacency, new_game, play_game, random_agent  # noqa
from hgnn import DEFAULT_WEIGHTS, Engine                              # noqa
from mcts import MCTS, policy_target                                  # noqa
from zeronet import ZeroNet, features, mean_matrix                    # noqa

passed = failed = 0


def ok(cond, name):
    global passed, failed
    print(("PASS  " if cond else "FAIL  ") + name)
    passed, failed = passed + bool(cond), failed + (not cond)


# ---- environment ---------------------------------------------------------

results = []
for key in LOWEST + ["plane9", "sphere-h", "torus-h"]:
    g = new_game(key)
    w, m = play_game(g, random_agent(random.Random(1)),
                     random_agent(random.Random(2)))
    results.append(w in ("B", "W", "draw") and m <= 4 * g.board.n)
ok(all(results), f"env: {len(results)} specs build; random games terminate "
   "legally with a scored winner")

# ---- numpy net: full numerical gradient check ------------------------------

rng = np.random.default_rng(0)
adj = [[1, 2], [0, 2, 3], [0, 1], [1, 4, 5], [3, 5], [3, 4], []]
A = mean_matrix(adj)
net = ZeroNet(hidden=6, layers=2, seed=3)
stones = [0, 1, 0, 2, 0, 1, 0]
X = features(adj, stones, 1)
mask = np.array([1, 0, 1, 0, 1, 1, 1, 1], dtype=float)
pi = np.array([0.3, 0, 0.2, 0, 0.25, 0.15, 0.05, 0.05])
z = 0.6

grads = net.zero_grads()
_, _, cache = net.forward(A, X, mask)
net.backward(cache, pi, z, grads)


def loss_at():
    _, _, c = net.forward(A, X, mask)
    p, v = c["probs"], c["v"]
    return -float(np.sum(pi * np.log(p + 1e-12))) + (z - v) ** 2


h = 1e-5
worst = 0.0
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
        ana = grads[k][i]
        rel = abs(num - ana) / max(1e-6, abs(num) + abs(ana))
        worst = max(worst, rel)
ok(worst < 1e-4, f"zeronet: analytic gradients match numerical over all "
   f"{sum(w.size for w in net.p.values())} parameters (worst rel err "
   f"{worst:.1e})")

# ---- hgnn <-> ai.js agreement -----------------------------------------------

if shutil.which("node"):
    def grid(w, hgt):
        N = []
        for r in range(hgt):
            for c in range(w):
                a = []
                if c > 0: a.append(r * w + c - 1)          # noqa: E701
                if c < w - 1: a.append(r * w + c + 1)      # noqa: E701
                if r > 0: a.append((r - 1) * w + c)        # noqa: E701
                if r < hgt - 1: a.append((r + 1) * w + c)  # noqa: E701
                N.append(a)
        return N

    def idx(r, c):
        return r * 7 + c

    positions = [
        [], [[idx(3, 3), 1]],
        [[idx(2, 2), 2], [idx(2, 1), 1], [idx(1, 2), 1], [idx(3, 2), 1]],
        [[idx(3, 3), 1], [idx(3, 2), 2], [idx(2, 3), 2], [idx(4, 3), 2]],
        [[idx(1, 1), 1], [idx(1, 2), 1], [idx(2, 1), 1], [idx(5, 5), 2],
         [idx(5, 4), 2], [idx(4, 5), 2], [idx(3, 3), 1], [idx(3, 4), 2]],
        [[idx(0, 0), 1], [idx(0, 1), 1], [idx(1, 0), 1], [idx(1, 1), 2],
         [idx(6, 6), 2], [idx(6, 5), 2], [idx(5, 6), 1]],
    ]
    probe = """
import { createRequire } from "module";
const require = createRequire(import.meta.url);
const GeoAI = require(%r);
function gridN(w,h){const N=[];for(let r=0;r<h;r++)for(let c=0;c<w;c++){const a=[];
if(c>0)a.push(r*w+c-1);if(c<w-1)a.push(r*w+c+1);if(r>0)a.push((r-1)*w+c);
if(r<h-1)a.push((r+1)*w+c);N.push(a);}return N;}
const N=gridN(7,7); const out=[];
for (const pos of JSON.parse(process.argv[2])) {
  const e=GeoAI.createEngine(N,{level:"strong",seed:1,select:{temp:0,noise:0}});
  const s=new Int8Array(49); for(const [v,c] of pos) s[v]=c;
  out.push({move:e.pickMove(s,1).move, value:e.evaluate(s,1).value});
}
console.log(JSON.stringify(out));
""" % os.path.join(ROOT, "web", "ai.js")
    pf = "/tmp/geo_agree_probe.mjs"
    with open(pf, "w") as f:
        f.write(probe)
    node = json.loads(subprocess.run(
        ["node", pf, json.dumps(positions)],
        capture_output=True, text=True).stdout)
    agree = True
    N7 = grid(7, 7)
    for i, pos in enumerate(positions):
        e = Engine(N7, level="strong", seed=1, select={"temp": 0, "noise": 0})
        s = [0] * 49
        for v, c in pos:
            s[v] = c
        r = e.pick_move(s, 1)
        ev = e.evaluate(s, 1)
        agree &= (r["move"] == node[i]["move"]
                  and abs(ev["V"] - node[i]["value"]) < 1e-4)
    ok(agree, "hgnn port: chosen moves and values identical to web/ai.js "
       f"under node on {len(positions)} positions")
else:
    print("skip  hgnn<->node agreement (node not on PATH)")

# ---- CEM trainer end-to-end ---------------------------------------------------

r = subprocess.run(
    [sys.executable, os.path.join(HERE, "train_hgnn.py"), "--boards", "plane",
     "--generations", "2", "--pop", "4", "--elite", "2", "--games", "2",
     "--eval-games", "2", "--seed", "7", "--out", "hgnn-smoke"],
    capture_output=True, text=True)
wj = os.path.join(ROOT, "web", "models", "hgnn-smoke.weights.json")
mj = os.path.join(ROOT, "web", "models", "hgnn-smoke.js")
cem_ok = r.returncode == 0 and os.path.isfile(wj) and os.path.isfile(mj)
loaded = None
if cem_ok:
    loaded = json.load(open(wj))
    cem_ok = "head" in loaded and "msc" in loaded
ok(cem_ok, "train_hgnn: 2-generation CEM run exports a model to web/models/")

if cem_ok and shutil.which("node"):
    check = """
const GeoAI = require(%r);
const W = require(%r);
function gridN(w,h){const N=[];for(let r=0;r<h;r++)for(let c=0;c<w;c++){const a=[];
if(c>0)a.push(r*w+c-1);if(c<w-1)a.push(r*w+c+1);if(r>0)a.push((r-1)*w+c);
if(r<h-1)a.push((r+1)*w+c);N.push(a);}return N;}
const e = GeoAI.createEngine(gridN(5,5), {seed:2});
e.setWeights(W);
const r = e.pickMove(new Int8Array(25), 1);
if (!(r.move >= 0 && r.move < 25)) process.exit(1);
""" % (os.path.join(ROOT, "web", "ai.js"), wj)
    rr = subprocess.run(["node", "-e", check], capture_output=True)
    ok(rr.returncode == 0, "trained weights load via setWeights in ai.js "
       "and produce a legal move")
for f in (wj, mj):
    if os.path.isfile(f):
        os.remove(f)

# ---- zero trainer end-to-end ----------------------------------------------------

ck = "/tmp/zero_smoke.npz"
if os.path.isfile(ck):
    os.remove(ck)
r = subprocess.run(
    [sys.executable, os.path.join(HERE, "train_zero.py"), "--specs", "plane",
     "--iters", "1", "--games-per-iter", "2", "--sims", "12",
     "--steps-per-iter", "10", "--batch", "16", "--eval-every", "1",
     "--seed", "5", "--checkpoint", ck],
    capture_output=True, text=True)
zero_ok = r.returncode == 0 and os.path.isfile(ck)
if not zero_ok:
    print(r.stdout[-800:], r.stderr[-800:])
else:
    net2 = ZeroNet.load(ck)
    zero_ok = net2.meta.get("specs") == ["plane"] and "loss" in r.stdout
ok(zero_ok, "train_zero: one self-play iteration trains, evaluates, and "
   "checkpoints (meta carries trained specs)")

# ---- MCTS sanity ------------------------------------------------------------------

g = new_game("plane")
board = g.board
tree = MCTS(ZeroNet(hidden=8, layers=2, seed=1), board,
            mean_matrix(board.adj), seed=2)
counts = tree.run(g, 32)
a = int(np.argmax(counts))
ok(counts.sum() == 32 and (a == board.n or g.is_legal(a)),
   "mcts: 32 simulations distribute over legal actions; argmax is playable")
# regression: live-ko positions must search in bounded time (the tree once
# cycled forever on ko recaptures because transpositions reset ko history)
import signal                                                          # noqa
gk = new_game("plane")
for mv in [6, 1, 8, 13, 12, 11, -1, 7]:
    gk.play_pass() if mv == -1 else gk.play(mv)
tk = MCTS(ZeroNet(hidden=16, layers=2, seed=1), gk.board,
          mean_matrix(gk.board.adj), seed=3)
signal.alarm(60)
ck_counts = tk.run(gk, 300)
signal.alarm(0)
ak = int(np.argmax(ck_counts))
ok(ck_counts.sum() == 300 and (ak == gk.board.n or gk.is_legal(ak)),
   "mcts: 300 simulations through a live ko complete in bounded time "
   "(positional-superko guard)")

g.play_pass()
g.play_pass()
counts2 = tree.run(g, 4)
ok(int(np.argmax(counts2)) == board.n,
   "mcts: terminal (double-pass) position returns pass")
pt = policy_target(counts, 0.0)
ok(abs(pt.sum() - 1) < 1e-9 and pt.max() == 1.0,
   "policy_target: temperature 0 is a one-hot on the visit argmax")

# ---- browser export: JS forward numerically matches python ----------------------------

from train_zero import export_web                                     # noqa
netw = ZeroNet.load(ck)
export_web(netw, "zero-webtest")
wj2 = os.path.join(ROOT, "web", "models", "zero-webtest.weights.json")
mj2 = os.path.join(ROOT, "web", "models", "zero-webtest.js")
web_ok = os.path.isfile(wj2) and os.path.isfile(mj2)
if web_ok and shutil.which("node"):
    gp = new_game("plane")
    adj_p = [list(a) for a in gp.board.adj]
    stones_p = [0] * gp.board.n
    stones_p[6], stones_p[12], stones_p[8] = 1, 2, 1
    mask_p = np.ones(gp.board.n + 1)
    mask_p[12] = mask_p[6] = mask_p[8] = 0
    probs_p, val_p, _ = netw.forward(mean_matrix(adj_p), 
                                     features(adj_p, stones_p, 1), mask_p)
    probe = """
const GeoAI = { models: [] };
const WEIGHTS = require(%r);
%s
const m = GeoAI.models[0];
const eng = m.create(%s, { level: "standard" });
const r = eng.forward(%s, 1, %s);
const pick = eng.pickMove(%s, 1, { legalMask: %s });
console.log(JSON.stringify({ probs: Array.from(r.probs), value: r.value,
                             move: pick.move, supports: m.supports }));
""" % (wj2, open(mj2).read(), json.dumps(adj_p), json.dumps(stones_p),
       json.dumps(list(mask_p)), json.dumps(stones_p),
       json.dumps([1 if stones_p[v] == 0 else 0 for v in range(25)]))
    pf2 = "/tmp/zero_web_probe.js"
    open(pf2, "w").write(probe)
    out = subprocess.run(["node", pf2], capture_output=True, text=True)
    try:
        j = json.loads(out.stdout)
        dp = float(np.abs(np.array(j["probs"]) - probs_p).max())
        dv = abs(j["value"] - val_p)
        legal_pick = j["move"] == -1 or (0 <= j["move"] < 25
                                         and stones_p[j["move"]] == 0)
        web_ok = dp < 1e-6 and dv < 1e-6 and legal_pick             and j["supports"]["surfaces"] == netw.meta["surfaces"]
        detail = f"max|dp|={dp:.1e}, |dv|={dv:.1e}"
    except Exception as e:
        web_ok, detail = False, f"{e}: {out.stdout[-200:]} {out.stderr[-200:]}"
    ok(web_ok, "browser export: the JS model's forward pass matches the "
       f"python net ({detail}) and plays legally")
for f in (wj2, mj2):
    if os.path.isfile(f):
        os.remove(f)

# ---- bridge bot ---------------------------------------------------------------------

os.environ["ZERO_CKPT"] = ck
sys.path.insert(0, os.path.join(ROOT, "bridge", "bots"))
import importlib                                                       # noqa
zm = importlib.import_module("zero_mini")
importlib.reload(zm)
bot = zm.BOT
avail = bot.available()
nb5 = []
for rr_ in range(5):
    for cc in range(5):
        a_ = []
        if cc > 0: a_.append(rr_ * 5 + cc - 1)          # noqa: E701
        if cc < 4: a_.append(rr_ * 5 + cc + 1)          # noqa: E701
        if rr_ > 0: a_.append((rr_ - 1) * 5 + cc)       # noqa: E701
        if rr_ < 4: a_.append((rr_ + 1) * 5 + cc)       # noqa: E701
        nb5.append(a_)
req = {"level": "standard",
       "spec": {"surface": "plane", "mesh": "square", "nx": 5, "ny": 5},
       "board": {"n": 25, "neighbors": nb5, "stones": [0] * 25,
                 "toMove": 1, "legalMask": [1] * 25, "moves": []}}
moves_ok = avail and bot.supports.get("surfaces") == ["plane"]
for lvl in ("casual", "standard", "strong"):
    req["level"] = lvl
    mv, info = bot.genmove(req)
    moves_ok = moves_ok and (mv == -1 or (0 <= mv < 25))
ok(moves_ok, "bridge bot zero_mini: loads the checkpoint, advertises "
   "trained surfaces, plays legally at all three levels")

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
