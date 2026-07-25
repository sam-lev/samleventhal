# Geodesics — training, loading, and playing every model

Two independent things happen to every model: it is **trained** (once, in
Python), and it is **deployed** so the app can play it. There are two
deployment channels, and most models can use either:

| Channel | Runs where | Entry point | Cost / strength | Needs a server? |
|---|---|---|---|---|
| **Online** | in the browser, compiled into `geodesics.html` | `web/models/<id>.js` (+ `.weights.json`), bundled by `node build.mjs` | light — policy + shallow lookahead; no deep search | no |
| **Bridge** | local Python, over WebSocket | `bridge/bots/<id>.py` exposing `BOT`, served by `python3 bridge/serve.py` | heavy — can run full PUCT/MCTS at many sims | yes (`serve.py`) |

The same trained weights can usually be shipped **both** ways. Which channel
is a model's *native* home is just a matter of weight — a graph net small
enough to hand-port to JS lives online; one that needs MCTS or expensive
pooling lives on the bridge.

### Model zoo at a glance

| Model | Trained by | Bridge | Online | Native home |
|---|---|---|---|---|
| **HGNN** — hierarchical Morse–Smale GNN | CEM, `train_hgnn.py` | drop-in bot (below) | ✅ `ai.js` built-in + trained `hgnn-*.js` | **Online** |
| **Zero** — AlphaZero-method graph net | SGD self-play, `train_zero.py` | ✅ `zero_mini.py` | ✅ `--export-web` | **Both** |
| **HATZ** — holonomy-aware topological AZ | SGD self-play, `train_hatz.py` / `train_hatz_vs.py` | ✅ `hatz_mini.py` | ✖ not shipped (see §HATZ) | **Bridge** |
| **KataGo** — external CNN engine | not trained here | ✅ `katago*.py` | ✖ impossible (rectangles only) | **Bridge only** |
| **Random** | — | ✅ `random_bot.py` (`py-random`) | ✅ `random.js` | **Both** |

Repository layout referenced throughout:

```
geodesics/
├── geodesics.html            # the built app (open with file://)
├── python/train/             # all training; run commands from here
│   ├── train_hgnn.py  train_zero.py  train_hatz.py  train_hatz_vs.py
│   ├── hgnn.py  zeronet.py  hatz.py  mcts.py  env.py
│   └── checkpoints/          # *.npz land here
├── web/                      # run `node build.mjs` from here
│   ├── build.mjs  shell.html  core.js  cells.js  ai.js  app.js
│   └── models/               # <id>.js + <id>.weights.json (online models)
└── bridge/                   # run `python3 serve.py` from here
    ├── serve.py  analysis.cfg
    ├── bots/                 # <id>.py exposing BOT (bridge models)
    └── nets/                 # KataGo *.bin.gz networks
```

---

## How loading & playing works (the 30-second version)

**Online.** `node build.mjs ../geodesics.html` (run in `web/`) inlines
`shell.html` + the four core modules + every `web/models/*.js`, each wrapped
with its sibling `<id>.weights.json` as a `WEIGHTS` constant. Each model file
calls `GeoAI.models.push({ id, name, levels, create })`. The app reads that
registry straight into the Opponent menu and runs the chosen engine
client-side — no network, works from a `file://` open. The built-in
`ai.js` engine (`id: "hgnn1"`) is always present even with no trained models.

**Bridge.** `python3 bridge/serve.py` imports every `bridge/bots/*.py`,
registers each module-level `BOT`, and listens on `ws://127.0.0.1:8765`. Open
`geodesics.html`; the app connects, calls `{"type":"hello"}`, and adds every
advertised bot to the menu ("bridge connected — N engines"). Each move it
sends a **stateless** `genmove` request — full stones, host-computed legal
mask, move history, and the adjacency list of the current play graph — and
the bot returns one site index (or `-1` to pass). The app re-validates every
move with its own rules engine, so a bot can never make an illegal move.
Stop `serve.py` and the bridge bots quietly vanish from the menu.

A bot hides itself (`available()` → `False`) when its weights or binary are
missing, and gates itself to boards it understands via
`supports = {"surfaces":[…], "meshes":[…], "incidence":[…]}` (omitted list =
"any"). The full annotated contract is `bridge/bots/random_bot.py`.

---

## HGNN — online-native, bridgeable

The hierarchical Morse–Smale graph engine. Its ~35 scalar weights are
optimized by a **cross-entropy method** (derivative-free: `pick_move`
contains argmax lookahead, so there is nothing to differentiate). `hgnn.py`
is the Python twin of the browser's `ai.js`, verified move-for-move.

**Train** (one instance per topology, or one across a pool):

```bash
cd python/train

# a sphere specialist
python3 train_hgnn.py --boards sphere --generations 60 --out hgnn-sphere

# one model across all lowest-complexity boards
python3 train_hgnn.py --boards lowest --generations 80 --out hgnn-all

# a Möbius/Klein/RP² non-orientable specialist
python3 train_hgnn.py --boards mobius,klein,rp2 --generations 80 --out hgnn-nonor
```

`--out <id>` writes `web/models/<id>.weights.json` + `<id>.js`. Omit `--out`
to just watch the fitness-vs-default curve without exporting.

**Play online** (its native channel):

```bash
cd web
node build.mjs ../geodesics.html      # bundles web/models/hgnn-*.js
```

Reopen `geodesics.html` → the Opponent menu now lists **HGNN sphere**, **HGNN
all**, etc. (trained models ship at the `standard` level). The always-present
**Hierarchical GNN** (`hgnn1`, hand-tuned Go priors, no training) offers
`casual` / `standard` / `strong`, where `strong` adds top-14 1-ply lookahead.

**Play over the bridge** (same engine, run in Python instead of compiled into
the page — handy for parity checks or serving without a rebuild). This bot
isn't in the repo by default; drop in `bridge/bots/hgnn_bridge.py`:

```python
"""bots/hgnn_bridge.py — serve the HGNN Python engine (hgnn.py) over the
bridge. HGNN_WEIGHTS points at any web/models/<id>.weights.json from
train_hgnn.py; with none it serves the built-in Go-prior weights."""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(os.path.dirname(os.path.dirname(HERE)), "python", "train")
if TRAIN not in sys.path:
    sys.path.insert(0, TRAIN)
WEIGHTS_PATH = os.environ.get("HGNN_WEIGHTS", "")


class HGNNBot:
    id = "hgnn"
    name = "HGNN (bridge)"
    levels = ["casual", "standard", "strong"]
    supports = None                          # plays every board

    def __init__(self):
        self._w = None
        if WEIGHTS_PATH and os.path.isfile(WEIGHTS_PATH):
            with open(WEIGHTS_PATH) as f:
                self._w = json.load(f)

    def genmove(self, req):
        from hgnn import Engine
        bd = req["board"]
        adj = [tuple(a) for a in bd["neighbors"]]
        legal = bd.get("legalMask") or [1] * bd["n"]
        eng = Engine(adj, level=req.get("level", "standard"), weights=self._w)
        out = eng.pick_move(list(bd["stones"]), bd["toMove"], legal_mask=legal)
        return out["move"], f"{self.name} · {out.get('reason', '')}"


BOT = HGNNBot()
```

```bash
cd bridge
HGNN_WEIGHTS=../web/models/hgnn-sphere.weights.json python3 serve.py
```

→ the menu gains **HGNN (bridge)** alongside the online models.

---

## Zero — the canonical both-channel model

An AlphaZero-method policy+value graph net, trained by **SGD self-play** with
a PUCT replay buffer. Because its forward pass is small, it ships identically
to the bridge (`zero_mini.py`) and to the browser (`--export-web`), and the JS
forward pass mirrors `zeronet.py` exactly.

**Train:**

```bash
cd python/train

python3 train_zero.py --iters 100                        # all lowest topologies
python3 train_zero.py --specs sphere,klein --sims 64     # a focused run
```

This writes `checkpoints/zero_mini.npz`. Continue onto bigger boards with the
curriculum warm-start:

```bash
python3 train_zero.py --resume checkpoints/zero_mini.npz \
    --specs torus2,klein2 --iters 120
```

**Play over the bridge** (native, automatic):

```bash
cd bridge && python3 serve.py       # 'Zero mini' appears once the .npz exists
```

Levels: `casual` samples the policy, `standard` plays its argmax, `strong`
runs a 96-simulation PUCT search on the request's own graph.

**Play online** — add one flag at train time to also emit a browser model:

```bash
cd python/train
python3 train_zero.py --iters 100 --export-web zero-all   # -> web/models/zero-all.*
cd ../web && node build.mjs ../geodesics.html
```

→ **Zero all** appears in the menu, running in-page (`casual` / `standard`;
no MCTS in the browser). You can also export from an already-trained
checkpoint by re-running with `--iters 0 --resume … --export-web <id>`.

---

## HATZ — bridge-native

Holonomy-Aware Topological AlphaZero: the Zero method plus an orientation
cocycle, cell incidences, a learned co-ownership **filtration**, and
**hierarchical Morse–Smale pooling**, with gauge augmentation and an
ownership auxiliary head. Trained by SGD self-play.

**Train (self-play):**

```bash
cd python/train

python3 train_hatz.py --iters 100                     # all lowest topologies
python3 train_hatz.py --specs mobius,klein,rp2        # non-orientable focus
python3 train_hatz.py --resume checkpoints/hatz.npz \
    --specs torus2,klein2 --iters 150                 # curriculum onto bigger boards
```

**Train against a chosen opponent** (e.g. bootstrap Go from KataGo on the
plane, then continue self-play onto the exotic surfaces — see
`train_hatz_vs.py`):

```bash
# Phase 1 — learn from KataGo on the 9×9 (records both seats)
python3 train_hatz_vs.py --specs plane9 --opponent katago-9x9 \
    --opponent-level standard --learn-from both --iters 120 \
    --checkpoint checkpoints/hatz_planar_katago.npz

# Phase 2 — carry those weights onto the non-orientable boards, self-play
python3 train_hatz_vs.py --specs sphere,cylinder,torus,mobius,klein,rp2 \
    --opponent self --resume checkpoints/hatz_planar_katago.npz \
    --iters 300 --checkpoint checkpoints/hatz_full.npz
```

Both write a `*.npz` checkpoint.

**Play over the bridge** (native, automatic):

```bash
cd bridge && python3 serve.py                          # serves checkpoints/hatz.npz
# or point at any checkpoint:
HATZ_CKPT=../python/train/checkpoints/hatz_full.npz python3 serve.py
```

→ **HATZ** appears (`casual` samples, `standard` argmax, `strong` = 64-sim
PUCT). It reconstructs the orientation cocycle from `nx`/`ny` so the
Möbius/Klein/RP² seam lands correctly on remote boards.

**Play online — not shipped.** Unlike Zero, HATZ has no browser port. Its
strength comes from things that don't exist in the page: MCTS search, the
Morse–Smale basin pooling, Cayley-geometric parallel transport, and the
filtration levels. To run HATZ in-browser you would hand-port `hatz.py`'s
`forward` to JS the way `zeronet.py` was ported for Zero — and even then only
the **raw policy head** (value/ownership optional), with no search, so it
would be a weak shadow of the bridged engine. For real HATZ play, use the
bridge.

---

## KataGo — bridge-only, external

Not trained here — KataGo is an external CNN engine over rectangles, so it is
gated to `plane:square:vertices` and can only run on the bridge. Install the
binary (`brew install katago` or a release build) and drop runtime
`*.bin.gz` networks (from katagotraining.org — the *network file* links, not
`.ckpt` checkpoints) into `bridge/nets/`:

```
bots/katago_b28.py        -> nets/katago_b28.bin.gz        (strong general)
bots/katago_zhizi-b40.py  -> nets/katago_zhizi-b40.bin.gz  (strong general)
bots/katago_9x9-b18.py    -> nets/katago_9x9-b18.bin.gz    (9×9 specialist)
bots/katago-humanv0.py    -> nets/katago-humanv0.bin.gz    (human ranks)
```

```bash
cd bridge && python3 serve.py     # each bot appears when its net + binary resolve
```

Strength maps to 24 / 200 / 1600 visits (`casual` / `standard` / `strong`);
the human net instead offers rank levels (20 kyu … 9 dan), sampling the raw
human policy at 1 visit under a `humanSLProfile`. A generic env slot exists
too: `KATAGO_MODEL=/path/net.bin.gz python3 serve.py`. Test the adapter
without a real KataGo: `python3 bridge/test_katago.py`.

---

## Random — both channels, no training

- **Online:** `random.js` is bundled by `build.mjs` (built-in, always there).
- **Bridge:** `bridge/bots/random_bot.py` serves it as **Random (bridge)**
  (`py-random`) — this is also the annotated reference for the bot contract.

Nothing to train.

---

## Cheat sheet

| Goal | Command (run from) |
|---|---|
| Train HGNN specialist | `python3 train_hgnn.py --boards sphere --generations 60 --out hgnn-sphere` *(python/train)* |
| Train Zero (all boards) | `python3 train_zero.py --iters 100` *(python/train)* |
| Train Zero + browser export | `python3 train_zero.py --iters 100 --export-web zero-all` *(python/train)* |
| Train HATZ self-play | `python3 train_hatz.py --iters 100` *(python/train)* |
| Train HATZ vs KataGo → self-play | `train_hatz_vs.py --opponent katago-9x9 …` then `--opponent self --resume …` *(python/train)* |
| Build the online app | `node build.mjs ../geodesics.html` *(web)* |
| Start the bridge | `python3 serve.py` *(bridge)* |
| Bridge a specific HATZ ckpt | `HATZ_CKPT=…/hatz_full.npz python3 serve.py` *(bridge)* |
| Bridge a specific Zero ckpt | `ZERO_CKPT=…/zero_mini.npz python3 serve.py` *(bridge)* |
| Bridge HGNN weights | `HGNN_WEIGHTS=…/hgnn-sphere.weights.json python3 serve.py` *(bridge)* |

## End-to-end: everything at once

```bash
# 1. train (python/train/)
cd python/train
python3 train_hgnn.py --boards lowest --generations 80 --out hgnn-all
python3 train_zero.py --iters 100 --export-web zero-all
python3 train_hatz_vs.py --specs plane9 --opponent katago-9x9 --learn-from both \
    --iters 120 --checkpoint checkpoints/hatz_planar_katago.npz
python3 train_hatz_vs.py --specs sphere,torus,mobius,klein,rp2 --opponent self \
    --resume checkpoints/hatz_planar_katago.npz --iters 300 \
    --checkpoint checkpoints/hatz.npz

# 2. bundle the online models into the page (web/)
cd ../../web && node build.mjs ../geodesics.html
#    -> menu gains: HGNN all, Zero all, Random, Hierarchical GNN (built-in)

# 3. start the bridge for the heavy engines (bridge/)
cd ../bridge && python3 serve.py
#    -> menu also gains: HATZ, Zero mini, KataGo* (if nets present), py-random

# 4. open geodesics.html — online models play with the bridge down;
#    bridge models appear whenever serve.py is running.
```

Online and bridge coexist in the same Opponent menu: online models play with
no server, and the bridge models fade in and out as `serve.py` starts and
stops.
