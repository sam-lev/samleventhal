# python/train — self-play training

Two trainers, one environment. The environment is just the geodesics
package itself: `make_board()` builds any surface × mesh, `Game` enforces
capture/superko/Tromp-Taylor, and `env.py` names the **lowest-complexity
variant of every topology** (`sphere` 42v geodesic, `plane` 5×5, `cylinder`,
`torus`, `mobius`, `klein`, `rp2` at 25v, plus `plane9`, `sphere-h`,
`torus-h`, …). Everything runs headless on numpy alone — nothing to install.

**An honesty note.** Real KataGo cannot train on these boards: its networks
are CNNs over rectangular tensors, so a sphere or a Möbius band cannot even
be fed in. `train_zero.py` is therefore the KataGo *method* — policy+value
network, PUCT self-play, replay buffer — applied to a graph network
(`zeronet.py`) that reads only the adjacency graph and hence plays every
surface. If you want literal KataGo on the classical boards, use the bridge
bots and katagotraining.org networks; those are already stronger than
anything trainable on a laptop.

## 1. Hierarchical (MSC) engine — `train_hgnn.py`

`hgnn.py` is an op-for-op Python port of the browser engine (`web/ai.js`):
same multi-persistence Morse–Smale hierarchy, same pyramid, same tactical
head, same float32 rounding — `test_train.py` verifies the two pick
**identical moves** on identical positions. Its ~34 scalar weights are
optimized by a cross-entropy method: sample around the current mean, score
by self-play win-rate against the mean, move toward the elites (pickMove
contains argmax lookahead, so derivative-free is the honest optimizer).

    # one instance per topology, trained independently:
    python3 train_hgnn.py --boards sphere --generations 60 --out hgnn-sphere
    python3 train_hgnn.py --boards torus  --generations 60 --out hgnn-torus
    # or one instance across the whole pool:
    python3 train_hgnn.py --boards lowest --generations 80 --out hgnn-all

`--out` writes `web/models/<out>.weights.json` + `<out>.js`; the next
`node build.mjs` in `web/` puts the trained instance in the Opponent menu,
fully in-browser. Expect ~6 s/generation at the defaults on a 42v board.

## 2. Zero-style opponent — `train_zero.py`

    python3 train_zero.py --iters 100                # all lowest topologies
    python3 train_zero.py --specs sphere,klein --sims 64

One net trains across all chosen specs (it is graph-native). Self-play games
use PUCT with Dirichlet root noise and visit-count policy targets;
`--eval-every` reports win-rate vs a random player and checkpoints to
`checkpoints/zero_mini.npz`. The bridge bot `bridge/bots/zero_mini.py`
serves the checkpoint automatically (hidden until one exists), advertising
exactly the surfaces/meshes in its training meta; its `strong` level runs a
96-sim search on the request's own graph. Backprop is hand-written numpy,
verified against numerical gradients over every parameter in the tests.

## Tests

    python3 test_train.py

Environment termination on every spec, the full gradient check, the
hgnn↔ai.js agreement suite (needs `node`), a two-generation CEM run whose
export round-trips through `setWeights`, a one-iteration zero run whose
checkpoint the bridge bot loads and plays, and MCTS sanity.
