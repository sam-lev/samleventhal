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

### Into the browser, no bridge

    python3 train_zero.py --iters 150 --specs lowest --export-web zero-all
    cd ../../web && node build.mjs ../geodesics.html

`--export-web` writes the trained net into `web/models/<id>.{js,weights.json}`
— the JS file implements the identical forward pass (verified to machine
precision in the tests), registers in the Opponent menu with the supports it
trained on, and runs fully client-side at casual (policy sampling) and
standard (argmax) levels. A ~6.5k-parameter net is ~80 KB of JSON; scale
`--hidden 64 --layers 4` and it is still tiny.

### Harder boards & generalization

`env.py` now carries a harder tier — `sphere2` (92v geodesic), `torus2`,
`klein2`, `rp22` (49v), `torus-h2`, `mobius-h`, and true 3D lattices `box3` /
`torus3` (4x4x4, 64v) — use `--specs sphere2,klein2,box3` etc. Recipes that
work: train one pool across many topologies for a generalist (the net is
graph-native, so it transfers); curriculum by resolution (train `torus`, then
fine-tune on `torus2` — same weights, bigger graph); per-family specialists
exported under separate ids so the menu gates each to its boards. The
highest-leverage upgrade remains symmetry augmentation by graph
automorphisms: every self-play position multiplies by |Aut| (60 rotations on
a geodesic sphere, all translations on a torus) at zero search cost.

## 3. Holonomy-Aware Topological AlphaZero — `train_hatz.py` (research draft)

A v1 of the Stage-2 architecture (`hatz.py`, ~35k params), honest about its
scope. What each contribution means here: **(1) reflection gauge structure**
— an orientation cocycle eps on edges, from face cycles (dual BFS) or the
quotient gluing rule; messages pass through eps-conditioned shared
restriction maps (W0 + eps*W1), so crossing the Mobius/Klein/RP2 seam applies
a different learned transport. Cocycle detection is verified against known
orientability on all 11 board types. **(2) Automorphism equivariance** is
exact and inherited (weight sharing + Aut-invariant inputs) — tested to
machine precision via WL-orbit constancy on sphere/torus and the
seam-preserving Mobius reflection; gauge augmentation trains away the
remaining seam-placement dependence. **(3) Morse-Smale pooling** — the same
basin hierarchy that powers the browser engine segments each position's
influence field into a region graph used as a U-Net-style pooling level
(position-dependent, parameter-independent, so backprop stays exact;
gradient-checked to 5e-6 over every parameter). **(4)** V/E/F cell ranks with
incidence message passing (faces where the board provides them), curvature
(angle-defect) inputs, PH-lite group features, ownership auxiliary head,
KataGo-style global pooling.

    python3 train_hatz.py --specs mobius,klein,rp2 --iters 100
    python3 train_hatz.py --iters 150        # all lowest topologies

Served by `bridge/bots/hatz_mini.py` (the bot reconstructs the seam from
nx/ny on remote boards and reports "non-orientable seam active"). Known v1
gaps: no continuous SO(2) part (isotropic aggregation), gauge invariance is
learned not guaranteed, PH features are local proxies. Note: self-play masks
passes in the first n/2 moves — under Tromp-Taylor + komi, early pass-pass
is a degenerate White win that otherwise collapses the policy (KataGo guards
the same way).

## Tests

    python3 test_train.py

Environment termination on every spec, the full gradient check, the
hgnn↔ai.js agreement suite (needs `node`), a two-generation CEM run whose
export round-trips through `setWeights`, a one-iteration zero run whose
checkpoint the bridge bot loads and plays, and MCTS sanity.
