#!/usr/bin/env python3
"""train_hatz.py — self-play training of the Holonomy-Aware Topological
AlphaZero (see hatz.py). Same loop as train_zero.py, three additions:

* the network reads the board through a Bundle (orientation cocycle,
  cell incidences, curvature, Morse–Smale pooling);
* an ownership auxiliary head trained on final Tromp-Taylor territory;
* gauge augmentation: each self-play game and each training step sees the
  cocycle in a random gauge, teaching invariance to seam placement.

    python3 train_hatz.py --iters 100                    # all lowest specs
    python3 train_hatz.py --specs mobius,klein,rp2       # non-orientable focus
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from collections import deque

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env import LOWEST, SPECS, board_for, new_game, play_game, random_agent  # noqa
from hatz import HATZ, Bundle, final_ownership                               # noqa
from mcts import MCTS, policy_target                                         # noqa

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(HERE, "checkpoints")


def self_play_game(net, key, boards, sims, seed, rng, gauge_aug,
                   temp_moves=10):
    game = new_game(key)
    board, bundle = boards[key]
    if gauge_aug:
        bundle = bundle.with_gauge(rng)
    tree = MCTS(net, board, bundle, seed=seed)
    records = []
    move_no = 0
    while not game.game_over and move_no < 4 * board.n:
        counts = tree.run(game, sims)
        if move_no < board.n // 2 and counts[:board.n].sum() > 0:
            counts = counts.copy()          # no early passing in self-play
            counts[board.n] = 0             # (Tromp-Taylor + komi makes
        tau = 1.0 if move_no < temp_moves else 0.0   # early pass-pass a
        pi = policy_target(counts, tau)              # degenerate White win)
        mask = np.zeros(board.n + 1)
        mask[board.n] = 1
        for v in game.legal_moves():
            mask[v] = 1
        records.append({"key": key, "stones": tuple(game.colors),
                        "to_move": game.to_move, "mask": mask, "pi": pi,
                        "player": game.to_move})
        a = int(np.random.default_rng(seed * 7 + move_no).choice(
            board.n + 1, p=pi)) if tau > 0 else int(np.argmax(pi))
        if a == board.n or not game.is_legal(a):
            game.play_pass()
        else:
            game.play(a)
        move_no += 1
    winner = game.score()["winner"]
    z = {"B": 1, "W": 2}.get(winner)
    own_black = final_ownership(board, game.colors)
    for r in records:
        r["z"] = 0.0 if z is None else (1.0 if r["player"] == z else -1.0)
        r["own"] = own_black if r["player"] == 1 else -own_black
    return records, winner, move_no


def eval_vs_random(net, key, boards, games, seed):
    board, bundle = boards[key]
    wins = 0.0
    for g in range(games):
        game = new_game(key)

        def net_agent(gm, color):
            legal = gm.legal_moves()
            if not legal:
                return -1
            mask = np.zeros(board.n + 1)
            mask[board.n] = 1
            for v in legal:
                mask[v] = 1
            probs, _ = net.policy_value(bundle, gm.colors, color, mask)
            a = int(np.argmax(probs[:board.n]))     # prefer playing to passing
            return a if probs[a] > 0 else -1

        rnd = random_agent(random.Random(seed * 31 + g))
        black, white = (net_agent, rnd) if g % 2 == 0 else (rnd, net_agent)
        winner, _ = play_game(game, black, white)
        if winner == "draw":
            wins += 0.5
        elif (winner == "B") == (g % 2 == 0):
            wins += 1.0
    return wins / games


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", default="lowest")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--games-per-iter", type=int, default=6)
    ap.add_argument("--sims", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps-per-iter", type=int, default=60)
    ap.add_argument("--buffer", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--gauge-aug", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--checkpoint", default=os.path.join(CKPT_DIR,
                                                         "hatz.npz"))
    args = ap.parse_args()

    keys = LOWEST if args.specs == "lowest" else args.specs.split(",")
    for k in keys:
        if k not in SPECS:
            sys.exit(f"unknown spec '{k}'")
    boards = {}
    for k in keys:
        b = board_for(k)
        boards[k] = (b, Bundle(b, SPECS[k][0]))
    net = HATZ(hidden=args.hidden, layers=args.layers, seed=args.seed)
    net.meta = {"arch": "hatz", "specs": keys,
                "surfaces": sorted({SPECS[k][0] for k in keys}),
                "meshes": sorted({SPECS[k][1] for k in keys})}
    buf = deque(maxlen=args.buffer)
    rng = np.random.default_rng(args.seed)
    os.makedirs(CKPT_DIR, exist_ok=True)
    nparam = sum(v.size for v in net.p.values())
    nonor = [k for k in keys if boards[k][1].nonorientable]
    print(f"HATZ training on {keys} ({nparam} params, "
          f"non-orientable: {nonor or 'none'}); {args.sims} sims, "
          f"gauge-aug {'on' if args.gauge_aug else 'off'}")
    for it in range(1, args.iters + 1):
        t0 = time.time()
        winners = []
        for g in range(args.games_per_iter):
            key = keys[(it * args.games_per_iter + g) % len(keys)]
            recs, winner, _ = self_play_game(
                net, key, boards, args.sims,
                seed=args.seed * 100 + it * 17 + g,
                rng=rng, gauge_aug=args.gauge_aug)
            buf.extend(recs)
            winners.append(winner)
        losses = []
        for _ in range(args.steps_per_iter):
            batch = [buf[i] for i in rng.integers(0, len(buf), args.batch)]
            gauged = {k: (boards[k][1].with_gauge(rng) if args.gauge_aug
                          else boards[k][1]) for k in keys}
            grads = net.zero_grads()
            total = 0.0
            for r in batch:
                _, _, _, C = net.forward(gauged[r["key"]], r["stones"],
                                         r["to_move"], r["mask"])
                total += net.backward(C, r["pi"], r["z"], r["own"], grads)
            for k2 in grads:
                grads[k2] /= len(batch)
            net.adam_step(grads, lr=args.lr)
            losses.append(total / len(batch))
        line = (f"iter {it:3d}  buffer {len(buf):5d}  "
                f"loss {np.mean(losses):.3f}  "
                f"B/W {winners.count('B')}/{winners.count('W')}  "
                f"{time.time() - t0:.0f}s")
        if it % args.eval_every == 0 or it == args.iters:
            wr = np.mean([eval_vs_random(net, k, boards, 6,
                                         seed=args.seed + it)
                          for k in keys])
            line += f"  | vs random: {wr:.2f}"
            net.save(args.checkpoint)
            line += "  [checkpoint]"
        print(line, flush=True)
    net.save(args.checkpoint)
    print(f"saved {args.checkpoint} — bridge bot bots/hatz_mini.py serves it")


if __name__ == "__main__":
    main()
