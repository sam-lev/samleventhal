#!/usr/bin/env python3
"""train_zero.py — AlphaZero/KataGo-style self-play training on the
lowest-complexity variant of every topology, headless.

(A note on names: real KataGo networks are CNNs over rectangles and cannot
see these boards at all. This is the KataGo *method* — policy+value network,
PUCT self-play, replay buffer — applied to a graph network that plays every
surface. One net trains across all specs because it only ever reads the
adjacency graph.)

    python3 train_zero.py --iters 100                 # all lowest topologies
    python3 train_zero.py --specs sphere,klein --sims 64

Checkpoints land in python/train/checkpoints/zero_mini.npz; the bridge bot
bots/zero_mini.py serves the latest automatically.
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
from env import (LOWEST, SPECS, adjacency, board_for, new_game,   # noqa: E402
                 play_game, random_agent)
from mcts import MCTS, policy_target                              # noqa: E402
from zeronet import ZeroNet, features, mean_matrix                # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(HERE, "checkpoints")


def self_play_game(net, key, boards, sims, seed, temp_moves=10):
    game = new_game(key)
    board, A = boards[key]
    tree = MCTS(net, board, A, seed=seed)
    records = []
    move_no = 0
    while not game.game_over and move_no < 4 * board.n:
        counts = tree.run(game, sims)
        tau = 1.0 if move_no < temp_moves else 0.0
        pi = policy_target(counts, tau)
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
    for r in records:
        r["z"] = 0.0 if z is None else (1.0 if r["player"] == z else -1.0)
    return records, winner, move_no


def eval_vs_random(net, key, boards, games, seed):
    board, A = boards[key]
    wins = 0.0
    for g in range(games):
        game = new_game(key)

        def net_agent(gm, color):
            mask = np.zeros(board.n + 1)
            mask[board.n] = 1
            legal = gm.legal_moves()
            for v in legal:
                mask[v] = 1
            if not legal:
                return -1
            X = features(board.adj, gm.colors, color)
            probs, _, _ = net.forward(A, X, mask)
            a = int(np.argmax(probs))
            return -1 if a == board.n else a

        rnd = random_agent(random.Random(seed * 31 + g))
        black, white = (net_agent, rnd) if g % 2 == 0 else (rnd, net_agent)
        winner, _ = play_game(game, black, white)
        net_is_black = g % 2 == 0
        if winner == "draw":
            wins += 0.5
        elif (winner == "B") == net_is_black:
            wins += 1.0
    return wins / games


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", default="lowest",
                    help=f"'lowest' or comma list of {', '.join(SPECS)}")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--games-per-iter", type=int, default=8)
    ap.add_argument("--sims", type=int, default=48)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--steps-per-iter", type=int, default=80)
    ap.add_argument("--buffer", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--checkpoint", default=os.path.join(CKPT_DIR,
                                                         "zero_mini.npz"))
    args = ap.parse_args()

    keys = LOWEST if args.specs == "lowest" else args.specs.split(",")
    for k in keys:
        if k not in SPECS:
            sys.exit(f"unknown spec '{k}'")
    boards = {}
    for k in keys:
        b = board_for(k)
        boards[k] = (b, mean_matrix(b.adj))
    net = ZeroNet(hidden=args.hidden, layers=args.layers, seed=args.seed)
    net.meta = {"specs": keys,
                "surfaces": sorted({SPECS[k][0] for k in keys}),
                "meshes": sorted({SPECS[k][1] for k in keys})}
    buf = deque(maxlen=args.buffer)
    rng = np.random.default_rng(args.seed)
    os.makedirs(CKPT_DIR, exist_ok=True)

    print(f"zero-style training on {keys} "
          f"({', '.join(f'{k}:{boards[k][0].n}v' for k in keys)}); "
          f"{args.sims} sims, hidden {args.hidden}x{args.layers}")
    for it in range(1, args.iters + 1):
        t0 = time.time()
        winners = []
        for g in range(args.games_per_iter):
            key = keys[(it * args.games_per_iter + g) % len(keys)]
            recs, winner, moves = self_play_game(
                net, key, boards, args.sims,
                seed=args.seed * 100 + it * 17 + g)
            buf.extend(recs)
            winners.append(winner)
        losses = []
        for _ in range(args.steps_per_iter):
            batch = [buf[i] for i in rng.integers(0, len(buf), args.batch)]
            grads = net.zero_grads()
            total = 0.0
            for r in batch:
                board, A = boards[r["key"]]
                X = features(board.adj, r["stones"], r["to_move"])
                _, _, cache = net.forward(A, X, r["mask"])
                total += net.backward(cache, r["pi"], r["z"], grads)
            for k in grads:
                grads[k] /= len(batch)
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
    print(f"saved {args.checkpoint} — the bridge bot bots/zero_mini.py "
          "now serves it")


if __name__ == "__main__":
    main()
