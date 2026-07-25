#!/usr/bin/env python3
"""train_hatz_vs.py — HATZ training with a *configurable opponent*.

This is a strict superset of train_hatz.py. Everything that made HATZ HATZ is
untouched and lives where it always did:

  * the hierarchical Morse-Smale pooling and the learned co-ownership
    filtration (hatz.py: forward/backward);
  * gauge augmentation of the orientation cocycle (Bundle.with_gauge);
  * the ownership / co-ownership auxiliary heads;
  * PUCT self-play search (mcts.py) and the curriculum warm-start (--resume).

The one addition is that the *other* player in each training game need no
longer be HATZ. With --opponent you can point the black/white seat at any
bridge bot in bridge/bots/ — in particular KataGo. That lets you bootstrap
strong planar Go into the network before the topology ever leaves the plane,
then --resume onto the non-orientable boards where no external teacher exists.

The intended two-phase regimen
------------------------------
Phase 1 — learn Go from KataGo on the plane (9x9):

    python3 train_hatz_vs.py \
        --specs plane9 --opponent katago-9x9 --opponent-level standard \
        --learn-from both --iters 120 \
        --checkpoint checkpoints/hatz_planar_katago.npz

Phase 2 — carry those weights onto the exotic surfaces, self-play only:

    python3 train_hatz_vs.py \
        --specs sphere,cylinder,torus,mobius,klein,rp2 --opponent self \
        --resume checkpoints/hatz_planar_katago.npz --iters 300 \
        --checkpoint checkpoints/hatz_full.npz

(Phase 2 is ordinary HATZ self-play — you can equally run train_hatz.py
--resume; this script just lets you use one entry point for both phases.)

Why "both" is the default record mode against an external opponent
------------------------------------------------------------------
A fresh net thrown against a 1600-visit KataGo loses essentially every game,
so if only HATZ's own moves are recorded the value target z is ~ -1 for the
whole buffer and the value head collapses onto that base rate (exactly the
"confident-but-wrong attractor" seen in earlier runs). Recording *both*
seats fixes this two ways: KataGo's moves enter the policy head as expert
targets (the AlphaGo-style supervised bootstrap), and the value head now
sees +1 labels on KataGo-to-move positions and -1 on HATZ-to-move ones — a
balanced, skill-bearing signal instead of a constant. Use --learn-from hatz
only when the opponent is strength-matched to HATZ (e.g. a low-kyu KataGo
human profile), where pure AlphaZero-against-a-sparring-partner is sound.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import random
import sys
import threading
import time
from collections import deque

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env import LOWEST, SPECS, board_for, new_game, play_game, random_agent  # noqa
from hatz import HATZ, Bundle, co_ownership_targets, final_ownership          # noqa
from mcts import MCTS, policy_target                                          # noqa
# reuse the canonical self-play game and the vs-random probe verbatim so the
# two scripts can never drift apart on the self-play path
from train_hatz import self_play_game, eval_vs_random                         # noqa

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(HERE, "checkpoints")


# --------------------------------------------------------------------------- #
#  Opponent: any bridge bot, driven in-process                                #
# --------------------------------------------------------------------------- #

def find_bots_dir(explicit=None):
    """Locate bridge/bots. Works from python/train/ (repo layout) or any
    ancestor that has a bridge/bots beneath it."""
    if explicit:
        if not os.path.isdir(explicit):
            sys.exit(f"--bots-dir: no such directory {explicit}")
        return explicit
    d = HERE
    for _ in range(6):                       # walk up a few levels
        cand = os.path.join(d, "bridge", "bots")
        if os.path.isdir(cand):
            return cand
        nd = os.path.dirname(d)
        if nd == d:
            break
        d = nd
    return None


def load_bridge_bots(bots_dir):
    """Import every bots/*.py exactly as serve.py does and return {id: BOT}.
    The bots directory goes on sys.path first so variant files can import
    their siblings (e.g. `from katago import KataGoBot`)."""
    bots = {}
    if bots_dir is None:
        return bots
    if bots_dir not in sys.path:
        sys.path.insert(0, bots_dir)
    for fname in sorted(os.listdir(bots_dir)):
        if not fname.endswith(".py") or fname.startswith("_"):
            continue
        path = os.path.join(bots_dir, fname)
        try:
            spec = importlib.util.spec_from_file_location(
                "geobot_" + fname[:-3].replace("-", "_"), path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            bot = getattr(mod, "BOT", None)
            if bot is not None and ":" not in getattr(bot, "id", ":"):
                bots[bot.id] = bot
        except Exception as e:               # a broken/optional bot is not fatal
            print(f"  (skip bot {fname}: {e})")
    return bots


def build_req(game, level, spec_meta):
    """Assemble the stateless bridge request from a live Game — the exact
    envelope documented in bots/random_bot.py. KataGo consumes `moves`
    (replaying under tromp-taylor); graph bots consume neighbors/stones/mask;
    we supply all of it so any bot is a drop-in opponent."""
    b = game.board
    n = b.n
    legal = set(game.legal_moves())
    stones = list(game.colors)
    cmap = {"B": 1, "W": 2}
    moves = [[cmap[c], (-1 if v is None else v)] for c, v in game.moves]
    # make_board keeps the raw grid dims under meta["resolved_params"]
    # (the top-level params echo the abstract surface/mesh/resolution spec);
    # KataGo needs the actual nx/ny of the goban it is being asked about
    rp = (b.meta or {}).get("resolved_params", {})
    nx = rp.get("nx") or b.params.get("nx", 0)
    ny = rp.get("ny") or b.params.get("ny", 0)
    return {
        "type": "genmove",
        "level": level,
        "spec": {
            "surface": spec_meta.get("surface") or b.params.get("surface"),
            "mesh": spec_meta.get("mesh") or rp.get("mesh"),
            "incidence": "vertices",
            "scaleIdx": spec_meta.get("scaleIdx", 0),
            "nx": nx,
            "ny": ny,
        },
        "board": {
            "n": n,
            "neighbors": [list(a) for a in b.adj],
            "stones": stones,
            "toMove": game.to_move,
            "legalMask": [1 if v in legal else 0 for v in range(n)],
            "moves": moves,
        },
    }


class BridgeOpponent:
    """Adapts a bridge BOT to the (game, level) -> (vertex, policy) contract
    used by the training loop. `policy` is a length-(n+1) target distribution
    when the bot exposes one, else None (the loop then falls back to a one-hot
    on the played site)."""

    def __init__(self, bot, level, spec_meta):
        self.bot = bot
        self.name = getattr(bot, "name", getattr(bot, "id", "opponent"))
        self.level = level or (getattr(bot, "levels", ["standard"]) or
                               ["standard"])[0]
        self.spec_meta = spec_meta

    def available(self):
        av = getattr(self.bot, "available", None)
        return True if av is None else bool(av())

    def move(self, game):
        req = build_req(game, self.level, self.spec_meta)
        result = self.bot.genmove(req)
        v = result[0] if isinstance(result, tuple) else result
        v = int(v)
        # optional richer signal: a bot may return ("move", info, policy) or
        # expose .policy(req); we keep the door open without requiring it
        pol = None
        if isinstance(result, tuple) and len(result) >= 3 and \
                isinstance(result[2], (list, tuple, np.ndarray)):
            pol = np.asarray(result[2], float)
        return v, pol


def onehot(n_plus_1, action):
    t = np.zeros(n_plus_1)
    t[action] = 1.0
    return t


def watched_move(opponent, game, warn_after, move_no):
    """Call opponent.move in a worker thread. If it hasn't returned within
    warn_after seconds, print a heads-up (KataGo silences its own stderr, so a
    stuck engine is otherwise invisible) and keep waiting. Never kills the
    subprocess — it only makes a hang legible."""
    if not warn_after or warn_after <= 0:
        return opponent.move(game)
    box = {}

    def worker():
        try:
            box["out"] = opponent.move(game)
        except Exception as e:                    # surface bot errors to caller
            box["err"] = e

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    th.join(warn_after)
    if th.is_alive():
        print(f"    [waiting] {opponent.name} has not replied in "
              f"{warn_after:.0f}s at move {move_no}. If it never returns it is "
              f"probably stuck loading its network or misconfigured — set "
              f"logToStderr=true in bridge/analysis.cfg to see why, or drop to "
              f"--opponent-level casual.", flush=True)
        th.join()
    if "err" in box:
        raise box["err"]
    return box["out"]


# --------------------------------------------------------------------------- #
#  A single training game against a fixed opponent                            #
# --------------------------------------------------------------------------- #

def play_vs_game(net, key, boards, opponent, sims, seed, rng, gauge_aug,
                 learn_from="both", hatz_color=1, temp_moves=10,
                 root_noise=True, progress=0, opponent_timeout=45):
    """HATZ (playing `hatz_color`) vs a fixed opponent. HATZ moves are chosen
    by its own PUCT search, giving genuine improved-policy targets; opponent
    moves are played by the bot. Records are emitted per --learn-from and get
    z / ownership targets filled in from the finished game, identically to
    self-play."""
    game = new_game(key)
    board, bundle = boards[key]
    if gauge_aug:
        bundle = bundle.with_gauge(rng)
    tree = MCTS(net, board, bundle, seed=seed)
    records = []
    move_no = 0
    while not game.game_over and move_no < 4 * board.n:
        to_move = game.to_move
        mask = np.zeros(board.n + 1)
        mask[board.n] = 1
        for v in game.legal_moves():
            mask[v] = 1

        if to_move == hatz_color:
            counts = tree.run(game, sims, root_noise=root_noise)
            if move_no < board.n // 2 and counts[:board.n].sum() > 0:
                counts = counts.copy()          # suppress early passes so a
                counts[board.n] = 0             # degenerate pass-pass White
            tau = 1.0 if move_no < temp_moves else 0.0   # win can't dominate
            pi = policy_target(counts, tau)
            records.append({"key": key, "stones": tuple(game.colors),
                            "to_move": to_move, "mask": mask, "pi": pi,
                            "player": to_move})
            a = int(np.random.default_rng(seed * 7 + move_no).choice(
                board.n + 1, p=pi)) if tau > 0 else int(np.argmax(pi))
        else:
            v, pol = watched_move(opponent, game, opponent_timeout, move_no)
            legal_v = (v is not None and 0 <= v < board.n and game.is_legal(v))
            a = v if legal_v else board.n       # anything illegal -> pass
            if learn_from == "both":
                # target = the opponent's policy if it gave one, else a
                # one-hot on the site it actually played (AlphaGo-style
                # expert imitation)
                if pol is not None and pol.shape[0] == board.n + 1 \
                        and pol.sum() > 0:
                    pi = (pol * mask)
                    pi = pi / pi.sum() if pi.sum() > 0 else onehot(
                        board.n + 1, a)
                else:
                    pi = onehot(board.n + 1, a)
                records.append({"key": key, "stones": tuple(game.colors),
                                "to_move": to_move, "mask": mask, "pi": pi,
                                "player": to_move})

        if a == board.n or not game.is_legal(a):
            game.play_pass()
        else:
            game.play(a)
        move_no += 1
        if progress and move_no % progress == 0:
            side = "H" if to_move == hatz_color else "K"
            print(f"      · {key} move {move_no:3d} ({side})", flush=True)

    winner = game.score()["winner"]
    z = {"B": 1, "W": 2}.get(winner)
    own_black = final_ownership(board, game.colors)
    eown = co_ownership_targets(bundle, own_black)     # perspective-invariant
    for r in records:
        r["z"] = 0.0 if z is None else (1.0 if r["player"] == z else -1.0)
        r["own"] = own_black if r["player"] == 1 else -own_black
        r["eown"] = eown
    return records, winner, move_no, hatz_color


def hatz_result(winner, hatz_color):
    """HATZ's game score in [0,1] (0.5 for a draw)."""
    if winner == "draw":
        return 0.5
    return 1.0 if {"B": 1, "W": 2}[winner] == hatz_color else 0.0


# --------------------------------------------------------------------------- #
#  Main                                                                        #
# --------------------------------------------------------------------------- #

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
    ap.add_argument("--temp-moves", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--resume", default=None, metavar="CKPT",
                    help="warm-start weights (+Adam state) from a checkpoint; "
                         "its --hidden/--layers win, so a curriculum onto new "
                         "boards just works")
    ap.add_argument("--checkpoint", default=os.path.join(CKPT_DIR,
                                                         "hatz.npz"))
    # --- opponent controls (the new surface area) ---
    ap.add_argument("--opponent", default="self",
                    help="'self' for ordinary HATZ self-play, or a bridge bot "
                         "id such as katago-9x9 / katago-b28 / katago-human / "
                         "py-random")
    ap.add_argument("--opponent-level", default=None,
                    help="strength level passed to the opponent bot "
                         "(e.g. casual|standard|strong, or '5 kyu' for the "
                         "human net); defaults to the bot's first level")
    ap.add_argument("--learn-from", choices=["hatz", "both"], default=None,
                    help="whose moves become training records against an "
                         "external opponent (default: both)")
    ap.add_argument("--hatz-plays", choices=["alternate", "black", "white"],
                    default="alternate",
                    help="which seat HATZ takes in vs-opponent games")
    ap.add_argument("--root-noise", type=int, default=1,
                    help="Dirichlet root exploration on HATZ's own moves")
    ap.add_argument("--progress", type=int, default=0, metavar="N",
                    help="against an external opponent, print a heartbeat "
                         "every N moves within a game (0 = per-game lines "
                         "only). Use it to tell 'slow' from 'stuck'.")
    ap.add_argument("--opponent-timeout", type=float, default=45,
                    help="warn (do not kill) if the opponent takes longer "
                         "than this many seconds for one move; 0 disables")
    ap.add_argument("--bots-dir", default=None,
                    help="override auto-detected bridge/bots location")
    args = ap.parse_args()

    keys = LOWEST if args.specs == "lowest" else args.specs.split(",")
    for k in keys:
        if k not in SPECS:
            sys.exit(f"unknown spec '{k}'")
    boards = {}
    spec_meta = {}
    for k in keys:
        b = board_for(k)
        boards[k] = (b, Bundle(b, SPECS[k][0]))
        surface, mesh, _ = SPECS[k]
        spec_meta[k] = {"surface": surface, "mesh": mesh, "scaleIdx": 0}

    vs_external = args.opponent != "self"
    learn_from = args.learn_from or ("both" if vs_external else "hatz")

    opponent = None
    if vs_external:
        bots = load_bridge_bots(find_bots_dir(args.bots_dir))
        bot = bots.get(args.opponent)
        if bot is None:
            have = ", ".join(sorted(bots)) or "(none found)"
            sys.exit(f"unknown opponent '{args.opponent}'. "
                     f"available bridge bots: {have}")
        # every board in this run must be one the opponent actually plays
        sup = getattr(bot, "supports", None)
        if sup is not None:
            for k in keys:
                surface, mesh, _ = SPECS[k]
                if surface not in (sup.get("surfaces") or []) or \
                        mesh not in (sup.get("meshes") or []):
                    sys.exit(
                        f"opponent '{args.opponent}' does not support "
                        f"spec '{k}' ({surface}/{mesh}). It supports "
                        f"{sup.get('surfaces')} x {sup.get('meshes')}. "
                        f"Train planar boards against KataGo, then --resume "
                        f"onto the rest with --opponent self.")
        opponent = BridgeOpponent(bot, args.opponent_level, {})
        if not opponent.available():
            sys.exit(
                f"opponent '{args.opponent}' is not available (missing "
                f"binary or network). Check bridge/bots/ setup, or test the "
                f"pipeline with --opponent py-random first.")

    if args.resume:
        if not os.path.isfile(args.resume):
            sys.exit(f"--resume: no checkpoint at {args.resume}")
        net = HATZ.load(args.resume)
        if net.hidden != args.hidden or net.layers != args.layers:
            print(f"resume: using checkpoint architecture "
                  f"hidden={net.hidden} layers={net.layers} "
                  f"(CLI --hidden/--layers ignored)")
        prev = net.meta.get("specs", [])
        print(f"resumed from {args.resume} "
              f"(previously trained on {prev or 'unknown'})")
    else:
        net = HATZ(hidden=args.hidden, layers=args.layers, seed=args.seed)

    seen_specs = sorted(set(net.meta.get("specs", [])) | set(keys)) \
        if args.resume else keys
    trained_vs = sorted(set(net.meta.get("trained_vs", [])) |
                        ({args.opponent} if vs_external else set()))
    net.meta = {"arch": "hatz", "version": 2, "specs": seen_specs,
                "surfaces": sorted({SPECS[k][0] for k in seen_specs}),
                "meshes": sorted({SPECS[k][1] for k in seen_specs}),
                "trained_vs": trained_vs}

    buf = deque(maxlen=args.buffer)
    rng = np.random.default_rng(args.seed)
    os.makedirs(CKPT_DIR, exist_ok=True)
    nparam = sum(v.size for v in net.p.values())
    nonor = [k for k in keys if boards[k][1].nonorientable]
    if vs_external:
        opp_desc = (f"vs {opponent.name} [{opponent.level}], "
                    f"record: {learn_from}, HATZ plays {args.hatz_plays}")
    else:
        opp_desc = "self-play"
    print(f"HATZ training on {keys} ({nparam} params, "
          f"non-orientable: {nonor or 'none'}); {args.sims} sims, "
          f"gauge-aug {'on' if args.gauge_aug else 'off'}; {opp_desc}")

    for it in range(1, args.iters + 1):
        t0 = time.time()
        winners = []
        hatz_scores = []
        for g in range(args.games_per_iter):
            key = keys[(it * args.games_per_iter + g) % len(keys)]
            gseed = args.seed * 100 + it * 17 + g
            if vs_external:
                if args.hatz_plays == "black":
                    hcolor = 1
                elif args.hatz_plays == "white":
                    hcolor = 2
                else:
                    hcolor = 1 if (it + g) % 2 == 0 else 2
                recs, winner, gmoves, hcolor = play_vs_game(
                    net, key, boards, opponent, args.sims, seed=gseed,
                    rng=rng, gauge_aug=args.gauge_aug, learn_from=learn_from,
                    hatz_color=hcolor, temp_moves=args.temp_moves,
                    root_noise=bool(args.root_noise), progress=args.progress,
                    opponent_timeout=args.opponent_timeout)
                hatz_scores.append(hatz_result(winner, hcolor))
                print(f"    game {g + 1}/{args.games_per_iter} {key}: "
                      f"{gmoves} moves, winner {winner}, "
                      f"HATZ={'B' if hcolor == 1 else 'W'} "
                      f"({hatz_result(winner, hcolor):.1f}), "
                      f"{time.time() - t0:.0f}s elapsed", flush=True)
            else:
                recs, winner, _ = self_play_game(
                    net, key, boards, args.sims, seed=gseed, rng=rng,
                    gauge_aug=args.gauge_aug, temp_moves=args.temp_moves)
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
                total += net.backward(C, r["pi"], r["z"], r["own"], grads,
                                      eown_t=r["eown"])
            for k2 in grads:
                grads[k2] /= len(batch)
            net.adam_step(grads, lr=args.lr)
            losses.append(total / len(batch))

        line = (f"iter {it:3d}  buffer {len(buf):5d}  "
                f"loss {np.mean(losses):.3f}  "
                f"B/W {winners.count('B')}/{winners.count('W')}  "
                f"{time.time() - t0:.0f}s")
        if vs_external and hatz_scores:
            line += f"  vs {args.opponent}: {np.mean(hatz_scores):.2f}"
        if it % args.eval_every == 0 or it == args.iters:
            wr = np.mean([eval_vs_random(net, k, boards, 6,
                                         seed=args.seed + it) for k in keys])
            line += f"  | vs random: {wr:.2f}"
            net.save(args.checkpoint)
            line += "  [checkpoint]"
        print(line, flush=True)

    net.save(args.checkpoint)
    print(f"saved {args.checkpoint} — bridge bot bots/hatz_mini.py serves it")


if __name__ == "__main__":
    main()
