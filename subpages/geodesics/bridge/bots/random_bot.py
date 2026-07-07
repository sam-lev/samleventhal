"""bots/random_bot.py — the bot contract, demonstrated minimally.

A bot module exposes a module-level BOT object with:

  id        str, unique, no ':'
  name      str shown in the app's Opponent menu
  levels    list of strength names; the chosen one arrives in req["level"]
  supports  None for "any board", or a whitelist dict such as
            {"surfaces": ["plane"], "meshes": ["square"],
             "incidence": ["vertices"]}
  available()   optional; return False to hide the bot (missing binary, etc.)
  genmove(req)  return a site index, or -1 to pass, or (move, info_string)

The request is stateless and complete:

  req["spec"]   {surface, mesh, scaleIdx, incidence, nx, ny}
  req["board"]  {n, neighbors, stones, toMove, legalMask, moves}
                neighbors: adjacency list of the *current play graph* —
                  works verbatim on every surface, mesh and incidence mode
                stones:    0 empty / 1 black / 2 white, per site
                toMove:    the color you are choosing for (1 or 2)
                legalMask: 1 where the host rules allow a move (ko, superko
                  and suicide already excluded) — respect it
                moves:     full history [[color, site-or--1], ...]

Pretrained models: load weights at import time (e.g. torch.load beside this
file), build features from `neighbors`/`stones`, and return an argmax over
legal sites. Anything importable in your Python environment is fair game —
the app never sees more than the returned integer.
"""

import random


class RandomBot:
    id = "py-random"
    name = "Random (bridge)"
    levels = ["uniform"]
    supports = None                     # plays every board

    def genmove(self, req):
        board = req["board"]
        mask = board.get("legalMask") or []
        legal = [v for v in range(board["n"])
                 if board["stones"][v] == 0 and (not mask or mask[v])]
        if not legal:
            return -1
        return random.choice(legal)


BOT = RandomBot()
