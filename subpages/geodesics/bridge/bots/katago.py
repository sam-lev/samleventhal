"""bots/katago.py — configurable KataGo adapter for the Geodesics bridge.

KataGoBot instances differ only in id/name/network (and optionally a
human-rank table), so a variant is a five-line file:

    from katago import KataGoBot, net
    BOT = KataGoBot("katago-b28", "KataGo b28", model=net("katago_b28.bin.gz"))

Networks live in bridge/nets/ (runtime *.bin.gz files from
katagotraining.org — not the raw .ckpt training checkpoints). A bot hides
itself from the app whenever its binary or network is missing.

Standard mode drives `katago analysis` with per-strength maxVisits and plays
the top move. Human mode (human_profiles given) instead asks for the raw
policy at 1 visit under a humanSLProfile (e.g. preaz_5k) and samples from it
at full temperature over the legal moves — per KataGo's Human SL guide, the
closest match to how a player of that rank actually plays.

This module also registers an env-driven generic slot: set
KATAGO_MODEL=/path/to/net.bin.gz (optionally KATAGO_BIN, KATAGO_CFG) to
enable it without writing a variant file.
"""

import json
import os
import random
import shutil
import subprocess
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
COLS = "ABCDEFGHJKLMNOPQRSTUVWXYZ"       # GTP columns skip 'I'
VISITS = {"casual": 24, "standard": 200, "strong": 1600}


def net(fname):
    """Path of a network file in bridge/nets/."""
    return os.path.join(HERE, "..", "nets", fname)


def v_to_gtp(v, nx, ny):
    if v < 0:
        return "pass"
    x, y = v % nx, v // nx
    return COLS[x] + str(ny - y)          # row numbers count from the bottom


def gtp_to_v(s, nx, ny):
    s = s.strip().upper()
    if s == "PASS":
        return -1
    x = COLS.index(s[0])
    y = ny - int(s[1:])
    return y * nx + x


class KataGoBot:
    def __init__(self, id, name, model=None, human_profiles=None,
                 supports=None, binary=None, cfg=None):
        self.id = id
        self.name = name
        self.model = model if model is not None \
            else os.environ.get("KATAGO_MODEL", "")
        self.human = human_profiles          # {level label: humanSLProfile}
        self.levels = (list(human_profiles) if human_profiles
                       else list(VISITS))
        self.supports = supports if supports is not None else {
            "surfaces": ["plane"], "meshes": ["square"],
            "incidence": ["vertices"]}
        self.binary = binary or os.environ.get("KATAGO_BIN", "katago")
        self.cfg = cfg or os.environ.get(
            "KATAGO_CFG", os.path.join(HERE, "..", "analysis.cfg"))
        self.proc = None
        self.lock = threading.Lock()
        self.qid = 0

    def available(self):
        binary_ok = (shutil.which(self.binary) is not None
                     or os.path.isfile(self.binary))
        if not binary_ok:
            return False
        if not (self.model and os.path.isfile(self.model)):
            if self.model:
                print(f"  {self.id}: network not found at {self.model}")
            return False
        return True

    def _ensure_proc(self):
        if self.proc is not None and self.proc.poll() is None:
            return
        self.proc = subprocess.Popen(
            [self.binary, "analysis", "-model", self.model,
             "-config", self.cfg],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1)

    def _query(self, query):
        self._ensure_proc()
        self.proc.stdin.write(json.dumps(query) + "\n")
        self.proc.stdin.flush()
        while True:
            line = self.proc.stdout.readline()
            if not line:
                self.proc = None
                raise RuntimeError(
                    f"{self.id}: katago exited (check network/config paths)")
            try:
                resp = json.loads(line)
            except ValueError:
                continue
            if resp.get("id") != query["id"]:
                continue
            if "error" in resp:
                raise RuntimeError(f"{self.id}: " + str(resp["error"]))
            return resp

    def genmove(self, req):
        spec, board = req["spec"], req["board"]
        nx, ny = spec["nx"], spec["ny"]
        cmap = {1: "B", 2: "W"}
        moves = [[cmap[c], v_to_gtp(v, nx, ny)] for c, v in board["moves"]]
        self.qid += 1
        query = {
            "id": f"q{self.qid}",
            "moves": moves,
            "rules": "tromp-taylor",
            "komi": 7.5,
            "boardXSize": nx,
            "boardYSize": ny,
        }
        level = req.get("level")

        if self.human:                        # rank-faithful policy sampling
            profile = self.human.get(level) or next(iter(self.human.values()))
            query["maxVisits"] = 1
            query["includePolicy"] = True
            query["overrideSettings"] = {"humanSLProfile": profile}
            resp = self._query(query)
            policy = resp.get("policy") or []
            # policy is row-major from the top-left — the same ordering as
            # our vertex index — with the pass logit last; illegal moves -1
            mask = board.get("legalMask") or []
            weights, sites = [], []
            for v in range(board["n"]):
                if board["stones"][v] == 0 and (not mask or mask[v]) \
                        and v < len(policy) and policy[v] > 0:
                    weights.append(policy[v])
                    sites.append(v)
            if not sites:
                return -1, f"{self.name} passes"
            mv = random.choices(sites, weights=weights, k=1)[0]
            return mv, f"{self.name} ({profile})"

        query["maxVisits"] = VISITS.get(level, VISITS["standard"])
        query["includePolicy"] = False
        resp = self._query(query)
        infos = resp.get("moveInfos") or []
        if not infos:
            return -1, f"{self.name} passes"
        mv = gtp_to_v(infos[0]["move"], nx, ny)
        wr = infos[0].get("winrate")
        info = (f"{self.name} {query['maxVisits']}v"
                + (f", winrate {wr:.0%}" if wr is not None else ""))
        return mv, info


# env-driven generic slot (hidden unless KATAGO_MODEL is set and valid)
BOT = KataGoBot("katago", "KataGo")
