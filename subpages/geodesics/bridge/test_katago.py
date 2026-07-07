#!/usr/bin/env python3
"""Adapter tests that need no real KataGo: a fake `katago` binary speaks the
analysis-engine protocol, records the queries it receives, and returns canned
responses. Run:  python3 bridge/test_katago.py
"""

import json
import os
import random
import stat
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "bots"))
import katago  # noqa: E402

FAKE = r'''#!/usr/bin/env python3
import json, os, sys
log = open(os.environ["FAKE_LOG"], "a")
for line in sys.stdin:
    q = json.loads(line)
    log.write(line); log.flush()
    n = q["boardXSize"] * q["boardYSize"]
    if q.get("includePolicy"):
        pol = [0.01] * (n + 1)
        pol[7] = 0.9          # fake human favourite
        pol[3] = -1           # illegal per KataGo convention
        r = {"id": q["id"], "policy": pol,
             "moveInfos": [{"move": "C3", "winrate": 0.5}]}
    else:
        r = {"id": q["id"], "moveInfos": [{"move": "D4", "winrate": 0.61}]}
    sys.stdout.write(json.dumps(r) + "\n"); sys.stdout.flush()
'''

passed = failed = 0
def ok(cond, name):
    global passed, failed
    print(("PASS  " if cond else "FAIL  ") + name)
    passed, failed = passed + (1 if cond else 0), failed + (0 if cond else 1)

with tempfile.TemporaryDirectory() as td:
    fake_bin = os.path.join(td, "katago")
    with open(fake_bin, "w") as f:
        f.write(FAKE)
    os.chmod(fake_bin, os.stat(fake_bin).st_mode | stat.S_IEXEC)
    fake_net = os.path.join(td, "net.bin.gz")
    open(fake_net, "w").write("x")
    log = os.path.join(td, "queries.log")
    os.environ["FAKE_LOG"] = log

    def request(n=25):
        return {"level": "standard",
                "spec": {"nx": 5, "ny": 5},
                "board": {"n": n, "stones": [0] * n,
                          "legalMask": [1] * n, "toMove": 1,
                          "moves": [[1, 12], [2, -1]]}}

    # availability gating
    ghost = katago.KataGoBot("g", "G", model=os.path.join(td, "missing.bin.gz"),
                             binary=fake_bin)
    live = katago.KataGoBot("kg", "KG", model=fake_net, binary=fake_bin)
    ok(not ghost.available() and live.available(),
       "bot hides without its network, appears with it")

    # standard mode: top move, correct coordinates, correct query fields
    mv, info = live.genmove(request())
    q = json.loads(open(log).readlines()[-1])
    ok(mv == katago.gtp_to_v("D4", 5, 5) and q["rules"] == "tromp-taylor"
       and q["komi"] == 7.5 and q["maxVisits"] == 200
       and q["moves"] == [["B", "C3"], ["W", "pass"]],
       f"standard mode: plays D4 -> site {mv}; history/rules/visits correct")

    # human mode: profile override, 1 visit, samples only legal positive-policy
    human = katago.KataGoBot("kh", "KH", model=fake_net, binary=fake_bin,
                             human_profiles={"5 kyu": "preaz_5k"})
    random.seed(0)
    req = request()
    req["level"] = "5 kyu"
    req["board"]["stones"][7] = 0
    req["board"]["legalMask"][9] = 0          # forbid one site
    counts = {}
    for _ in range(40):
        mv, info = human.genmove(req)
        counts[mv] = counts.get(mv, 0) + 1
    q = json.loads(open(log).readlines()[-1])
    ok(q["maxVisits"] == 1 and q["includePolicy"] is True
       and q["overrideSettings"] == {"humanSLProfile": "preaz_5k"},
       "human mode: 1 visit, includePolicy, humanSLProfile override sent")
    ok(counts.get(7, 0) > 25 and 3 not in counts and 9 not in counts
       and all(0 <= m < 25 for m in counts),
       f"human mode: samples the policy (site 7 x{counts.get(7,0)}/40), "
       "never illegal or masked sites")

    ok(human.levels == ["5 kyu"] and live.levels == ["casual", "standard",
                                                     "strong"],
       "levels derive from mode")

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
