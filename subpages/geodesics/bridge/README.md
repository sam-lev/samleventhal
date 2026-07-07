# Geodesics bridge — local engines over WebSocket

Run beside the app, zero dependencies (Python 3.8+):

    python3 bridge/serve.py

Open `geodesics.html` (a `file://` open is fine) and the Opponent menu gains
every bot the bridge advertises — the app shows "bridge connected — N
engines". Stop the bridge and they vanish; the app quietly re-probes every
few seconds. The server binds 127.0.0.1 only.

## Adding a bot

Drop a file in `bridge/bots/` exposing a module-level `BOT` — see
`bots/random_bot.py` for the full annotated contract. In short: `genmove(req)`
receives the complete position (stones, host-computed legal mask, move
history, and the adjacency list of the *current play graph*, so one bot can
play every surface, mesh and incidence mode) and returns a site index or -1.
Restart the bridge; the bot appears. Pretrained PyTorch/JAX/etc. models load
at import time — the app only ever sees the returned integer, and its own
rules engine re-validates every move.

Declare `supports = {"surfaces": [...], "meshes": [...], "incidence": [...]}`
to gate a bot to boards it understands (omitted lists mean "any"); the app
greys it out elsewhere and falls back to Human if the board changes under it.
Define `available()` returning False to hide a bot whose binary or weights
are missing.

## KataGo

KataGo bots live on the classical plane boards (9×9 / 13×13 / 19×19 — its
CNNs only understand rectangles, so they are gated to
`plane:square:vertices`). Install KataGo (`brew install katago`, or a
release binary), then drop runtime **`*.bin.gz` network files** (from
katagotraining.org — the "Network file" links, not the raw `.ckpt`
checkpoints) into `bridge/nets/` under the names the variant bots expect:

    bots/katago_b28.py        -> nets/katago_b28.bin.gz        (strong)
    bots/katago_zhizi-b40.py  -> nets/katago_zhizi-b40.bin.gz  (strong)
    bots/katago_9x9-b18.py    -> nets/katago_9x9-b18.bin.gz    (9×9 only)
    bots/katago-humanv0.py    -> nets/katago-humanv0.bin.gz    (human ranks)

Restart the bridge; each bot appears exactly when its network (and the
`katago` binary) resolve. Strength levels map to 24 / 200 / 1600 visits.
The human bot instead offers rank levels (20 kyu … 9 dan) and samples the
raw human policy at 1 visit under a `humanSLProfile` — KataGo's recommended
recipe for rank-faithful play; adjust the profile table in the bot file if
your KataGo version names profiles differently. A generic env-driven slot
also exists: `export KATAGO_MODEL=/path/to/net.bin.gz` (optional
`KATAGO_BIN`, `KATAGO_CFG`) enables `bots/katago.py` without writing a file.
New variants are five lines — see any variant bot. History is replayed in
full with `rules: tromp-taylor`, komi 7.5, so KataGo's superko view matches
the app's. Test the adapter plumbing without a real KataGo:
`python3 bridge/test_katago.py`.

## Protocol (for other clients or servers)

    → {"type":"hello"}
    ← {"type":"models","models":[{"id","name","levels",[...],"supports"}...]}
    → {"type":"genmove","id":N,"model":"katago","level":"standard",
       "spec":{"surface","mesh","scaleIdx","incidence","nx","ny"},
       "board":{"n","neighbors","stones","toMove","legalMask","moves"}}
    ← {"type":"move","id":N,"move":site-or--1,"info":"..."}   (or "error")

Requests are stateless; the bridge may be restarted between moves.
