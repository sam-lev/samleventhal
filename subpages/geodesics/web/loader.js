/*! GeoModels v0.1 — portable model cards for Geodesics.
 *
 * A model is DATA, not code: a single self-describing JSON "card"
 *
 *   { "format": "geo-model-1",          the schema tag (required)
 *     "arch":   "zero" | "hatz",        which shipped RUNTIME evaluates it
 *     "id":     "zero-all",             unique; re-loading replaces
 *     "name":   "Zero (all surfaces)",  shown in the Opponent menu
 *     "levels": ["casual", "standard"], strength names
 *     "supports": { surfaces:[..], meshes:[..], incidence:[..],
 *                   scaleIdx:[..] } | null,      board whitelist (null = any)
 *     "hidden": 32, "layers": 3,        architecture shape
 *     "meta":   { ... },                training provenance (informational)
 *     "params": { name: nested lists }, the weights themselves
 *     "code":   { hatz: url, hgnn: url } }       (hatz cards only, see below)
 *
 * Cards are produced by the training scripts (train_zero.py --export-model,
 * train_hatz.py --export-model) and can be loaded three ways in the app:
 * picked as a file, fetched from a pasted URL, or auto-fetched from a
 * #model=<url> link — which is the whole phone story: tap a link, the model
 * appears in the Opponent menu.
 *
 * SECURITY MODEL. Only weights are ever loaded, never JavaScript: the
 * "arch" field dispatches to a runtime that shipped WITH this page. An
 * untrusted card can at worst supply bad numbers; the app's pickMove path
 * already degrades malformed replies to a pass, never to a corrupted board.
 * (The one nuance is the hatz runtime, which fetches the two *published*
 * python source files named by the card into a sandboxed WebAssembly
 * interpreter — see the hatz section below for why that is still
 * weights-shaped trust, and how to pin the sources.)
 *
 * RUNTIMES.
 *   zero   pure JavaScript. The graph-net forward pass below mirrors
 *          python/train/zeronet.py line for line (it is the same code that
 *          train_zero.py used to inline into every exported model; now it
 *          lives here once and models are data).
 *   hatz   the real python. HATZ's forward pass (Kruskal sweeps,
 *          persistence pairing, Morse routing) has data-dependent control
 *          flow that no tensor-graph export survives, so instead of a
 *          drift-prone JS port the runtime runs hatz.py itself in a Web
 *          Worker under Pyodide (CPython compiled to WebAssembly): exact
 *          training parity, zero porting, zero server. First use lazily
 *          downloads the interpreter (~10-20 MB, cached by the browser
 *          thereafter); boards here are <= a few hundred sites, so per-move
 *          latency stays comfortable even on a phone.
 *
 * No dependencies. Browser: window.GeoModels. Node: module.exports (the
 * zero runtime and all validation are testable headlessly; the hatz
 * runtime requires Worker/fetch and reports itself unavailable without
 * them rather than throwing).
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.GeoModels = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  const FORMAT = "geo-model-1";

  // ---------------------------------------------------------------------
  // small helpers
  // ---------------------------------------------------------------------

  // Is x a rectangular numeric matrix of shape [rows][cols]?
  function isMatrix(x, rows, cols) {
    if (!Array.isArray(x) || x.length !== rows) return false;
    for (const r of x)
      if (!Array.isArray(r) || r.length !== cols ||
          r.some(v => typeof v !== "number" || !isFinite(v))) return false;
    return true;
  }
  function isVector(x, n) {
    return Array.isArray(x) && x.length === n &&
           x.every(v => typeof v === "number" && isFinite(v));
  }

  // ---------------------------------------------------------------------
  // runtime: zero — the graph policy/value net, natively in JS
  // ---------------------------------------------------------------------
  // Forward pass identical to python/train/zeronet.py (and to the model
  // template train_zero.py previously inlined per export): five input
  // features per vertex -> D hidden, L rounds of self+mean-neighbor message
  // passing, then a per-vertex policy head, a global pass logit and a tanh
  // value head, masked softmax over n+1 actions (index n = pass).

  function zeroForward(card, adj, stones, toMove, mask) {
    const P = card.params, L = card.layers, D = card.hidden;
    const n = adj.length;
    let md = 1;                                  // max degree, for the
    for (let v = 0; v < n; v++)                  // normalized degree feature
      md = Math.max(md, adj[v].length);
    let H = new Array(n);                        // features -> first layer
    for (let v = 0; v < n; v++) {
      const c = stones[v];
      const x = [c === toMove ? 1 : 0,           // own stone
                 (c && c !== toMove) ? 1 : 0,    // opponent stone
                 c === 0 ? 1 : 0,                // empty
                 adj[v].length / md,             // relative degree
                 1];                             // bias channel
      const h = new Float64Array(D);
      for (let j = 0; j < D; j++) {
        let s = P.b0[j];
        for (let i = 0; i < 5; i++) s += x[i] * P.W0[i][j];
        h[j] = s > 0 ? s : 0;                    // ReLU
      }
      H[v] = h;
    }
    for (let l = 0; l < L; l++) {                // message-passing rounds
      const Ws = P["Ws" + l], Wn = P["Wn" + l], b = P["b" + (l + 1)];
      const AH = new Array(n);                   // mean over neighbors
      for (let v = 0; v < n; v++) {
        const m = new Float64Array(D), a = adj[v];
        for (let k = 0; k < a.length; k++) {
          const hu = H[a[k]];
          for (let j = 0; j < D; j++) m[j] += hu[j];
        }
        if (a.length) for (let j = 0; j < D; j++) m[j] /= a.length;
        AH[v] = m;
      }
      const H2 = new Array(n);                   // self + neighbor mix
      for (let v = 0; v < n; v++) {
        const h = new Float64Array(D);
        for (let j = 0; j < D; j++) {
          let s = b[j];
          for (let i = 0; i < D; i++)
            s += H[v][i] * Ws[i][j] + AH[v][i] * Wn[i][j];
          h[j] = s > 0 ? s : 0;
        }
        H2[v] = h;
      }
      H = H2;
    }
    const g = new Float64Array(D);               // global mean readout
    for (let v = 0; v < n; v++)
      for (let j = 0; j < D; j++) g[j] += H[v][j] / n;
    const logits = new Float64Array(n + 1);
    for (let v = 0; v < n; v++) {                // per-vertex policy head
      let s = P.bp[0];
      for (let j = 0; j < D; j++) s += H[v][j] * P.wp[j];
      logits[v] = s;
    }
    let pl = P.bq[0], u = P.bv[0];               // pass logit + value head
    for (let j = 0; j < D; j++) { pl += g[j] * P.wq[j]; u += g[j] * P.wv[j]; }
    logits[n] = pl;
    let mx = -Infinity;                          // masked softmax
    for (let i = 0; i <= n; i++) if (mask[i]) mx = Math.max(mx, logits[i]);
    let Z = 0;
    const probs = new Float64Array(n + 1);
    for (let i = 0; i <= n; i++) {
      probs[i] = mask[i] ? Math.exp(logits[i] - mx) : 0;
      Z += probs[i];
    }
    for (let i = 0; i <= n; i++) probs[i] /= Z || 1;
    return { probs, value: Math.tanh(u) };
  }

  const zeroRuntime = {
    // Structural validation: every named parameter must exist with the
    // shape the forward pass will index, so a truncated or hand-edited
    // card fails at load time with a message, not mid-game with NaNs.
    validate(card) {
      const D = card.hidden, L = card.layers, P = card.params;
      if (!Number.isInteger(D) || D < 1 || !Number.isInteger(L) || L < 0)
        return "hidden/layers must be positive integers";
      if (!P || typeof P !== "object") return "missing params";
      if (!isMatrix(P.W0, 5, D) || !isVector(P.b0, D))
        return "input layer W0/b0 has the wrong shape";
      for (let l = 0; l < L; l++)
        if (!isMatrix(P["Ws" + l], D, D) || !isMatrix(P["Wn" + l], D, D) ||
            !isVector(P["b" + (l + 1)], D))
          return "message-passing layer " + l + " has the wrong shape";
      for (const [w, b] of [["wp", "bp"], ["wq", "bq"], ["wv", "bv"]])
        if (!isVector(P[w], D) || !isVector(P[b], 1))
          return "head " + w + "/" + b + " has the wrong shape";
      return null;
    },
    // create() has the same contract as every GeoAI model entry: it is
    // handed the adjacency lists and returns { pickMove }. `hooks` is
    // unused here — the zero net reads nothing but the graph.
    create(card, neighbors, opts /* , hooks */) {
      const level = (opts && opts.level) || "standard";
      return {
        pickMove(stones, color, o) {
          const n = neighbors.length;
          const legal = (o && o.legalMask) || null;
          const mask = new Uint8Array(n + 1);
          mask[n] = 1;                            // passing is always legal
          for (let v = 0; v < n; v++)
            if (stones[v] === 0 && (!legal || legal[v])) mask[v] = 1;
          const { probs, value } = zeroForward(card, neighbors, stones,
                                               color, mask);
          let a = n;
          if (level === "casual") {               // sample the raw policy
            let r = Math.random();
            for (let i = 0; i <= n; i++) { r -= probs[i]; if (r <= 0) { a = i; break; } }
          } else {                                // argmax
            let best = -1;
            for (let i = 0; i <= n; i++)
              if (probs[i] > best) { best = probs[i]; a = i; }
          }
          return { move: a === n ? -1 : a,
                   value, reason: "zero policy v" + value.toFixed(2) };
        },
      };
    },
  };

  // ---------------------------------------------------------------------
  // runtime: hatz — python-in-the-browser via a Pyodide Web Worker
  // ---------------------------------------------------------------------
  // Lifecycle: ONE worker per page, created lazily the first time a HATZ
  // opponent actually has to move (so the base page pays nothing).
  // Boot sequence, all inside the worker: import the Pyodide loader from
  // its CDN, initialize the WebAssembly interpreter, install numpy, fetch
  // the two published source files (hatz.py, hgnn.py) named by the card
  // and write them into the interpreter's virtual filesystem, then run a
  // small glue script (below) that knows how to register cards and answer
  // genmove requests. Cards register their weights by rebuilding a HATZ
  // instance and assigning net.p — the exact inverse of --export-model.
  //
  // The protocol is deliberately the bridge protocol's shape: stateless
  // genmove requests carrying the whole position, so the worker (like a
  // bridge bot) can never desynchronize from the page, and Bundles are
  // cached inside python keyed by (surface, adjacency) since they are pure
  // functions of the board.
  //
  // Trust note: the card's code URLs decide WHICH hatz.py runs. That is
  // still weights-shaped trust in practice — the interpreter is sandboxed
  // WebAssembly inside a Worker with no DOM access, and its only output is
  // a move index that the host rules re-validate — but a deployment that
  // wants to pin the sources can set window.GEO_HATZ_CODE and cards'
  // "code" fields are then ignored.

  const PYODIDE_URL_DEFAULT = "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/";

  // Python glue, executed once inside the worker after hatz.py / hgnn.py
  // are on the virtual filesystem. Kept as a plain string so this module
  // stays a single dependency-free file.
  const HATZ_GLUE = [
    "import json",
    "import numpy as np",
    "from types import SimpleNamespace",
    "import hatz",
    "",
    "NETS = {}      # model id -> HATZ instance",
    "BUNDLES = {}   # (surface, adjacency-json) -> Bundle (pure board data)",
    "",
    "def register(card_json):",
    "    # Rebuild a HATZ net from a geo-model card: fresh architecture,",
    "    # then overwrite every parameter array (the inverse of",
    "    # train_hatz.py --export-model).",
    "    c = json.loads(card_json)",
    "    net = hatz.HATZ(hidden=int(c['hidden']), layers=int(c['layers']))",
    "    for k in net.p:",
    "        if k not in c['params']:",
    "            raise ValueError('card is missing parameter ' + k)",
    "        w = np.asarray(c['params'][k], dtype=float)",
    "        if w.shape != net.p[k].shape:",
    "            raise ValueError('parameter %s has shape %s, expected %s'",
    "                             % (k, w.shape, net.p[k].shape))",
    "        net.p[k] = w",
    "    net.meta = c.get('meta', {})",
    "    NETS[c['id']] = net",
    "    return 'ok'",
    "",
    "def _bundle(surface, neighbors, nx, ny, faces, coords):",
    "    # Bundles are static per board; cache them. When the page supplies",
    "    # the mesh's face list and coordinates the Bundle here is",
    "    # IDENTICAL to the one training built (face ranks, face-based",
    "    # orientation cocycle, curvature) — the web app's faces are",
    "    # verified face-for-face against python topology. Without them we",
    "    # fall back to synthesizing grid coordinates from nx/ny, which is",
    "    # still enough for the name-based cocycle to place the",
    "    # Mobius/Klein/RP2 seam (the bridge bots' trick).",
    "    key = (surface or '', len(faces or []), json.dumps(neighbors))",
    "    if key not in BUNDLES:",
    "        n = len(neighbors)",
    "        if coords is None and nx and ny:",
    "            coords = [(v % nx, v // nx) for v in range(n)]",
    "        board = SimpleNamespace(name=surface or 'remote', n=n,",
    "                                adj=[tuple(a) for a in neighbors],",
    "                                coords=coords,",
    "                                faces=[tuple(f) for f in faces]",
    "                                      if faces else None,",
    "                                params={}, meta={})",
    "        BUNDLES[key] = hatz.Bundle(board, surface)",
    "    return BUNDLES[key]",
    "",
    "def genmove(req_json):",
    "    # Stateless, bridge-shaped: the request carries the entire",
    "    # position. Returns {move, value, reason}; move -1 is a pass.",
    "    r = json.loads(req_json)",
    "    net = NETS[r['model']]",
    "    nbrs = r['neighbors']",
    "    n = len(nbrs)",
    "    stones = list(r['stones'])",
    "    legal = r.get('legalMask') or [1] * n",
    "    mask = np.zeros(n + 1)",
    "    mask[n] = 1",
    "    for v in range(n):",
    "        if stones[v] == 0 and legal[v]:",
    "            mask[v] = 1",
    "    bundle = _bundle(r.get('surface'), nbrs, r.get('nx', 0),",
    "                     r.get('ny', 0), r.get('faces'), r.get('coords'))",
    "    probs, value = net.policy_value(bundle, stones, r['toMove'], mask)",
    "    probs = probs * mask",
    "    if probs.sum() <= 0:",
    "        return json.dumps({'move': -1, 'value': 0.0,",
    "                           'reason': 'HATZ passes'})",
    "    if r.get('level') == 'casual':",
    "        probs = probs / probs.sum()",
    "        a = int(np.random.default_rng().choice(n + 1, p=probs))",
    "    else:",
    "        a = int(np.argmax(probs))",
    "    tag = ' \\u00b7 non-orientable seam active' \\",
    "        if bundle.nonorientable else ''",
    "    return json.dumps({'move': -1 if a == n else a,",
    "                       'value': float(value),",
    "                       'reason': 'HATZ policy, value %+.2f%s'",
    "                                 % (value, tag)})",
  ].join("\n");

  // The worker's own JavaScript, instantiated from a Blob URL so the
  // single-file build needs no separate worker asset. It boots Pyodide on
  // the first message, then answers {register} and {genmove} requests;
  // every reply carries the request id so the page can match promises.
  function hatzWorkerSource() {
    return [
      "let pyodide = null, booted = null;",
      "async function boot(cfg) {",
      "  importScripts(cfg.pyodideURL + 'pyodide.js');",
      "  pyodide = await loadPyodide({ indexURL: cfg.pyodideURL });",
      "  await pyodide.loadPackage('numpy');",
      "  for (const [name, url] of Object.entries(cfg.code)) {",
      "    const resp = await fetch(url);",
      "    if (!resp.ok) throw new Error('fetch ' + url + ': ' + resp.status);",
      "    pyodide.FS.writeFile(name + '.py', await resp.text());",
      "  }",
      "  pyodide.runPython(cfg.glue);",
      "}",
      "onmessage = async (e) => {",
      "  const m = e.data;",
      "  try {",
      "    if (m.type === 'boot') {",
      "      booted = boot(m.cfg);",
      "      await booted;",
      "      postMessage({ type: 'ready', id: m.id });",
      "      return;",
      "    }",
      "    await booted;                       // queue behind the boot",
      "    if (m.type === 'register') {",
      "      pyodide.globals.get('register')(JSON.stringify(m.card));",
      "      postMessage({ type: 'registered', id: m.id });",
      "    } else if (m.type === 'genmove') {",
      "      const out = pyodide.globals.get('genmove')(JSON.stringify(m.req));",
      "      postMessage({ type: 'move', id: m.id, result: JSON.parse(out) });",
      "    }",
      "  } catch (err) {",
      "    postMessage({ type: 'error', id: m.id,",
      "                  message: String(err && err.message || err) });",
      "  }",
      "};",
    ].join("\n");
  }

  // Page-side worker handle: lazy singleton + promise-per-request table.
  const hz = { worker: null, next: 1, pending: new Map(),
               registered: new Set() };

  function hatzAvailable() {
    return typeof Worker !== "undefined" && typeof Blob !== "undefined" &&
           typeof URL !== "undefined" && typeof fetch !== "undefined";
  }

  // Send one message and get one matching reply as a promise. The timeout
  // is generous on the first call (the interpreter download) and normal
  // afterwards; a timed-out or crashed worker rejects, which the app's
  // aiMove path turns into "AI unavailable — your move".
  function hzCall(msg, timeoutMs) {
    return new Promise((resolve, reject) => {
      const id = hz.next++;
      hz.pending.set(id, { resolve, reject });
      hz.worker.postMessage({ ...msg, id });
      setTimeout(() => {
        if (hz.pending.has(id)) {
          hz.pending.delete(id);
          reject(new Error("python runtime timeout"));
        }
      }, timeoutMs);
    });
  }

  function hzEnsureWorker(card) {
    if (hz.worker) return;
    const url = URL.createObjectURL(
      new Blob([hatzWorkerSource()], { type: "text/javascript" }));
    hz.worker = new Worker(url);
    hz.worker.onmessage = (e) => {
      const m = e.data;
      const p = hz.pending.get(m.id);
      if (!p) return;
      hz.pending.delete(m.id);
      if (m.type === "error") p.reject(new Error(m.message));
      else p.resolve(m.result);
    };
    hz.worker.onerror = (e) => {           // catastrophic: fail everything
      for (const p of hz.pending.values())
        p.reject(new Error(e.message || "python runtime crashed"));
      hz.pending.clear();
    };
    // Source pinning: a page-level override beats the card (deployments
    // that host their own hatz.py/hgnn.py set window.GEO_HATZ_CODE).
    const pageCode = (typeof self !== "undefined" && self.GEO_HATZ_CODE) ||
                     null;
    const code = pageCode || card.code ||
                 { hatz: "models/hatz.py", hgnn: "models/hgnn.py" };
    const pyodideURL = (typeof self !== "undefined" && self.GEO_PYODIDE_URL)
                       || card.pyodide || PYODIDE_URL_DEFAULT;
    // fire-and-forget: replies to later calls queue behind the boot
    hzCall({ type: "boot", cfg: { pyodideURL, code, glue: HATZ_GLUE } },
           300000).catch(() => { /* surfaced by the first genmove */ });
  }

  const hatzRuntime = {
    validate(card) {
      if (!hatzAvailable())
        return "this environment has no Worker/fetch (python runtime " +
               "unavailable)";
      if (!Number.isInteger(card.hidden) || !Number.isInteger(card.layers))
        return "hidden/layers must be integers";
      if (!card.params || typeof card.params !== "object")
        return "missing params";
      // Shape-checking HATZ's ~30 parameter tensors client-side would
      // duplicate the python constructor; register() inside the worker
      // does it exactly (against the real shapes) and its error message
      // is surfaced on the first move instead.
      return null;
    },
    create(card, neighbors, opts, hooks) {
      hzEnsureWorker(card);
      // Register lazily and only once per model id; the promise is shared
      // so concurrent games on the same model do not double-register.
      if (!hz.registered.has(card.id)) {
        hz.registered.add(card.id);
        hz.regPromise = hzCall({ type: "register", card }, 300000)
          .catch((e) => { hz.registered.delete(card.id); throw e; });
      }
      const level = (opts && opts.level) || "standard";
      return {
        pickMove(stones, color, o) {
          // The cocycle needs to know which surface it is on and, for
          // grid boards, the nx/ny lattice shape — the app supplies both
          // through the hooks it passed to install() (loader.js itself
          // never reads app state).
          const info = (hooks && hooks.boardInfo && hooks.boardInfo()) || {};
          const req = {
            model: card.id, level,
            surface: info.surface || null,
            nx: info.nx || 0, ny: info.ny || 0,
            // mesh faces + coordinates when the page has them: with these
            // the worker's Bundle matches the training Bundle exactly
            faces: info.faces || null,
            coords: info.coords || null,
            neighbors,
            stones: Array.from(stones),
            toMove: color,
            legalMask: Array.from((o && o.legalMask) || []),
          };
          return (hz.regPromise || Promise.resolve())
            .then(() => hzCall({ type: "genmove", req }, 300000));
        },
      };
    },
  };

  const runtimes = { zero: zeroRuntime, hatz: hatzRuntime };

  // ---------------------------------------------------------------------
  // validation + installation
  // ---------------------------------------------------------------------

  // Structural check of a parsed card. Returns an error string (for the
  // UI) or null when the card is sound.
  function validate(card) {
    if (!card || typeof card !== "object") return "not a JSON object";
    if (card.format !== FORMAT)
      return "not a " + FORMAT + " card (format: " +
             JSON.stringify(card.format) + ")";
    if (!runtimes[card.arch])
      return "unknown architecture '" + card.arch + "' (this page ships: " +
             Object.keys(runtimes).join(", ") + ")";
    if (typeof card.id !== "string" || !card.id || card.id.includes(":"))
      return "card needs a string id without ':'";
    if (card.supports !== undefined && card.supports !== null &&
        typeof card.supports !== "object")
      return "supports must be an object or null";
    return runtimes[card.arch].validate(card);
  }

  // Install a validated card into a GeoAI.models registry. `hooks` are
  // app-supplied callbacks runtimes may need (currently boardInfo() for
  // the hatz cocycle). Re-installing an id replaces the previous entry, so
  // "load a newer checkpoint" is just loading again. Returns the entry.
  function install(card, GeoAI, hooks) {
    const err = validate(card);
    if (err) throw new Error(err);
    const entry = {
      id: card.id,
      name: card.name || card.id,
      levels: (card.levels && card.levels.length) ? card.levels
                                                  : ["standard"],
      supports: card.supports || null,
      loaded: true,                    // marks user-loaded (listable) models
      arch: card.arch,
      create: (neighbors, opts) =>
        runtimes[card.arch].create(card, neighbors, opts, hooks || {}),
    };
    const i = GeoAI.models.findIndex(m => m.id === entry.id);
    if (i >= 0) GeoAI.models.splice(i, 1, entry);
    else GeoAI.models.push(entry);
    return entry;
  }

  // Parse card text (from a file or textarea). Throws with a message the
  // UI can show verbatim.
  function fromText(text) {
    let card;
    try { card = JSON.parse(text); }
    catch (e) { throw new Error("not valid JSON: " + e.message); }
    const err = validate(card);
    if (err) throw new Error(err);
    return card;
  }

  // Fetch a card by URL (the #model= path). Only http(s) is followed.
  async function fromURL(url) {
    if (!/^https?:\/\//i.test(url))
      throw new Error("model URLs must be http(s)");
    const resp = await fetch(url);
    if (!resp.ok) throw new Error("fetch failed: HTTP " + resp.status);
    return fromText(await resp.text());
  }

  // The user-loaded entries currently installed (for the Models sheet).
  function list(GeoAI) { return GeoAI.models.filter(m => m.loaded); }

  // Remove a loaded model by id. Returns true when something was removed.
  function remove(GeoAI, id) {
    const i = GeoAI.models.findIndex(m => m.loaded && m.id === id);
    if (i < 0) return false;
    GeoAI.models.splice(i, 1);
    return true;
  }

  return { version: "0.1.0", FORMAT, runtimes, validate, install,
           fromText, fromURL, list, remove,
           // internal, exposed for the headless test suite: the exact
           // python glue the worker runs, testable under plain CPython
           _hatzGlue: HATZ_GLUE };
});
