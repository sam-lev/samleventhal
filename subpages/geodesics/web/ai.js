/*! GeoAI v0.1 — Hierarchical Graph Neural Network opponent for Geodesics.
 *
 * A forward-pass graph neural network over an arbitrary board graph (any
 * surface / mesh / resolution / dimension produced by make_board). The
 * adjacency list is the complete rule-relevant structure, so one engine
 * plays every topology, orientable or not.
 *
 *   tactical layer   exact per-candidate features from group analysis:
 *                    captures, atari escapes/threats, liberties-after,
 *                    self-atari, own-eye detection
 *   message passing  higher-order aggregation over BOTH the 1-ring (N1)
 *                    and the exact 2-ring (N2) neighborhoods, with tanh
 *                    nonlinearities between blocks
 *   hierarchy        two cooperating multi-scale structures:
 *                    (1) a multi-persistence Morse–Smale hierarchy of the
 *                        influence function — after Leventhal, Gyulassy,
 *                        Pascucci & Heimann, "Modeling Hierarchical
 *                        Topological Structure in Scientific Images with
 *                        Graph Neural Networks" (NeurIPS 2022) — treating
 *                        the board graph as the domain: steepest-ascent
 *                        basins are cancelled by topological persistence
 *                        into NESTED partitions on one shared vertex set
 *                        (same label at p_i ⇒ same label at p_j ≥ p_i),
 *                        and every message-passing block combines fine
 *                        neighborhoods with region messages from all
 *                        persistence levels — the paper's hierarchical
 *                        joint training (HJT) scheme with fixed weights
 *                    (2) a structural graph-U-Net coarsening pyramid via
 *                        greedy maximal matching (roughly the inverse of
 *                        geodesic subdivision), which still applies when
 *                        the board is empty and the field has no topology
 *   heads            per-vertex policy logits + a scalar value in (-1, 1)
 *   future estimate  optional 1-ply lookahead: top-k policy moves are
 *                    simulated exactly and re-ranked by value gain minus
 *                    the opponent's immediate reply threat
 *
 * Weights are hand-initialised Go priors. engine.setWeights(w) accepts a
 * trained parameter set with the identical schema (e.g. exported from the
 * Python layer after self-play training) — the architecture, not the
 * numbers, is fixed.
 *
 * Colors: 0 empty, 1 black, 2 white. Pass = -1.
 * Ko/superko legality belongs to the host: pass a legalMask into
 * pickMove(). Without one the engine only forbids occupied vertices,
 * suicide, and filling its own eyes.
 *
 * No dependencies. Browser: window.GeoAI. Node: module.exports.
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.GeoAI = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  // ---------- utilities ------------------------------------------------------

  function mulberry32(seed) {
    let t = seed >>> 0;
    return function () {
      t += 0x6d2b79f5;
      let r = Math.imul(t ^ (t >>> 15), 1 | t);
      r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
      return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
    };
  }

  function tanhInPlace(f) {
    for (let i = 0; i < f.length; i++) f[i] = Math.tanh(f[i]);
    return f;
  }

  function deepMerge(dst, src) {
    for (const k in src) {
      if (src[k] && typeof src[k] === "object" && !Array.isArray(src[k])) {
        if (!dst[k] || typeof dst[k] !== "object") dst[k] = {};
        deepMerge(dst[k], src[k]);
      } else dst[k] = src[k];
    }
    return dst;
  }

  // ---------- graph structure: rings and pyramid -----------------------------

  /** Sanitize an adjacency list: drop self-loops, duplicates, out-of-range. */
  function cleanAdjacency(neighbors) {
    const n = neighbors.length;
    const N1 = new Array(n);
    for (let v = 0; v < n; v++) {
      const s = new Set();
      const a = neighbors[v] || [];
      for (let i = 0; i < a.length; i++) {
        const u = a[i] | 0;
        if (u !== v && u >= 0 && u < n) s.add(u);
      }
      N1[v] = Array.from(s);
    }
    return N1;
  }

  /** Exact 2-ring (distance exactly 2) per vertex — the higher-order hop. */
  function buildRings(N1) {
    const n = N1.length;
    const N2 = new Array(n);
    const mark = new Int32Array(n).fill(-1);
    for (let v = 0; v < n; v++) {
      mark[v] = v;
      const nb = N1[v];
      for (let i = 0; i < nb.length; i++) mark[nb[i]] = v;
      const ring = [];
      for (let i = 0; i < nb.length; i++) {
        const w1 = N1[nb[i]];
        for (let j = 0; j < w1.length; j++) {
          const w = w1[j];
          if (mark[w] !== v) {
            mark[w] = v;
            ring.push(w);
          }
        }
      }
      N2[v] = ring;
    }
    return N2;
  }

  /** One coarsening step by greedy maximal matching (deterministic). */
  function coarsen(N1) {
    const n = N1.length;
    const assign = new Int32Array(n).fill(-1);
    let c = 0;
    for (let v = 0; v < n; v++) {
      if (assign[v] !== -1) continue;
      let mate = -1;
      const nb = N1[v];
      for (let i = 0; i < nb.length; i++)
        if (assign[nb[i]] === -1) {
          mate = nb[i];
          break;
        }
      assign[v] = c;
      if (mate !== -1) assign[mate] = c;
      c++;
    }
    const sets = new Array(c);
    for (let i = 0; i < c; i++) sets[i] = new Set();
    for (let v = 0; v < n; v++) {
      const nb = N1[v];
      for (let i = 0; i < nb.length; i++) {
        const a = assign[v], b = assign[nb[i]];
        if (a !== b) sets[a].add(b);
      }
    }
    return { assign, N1: sets.map((s) => Array.from(s)), size: c };
  }

  /** Coarsening pyramid + composed fine→level-l vertex maps. */
  function buildPyramid(N1, maxLevels, minSize) {
    const levels = [{ N1: N1, n: N1.length }];
    while (levels.length < maxLevels && levels[levels.length - 1].n > minSize) {
      const top = levels[levels.length - 1];
      const c = coarsen(top.N1);
      if (c.size >= top.n) break;
      top.assign = c.assign;
      levels.push({ N1: c.N1, n: c.size });
    }
    const mapTo = [null]; // level 0 is the identity
    let prev = null;
    for (let l = 1; l < levels.length; l++) {
      const a = levels[l - 1].assign;
      const m = new Int32Array(levels[0].n);
      for (let v = 0; v < m.length; v++) m[v] = a[prev ? prev[v] : v];
      mapTo.push(m);
      prev = m;
    }
    return { levels, mapTo };
  }

  // ---------- multi-persistence Morse–Smale hierarchy ------------------------
  // Graph-native analogue of the topological graph hierarchy in Leventhal,
  // Gyulassy, Pascucci & Heimann (NeurIPS 2022), with the board graph as the
  // discrete domain and the influence function as the scalar field.

  /**
   * Ascending-basin hierarchy of a scalar function f on the graph.
   * Steepest ascent assigns every vertex to a local maximum (ties broken by
   * index, giving a total order on plateaus). A descending union-find sweep
   * then pairs each maximum with the merge value that cancels it — its
   * topological persistence — recording which surviving maximum absorbs it.
   * labelsAt(p) cancels every maximum of persistence < p, so raising p only
   * merges regions: the nested multi-persistence hierarchy of the paper,
   * with all levels living on one shared vertex set.
   */
  function basinHierarchy(f, N1) {
    const n = f.length;
    const higher = (a, b) => f[a] > f[b] || (f[a] === f[b] && a > b);
    const up = new Int32Array(n).fill(-1);
    for (let v = 0; v < n; v++) {
      let best = v;
      const nb = N1[v];
      for (let i = 0; i < nb.length; i++) if (higher(nb[i], best)) best = nb[i];
      if (best !== v) up[v] = best;
    }
    const basin = new Int32Array(n).fill(-1);
    for (let v = 0; v < n; v++) {
      let x = v;
      const path = [];
      while (basin[x] === -1 && up[x] !== -1) {
        path.push(x);
        x = up[x];
      }
      const m = basin[x] === -1 ? x : basin[x];
      basin[x] = m;
      for (let i = 0; i < path.length; i++) basin[path[i]] = m;
    }
    // descending sweep: components labelled by their highest maximum
    const order = Array.from({ length: n }, (_, v) => v).sort((a, b) => (higher(a, b) ? -1 : 1));
    const parent = new Int32Array(n).fill(-1);
    const cmax = new Int32Array(n).fill(-1);
    const pers = new Float64Array(n).fill(Infinity); // indexed by maximum vertex
    const mergedInto = new Int32Array(n).fill(-1);
    const find = (x) => {
      while (parent[x] !== x) {
        parent[x] = parent[parent[x]];
        x = parent[x];
      }
      return x;
    };
    const maxima = [];
    for (let oi = 0; oi < n; oi++) {
      const v = order[oi];
      parent[v] = v;
      cmax[v] = v;
      if (up[v] === -1) maxima.push(v);
      const nb = N1[v];
      for (let i = 0; i < nb.length; i++) {
        const u = nb[i];
        if (parent[u] === -1) continue; // still below the sweep
        const ru = find(u);
        const rv = find(v);
        if (ru === rv) continue;
        let live = ru;
        let die = rv;
        if (higher(cmax[die], cmax[live])) {
          live = rv;
          die = ru;
        }
        pers[cmax[die]] = f[cmax[die]] - f[v];
        mergedInto[cmax[die]] = cmax[live];
        parent[die] = live;
      }
    }
    const finitePers = [];
    for (let i = 0; i < maxima.length; i++)
      if (pers[maxima[i]] < Infinity) finitePers.push(pers[maxima[i]]);
    finitePers.sort((a, b) => a - b);
    function labelsAt(p) {
      const lab = new Int32Array(n);
      for (let v = 0; v < n; v++) {
        let m = basin[v];
        while (mergedInto[m] !== -1 && pers[m] < p) m = mergedInto[m];
        lab[v] = m;
      }
      return lab;
    }
    return { basin, up, maxima, finitePers, labelsAt };
  }

  // ---------- message passing -------------------------------------------------

  /**
   * `steps` rounds of h' = wSelf·h + w1·mean_{N1} h + w2·mean_{N2} h.
   * N2 may be null (1-hop only). Returns a fresh Float32Array.
   */
  function diffuse(h, N1, N2, steps, wSelf, w1, w2) {
    const n = h.length;
    let a = Float32Array.from(h);
    let b = new Float32Array(n);
    for (let t = 0; t < steps; t++) {
      for (let v = 0; v < n; v++) {
        const l1 = N1[v];
        let s1 = 0;
        for (let i = 0; i < l1.length; i++) s1 += a[l1[i]];
        const m1 = l1.length ? s1 / l1.length : 0;
        let m2 = 0;
        if (N2 && w2 !== 0) {
          const l2 = N2[v];
          if (l2.length) {
            let s2 = 0;
            for (let i = 0; i < l2.length; i++) s2 += a[l2[i]];
            m2 = s2 / l2.length;
          }
        }
        b[v] = wSelf * a[v] + w1 * m1 + w2 * m2;
      }
      const tmp = a;
      a = b;
      b = tmp;
    }
    return a;
  }

  // ---------- rules kernel (exact; used for features and lookahead) ----------

  function makeKernel(N1) {
    const n = N1.length;
    const seen = new Int32Array(n);
    let stamp = 0;

    /** Vertices of the group at `start` if it has NO liberties, else null. */
    function deadGroup(s, start, color) {
      stamp++;
      const st = [start];
      seen[start] = stamp;
      const g = [];
      while (st.length) {
        const x = st.pop();
        g.push(x);
        const nb = N1[x];
        for (let i = 0; i < nb.length; i++) {
          const u = nb[i];
          if (s[u] === 0) return null;
          if (s[u] === color && seen[u] !== stamp) {
            seen[u] = stamp;
            st.push(u);
          }
        }
      }
      return g;
    }

    /** Exact move application with captures. Returns null on suicide/occupied. */
    function applyMove(stones, v, color) {
      if (stones[v] !== 0) return null;
      const s = Int8Array.from(stones);
      s[v] = color;
      const opp = 3 - color;
      const captured = [];
      const nb = N1[v];
      for (let i = 0; i < nb.length; i++) {
        const u = nb[i];
        if (s[u] === opp) {
          const g = deadGroup(s, u, opp);
          if (g) for (let j = 0; j < g.length; j++) { s[g[j]] = 0; captured.push(g[j]); }
        }
      }
      if (deadGroup(s, v, color)) return null; // suicide
      return { stones: s, captured };
    }

    /** Group decomposition with liberty sets. */
    function analyze(stones) {
      const gid = new Int32Array(n).fill(-1);
      const groups = [];
      for (let v = 0; v < n; v++) {
        if (stones[v] === 0 || gid[v] !== -1) continue;
        const color = stones[v];
        const id = groups.length;
        const g = { color, size: 0, libs: new Set() };
        groups.push(g);
        const st = [v];
        gid[v] = id;
        while (st.length) {
          const x = st.pop();
          g.size++;
          const nb = N1[x];
          for (let i = 0; i < nb.length; i++) {
            const u = nb[i];
            if (stones[u] === 0) g.libs.add(u);
            else if (stones[u] === color && gid[u] === -1) {
              gid[u] = id;
              st.push(u);
            }
          }
        }
      }
      return { gid, groups };
    }

    return { n, N1, deadGroup, applyMove, analyze };
  }

  // ---------- default parameters (hand-set Go priors) ------------------------

  const DEFAULT_WEIGHTS = {
    diffusion: {
      blocks: [
        { steps: 3, self: 0.5, n1: 0.36, n2: 0.14 },
        { steps: 2, self: 0.55, n1: 0.33, n2: 0.12 },
      ],
    },
    pyramid: { maxLevels: 4, minSize: 24, steps: 3, self: 0.5, n1: 0.5, mix: [1.0, 0.6, 0.38, 0.24] },
    msc: {
      enabled: true,
      minVerts: 8,   // skip the topological hierarchy on tiny boards
      minBasins: 3,  // ...and on fields too flat to have structure
      quantiles: [0, 0.6, 0.9], // persistence levels, finest → coarsest
      mix: [0.4, 0.28, 0.18],   // HJT combine weight per persistence level
      regionSmooth: 0.3,        // one message-passing step on each region graph
      smooth: { steps: 2, self: 0.5, n1: 0.36, n2: 0.14 }, // builds the scalar field
    },
    sigma: 0.35, // width of the "contested" band around M = 0
    head: {
      capture: 6.0,     // per stone captured
      escape: 5.0,      // per own stone rescued from atari
      atari: 1.2,       // per opponent stone put in atari
      libs: 1.2,        // liberties of the resulting group (saturating)
      frontier: 1.6,    // play where influence is contested
      grad: 1.1,        // play on steep influence gradients (moyo borders)
      mscBoundary: 0.9, // play on contested separatrices between opposing basins
      expand: 0.8,      // empty space in the 1- and 2-rings
      lowDeg: 0.35,     // penalty per missing degree vs the median (boundary lines)
      selfAtari: 5.0,   // penalty, scaled by the group being endangered
      passThresh: 0.35, // pass when ahead and nothing scores above this
      valueGain: 2.2,   // lookahead: weight of the value swing
      replyFear: 1.4,   // lookahead: weight of opponent's immediate reply threat
      lookBias: 0.15,   // lookahead: retain a little of the raw policy score
    },
    select: {
      casual:   { topk: 0,  temp: 1.1,  noise: 0.3 },
      standard: { topk: 8,  temp: 0.45, noise: 0.1 },
      strong:   { topk: 14, temp: 0.15, noise: 0.03 },
    },
  };

  // ---------- engine ----------------------------------------------------------

  /**
   * createEngine(neighbors, options?)
   *   neighbors : adjacency list, Array<Array<int>> — the play graph
   *   options   : { level: 'casual'|'standard'|'strong', seed, weights, select }
   */
  function createEngine(neighbors, options) {
    options = options || {};
    const N1 = cleanAdjacency(neighbors);
    const n = N1.length;
    const N2 = buildRings(N1);
    const weights = deepMerge(JSON.parse(JSON.stringify(DEFAULT_WEIGHTS)), options.weights || {});
    const pyr = buildPyramid(N1, weights.pyramid.maxLevels, weights.pyramid.minSize);
    const kernel = makeKernel(N1);
    const rng = mulberry32(options.seed == null ? 0x9e3779b9 : options.seed);
    const level = options.level || "standard";

    const deg = new Int32Array(n);
    for (let v = 0; v < n; v++) deg[v] = N1[v].length;
    const sorted = Array.from(deg).sort((a, b) => a - b);
    const medianDeg = n ? sorted[n >> 1] : 0;

    function selectCfg() {
      const base = weights.select[level] || weights.select.standard;
      return options.select ? deepMerge(Object.assign({}, base), options.select) : base;
    }

    function toInt8(a) {
      if (a instanceof Int8Array && a.length === n) return a;
      const s = new Int8Array(n);
      for (let v = 0; v < n; v++) s[v] = a[v] | 0;
      return s;
    }

    function normalizeMask(m) {
      if (!m) return null;
      if (typeof m === "function") return m;
      return (v) => !!m[v];
    }

    // -- multi-persistence topological hierarchy (per evaluation) --------------

    /** Compact region structure for one basin partition. */
    function makeRegions(labels, f) {
      const idOf = new Map();
      const rid = new Int32Array(n);
      for (let v = 0; v < n; v++) {
        let id = idOf.get(labels[v]);
        if (id === undefined) {
          id = idOf.size;
          idOf.set(labels[v], id);
        }
        rid[v] = id;
      }
      const R = idOf.size;
      const adjS = new Array(R);
      for (let i = 0; i < R; i++) adjS[i] = new Set();
      for (let v = 0; v < n; v++) {
        const nb = N1[v];
        for (let i = 0; i < nb.length; i++)
          if (rid[nb[i]] !== rid[v]) adjS[rid[v]].add(rid[nb[i]]);
      }
      const peak = new Float32Array(R);
      idOf.forEach((id, maxV) => {
        peak[id] = f[maxV];
      });
      return { R, rid, adj: adjS.map((s) => Array.from(s)), peak };
    }

    /**
     * The paper's nested hierarchy, computed on the fly for the current
     * position: ascending and descending basins of the influence function at
     * every persistence quantile, plus the finest-level contested
     * separatrices (vertices bordering a basin whose peak takes the opposite
     * sign — the black/white watershed).
     */
    function buildMscHierarchy(f0) {
      const C = weights.msc;
      if (!C.enabled || n < C.minVerts) return null;
      const asc = basinHierarchy(f0, N1);
      const neg = new Float32Array(n);
      for (let v = 0; v < n; v++) neg[v] = -f0[v];
      const desc = basinHierarchy(neg, N1);
      if (asc.maxima.length + desc.maxima.length < C.minBasins) return null;
      const persAll = asc.finitePers.concat(desc.finitePers).sort((a, b) => a - b);
      const q = (x) =>
        persAll.length ? persAll[Math.min(persAll.length - 1, Math.floor(x * persAll.length))] : Infinity;
      const levels = [];
      for (let i = 0; i < C.quantiles.length; i++) {
        const p = C.quantiles[i] <= 0 ? 0 : q(C.quantiles[i]);
        levels.push({ p, asc: makeRegions(asc.labelsAt(p), f0), desc: makeRegions(desc.labelsAt(p), neg) });
      }
      const L0 = levels[0];
      const boundary = new Float32Array(n);
      for (let v = 0; v < n; v++) {
        const mine = L0.asc.peak[L0.asc.rid[v]];
        const nb = N1[v];
        for (let i = 0; i < nb.length; i++) {
          const r = L0.asc.rid[nb[i]];
          if (r !== L0.asc.rid[v] && (mine * L0.asc.peak[r] <= 0 || Math.abs(L0.asc.peak[r]) < 0.02)) {
            boundary[v] = 1;
            break;
          }
        }
      }
      return { levels, boundary };
    }

    /** Pool h over a region partition, one smoothing step on the region graph, unpool. */
    function regionMessage(h, reg, smooth) {
      const sum = new Float32Array(reg.R);
      const cnt = new Float32Array(reg.R);
      for (let v = 0; v < n; v++) {
        sum[reg.rid[v]] += h[v];
        cnt[reg.rid[v]]++;
      }
      const m = new Float32Array(reg.R);
      for (let i = 0; i < reg.R; i++) m[i] = cnt[i] ? sum[i] / cnt[i] : 0;
      const m2 = new Float32Array(reg.R);
      for (let i = 0; i < reg.R; i++) {
        const a = reg.adj[i];
        let s = 0;
        for (let j = 0; j < a.length; j++) s += m[a[j]];
        m2[i] = a.length ? (1 - smooth) * m[i] + (smooth * s) / a.length : m[i];
      }
      const out = new Float32Array(n);
      for (let v = 0; v < n; v++) out[v] = m2[reg.rid[v]];
      return out;
    }

    // -- feature network ------------------------------------------------------

    /** Health-weighted ±1 stone field from the mover's perspective. */
    function stoneField(stones, color, A) {
      const f = new Float32Array(n);
      for (let v = 0; v < n; v++) {
        const c = stones[v];
        if (!c) continue;
        const g = A.groups[A.gid[v]];
        const health = Math.min(g.libs.size, 3) / 3;
        f[v] = (c === color ? 1 : -1) * (0.4 + 0.6 * health);
      }
      return f;
    }

    /** Multi-scale influence field M, contested-frontier map, gradient, value. */
    function computeFields(stones, color, A) {
      const base = stoneField(stones, color, A);

      // the scalar function whose topology drives the hierarchy: a light
      // higher-order smoothing of the stone field
      const S = weights.msc.smooth;
      const f0 = diffuse(base, N1, N2, S.steps, S.self, S.n1, S.n2);
      const msc = buildMscHierarchy(f0);

      // fine level: higher-order (N1 + N2) message-passing blocks, each one
      // jointly combined — HJT-style — with region messages from every
      // persistence level of the hierarchy before the nonlinearity
      let fine = base;
      for (let bi = 0; bi < weights.diffusion.blocks.length; bi++) {
        const b = weights.diffusion.blocks[bi];
        fine = diffuse(fine, N1, N2, b.steps, b.self, b.n1, b.n2);
        if (msc) {
          const mixes = weights.msc.mix;
          for (let li = 0; li < msc.levels.length; li++) {
            const w = mixes[Math.min(li, mixes.length - 1)];
            if (!w) continue;
            const L = msc.levels[li];
            const ma = regionMessage(fine, L.asc, weights.msc.regionSmooth);
            const md = regionMessage(fine, L.desc, weights.msc.regionSmooth);
            for (let v = 0; v < n; v++) fine[v] += w * 0.5 * (ma[v] + md[v]);
          }
        }
        tanhInPlace(fine);
      }

      // pyramid: pool base to every coarse level, diffuse there, unpool
      const P = weights.pyramid;
      const M = new Float32Array(n);
      for (let v = 0; v < n; v++) M[v] = P.mix[0] * fine[v];
      for (let l = 1; l < pyr.levels.length; l++) {
        const mixW = P.mix[Math.min(l, P.mix.length - 1)];
        const map = pyr.mapTo[l];
        const cn = pyr.levels[l].n;
        const sum = new Float32Array(cn);
        const cnt = new Float32Array(cn);
        for (let v = 0; v < n; v++) {
          sum[map[v]] += base[v];
          cnt[map[v]]++;
        }
        let cf = new Float32Array(cn);
        for (let i = 0; i < cn; i++) cf[i] = cnt[i] ? sum[i] / cnt[i] : 0;
        cf = diffuse(cf, pyr.levels[l].N1, null, P.steps, P.self, P.n1, 0);
        tanhInPlace(cf);
        for (let v = 0; v < n; v++) M[v] += mixW * cf[map[v]];
      }
      tanhInPlace(M);

      const frontier = new Float32Array(n);
      const grad = new Float32Array(n);
      const inv2s2 = 1 / (2 * weights.sigma * weights.sigma);
      let V = 0;
      for (let v = 0; v < n; v++) {
        frontier[v] = Math.exp(-M[v] * M[v] * inv2s2);
        const nb = N1[v];
        let g = 0;
        for (let i = 0; i < nb.length; i++) g += Math.abs(M[v] - M[nb[i]]);
        grad[v] = nb.length ? g / nb.length : 0;
        V += M[v];
      }
      V /= Math.max(1, n);
      return { M, frontier, grad, V, msc };
    }

    /** Exact tactical features per empty vertex (null = occupied or suicide). */
    function candidateFeatures(stones, color, A) {
      const feats = new Array(n).fill(null);
      const libSet = new Set();
      for (let v = 0; v < n; v++) {
        if (stones[v] !== 0) continue;
        const nb = N1[v];
        let emptyN1 = 0;
        const ownG = new Set();
        const oppG = new Set();
        for (let i = 0; i < nb.length; i++) {
          const u = nb[i];
          const c = stones[u];
          if (c === 0) emptyN1++;
          else if (c === color) ownG.add(A.gid[u]);
          else oppG.add(A.gid[u]);
        }
        let captures = 0;
        let atariThreat = 0;
        oppG.forEach((gi) => {
          const g = A.groups[gi];
          if (g.libs.size === 1) captures += g.size;
          else if (g.libs.size === 2) atariThreat += g.size;
        });
        libSet.clear();
        for (let i = 0; i < nb.length; i++) if (stones[nb[i]] === 0) libSet.add(nb[i]);
        let mergedSize = 1;
        let ownInAtari = 0;
        let minOwnLibs = Infinity;
        ownG.forEach((gi) => {
          const g = A.groups[gi];
          mergedSize += g.size;
          if (g.libs.size === 1) ownInAtari += g.size;
          if (g.libs.size < minOwnLibs) minOwnLibs = g.libs.size;
          g.libs.forEach((l) => libSet.add(l));
        });
        libSet.delete(v);
        let libsAfter = libSet.size;
        if (captures > 0) libsAfter = Math.max(libsAfter, Math.min(captures, 2));
        if (captures === 0 && libsAfter === 0) continue; // suicide → stays null
        const selfAtari = captures === 0 && libsAfter === 1;
        const escape = ownInAtari > 0 && (libsAfter >= 2 || captures > 0) ? ownInAtari : 0;
        const eyeFill =
          emptyN1 === 0 && oppG.size === 0 && captures === 0 && ownG.size > 0 && minOwnLibs >= 2;
        const r2 = N2[v];
        let e2 = 0;
        for (let i = 0; i < r2.length; i++) if (stones[r2[i]] === 0) e2++;
        const expansion = (emptyN1 + 0.5 * e2) / Math.max(1, nb.length + 0.5 * r2.length);
        feats[v] = { captures, atariThreat, libsAfter, selfAtari, escape, eyeFill, expansion, mergedSize };
      }
      return feats;
    }

    /** Policy head: linear readout over tactical features + network fields. */
    function headScore(f, fields, v) {
      const W = weights.head;
      let s = 0;
      s += W.capture * Math.min(f.captures, 10);
      s += W.escape * Math.min(f.escape, 10);
      s += W.atari * Math.min(f.atariThreat, 6);
      s += (W.libs * Math.min(f.libsAfter, 6)) / 6;
      s += W.frontier * fields.frontier[v];
      s += W.grad * Math.min(fields.grad[v] * 4, 1);
      if (fields.msc) s += W.mscBoundary * fields.msc.boundary[v];
      s += W.expand * f.expansion;
      s -= W.lowDeg * Math.max(0, medianDeg - deg[v]);
      if (f.selfAtari) s -= W.selfAtari * (1 + f.mergedSize / 4);
      return s;
    }

    /** Cheap position value used inside the 1-ply lookahead. */
    function quickValue(stones, color) {
      const A = kernel.analyze(stones);
      const base = stoneField(stones, color, A);
      const f = diffuse(base, N1, null, 2, 0.55, 0.45, 0);
      let V = 0;
      for (let v = 0; v < n; v++) V += Math.tanh(1.8 * f[v]);
      V /= Math.max(1, n);
      if (pyr.levels.length > 1) {
        const map = pyr.mapTo[1];
        const cn = pyr.levels[1].n;
        const sum = new Float32Array(cn);
        const cnt = new Float32Array(cn);
        for (let v = 0; v < n; v++) {
          sum[map[v]] += base[v];
          cnt[map[v]]++;
        }
        let cf = new Float32Array(cn);
        for (let i = 0; i < cn; i++) cf[i] = cnt[i] ? sum[i] / cnt[i] : 0;
        cf = diffuse(cf, pyr.levels[1].N1, null, 3, 0.5, 0.5, 0);
        let CV = 0;
        for (let i = 0; i < cn; i++) CV += Math.tanh(2.2 * cf[i]);
        CV /= cn;
        V = 0.5 * V + 0.5 * CV;
      }
      return V;
    }

    /** Fraction of own stones the opponent could capture immediately. */
    function replyThreat(stones, color) {
      const A = kernel.analyze(stones);
      let t = 0;
      for (let i = 0; i < A.groups.length; i++) {
        const g = A.groups[i];
        if (g.color === color && g.libs.size === 1) t += g.size;
      }
      return Math.min(t, 8) / 8;
    }

    /** Score every playable vertex. eyeFill entries carry score -1e9. */
    function evaluatePolicy(stones, color, isLegal, noise) {
      const A = kernel.analyze(stones);
      const fields = computeFields(stones, color, A);
      const feats = candidateFeatures(stones, color, A);
      const list = [];
      let sawLegal = false;
      for (let v = 0; v < n; v++) {
        if (stones[v] !== 0) continue;
        if (isLegal && !isLegal(v)) continue;
        const f = feats[v];
        if (!f) continue; // suicide
        sawLegal = true;
        const s = f.eyeFill ? -1e9 : headScore(f, fields, v) + noise * (rng() - 0.5) * 2;
        list.push({ v, s, f });
      }
      list.sort((a, b) => b.s - a.s);
      return { fields, list, sawLegal };
    }

    function softmaxSample(cands, temp) {
      if (cands.length === 1 || temp <= 1e-4) return cands[0].v;
      const m = cands[0].s;
      const ps = new Array(cands.length);
      let Z = 0;
      for (let i = 0; i < cands.length; i++) {
        const p = Math.exp((cands[i].s - m) / temp);
        ps[i] = p;
        Z += p;
      }
      let r = rng() * Z;
      for (let i = 0; i < cands.length; i++) {
        r -= ps[i];
        if (r <= 0) return cands[i].v;
      }
      return cands[0].v;
    }

    // -- public API -----------------------------------------------------------

    /**
     * pickMove(stones, color, { legalMask }) → { move, value, reason?, candidates? }
     *   stones    : array-like of 0/1/2 per vertex
     *   color     : 1 or 2 (the engine's color, to move)
     *   legalMask : Uint8Array | Array | (v)=>bool — host ko/superko legality
     *   move      : vertex index, or -1 for pass
     */
    function pickMove(stonesIn, color, opts) {
      opts = opts || {};
      const stones = toInt8(stonesIn);
      const isLegal = normalizeMask(opts.legalMask);
      const sel = selectCfg();
      const { fields, list, sawLegal } = evaluatePolicy(stones, color, isLegal, sel.noise);
      const cands = list.filter((c) => c.s > -1e8);
      if (!cands.length)
        return { move: -1, value: fields.V, reason: sawLegal ? "only-eye-fills" : "no-legal-moves" };
      if (cands[0].s < weights.head.passThresh && fields.V > 0.12)
        return { move: -1, value: fields.V, reason: "nothing-gains" };

      let ranked = cands;
      if (sel.topk > 0 && n > 2) {
        const K = Math.min(sel.topk, cands.length);
        const baseQ = quickValue(stones, color);
        const evals = [];
        for (let i = 0; i < K; i++) {
          const c = cands[i];
          const sim = kernel.applyMove(stones, c.v, color);
          if (!sim) continue;
          const dq = quickValue(sim.stones, color) - baseQ;
          const threat = replyThreat(sim.stones, color);
          evals.push({
            v: c.v,
            s: weights.head.valueGain * dq - weights.head.replyFear * threat + weights.head.lookBias * c.s,
          });
        }
        if (evals.length) {
          evals.sort((a, b) => b.s - a.s);
          ranked = evals;
        }
      }
      const move = softmaxSample(ranked, sel.temp);
      return {
        move,
        value: fields.V,
        candidates: ranked.slice(0, 5).map((c) => ({ v: c.v, score: c.s })),
      };
    }

    /** Raw policy scores (sorted desc) — handy for tests and heatmaps. */
    function scoreMoves(stonesIn, color, opts) {
      opts = opts || {};
      const stones = toInt8(stonesIn);
      const { list } = evaluatePolicy(stones, color, normalizeMask(opts.legalMask), 0);
      return list.map((c) => ({ v: c.v, score: c.s, eyeFill: !!c.f.eyeFill, selfAtari: !!c.f.selfAtari }));
    }

    /** Influence field + value for the given side — for UI heatmaps. */
    function evaluate(stonesIn, color) {
      const stones = toInt8(stonesIn);
      const A = kernel.analyze(stones);
      const fields = computeFields(stones, color, A);
      return {
        value: fields.V,
        M: fields.M,
        frontier: fields.frontier,
        msc: fields.msc
          ? {
              boundary: fields.msc.boundary,
              ascRid: fields.msc.levels[0].asc.rid,
              descRid: fields.msc.levels[0].desc.rid,
              levels: fields.msc.levels.map((L) => ({
                p: L.p,
                ascRegions: L.asc.R,
                descRegions: L.desc.R,
              })),
            }
          : null,
      };
    }

    return {
      n,
      levels: pyr.levels.map((l) => l.n),
      level,
      pickMove,
      scoreMoves,
      evaluate,
      kernel,
      setWeights: (w) => deepMerge(weights, w || {}),
      getWeights: () => JSON.parse(JSON.stringify(weights)),
    };
  }

  // ---------- model registry ---------------------------------------------------

  const models = [
    {
      id: "hgnn1",
      name: "Hierarchical GNN",
      description:
        "Fixed-weight hierarchical graph network: higher-order (1- and 2-ring) message passing joined across a multi-persistence Morse–Smale hierarchy of the influence field (Leventhal et al., NeurIPS 2022), a matching-coarsened structural pyramid, exact tactical features, and 1-ply future estimation.",
      levels: ["casual", "standard", "strong"],
      create: (neighbors, opts) => createEngine(neighbors, opts),
    },
  ];

  return {
    version: "0.2.0",
    createEngine,
    makeKernel: (neighbors) => makeKernel(cleanAdjacency(neighbors)),
    models,
    _internals: { buildRings, buildPyramid, coarsen, diffuse, cleanAdjacency, basinHierarchy },
  };
});
