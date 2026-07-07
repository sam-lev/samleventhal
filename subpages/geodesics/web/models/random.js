// models/random.js — a drop-in opponent, and the template for writing more.
//
// Every models/*.js file is bundled by `node build.mjs` and runs after
// GeoAI loads. The build wraps this file in a closure that defines WEIGHTS:
// the parsed contents of a sibling `random.weights.json` if one exists,
// otherwise null. To ship a pretrained variant of the built-in engine:
//
//   GeoAI.models.push({
//     id: "hgnn1-sp1", name: "HGNN self-play v1", levels: ["standard"],
//     create(neighbors, opts) {
//       const e = GeoAI.createEngine(neighbors, opts);
//       if (WEIGHTS) e.setWeights(WEIGHTS);   // schema: engine.getWeights()
//       return e;
//     },
//   });
//
// The only contract: create(neighbors, opts) returns an object with
// pickMove(stones, color, { legalMask }) -> { move, reason? } where move is
// a site index or -1 for a pass (a Promise of that shape is also accepted).
// stones is 0/1/2 per site; legality is enforced by the host either way.
// Model ids must not contain ":". Optionally set supports:
// { surfaces: [...], meshes: [...], incidence: [...] } to gate availability.

GeoAI.models.push({
  id: "random",
  name: "Random",
  levels: ["uniform"],
  create(neighbors) {
    // deterministic per board size, so games are reproducible-ish
    let s = 0x9e3779b9 ^ neighbors.length;
    const rnd = () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 4294967296);
    return {
      pickMove(stones, color, opts) {
        const mask = opts && opts.legalMask;
        const legal = [];
        for (let v = 0; v < stones.length; v++)
          if (stones[v] === 0 && (!mask || mask[v])) legal.push(v);
        if (!legal.length) return { move: -1, reason: "no-legal-moves" };
        return { move: legal[(rnd() * legal.length) | 0], reason: "uniform" };
      },
    };
  },
});
