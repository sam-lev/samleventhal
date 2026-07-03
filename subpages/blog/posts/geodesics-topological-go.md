*Geodesics is Go, but the board is no longer a flat 19×19 grid — it's a surface you choose. A sphere, a torus, a Möbius band, a 3D lattice; tiled with triangles, squares, or hexagons; at whatever resolution you like. The rules never change, because the rules never actually needed the geometry. **[Play it here.](../geodesics/geodesics.html)*** *(single self-contained HTML file — no install, works in any modern browser).*

## The idea

Go's rules are astonishingly simple: stones connect to their neighbors, a group with no adjacent empty points is captured, you can't repeat a whole-board position. Notice what's *missing* from that description — any mention of a square, an edge, a corner, or even two dimensions. The rules only ever refer to which points are **adjacent** to which. Everything strategic that we think of as intrinsic to Go — the value of the corner, crawling on the first line, the edge as a wall — is a consequence of one specific choice of board, not of the game itself.

So: keep the rules, swap the board. That's Geodesics. The engine only ever sees an **adjacency graph**, so the identical Tromp–Taylor rule kernel runs on a sphere, a torus, a non-orientable Möbius band, or a solid 3D box without a single special case.

## The design space

The board is assembled from four independent choices, each exposed as a control:

- **Surface** — the manifold the game lives on. A **sphere** has no edges and no corners at all, so there's no first line to crawl on and no cheap corner to live in; everything must be enclosed in the open. A **torus** is edgeless *and* vertex-transitive — every point is identical, so opening theory has to start from pure symmetry. A **Möbius band** is one-sided: a chain that crosses the seam comes back mirrored, and what looks like two rims is a single boundary circle of double length. A **box** puts Go in 3D, where a stone in open space has six liberties.
- **Mesh** — the tiling whose vertices are the playable points. **Triangular** gives every point degree 6 (thick, hard-to-kill groups); **square** is the classical degree-4 goban; **hexagonal** gives degree 3 (razor-sharp, first-line Go everywhere). Mesh type is arguably the biggest strategic dial after the topology itself, because it sets how many liberties a lone stone starts with.
- **Scale** — mesh resolution, from a few dozen points to a few hundred.
- **Paint** — how the board's structure is colored (more on this below).

On the sphere, these tilings turn into objects a chemist or a geodesic-dome builder would recognize. The triangular sphere is a **geodesic polyhedron** — the same construction as a geodesic dome. Its hexagonal dual is a **Goldberg polyhedron**: the soccer-ball / buckminsterfullerene pattern, hexagons with exactly twelve pentagons mixed in. The square sphere is a **cube-sphere**, a subdivided cube pushed out to a ball.

## The mathematics is the point, and it's on hover

Every one of those "exactly twelve pentagons" facts is forced by a theorem, and I wanted the game to *teach* that rather than hide it. So the board carries a readout — something like `S² · {3,5+}(4,0) · V 162 · χ 2 · ∂ 0` — and **every term is explainable in place**: hover for a one-line gloss, right-click (or long-press on mobile) for the full explanation.

Why can't you build a perfectly regular degree-6 triangular mesh on a sphere? Because the **Euler characteristic** χ = V − E + F = 2 forbids it. The discrete Gauss–Bonnet theorem says the total "defect" summed over the whole surface must equal 6χ = 12 — which is why a geodesic sphere always has exactly twelve degree-5 vertices, and a Goldberg sphere exactly twelve pentagons, no matter how finely you subdivide. On a torus, χ = 0, so the defect budget is zero and the lattice closes up perfectly regular. The game marks those forced defects in brass so you can see the topology asserting itself. The same hover-to-explain treatment covers the game rules (capture, positional superko, Benson's pass-alive test, Tromp–Taylor scoring) — each stated in the graph-theoretic form that makes it survive the change of topology.

Before your first move, those board readouts are **click-editable**: tap the surface, mesh, or scale term to cycle it and rebuild the board on the spot.

## Coloring the structure

The **Paint** control colors the board's incidence relationships, which is a nice way to *see* some graph theory:

- **Vertices** — a proper coloring where adjacent points always differ. The number of colors you need is the graph's chromatic number.
- **Edges** — a proper edge coloring, where edges meeting at a point differ (Vizing's theorem says Δ or Δ+1 colors suffice).
- **Cells** — faces colored so that neighbors across an edge differ. On quotient surfaces you can watch a checkerboard *fail to close up* across a seam — the topology breaking a coloring that works fine locally.

## Correspondence play, no server

A game is fully described by its board spec plus its move list, so **Share** encodes exactly that into a short code (and a URL). Send it to someone; they load it, play a move, and send the new code back. It's serverless correspondence Go on any topology — and because loading a code re-runs every move through the rules engine, a corrupted or illegal code fails loudly instead of silently desyncing.

## Under the hood

There are two implementations. The web game is a single self-contained HTML file (framework-free engine + topology core, a Three.js renderer). Behind it sits a tested Python reference engine — a five-layer architecture where the rule kernel provably never touches geometry, a `make_board(surface, mesh, resolution, dimension)` spec API, and a mesh module implementing **Conway polyhedron operators** so you can build boards like `t(I)` (the truncated icosahedron — a literal soccer ball) or `aa(D)` (the rhombicosidodecahedron). Board counts are verified against the canonical polyhedron catalogue, and the whole thing is covered by a passing test suite.

But you don't need any of that to play. Pick a surface, pick a tiling, and put a stone down somewhere that has never had a first line.

**[→ Play Geodesics](../geodesics/geodesics.html)**
