# Geodesics — pilot implementation

Go, topologically abstracted. This is the **M0 reference engine** from the
design report: a pure-Python implementation of the five-layer architecture,
built to validate the abstract rule system and the mathematical spec before
the production Rust port. Zero runtime dependencies.

```
Layer 1  Topology      sphere, plane, cylinder/annulus, torus, Möbius band,
                       Klein bottle, projective plane; paths/cycles (1D)
                       and box / T^3 lattices (3D)
Layer 2  Mesh          three tilings per surface — tri (degree 6), square
                       (degree 4), hex (degree 3); Class-I geodesic and
                       quad subdivision; oriented-mesh module with Conway
                       operators (d k a t j) over Platonic seeds
Layer 3  Graph         Board = finite simple graph (+ optional coords/faces)
Layer 4  Rules         Tromp-Taylor kernel: capture, suicide policy,
                       positional/situational/simple superko (generalized
                       Zobrist), area scoring; Benson pass-alive
Layer 5  Session       Game: play/pass/undo/score, gSGF 0.1 serialization
```

The load-bearing property: **Layers 4–5 never see geometry.** A `Board`'s
adjacency lists are the complete rule-relevant structure, so every topology —
including non-orientable ones and 3D lattices — plays with the identical
engine.

## The board design space

`make_board` assembles a board from four orthogonal player choices:

```python
from geodesics import make_board, Game

b = make_board(
    surface="torus",     # sphere | plane | cylinder | torus | mobius | klein | rp2
    mesh="hex",          # tri (deg 6) | square (deg 4) | hex (deg 3)
    resolution=2,        # abstract scale, 1, 2, 3, ...
    dimension=2,         # 1 (path/cycle) | 2 (surfaces) | 3 (box, T^3)
)
Game(b).legal_moves()
```

Mesh type sets the local branching factor — the most strategy-shaping knob
after the topology itself. Degree-3 boards make liberties scarce everywhere
(the whole board plays like the first line); degree-6 boards make chains
hard to kill. On the sphere the three types become the three classical
"almost-regular" families, each with its Euler-mandated defects:

| mesh   | sphere realization                    | defects            |
|--------|---------------------------------------|--------------------|
| tri    | geodesic polyhedron {3,5+}_(r,0)      | 12 vertices, deg 5 |
| square | cube-sphere (subdivided cube)         | 8 vertices, deg 3  |
| hex    | Goldberg polyhedron GP(r,0)           | 12 pentagon faces  |

Approximate sizes: grid surfaces `(2r+3)^2`; sphere tri `10r^2+2`, square
`6(r+1)^2+2`, hex `20r^2`; 3D `(r+3)^3`; 1D `4r+5`. Exact sizes can be
pinned with `nx=`, `ny=`, `nz=`, `frequency=` overrides. Hex meshes carry
parity conditions across wrapped seams (even sides; odd width on the Möbius
band) and are obstructed on the Klein bottle / RP² in the pilot — the
constructor explains when refusing.

## Polyhedron boards (Conway operators)

`geodesics.mesh` implements oriented closed meshes with a rotation system
derived from face windings, Platonic seeds (T C O D I), and the Conway
operators **d**ual, **k**is, **a**mbo, **t**runcate (= dkd), **j**oin (= da)
— compositions give expand `aa`, bevel `ta`, ortho `jj`, meet `kj`. Boards
match the canonical catalogue at https://dmccooey.com/polyhedra:

```python
from geodesics import conway_board
conway_board("I", "t")     # truncated icosahedron — Go on the soccer ball
conway_board("D", "aa")    # rhombicosidodecahedron, 60 pts, degree 4
conway_board("D", "ta")    # great rhombicosidodecahedron, 120 pts
```

Subdivision is seed-generic too: `geodesic_subdivide(octahedron(), f)`
yields {3,4+} spheres with 6 degree-4 defects, `quad_subdivide(cube(), f)`
the cube-sphere.

## Quickstart

Environment (the engine is dependency-free; this covers the demo + tests):

```
conda env create -f environment.yml && conda activate geodesics
# or, with a plain venv:
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

```python
from geodesics import Game, mobius, sphere_geodesic, pass_alive, BLACK

g = Game(mobius(9, 5))
g.play(g.board.vertex_index(8, 0))   # a stone whose seam liberty is at (0, 4)
print(g.ascii())

s = Game(sphere_geodesic(3))         # 92 points, 12 of degree 5, chi = 2
s.play(0); s.play(1)
print(s.score(), pass_alive(s.board, s.colors, BLACK))
```

Run the narrated demo (`--png` renders the sphere position):

```
python -m geodesics.demo --png sphere.png
python -m pytest tests -q            # 24 tests
```

## The rule kernel, formally

State: a coloring `c : V -> {empty, black, white}` on graph `G = (V, E)`,
plus player-to-move and position history. For `S ⊆ V` monochromatic and
connected (a *chain*), its liberties are `L(S) = N(S) ∩ c⁻¹(empty)`.

A move by color `x` at empty vertex `v`:
1. set `c(v) = x`;
2. for each opponent chain `S` with `v ∈ N(S)` and `L(S) = ∅`: erase `S`;
3. if the chain containing `v` now has no liberties, the move is *suicide*
   — illegal unless the ruleset permits it, in which case the chain erases
   itself (Tromp-Taylor style);
4. the resulting coloring must not repeat any prior coloring
   (*positional superko*; situational adds player-to-move; detected via
   Zobrist hashing with one 64-bit key per (vertex, color)).

Two consecutive passes end the game; area score = stones + empty regions
whose stone-boundary is monochromatic. *Unconditional life* is Benson's
greatest fixpoint — chains keeping ≥ 2 vital enclosed regions — whose 1976
proof is purely graph-combinatorial and needs no modification here.

A consequence surfaced by the tests: under positional superko, single-stone
suicide is *always* illegal (it recreates the previous position) even when
suicide is allowed; only multi-stone suicide can be played. The engine gets
this right by construction rather than by special-casing.

## What the tests pin down

- **Invariants** — geodesic spheres satisfy `V = 10v²+2, χ = 2` with 12
  degree-5 defects; Goldberg spheres are 3-regular with exactly 12 pentagon
  faces; cube-spheres carry 8 degree-3 corners; Klein bottles are closed and
  regular under both square and tri meshes; the Möbius band has *one*
  boundary circle where the cylinder has two; T³ is 6-regular.
- **Conway operators** — tI, aC, aaD, taD, jC match the canonical polyhedron
  catalogue's V/E/F and face-size censuses; dual is an involution; the
  oriented-manifold validator gates every operator.
- **Topology-dependent play** — capturing the same point takes 2 stones on
  the plane, 4 on the torus, 6 in the interior of a 3D box; a Möbius capture
  lands across the seam on the orientation-flipped row.
- **Rules** — ko forbidden then legal after an exchange; suicide policy;
  scoring; undo; 150-move random self-play with per-move zero-liberty and
  hash-uniqueness invariants.
- **Persistence** — gSGF round-trips spec boards (surface/mesh/resolution)
  and Conway boards exactly, including position hashes.

## gSGF 0.1

JSON: `board` is either a registry constructor (`{"type": "torus",
"params": {"nx": 9, "ny": 9}}`) or an explicit adjacency list; `rules` is
the `RuleConfig`; `moves` are vertex ids (replayed with validation on load).

## Difficulty scaling

Board complexity is a constructor parameter, per the report's model:
vertex count (`frequency`, `nx`/`ny`), genus/orientability (plane → torus →
Klein), boundary structure, and degree irregularity (pentagon defects) all
scale independently. `sphere_geodesic(2)` (42 points) is a teaching board;
`sphere_geodesic(6)` (362) approaches 19×19 in scale.

## Next (per the report roadmap)

M1: port Layer 4 to Rust behind this exact API (property-test parity against
this reference); M2: geometry service for Class-II/III subdivisions, Goldberg
duals, and surface Voronoi via CGAL/geometry-central; discrete-geodesic
connectivity policies (heat method, straightest geodesics) as pluggable
adjacency generators; M3: GNN+MCTS agents (CNNs assume grid structure and do
not transfer).
