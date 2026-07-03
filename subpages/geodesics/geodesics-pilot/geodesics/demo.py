"""Narrated demo of the Geodesics pilot.

    python -m geodesics.demo [--png OUT.png]

Walks the same point through three topologies to show rule invariance vs.
connectivity dependence, then plays a random game on a frequency-3 geodesic
sphere, reports Tromp-Taylor score and Benson pass-alive chains, and
optionally renders the sphere position to PNG.
"""

from __future__ import annotations

import argparse
import random

from . import (Game, RuleConfig, plane, torus, mobius, sphere_geodesic,
               pass_alive, BLACK, WHITE, EMPTY)


def banner(t: str) -> None:
    print(f"\n{'=' * 62}\n{t}\n{'=' * 62}")


def demo_same_point_three_topologies() -> None:
    banner("One point, three topologies: capturing the stone at (0,0)")
    for make, need in ((plane, 2), (torus, 4)):
        b = make(5, 5)
        g = Game(b)
        w = b.vertex_index(0, 0)
        deg = len(b.adj[w])
        print(f"\n[{b.name}] degree of (0,0) = {deg} -> needs {need} stones")
        g.set_position({w: WHITE}, to_move=BLACK)
        for x, y in [(1, 0), (0, 1), (4, 0), (0, 4)][:need]:
            g.play(b.vertex_index(x, y))       # Black surrounds
            if g.colors[w] == EMPTY:
                break
            g.play_pass()                      # White passes in this demo
        print(g.ascii())
        print(f"captured: {g.colors[w] == EMPTY}, "
              f"black captures = {g.captures[BLACK]}")

    banner("Möbius seam: the liberty that lives on the other row")
    b = mobius(6, 5)
    g = Game(b)
    tgt = b.vertex_index(5, 0)
    nb = [b.vertex_xy(u) for u in b.adj[tgt]]
    print(f"neighbors of (5,0): {nb}   <- note (0,4): orientation-reversed seam")
    g.set_position({b.vertex_index(4, 0): BLACK,
                    b.vertex_index(5, 1): BLACK,
                    tgt: WHITE}, to_move=BLACK)
    print("before:\n" + g.ascii())
    g.play(b.vertex_index(0, 4))
    print("Black plays (0,4) — across the seam:\n" + g.ascii())
    print(f"white stone at (5,0) captured: {g.colors[tgt] == EMPTY}")


def demo_sphere(png: str | None, moves: int = 40, seed: int = 7):
    banner(f"Geodesic sphere {{3,5+}}_(3,0): random self-play, {moves} moves")
    b = sphere_geodesic(3)
    hist = b.degree_histogram()
    print(f"V={b.n} E={len(b.edges())} F={len(b.faces)} "
          f"chi={b.euler_characteristic()}  degrees={hist}")
    g = Game(b, RuleConfig(komi=0.0))
    rng = random.Random(seed)
    for _ in range(moves):
        legal = g.legal_moves()
        if not legal:
            break
        g.play(rng.choice(legal))
    s = g.score()
    print(f"score: B={s['black']} W={s['white']}  "
          f"(stones {s['stones']}, territory {s['territory']})")
    pa_b = pass_alive(b, g.colors, BLACK)
    pa_w = pass_alive(b, g.colors, WHITE)
    print(f"Benson pass-alive: black {len(pa_b)} stones, white {len(pa_w)} stones"
          f"  (random play rarely builds pass-alive shape — expected 0/0)")
    if png:
        render_sphere_png(b, g, png)
        print(f"rendered -> {png}")
    return b, g


def render_sphere_png(b, g, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    fig = plt.figure(figsize=(7, 7), facecolor="#101418")
    ax = fig.add_subplot(111, projection="3d", facecolor="#101418")
    segs = [(b.coords[a], b.coords[c]) for a, c in b.edges()]
    ax.add_collection3d(Line3DCollection(segs, colors="#4a5561", linewidths=0.7))
    for color, face, edge in ((BLACK, "#141414", "#3d3d3d"),
                              (WHITE, "#f2ede2", "#b8b2a4")):
        pts = [b.coords[v] for v in range(b.n) if g.colors[v] == color]
        if pts:
            xs, ys, zs = zip(*pts)
            ax.scatter(xs, ys, zs, s=120, c=face, edgecolors=edge,
                       linewidths=0.8, depthshade=True)
    penta = [b.coords[v] for v in range(b.n) if len(b.adj[v]) == 5]
    xs, ys, zs = zip(*penta)
    ax.scatter(xs, ys, zs, s=10, c="#c98a3d", depthshade=False)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.set_title("Geodesics — sphere {3,5+}(3,0), random position",
                 color="#c9d2da", fontsize=11, pad=0)
    lim = 0.72
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def demo_spec_gallery() -> None:
    banner("The board design space: make_board(surface, mesh, resolution, dim)")
    from . import make_board, conway_board
    specs = [
        ("torus", "hex", 1, 2), ("torus", "tri", 1, 2),
        ("mobius", "hex", 1, 2), ("sphere", "square", 2, 2),
        ("sphere", "hex", 2, 2), ("plane", "square", 1, 3),
        ("torus", "square", 1, 3), ("torus", "square", 1, 1),
    ]
    for surface, mesh_t, r, dim in specs:
        b = make_board(surface, mesh_t, r, dimension=dim)
        print(f"  {surface:8s} {mesh_t:6s} r={r} dim={dim} -> "
              f"{b.meta['resolved']:15s} V={b.n:4d} deg={b.degree_histogram()}")
    for seed, ops, note in (("I", "t", "soccer ball"),
                            ("D", "aa", "rhombicosidodecahedron"),
                            ("D", "ta", "great rhombicosidodecahedron")):
        b = conway_board(seed, ops)
        print(f"  conway {ops+seed:4s} ({note}): V={b.n} "
              f"deg={b.degree_histogram()}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--png", default=None, help="render sphere demo to PNG")
    args = ap.parse_args()
    demo_same_point_three_topologies()
    demo_spec_gallery()
    demo_sphere(args.png)


if __name__ == "__main__":
    main()
