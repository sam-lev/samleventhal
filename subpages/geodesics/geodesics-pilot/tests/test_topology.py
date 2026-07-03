import pytest

from geodesics import topology


# ---------------------------------------------------------------------------
# Geodesic spheres: the counts mandated by the Euler characteristic
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("f", [1, 2, 3, 4])
def test_sphere_counts(f):
    b = topology.sphere_geodesic(f)
    assert b.n == 10 * f * f + 2
    assert len(b.edges()) == 30 * f * f
    assert len(b.faces) == 20 * f * f
    assert b.euler_characteristic() == 2          # V - E + F = 2 on S^2
    hist = b.degree_histogram()
    assert hist.get(5) == 12                       # exactly 12 pentagon defects
    if f > 1:
        assert hist.get(6) == b.n - 12
    assert set(hist) <= {5, 6}


def test_sphere_vertices_on_unit_sphere():
    b = topology.sphere_geodesic(3)
    for x, y, z in b.coords:
        assert abs(x * x + y * y + z * z - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Grid quotients: seams, flips, degrees, boundaries
# ---------------------------------------------------------------------------

def _components(vertices, adj):
    """Connected components of the induced subgraph on `vertices`."""
    vs = set(vertices)
    seen, comps = set(), []
    for v in vs:
        if v in seen:
            continue
        comp, stack = {v}, [v]
        while stack:
            u = stack.pop()
            for w in adj[u]:
                if w in vs and w not in comp:
                    comp.add(w)
                    stack.append(w)
        seen |= comp
        comps.append(comp)
    return comps


def test_plane_degrees():
    b = topology.plane(5, 5)
    assert b.degree_histogram() == {2: 4, 3: 12, 4: 9}


def test_torus_is_homogeneous():
    b = topology.torus(5, 5)
    assert b.degree_histogram() == {4: 25}
    assert len(b.edges()) == 2 * 25
    # corner wraps both ways
    nbrs = set(b.adj[b.vertex_index(0, 0)])
    assert {b.vertex_index(1, 0), b.vertex_index(4, 0),
            b.vertex_index(0, 1), b.vertex_index(0, 4)} == nbrs


def test_mobius_seam_flips_orientation():
    b = topology.mobius(6, 5)
    # leaving right edge at row 0 re-enters left edge at row ny-1
    assert b.vertex_index(0, 4) in b.adj[b.vertex_index(5, 0)]
    assert b.vertex_index(0, 0) in b.adj[b.vertex_index(5, 4)]
    assert b.vertex_index(0, 2) in b.adj[b.vertex_index(5, 2)]  # center row


def test_mobius_has_one_boundary_circle_cylinder_has_two():
    mob = topology.mobius(6, 5)
    cyl = topology.cylinder(6, 5)
    mob_rim = [v for v in range(mob.n) if len(mob.adj[v]) == 3]
    cyl_rim = [v for v in range(cyl.n) if len(cyl.adj[v]) == 3]
    assert len(mob_rim) == len(cyl_rim) == 2 * 6
    # the defining invariant: the Möbius band's boundary is a single circle
    assert len(_components(mob_rim, mob.adj)) == 1
    assert len(_components(cyl_rim, cyl.adj)) == 2


def test_klein_bottle_is_closed():
    b = topology.klein(6, 5)
    assert b.degree_histogram() == {4: 30}       # no boundary anywhere
    assert len(b.edges()) == 2 * 30


def test_rp2_corner_identifications():
    b = topology.rp2(5, 5)
    # antipodal gluing doubles up two corner edges; dedup leaves 4 corners deg 3
    assert b.degree_histogram() == {3: 4, 4: 21}
    assert len(b.edges()) == 2 * 25 - 2


def test_registry_roundtrip():
    b = topology.build("mobius", nx=7, ny=5)
    assert b.name == "mobius" and b.params == {"nx": 7, "ny": 5, "mesh": "square"}
    with pytest.raises(KeyError):
        topology.build("hyperbolic_disk")
