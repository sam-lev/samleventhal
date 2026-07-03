import pytest

from geodesics import mesh as M


# ---------------------------------------------------------------------------
# seeds
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,v,e,f", [
    ("T", 4, 6, 4), ("C", 8, 12, 6), ("O", 6, 12, 8),
    ("D", 20, 30, 12), ("I", 12, 30, 20),
])
def test_platonic_seeds(name, v, e, f):
    m = M.seed(name)
    assert (m.v, m.e, m.f) == (v, e, f)
    assert m.euler() == 2
    M.check_closed_oriented(m)


# ---------------------------------------------------------------------------
# Conway operators against the canonical catalogue (dmccooey.com/polyhedra)
# ---------------------------------------------------------------------------

def test_truncated_icosahedron_is_the_soccer_ball():
    m = M.truncate(M.icosahedron())
    assert (m.v, m.e, m.f) == (60, 90, 32)
    assert m.face_sizes() == {5: 12, 6: 20}
    assert m.vertex_degrees() == {3: 60}


def test_ambo_cube_is_the_cuboctahedron():
    m = M.ambo(M.cube())
    assert (m.v, m.e, m.f) == (12, 24, 14)
    assert m.face_sizes() == {3: 8, 4: 6}


def test_expand_dodecahedron_is_the_rhombicosidodecahedron():
    m = M.conway("aa", M.seed("D"))          # e = aa
    assert (m.v, m.e, m.f) == (60, 120, 62)
    assert m.face_sizes() == {3: 20, 4: 30, 5: 12}


def test_bevel_dodecahedron_is_the_great_rhombicosidodecahedron():
    m = M.conway("ta", M.seed("D"))          # b = ta
    assert (m.v, m.e, m.f) == (120, 180, 62)
    assert m.face_sizes() == {4: 30, 6: 20, 10: 12}
    assert m.vertex_degrees() == {3: 120}


def test_dual_is_an_involution_and_kis_counts():
    d = M.seed("D")
    dd = M.dual(M.dual(d))
    assert (dd.v, dd.e, dd.f) == (d.v, d.e, d.f)
    k = M.kis(M.cube())                       # tetrakis hexahedron
    assert (k.v, k.e, k.f) == (8 + 6, 12 + 24, 24)
    assert all(len(f) == 3 for f in k.faces)


def test_join_and_unknown_op():
    j = M.join(M.cube())                      # rhombic dodecahedron
    assert (j.v, j.e, j.f) == (14, 24, 12)
    assert all(len(f) == 4 for f in j.faces)
    with pytest.raises(KeyError):
        M.conway("z", M.cube())


# ---------------------------------------------------------------------------
# subdivision generality
# ---------------------------------------------------------------------------

def test_geodesic_subdivision_of_other_deltahedra():
    # Euler forces q defects of degree q on {3,q+}_(f,0): 4 on T, 6 on O
    t = M.geodesic_subdivide(M.tetrahedron(), 3)
    assert t.vertex_degrees() == {3: 4, 6: t.v - 4}
    o = M.geodesic_subdivide(M.octahedron(), 3)
    assert o.vertex_degrees() == {4: 6, 6: o.v - 6}
    with pytest.raises(ValueError):
        M.geodesic_subdivide(M.cube(), 2)     # not a triangle mesh


def test_quad_subdivision_needs_quads():
    q = M.quad_subdivide(M.cube(), 4)
    assert (q.v, q.e, q.f) == (6 * 16 + 2, 12 * 16, 6 * 16)
    assert q.vertex_degrees() == {3: 8, 4: q.v - 8}
    with pytest.raises(ValueError):
        M.quad_subdivide(M.icosahedron(), 2)
