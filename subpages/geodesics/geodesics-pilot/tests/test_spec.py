import pytest

from geodesics import (make_board, conway_board, torus, mobius, klein,
                       sphere_goldberg, sphere_quad, torus3, box3,
                       Game, RuleConfig, gsgf, BLACK, EMPTY)


# ---------------------------------------------------------------------------
# mesh types on grid quotients
# ---------------------------------------------------------------------------

def test_tri_mesh_degrees():
    assert torus(7, 7, mesh="tri").degree_histogram() == {6: 49}
    assert klein(6, 6, mesh="tri").degree_histogram() == {6: 36}


def test_hex_mesh_is_three_regular_where_closed():
    b = torus(8, 6, mesh="hex")
    assert b.degree_histogram() == {3: 48}
    assert len(b.edges()) == 3 * 48 // 2


def test_hex_mobius_parity():
    b = mobius(8, 5, mesh="hex")                 # nx even, ny odd: allowed
    hist = b.degree_histogram()
    assert set(hist) == {2, 3}                    # boundary rows lose an edge
    with pytest.raises(ValueError):
        mobius(7, 5, mesh="hex")                  # odd nx across the seam
    with pytest.raises(ValueError):
        mobius(8, 6, mesh="hex")                  # even ny breaks the flip
    with pytest.raises(ValueError):
        klein(8, 6, mesh="hex")                   # obstructed outright


# ---------------------------------------------------------------------------
# sphere families
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("f", [1, 2, 3])
def test_goldberg_counts(f):
    b = sphere_goldberg(f)
    assert b.n == 20 * f * f
    assert len(b.edges()) == 30 * f * f
    assert b.degree_histogram() == {3: b.n}
    assert b.euler_characteristic() == 2
    sizes = {}
    for face in b.faces:
        sizes[len(face)] = sizes.get(len(face), 0) + 1
    assert sizes.get(5) == 12                     # the 12 pentagon defects


@pytest.mark.parametrize("f", [2, 3, 4])
def test_cube_sphere_counts(f):
    b = sphere_quad(f)
    assert b.n == 6 * f * f + 2
    assert len(b.edges()) == 12 * f * f
    assert b.degree_histogram() == {3: 8, 4: b.n - 8}
    assert b.euler_characteristic() == 2


# ---------------------------------------------------------------------------
# dimensions 1 and 3
# ---------------------------------------------------------------------------

def test_three_torus_is_homogeneous():
    b = torus3(4, 4, 4)
    assert b.degree_histogram() == {6: 64}
    v0 = b.vertex_index(0, 0, 0)
    assert b.vertex_index(3, 0, 0) in b.adj[v0]   # wraps in every axis
    assert b.vertex_index(0, 3, 0) in b.adj[v0]
    assert b.vertex_index(0, 0, 3) in b.adj[v0]


def test_box_capture_needs_six_stones():
    b = box3(4, 4, 4)
    assert b.degree_histogram() == {3: 8, 4: 24, 5: 24, 6: 8}
    g = Game(b)
    center = b.vertex_index(1, 1, 1)              # interior: degree 6
    g.set_position({center: 2}, to_move=BLACK)    # lone white stone
    nbrs = list(b.adj[center])
    for i, v in enumerate(nbrs):
        g.play(v)
        if i < len(nbrs) - 1:
            assert g.colors[center] != EMPTY
            g.play_pass()
    assert g.colors[center] == EMPTY              # captured on the 6th
    assert g.captures[BLACK] == 1


def test_dimension_one():
    b = make_board("torus", resolution=1, dimension=1)
    assert b.meta["resolved"] == "cycle" and b.degree_histogram() == {2: 9}
    p = make_board("plane", resolution=1, dimension=1)
    assert p.meta["resolved"] == "path"


# ---------------------------------------------------------------------------
# the spec itself
# ---------------------------------------------------------------------------

def test_spec_resolution_scaling_and_overrides():
    small = make_board("torus", "square", 1)
    big = make_board("torus", "square", 4)
    assert small.n == 25 and big.n == 121        # (2r+3)^2
    pinned = make_board("torus", "square", 1, nx=9, ny=5)
    assert pinned.n == 45                         # overrides win
    with pytest.raises(ValueError):
        make_board("torus", "hex", 2, dimension=3)


def test_spec_gsgf_roundtrip():
    b = make_board("sphere", "hex", 2)            # Goldberg GP(2,0)
    g = Game(b, RuleConfig(komi=1.5))
    g.play(0)
    g.play(b.adj[0][0])
    g.play_pass()
    text = gsgf.dumps(g)
    g2 = gsgf.loads(text)
    assert g2.board.params["mesh"] == "hex"
    assert g2.board.meta["resolved"] == "sphere_goldberg"
    assert g2.colors == g.colors
    assert g2._hash_history[-1] == g._hash_history[-1]


def test_conway_board_gsgf_roundtrip():
    b = conway_board("D", "aa")                   # rhombicosidodecahedron
    g = Game(b)
    g.play(10)
    g2 = gsgf.loads(gsgf.dumps(g))
    assert g2.board.params == {"seed": "D", "ops": "aa"}
    assert g2.colors == g.colors
