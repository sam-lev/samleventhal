from geodesics import Game, pass_alive, plane, torus, BLACK, WHITE


def v(board, x, y):
    return board.vertex_index(x, y)


def test_two_eyes_are_pass_alive():
    b = plane(7, 7)
    g = Game(b)
    chain = [(1, 0), (0, 1), (1, 1), (2, 1), (3, 1), (3, 0)]  # eyes at (0,0),(2,0)
    g.set_position({v(b, x, y): BLACK for x, y in chain})
    alive = pass_alive(b, g.colors, BLACK)
    assert alive == {v(b, x, y) for x, y in chain}


def test_one_eye_is_not_pass_alive():
    b = plane(7, 7)
    g = Game(b)
    chain = [(1, 0), (0, 1), (1, 1)]                          # only eye: (0,0)
    g.set_position({v(b, x, y): BLACK for x, y in chain})
    assert pass_alive(b, g.colors, BLACK) == set()
    assert pass_alive(b, g.colors, WHITE) == set()            # no white chains


def test_two_eyes_on_torus():
    # a ring around the torus's x-direction with two one-point eyes carved in
    b = torus(5, 5)
    g = Game(b)
    stones = {}
    for x in range(5):
        for y in (0, 2):
            stones[v(b, x, y)] = BLACK                        # two full rings
    for x in (0, 2):
        stones[v(b, x, 1)] = BLACK                            # eye walls
    # eyes at (1,1), (3,1)+(4,1) region — every empty in row 1 borders rings
    g.set_position(stones)
    alive = pass_alive(b, g.colors, BLACK)
    assert alive == set(stones)                               # whole structure lives
