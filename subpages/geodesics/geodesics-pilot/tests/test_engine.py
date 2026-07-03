import random

import pytest

from geodesics import (Game, RuleConfig, IllegalMove, chain_at, gsgf,
                       plane, torus, mobius, EMPTY, BLACK, WHITE)


def v(board, x, y):
    return board.vertex_index(x, y)


# ---------------------------------------------------------------------------
# Capture — and how topology changes it
# ---------------------------------------------------------------------------

def test_corner_capture_on_plane_needs_two_stones():
    b = plane(5, 5)
    g = Game(b)
    g.play(v(b, 1, 0))          # B
    g.play(v(b, 0, 0))          # W into the corner (2 liberties there)
    g.play(v(b, 0, 1))          # B captures
    assert g.colors[v(b, 0, 0)] == EMPTY
    assert g.captures[BLACK] == 1


def test_same_point_on_torus_needs_four_stones():
    b = torus(5, 5)
    g = Game(b)
    g.play(v(b, 1, 0))          # B
    g.play(v(b, 0, 0))          # W — no corners on a torus: 4 liberties
    g.play(v(b, 4, 0))          # B (wraps)
    g.play(v(b, 2, 2))          # W elsewhere
    g.play(v(b, 0, 1))          # B
    assert g.colors[v(b, 0, 0)] == WHITE          # still alive after 3
    g.play(v(b, 2, 3))          # W elsewhere
    g.play(v(b, 0, 4))          # B closes the wrap-around liberty
    assert g.colors[v(b, 0, 0)] == EMPTY
    assert g.captures[BLACK] == 1


def test_capture_across_mobius_seam():
    b = mobius(6, 5)
    g = Game(b)
    # W at (5,0): neighbors are (4,0), (5,1) and — across the flipped seam —
    # (0,4). Only 3 liberties despite sitting on the "edge column".
    target = v(b, 5, 0)
    g.set_position({v(b, 4, 0): BLACK, v(b, 5, 1): BLACK, target: WHITE},
                   to_move=BLACK)
    g.play(v(b, 0, 4))          # the seam liberty, on the *other* row
    assert g.colors[target] == EMPTY
    assert g.captures[BLACK] == 1


# ---------------------------------------------------------------------------
# Ko / positional superko lifecycle
# ---------------------------------------------------------------------------

def test_positional_superko_forbids_then_allows_retake():
    b = plane(5, 5)
    g = Game(b)
    g.set_position({
        v(b, 1, 0): BLACK, v(b, 0, 1): BLACK, v(b, 1, 2): BLACK,
        v(b, 2, 0): WHITE, v(b, 3, 1): WHITE, v(b, 2, 2): WHITE,
        v(b, 1, 1): WHITE,                      # the ko stone
    }, to_move=BLACK)
    g.play(v(b, 2, 1))                          # B captures the ko
    assert g.colors[v(b, 1, 1)] == EMPTY
    with pytest.raises(IllegalMove) as ex:
        g.play(v(b, 1, 1))                      # immediate retake = repetition
    assert "superko" in ex.value.reason
    g.play(v(b, 4, 4))                          # W ko threat elsewhere
    g.play(v(b, 0, 4))                          # B answers
    g.play(v(b, 1, 1))                          # retake now legal: new position
    assert g.colors[v(b, 2, 1)] == EMPTY


# ---------------------------------------------------------------------------
# Suicide
# ---------------------------------------------------------------------------

def test_suicide_illegal_by_default_legal_when_configured():
    # walls enclosing a 2-point space with one W stone already inside

    def setup(game):
        b_ = game.board
        game.set_position({
            v(b_, 2, 0): BLACK, v(b_, 0, 1): BLACK, v(b_, 1, 1): BLACK,
            v(b_, 0, 0): WHITE,
        }, to_move=WHITE)

    b = plane(5, 5)
    g = Game(b)
    setup(g)
    with pytest.raises(IllegalMove) as ex:
        g.play(v(b, 1, 0))                       # fills own last liberty
    assert ex.value.reason == "suicide"

    # Tromp-Taylor style: multi-stone suicide is legal (single-stone suicide
    # would recreate the previous position and is caught by superko instead)
    g2 = Game(b, RuleConfig(allow_suicide=True))
    setup(g2)
    g2.play(v(b, 1, 0))
    assert g2.colors[v(b, 0, 0)] == EMPTY        # both stones removed
    assert g2.colors[v(b, 1, 0)] == EMPTY
    assert g2.captures[BLACK] == 2

    g3 = Game(b, RuleConfig(allow_suicide=True))
    g3.set_position({v(b, 1, 0): BLACK, v(b, 0, 1): BLACK}, to_move=WHITE)
    with pytest.raises(IllegalMove) as ex:
        g3.play(v(b, 0, 0))                      # 1-stone suicide: repetition
    assert "superko" in ex.value.reason


# ---------------------------------------------------------------------------
# Scoring, passes, undo
# ---------------------------------------------------------------------------

def test_area_scoring_wall():
    b = plane(3, 3)
    g = Game(b)
    g.set_position({v(b, 1, 0): BLACK, v(b, 1, 1): BLACK, v(b, 1, 2): BLACK},
                   to_move=BLACK)
    g.play_pass()
    g.play_pass()
    assert g.game_over
    s = g.score()
    assert s["black"] == 9 and s["white"] == 0 and s["winner"] == "B"


def test_undo_restores_position_and_history():
    b = plane(5, 5)
    g = Game(b)
    h0 = g._hash_history[-1]
    g.play(v(b, 2, 2))
    g.play(v(b, 1, 1))
    g.undo()
    g.undo()
    assert g.colors == [EMPTY] * b.n
    assert g.to_move == BLACK
    assert g._hash_history[-1] == h0 and len(g._hash_history) == 1


# ---------------------------------------------------------------------------
# Invariant fuzz: random legal self-play on a torus
# ---------------------------------------------------------------------------

def test_random_selfplay_invariants():
    b = torus(5, 5)
    g = Game(b)
    rng = random.Random(11)
    stone_hashes = []
    for _ in range(150):
        moves = g.legal_moves()
        if not moves:
            break
        g.play(rng.choice(moves))
        stone_hashes.append(g._hash_history[-1])
        # no chain on the board may ever have zero liberties
        seen = set()
        for u in range(b.n):
            if g.colors[u] != EMPTY and u not in seen:
                stones, libs = chain_at(g.colors, b, u)
                seen |= stones
                assert libs, f"zero-liberty chain survived at {u}"
    # positional superko guarantees every post-move position is new
    assert len(set(stone_hashes)) == len(stone_hashes)
    g.score()                                    # scoring never crashes


# ---------------------------------------------------------------------------
# gSGF round-trip
# ---------------------------------------------------------------------------

def test_gsgf_roundtrip_on_mobius():
    b = mobius(6, 5)
    g = Game(b, RuleConfig(komi=2.5))
    g.play(v(b, 2, 2))
    g.play(v(b, 3, 2))
    g.play_pass()
    g.play(v(b, 5, 0))
    text = gsgf.dumps(g, meta={"event": "pilot"})
    g2 = gsgf.loads(text)
    assert g2.board.name == "mobius" and g2.board.params["nx"] == 6
    assert g2.colors == g.colors
    assert g2.to_move == g.to_move
    assert g2.rules.komi == 2.5
    assert g2._hash_history[-1] == g._hash_history[-1]
