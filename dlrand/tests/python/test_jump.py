"""Verifies the jump-ahead polynomials mathematically.

The xoshiro256 state transition is linear over GF(2): s' = T s for a 256x256 bit matrix T. Squaring T 128 (resp. 192)
times gives T^(2^128) (resp. T^(2^192)), the exact jump. We compare that with rl_jump / rl_long_jump.
"""
import ctypes
import random

import dlrand

M64 = (1 << 64) - 1


def _rotl(x, k):
    return ((x << k) | (x >> (64 - k))) & M64


def _step(w):
    s0, s1, s2, s3 = w
    t = (s1 << 17) & M64
    s2 ^= s0
    s3 ^= s1
    s1 ^= s2
    s0 ^= s3
    s2 ^= t
    s3 = _rotl(s3, 45)
    return [s0, s1, s2, s3]


def _to_vec(w):
    return w[0] | (w[1] << 64) | (w[2] << 128) | (w[3] << 192)


def _from_vec(v):
    return [(v >> (64 * i)) & M64 for i in range(4)]


def _apply(cols, v):
    r, k = 0, 0
    while v:
        if v & 1:
            r ^= cols[k]
        v >>= 1
        k += 1
    return r


def _power_of_two(k):
    cols = [_to_vec(_step(_from_vec(1 << i))) for i in range(256)]
    for _ in range(k):
        cols = [_apply(cols, c) for c in cols]
    return cols


def _check(jump_fn_name, exponent):
    lib = dlrand.load_library()
    jump = getattr(lib, jump_fn_name)
    matrix = _power_of_two(exponent)
    random.seed(1)
    for _ in range(4):
        g = dlrand.RandomGenerator(random.getrandbits(64))
        before = g.getstate()[:4]
        expected = _from_vec(_apply(matrix, _to_vec(list(before))))
        jump(g._st)
        assert list(g.getstate()[:4]) == expected
        g.close()


def test_jump_is_exactly_2_to_the_128_steps():
    _check("rl_jump", 128)


def test_long_jump_is_exactly_2_to_the_192_steps():
    _check("rl_long_jump", 192)


def test_spawn_children_are_successive_jumps():
    g = dlrand.RandomGenerator(77)
    kids = g.spawn(2)
    lib = dlrand.load_library()
    ref = dlrand.RandomGenerator(77)
    lib.rl_jump(ref._st)
    assert kids[0].getstate()[:4] == ref.getstate()[:4]
    lib.rl_jump(ref._st)
    assert kids[1].getstate()[:4] == ref.getstate()[:4]
