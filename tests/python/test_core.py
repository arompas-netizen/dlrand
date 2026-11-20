import pytest

import dlrand
from helpers import assert_pmf, mean_var


def test_same_seed_same_stream_regardless_of_other_instances():
    a, b, other = dlrand.RandomGenerator(7), dlrand.RandomGenerator(7), dlrand.RandomGenerator(99)
    seq_a = a.dl_box_muller(size=6)
    first = b.dl_box_muller(size=1)
    other.dl_box_muller(size=3)                      # used to corrupt a shared static cache
    rest = b.dl_box_muller(size=5)
    assert seq_a == first + rest


def test_different_seeds_differ():
    assert dlrand.RandomGenerator(1).uniform(size=5) != dlrand.RandomGenerator(2).uniform(size=5)


def test_entropy_seed_when_none():
    assert dlrand.RandomGenerator().uniform(size=3) != dlrand.RandomGenerator().uniform(size=3)


@pytest.mark.parametrize("bad", [-1, 2 ** 64, 1.5, "7"])
def test_bad_seed(bad):
    with pytest.raises((ValueError, TypeError)):
        dlrand.RandomGenerator(bad)


def test_state_roundtrip_including_cached_variate(rng):
    rng.dl_box_muller()                              # leaves a cached second variate in the state
    snap = rng.getstate()
    expected = rng.dl_box_muller(size=7)
    rng.setstate(snap)
    assert rng.dl_box_muller(size=7) == expected
    with pytest.raises(ValueError):
        rng.setstate((0, 0, 0, 0, 0, 0))
    with pytest.raises(ValueError):
        rng.setstate((1, 2, 3))


def test_reseed_resets_stream(rng):
    first = rng.uniform(size=4)
    rng.dl_box_muller()
    rng.reseed(rng.seed_value)
    assert rng.uniform(size=4) == first


def test_spawn_gives_distinct_streams_and_does_not_advance_parent():
    g = dlrand.RandomGenerator(5)
    before = g.getstate()
    kids = g.spawn(3)
    assert g.getstate() == before
    outs = [tuple(k.uniform(size=5)) for k in kids] + [tuple(dlrand.RandomGenerator(5).uniform(size=5))]
    assert len(set(outs)) == 4
    assert g.spawn(0) == []


def test_uniform_range_and_moments(rng):
    xs = rng.uniform(2.0, 5.0, size=100000)
    assert min(xs) >= 2.0 and max(xs) < 5.0
    m, v = mean_var(xs)
    assert abs(m - 3.5) < 0.02 and abs(v - 0.75) < 0.02
    assert rng.uniform(1.0, 1.0) == 1.0
    assert isinstance(rng.uniform(), float) and rng.uniform(size=0) == []


@pytest.mark.parametrize("args", [(2.0, 1.0), (float("nan"), 1.0), (0.0, float("inf"))])
def test_uniform_rejects_bad_ranges(rng, args):
    with pytest.raises(ValueError):
        rng.uniform(*args)


def test_integers_inclusive_and_unbiased(rng):
    xs = rng.integers(1, 6, size=120000)
    assert set(xs) == {1, 2, 3, 4, 5, 6}
    assert_pmf(xs, lambda k: 1 / 6, 1, 6)
    assert rng.integers(5, 5) == 5
    assert -3 <= rng.integers(-3, 3) <= 3
    big = rng.integers(-2 ** 63, 2 ** 63 - 1, size=10)          # the full 64-bit span works
    assert len(big) == 10
    with pytest.raises(ValueError):
        rng.integers(4, 3)
    with pytest.raises(ValueError):
        rng.integers(0, 2 ** 63)                                # does not fit in int64


def test_size_validation(rng):
    with pytest.raises(ValueError):
        rng.uniform(size=-1)
    with pytest.raises(TypeError):
        rng.uniform(size=2.5)


def test_closed_generator_raises():
    g = dlrand.RandomGenerator(1)
    g.close()
    g.close()                                                    # idempotent
    with pytest.raises(dlrand.DlrandError):
        g.uniform()


def test_context_manager():
    with dlrand.RandomGenerator(3) as g:
        assert 0 <= g.uniform() < 1
    with pytest.raises(dlrand.DlrandError):
        g.uniform()


def test_numpy_backend():
    np = pytest.importorskip("numpy")
    g = dlrand.RandomGenerator(4, numpy=True)
    arr = g.poisson(5, size=10)
    assert isinstance(arr, np.ndarray) and arr.dtype == np.int64 and len(arr) == 10
    assert isinstance(g.uniform(size=0), np.ndarray)


def test_version_matches_library():
    lib = dlrand.load_library()
    assert lib.rl_version().decode() == dlrand.__version__


def test_xoshiro256pp_known_answer_vector():
    """Published xoshiro256++ reference outputs for the initial state {1, 2, 3, 4}."""
    lib = dlrand.load_library()
    g = dlrand.RandomGenerator(0)
    g.setstate((1, 2, 3, 4, 0, 0))
    expected = [41943041, 58720359, 3588806011781223, 3591011842654386, 9228616714210784205,
                9973669472204895162, 14011001112246962877, 12406186145184390807, 15849039046786891736,
                10450023813501588000]
    assert [lib.rl_next_u64(g._st) for _ in range(10)] == expected
