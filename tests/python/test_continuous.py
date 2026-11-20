import math

import pytest

from helpers import assert_cdf, mean_var, norm_cdf


def test_exponential(rng):
    xs = rng.exponential(2.5, size=100000)
    assert min(xs) >= 0.0
    assert_cdf(xs, lambda x: 1 - math.exp(-x / 2.5))
    assert abs(sum(xs) / len(xs) - 2.5) < 0.05


def test_laplace(rng):
    xs = rng.laplace(1.0, 2.0, size=100000)
    assert all(math.isfinite(x) for x in xs)
    assert_cdf(xs, lambda x: 0.5 * math.exp((x - 1) / 2) if x < 1 else 1 - 0.5 * math.exp(-(x - 1) / 2))


def test_cauchy(rng):
    xs = rng.cauchy(-1.0, 0.5, size=100000)
    assert all(math.isfinite(x) for x in xs)
    assert_cdf(xs, lambda x: 0.5 + math.atan((x + 1.0) / 0.5) / math.pi)


def test_large_batches_are_always_finite(rng):
    for fn in (rng.laplace, rng.exponential, rng.cauchy, rng.dl_box_muller, rng.dl_ziggurat):
        assert all(math.isfinite(x) for x in fn(size=300000))


@pytest.mark.parametrize("call", [
    lambda r: r.exponential(0.0), lambda r: r.exponential(-1.0), lambda r: r.exponential(float("nan")),
    lambda r: r.laplace(0.0, -1.0), lambda r: r.laplace(float("inf"), 1.0),
    lambda r: r.cauchy(0.0, 0.0), lambda r: r.cauchy(float("nan"), 1.0),
    lambda r: r.dl_box_muller(0.0, -1.0), lambda r: r.dl_box_muller(float("nan"), 1.0),
    lambda r: r.dl_ziggurat(mu=0.0, sigma=-1.0), lambda r: r.dl_ziggurat(layers=3), lambda r: r.dl_ziggurat(layers=10 ** 6),
])
def test_invalid_parameters_raise(rng, call):
    with pytest.raises(ValueError):
        call(rng)


def test_dl_box_muller_distribution(rng):
    xs = rng.dl_box_muller(3.0, 2.0, size=200000)
    assert_cdf(xs, lambda x: norm_cdf((x - 3.0) / 2.0))
    m, v = mean_var(xs)
    k = sum((x - m) ** 4 for x in xs) / len(xs) / v ** 2
    assert abs(m - 3.0) < 0.03 and abs(v - 4.0) < 0.08 and abs(k - 3.0) < 0.1
    assert set(rng.dl_box_muller(5.0, 0.0, size=10)) == {5.0}
    assert rng.dl_box_muller(size=0) == []


def test_normal_bm_alias_still_works(rng):
    assert isinstance(rng.normal_bm(), float)


def test_dl_box_muller_odd_and_even_sizes_are_consistent():
    import dlrand
    a, b = dlrand.RandomGenerator(11), dlrand.RandomGenerator(11)
    one_by_one = [b.dl_box_muller() for _ in range(7)]
    assert a.dl_box_muller(size=7) == one_by_one


@pytest.mark.parametrize("layers", [8, 16, 64, 128, 256, 512, 4096])
def test_dl_ziggurat_exact_for_every_layer_count(rng, layers):
    # layers != 128 used to give a wrong distribution (64) or only worked through NaN rejection (256)
    xs = rng.dl_ziggurat(size=200000, layers=layers)
    assert_cdf(xs, norm_cdf)
    m, v = mean_var(xs)
    k = sum((x - m) ** 4 for x in xs) / len(xs) / v ** 2
    assert abs(m) < 0.012 and abs(v - 1.0) < 0.02 and abs(k - 3.0) < 0.1


@pytest.mark.parametrize("layers,threshold", [(128, 3.4426198558966), (256, 3.6541528853610), (256, 3.7), (64, 3.0)])
def test_dl_ziggurat_tail_mass(rng, layers, threshold):
    n = 1_000_000
    xs = rng.dl_ziggurat(size=n, layers=layers)
    expected = n * 2.0 * (1.0 - norm_cdf(threshold))
    got = sum(1 for x in xs if abs(x) > threshold)
    assert abs(got - expected) < 5.0 * math.sqrt(expected) + 2


def test_dl_ziggurat_location_scale_and_symmetry(rng):
    xs = rng.dl_ziggurat(mu=10.0, sigma=0.5, size=100000)
    assert_cdf(xs, lambda x: norm_cdf((x - 10.0) / 0.5))
    zs = rng.dl_ziggurat(size=100000)
    assert abs(sum(1 for z in zs if z > 0) / len(zs) - 0.5) < 0.01
