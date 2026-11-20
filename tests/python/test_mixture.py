import math

import pytest

import dlrand
from helpers import assert_pmf, mean_var, poisson_pmf


COMPONENTS = [{"weight": 0.2, "kind": "uniform", "start": 0, "range": 3},
              {"weight": 0.5, "kind": "uniform", "start": 100, "range": 2},
              {"weight": 0.3, "kind": "uniform", "start": 1000, "range": 1}]


def target(k):
    return {0: 0.2 / 3, 1: 0.2 / 3, 2: 0.2 / 3, 100: 0.25, 101: 0.25, 1000: 0.3}.get(k, 0.0)


@pytest.mark.parametrize("mode", ["manual", "residual", "alias"])
def test_mixture_weights_in_every_mode(rng, mode):
    xs = rng.sample_mixture(COMPONENTS, size=120000, mode=mode)
    support = sorted(k for k in set(xs))
    assert set(support) == {0, 1, 2, 100, 101, 1000}
    n = len(xs)
    for k in support:
        p = target(k)
        assert abs(xs.count(k) / n - p) < 5 * math.sqrt(p * (1 - p) / n) + 1e-4


def test_engine_is_built_once_and_reused(rng):
    mix = rng.mixture(COMPONENTS)
    a = [mix() for _ in range(5)]
    b = mix.sample(5)
    assert len(a) == len(b) == 5
    mix.close()
    with pytest.raises(ValueError):
        mix()


def test_native_workers_poisson_geometric_binomial(rng):
    mix = (dlrand.MixtureBuilder(rng).add_poisson(0.5, 30.0).add_geometric(0.25, 0.5)
           .add_binomial(0.25, 40, 0.5).build("alias"))
    xs = mix.sample(60000)
    assert min(xs) >= 0
    m, _ = mean_var(xs)
    assert abs(m - (0.5 * 30 + 0.25 * 2 + 0.25 * 20)) < 0.25


def test_poisson_only_mixture_matches_pmf(rng):
    mix = dlrand.MixtureBuilder(rng).add_poisson(0.5, 3.0).add_poisson(0.5, 20.0).build("manual")
    xs = mix.sample(150000)
    assert_pmf(xs, lambda k: 0.5 * poisson_pmf(3.0, k) + 0.5 * poisson_pmf(20.0, k), 0, 50)


def test_floating_point_weights_do_not_bias_component_zero(rng):
    # weights that sum to 1 only up to rounding: leftover probability must not fall onto component 0
    w = [0.1] * 10
    comps = [{"weight": x, "kind": "uniform", "start": i, "range": 1} for i, x in enumerate(w)]
    xs = rng.sample_mixture(comps, size=100000, mode="manual")
    assert abs(xs.count(0) / 1e5 - 0.1) < 0.005


def test_python_callable_worker_and_exceptions(rng):
    mix = dlrand.MixtureBuilder(rng).add_callable(0.5, lambda: 7).add_uniform(0.5, 0, 1).build("manual")
    xs = mix.sample(20000)
    assert abs(xs.count(7) / 20000 - 0.5) < 0.02 and set(xs) == {0, 7}

    def boom():
        raise KeyError("worker failed")
    bad = dlrand.MixtureBuilder(rng).add_callable(1.0, boom).build("manual")
    with pytest.raises(KeyError):
        bad()


@pytest.mark.parametrize("specs", [
    [],
    [{"weight": 0.5, "kind": "uniform", "start": 0, "range": 1}],                          # sums to 0.5
    [{"weight": 1.5, "kind": "uniform", "start": 0, "range": 1}, {"weight": -0.5, "kind": "uniform", "start": 0, "range": 1}],
    [{"weight": 1.0, "kind": "poisson"}],                                                  # missing 'lam'
    [{"weight": 1.0, "kind": "zipf"}],
    [{"weight": 1.0, "kind": "uniform", "start": 0, "range": 0}],                          # empty range
])
def test_invalid_mixtures(rng, specs):
    with pytest.raises(ValueError):
        mix = rng.mixture(specs)
        mix()


def test_bad_mode(rng):
    with pytest.raises(ValueError):
        rng.mixture(COMPONENTS, mode="fast")


def test_alias_sampler(rng):
    s = rng.alias_sampler([1, 0, 3, 6])
    xs = s.sample(100000)
    assert 1 not in xs
    for k, p in ((0, 0.1), (2, 0.3), (3, 0.6)):
        assert abs(xs.count(k) / 1e5 - p) < 0.01
    assert isinstance(s(), int)
    s.close()
    with pytest.raises(ValueError):
        s.sample()
    for bad in ([], [0, 0], [-1, 2], [float("nan"), 1]):
        with pytest.raises(ValueError):
            rng.alias_sampler(bad)


def test_alias_sampler_many_categories(rng):
    n = 1000
    w = [i + 1 for i in range(n)]
    xs = rng.alias_sampler(w).sample(300000)
    top = sum(1 for x in xs if x >= n // 2) / len(xs)
    expected = sum(w[n // 2:]) / sum(w)
    assert abs(top - expected) < 0.01
