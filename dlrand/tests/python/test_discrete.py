import math

import pytest

import dlrand
from helpers import assert_pmf, binom_pmf, geom_pmf_failures, mean_var, poisson_pmf


# ------------------------------------------------------------------ bernoulli
@pytest.mark.parametrize("method", ["standard", "jump", "auto"])
def test_bernoulli_inclusion_rate_and_order(rng, method):
    data = list(range(1000))
    counts = [0] * 1000
    for _ in range(300):
        kept = rng.bernoulli_sampling(data, 0.2, method=method)
        assert kept == sorted(kept) and len(set(kept)) == len(kept)
        for k in kept:
            counts[k] += 1
    assert abs(sum(counts) / (300 * 1000) - 0.2) < 0.005
    assert max(counts) < 300 * 0.2 + 6 * math.sqrt(300 * 0.2 * 0.8) + 5      # no index is favoured


def test_bernoulli_edges(rng):
    data = list("abcdef")
    assert rng.bernoulli_sampling(data, 0.0, method="jump") == []        # used to segfault
    assert rng.bernoulli_sampling(data, 0.0, method="standard") == []
    assert rng.bernoulli_sampling(data, 1.0, method="jump") == data
    assert rng.bernoulli_sampling([], 0.5) == []
    assert rng.bernoulli_sampling(data, 1e-300, method="jump") == []
    for bad in (-0.1, 1.5, float("nan")):
        with pytest.raises(ValueError):
            rng.bernoulli_sampling(data, bad)
    with pytest.raises(ValueError):
        rng.bernoulli_sampling(data, 0.5, method="nope")


# ------------------------------------------------------------------ permutation / choice
def test_permutation_uniform_over_all_orderings(rng):
    from itertools import permutations
    perms = list(permutations(range(4)))
    idx = {p: i for i, p in enumerate(perms)}
    samples = [idx[tuple(rng.permutation([0, 1, 2, 3]))] for _ in range(120000)]
    assert_pmf(samples, lambda k: 1 / 24, 0, 23)


def test_partial_permutation_is_sampling_without_replacement(rng):
    data = ["a", "b", "c", "d", "e"]
    firsts = []
    for _ in range(50000):
        out = rng.permutation(data, r=3)
        assert len(out) == 3 and len(set(out)) == 3
        firsts.append(data.index(out[0]))
    assert_pmf(firsts, lambda k: 0.2, 0, 4)
    assert data == ["a", "b", "c", "d", "e"]                              # input untouched
    assert rng.permutation(data, r=0) == []
    assert sorted(rng.permutation(data)) == data


def test_permutation_rejects_r_out_of_range(rng):
    for r in (4, 10, -1):
        with pytest.raises(ValueError):                                   # r > n used to corrupt memory
            rng.permutation([1, 2, 3], r=r)


def test_permutation_accepts_any_objects(rng):
    objs = [object(), "x", 3.5, (1, 2), None]
    out = rng.permutation(objs)
    assert sorted(map(id, out)) == sorted(map(id, objs))


def test_choice_uniform_and_permutation(rng):
    data = list("ABCDE")
    assert_pmf([data.index(c) for c in rng.choice(data, size=50000)], lambda k: 0.2, 0, 4)
    got = rng.choice(data, size=5, distribution="permutation")
    assert sorted(got) == data
    with pytest.raises(ValueError):
        rng.choice(data, size=6, distribution="permutation")
    with pytest.raises(ValueError):
        rng.choice([], size=1)
    with pytest.raises(ValueError):
        rng.choice(data, distribution="zipf")
    assert rng.choice(data, size=4, as_index=True) and all(0 <= i < 5 for i in rng.choice(data, size=20, as_index=True))


def test_choice_geometric_index_zero_is_reachable(rng):
    data = list("ABCDE")
    idx = rng.choice(data, size=60000, distribution="geometric", as_index=True, p=0.5)
    p0 = idx.count(0) / len(idx)
    assert abs(p0 - 0.5) < 0.02                                           # element A was never chosen before
    assert idx.count(4) / len(idx) > 0.05                                  # clamped tail lands on the last element
    rej = rng.choice(data, size=60000, distribution="geometric", as_index=True, p=0.5, overflow="reject")
    assert max(rej) == 4
    trunc = [0.5 * 0.5 ** k for k in range(5)]
    z = sum(trunc)
    assert_pmf(rej, lambda k: trunc[k] / z if 0 <= k < 5 else 0.0, 0, 4)    # exactly truncated geometric
    with pytest.raises(ValueError):
        rng.choice(data, distribution="geometric")                         # missing p


# ------------------------------------------------------------------ geometric
@pytest.mark.parametrize("p", [0.05, 0.3, 0.9])
def test_geometric_both_variants_match_pmf(rng, p):
    fails = rng.geometric(p, size=100000, method="failure")
    assert min(fails) == 0
    assert_pmf(fails, lambda k: geom_pmf_failures(p, k), 0, int(12 / p))
    trials = rng.geometric(p, size=100000, method="success")
    assert min(trials) == 1
    assert_pmf([t - 1 for t in trials], lambda k: geom_pmf_failures(p, k), 0, int(12 / p))


def test_geometric_edges_and_errors(rng):
    assert set(rng.geometric(1.0, size=50)) == {1}
    assert set(rng.geometric(1.0, size=50, method="failure")) == {0}
    assert all(v >= 1 for v in rng.geometric(1e-20, size=20))              # log1p keeps tiny p exact
    for bad in (0.0, -0.5, 1.5, float("nan")):
        with pytest.raises(ValueError):                                   # used to return -1 as if it were data
            rng.geometric(bad)
    with pytest.raises(ValueError):
        rng.geometric(0.5, method="nope")


# ------------------------------------------------------------------ poisson
@pytest.mark.parametrize("lam", [0.1, 3.0, 9.9, 10.0, 25.0, 100.0, 800.0, 5000.0])
def test_poisson_matches_exact_pmf(rng, lam):
    xs = rng.poisson(lam, size=200000)
    sd = math.sqrt(lam)
    lo, hi = max(0, int(lam - 8 * sd - 10)), int(lam + 8 * sd + 10)
    assert_pmf(xs, lambda k: poisson_pmf(lam, k), lo, hi)


def test_poisson_large_lambda_no_hang_no_garbage(rng):
    for lam in (745.0, 1000.0, 1e5, 1e9, 1e12, 1e15):
        xs = rng.poisson(lam, size=2000)
        m, v = mean_var(xs)
        assert abs(m - lam) < 6 * math.sqrt(lam / 2000) + 1
        assert abs(v / lam - 1) < 0.15
    assert min(rng.poisson(800.0, size=3000)) > 600                          # the old code returned 0 here


def test_poisson_methods_agree_and_validate(rng):
    a = rng.poisson(40.0, size=100000, method="sequential")
    b = rng.poisson(40.0, size=100000, method="ptrs")
    assert abs(sum(a) / len(a) - sum(b) / len(b)) < 0.15
    assert_pmf(a, lambda k: poisson_pmf(40.0, k), 5, 90)
    assert rng.poisson(0.0, size=5) == [0] * 5
    with pytest.raises(ValueError):
        rng.poisson(800.0, method="sequential")
    with pytest.raises(ValueError):
        rng.poisson(5.0, method="ptrs")
    with pytest.raises(ValueError, match="centered"):
        rng.poisson(5.0, method="centered")
    for bad in (-1.0, float("nan"), float("inf"), 1e16):
        with pytest.raises(ValueError):
            rng.poisson(bad)


# ------------------------------------------------------------------ binomial
CASES = [(5, 0.3, "summation"), (30, 0.5, "auto"), (50, 0.19, "auto"), (100, 0.5, "btrs"), (40, 0.25, "btrs"),
         (2000, 0.5, "auto"), (1000, 0.9, "auto"), (1000, 0.97, "auto"), (10 ** 6, 1e-5, "auto"),
         (10 ** 5, 3e-4, "inversion"), (60, 0.1, "inversion"), (20, 0.5, "btrs"), (7, 0.5, "inversion")]


@pytest.mark.parametrize("n,p,method", CASES)
def test_binomial_matches_exact_pmf(rng, n, p, method):
    xs = rng.binomial(n, p, size=200000, method=method)
    sd = math.sqrt(n * p * (1 - p))
    lo, hi = max(0, int(n * p - 8 * sd - 5)), min(n, int(n * p + 8 * sd + 5))
    assert_pmf(xs, lambda k: binom_pmf(n, p, k), lo, hi)


def test_binomial_huge_n_moments_and_old_failure(rng):
    xs = rng.binomial(2000, 0.5, size=4000)
    assert abs(sum(xs) / len(xs) - 1000) < 3                                 # BINV used to return n
    assert len(set(xs)) > 50
    for n, p in ((10 ** 9, 0.5), (2 ** 52, 0.3)):
        ys = rng.binomial(n, p, size=2000)
        m, v = mean_var(ys)
        assert abs(m - n * p) < 6 * math.sqrt(n * p * (1 - p) / 2000)
        assert abs(v / (n * p * (1 - p)) - 1) < 0.15


def test_binomial_edges_and_errors(rng):
    assert rng.binomial(20, 1.0, size=4) == [20] * 4                          # used to return 1
    assert rng.binomial(20, 0.0, size=4) == [0] * 4
    assert rng.binomial(0, 0.5, size=4) == [0] * 4
    assert rng.binomial(10, 1.0, method="inversion") == 10
    for n, p in ((20, 1.5), (20, -0.1), (-1, 0.5), (20, float("nan"))):
        with pytest.raises(ValueError):
            rng.binomial(n, p)
    with pytest.raises(ValueError):
        rng.binomial(19, 0.5, method="btrs")                                  # n*p < 10
    with pytest.raises(ValueError):
        rng.binomial(5000, 0.5, method="inversion")                           # would underflow
    with pytest.raises(ValueError):
        rng.binomial(10, 0.5, method="nope")


def test_poisson_approximation_is_explicit_and_inexact(rng):
    exact = rng.binomial(50, 0.19, size=200000)
    approx = rng.binomial(50, 0.19, size=200000, method="poisson_approx")
    assert abs(mean_var(exact)[1] - 50 * 0.19 * 0.81) < 0.1
    assert mean_var(approx)[1] > 9.0                                           # ~9.5 (Poisson) vs 7.7 (exact)
    assert max(approx) <= 50


# ------------------------------------------------------------------ multinomial
@pytest.mark.parametrize("method", ["auto", "alias", "sequencing"])
def test_multinomial_totals_and_means(rng, method):
    p = [0.1, 0.2, 0.3, 0.0, 0.4]
    n = 500
    acc = [0.0] * 5
    for _ in range(2000):
        c = rng.multinomial(n, p, method=method)
        assert sum(c) == n and c[3] == 0
        for i, v in enumerate(c):
            acc[i] += v
    for i in range(5):
        assert abs(acc[i] / 2000 - n * p[i]) < 6 * math.sqrt(n * p[i] * (1 - p[i]) / 2000) + 1e-9


def test_multinomial_large_n_was_broken(rng):
    c = rng.multinomial(5000, [0.5, 0.5])
    assert sum(c) == 5000 and 2300 < c[0] < 2700                               # used to be [5000, 0]
    c2 = rng.multinomial(10 ** 6, [0.25] * 4)
    assert sum(c2) == 10 ** 6 and all(abs(v - 250000) < 3000 for v in c2)


def test_multinomial_marginals_are_binomial(rng):
    firsts = [rng.multinomial(20, [0.3, 0.7])[0] for _ in range(100000)]
    assert_pmf(firsts, lambda k: binom_pmf(20, 0.3, k), 0, 20)


def test_multinomial_edges(rng):
    assert rng.multinomial(0, [0.5, 0.5]) == [0, 0]
    assert rng.multinomial(7, [1.0]) == [7]
    assert sum(rng.multinomial(3, [0.01] * 100)) == 3                          # alias path (r > n)
    for p in ([0.5, 0.6], [1.5, -0.5], [], [float("nan"), 1.0]):
        with pytest.raises(ValueError):
            rng.multinomial(10, p)                                              # r = 0 used to write counts[-1]
    with pytest.raises(ValueError):
        rng.multinomial(10, [0.5, 0.5], method="nope")
