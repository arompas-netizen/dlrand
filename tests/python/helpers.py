"""Statistical helpers. All tests use fixed seeds, so every outcome is deterministic (no flaky tests);
thresholds are ~5 sigma so a correct sampler passes with enormous margin and a wrong one fails clearly."""
import math


def binom_pmf(n, p, k):
    if k < 0 or k > n:
        return 0.0
    if p == 0.0:
        return 1.0 if k == 0 else 0.0
    if p == 1.0:
        return 1.0 if k == n else 0.0
    return math.exp(math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
                    + k * math.log(p) + (n - k) * math.log1p(-p))


def poisson_pmf(lam, k):
    if k < 0:
        return 0.0
    return math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1)) if lam > 0 else (1.0 if k == 0 else 0.0)


def geom_pmf_failures(p, k):
    return p * (1 - p) ** k if k >= 0 else 0.0


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def chi2_discrete(samples, pmf, lo, hi, min_expected=10.0):
    """Chi-square statistic and df for integer samples vs pmf over [lo, hi] (+ one 'outside' bin)."""
    n = len(samples)
    counts = {}
    outside = 0
    for s in samples:
        if lo <= s <= hi:
            counts[s] = counts.get(s, 0) + 1
        else:
            outside += 1
    probs = [pmf(k) for k in range(lo, hi + 1)]
    obs = [counts.get(k, 0) for k in range(lo, hi + 1)]
    probs.append(max(0.0, 1.0 - sum(probs)))
    obs.append(outside)
    # merge neighbouring bins until every expected count is >= min_expected
    mo, mp, co, cp = [], [], 0, 0.0
    for o, p in zip(obs, probs):
        co += o
        cp += p
        if cp * n >= min_expected:
            mo.append(co); mp.append(cp); co, cp = 0, 0.0
    if cp > 0 or co > 0:
        if mp:
            mo[-1] += co; mp[-1] += cp
        else:
            mo.append(co); mp.append(cp)
    stat = sum((o - n * p) ** 2 / (n * p) for o, p in zip(mo, mp) if p > 0)
    return stat, len(mo) - 1


def assert_pmf(samples, pmf, lo, hi, z=5.0):
    stat, df = chi2_discrete(samples, pmf, lo, hi)
    limit = df + z * math.sqrt(2.0 * df) + 5
    assert stat < limit, f"chi-square {stat:.1f} exceeds {limit:.1f} (df={df})"


def ks_stat(samples, cdf):
    xs = sorted(samples)
    n = len(xs)
    d = 0.0
    for i, x in enumerate(xs):
        f = cdf(x)
        d = max(d, (i + 1) / n - f, f - i / n)
    return d


def assert_cdf(samples, cdf, c=2.0):
    n = len(samples)
    d = ks_stat(samples, cdf)
    assert d < c / math.sqrt(n), f"KS distance {d:.5f} exceeds {c / math.sqrt(n):.5f}"


def mean_var(xs):
    n = len(xs)
    m = sum(xs) / n
    return m, sum((x - m) ** 2 for x in xs) / n
