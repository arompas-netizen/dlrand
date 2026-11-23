import math

import pytest

import dlrand
from helpers import assert_cdf, mean_var
from dlrand.optimizer import _density


def semicircle(x):
    return math.sqrt(1 - x * x) if -1 <= x <= 1 else 0.0


def semicircle_cdf(x):
    x = min(max(x, -1.0), 1.0)
    return (x * math.sqrt(1 - x * x) + math.asin(x) + math.pi / 2) / math.pi


def gamma2(x):
    return x * math.exp(-x) if x >= 0 else 0.0


def gamma2_cdf(x):
    return 1 - (1 + x) * math.exp(-x) if x > 0 else 0.0


def dl_pdf(x):
    return math.exp(-0.5 * x * x)


def true_sup_ratio(f, ptype, p1, p2, lo, hi, n=200001):
    best = 0.0
    for i in range(n):
        x = lo + (hi - lo) * i / (n - 1)
        fx = f(x)
        if fx > 0:
            g = _density(ptype, x, p1, p2)
            best = max(best, fx / g if g > 0 else float("inf"))
    return best


def test_analyze_explicit_domain(rng):
    a = dlrand.RejectionOptimizer(rng).analyze(semicircle, -1.0, 1.0)
    assert (a.a, a.b) == (-1.0, 1.0)
    assert abs(a.area - math.pi / 2) < 1e-3 and abs(a.f_max - 1.0) < 1e-6 and a.f_min == 0.0


def test_analyze_auto_support_and_mass_trim(rng):
    o = dlrand.RejectionOptimizer(rng)
    a = o.analyze(dl_pdf, tail_mass=0.01)
    assert abs(a.total_area - math.sqrt(2 * math.pi)) < 1e-6
    assert abs(a.area / a.total_area - 0.99) < 0.003            # the "99% of the mass" claim is now true
    assert -2.7 < a.a < -2.4 and 2.4 < a.b < 2.7


def test_analyze_off_center_target(rng):
    o = dlrand.RejectionOptimizer(rng)
    shifted = lambda x: math.exp(-0.5 * ((x - 100.0) / 2.0) ** 2)     # noqa: E731
    a = o.analyze(shifted, center=100.0)
    b = o.analyze(shifted)                                            # found even with the default centre 0
    for r in (a, b):
        assert abs(r.total_area - 2.0 * math.sqrt(2 * math.pi)) < 1e-3 and 90 < r.a < 100 < r.b < 110


def test_analyze_failure_modes(rng):
    o = dlrand.RejectionOptimizer(rng)
    with pytest.raises(ValueError):
        o.analyze(lambda x: x * x)                                    # no decaying support
    with pytest.raises(ValueError):
        o.analyze(lambda x: float("nan"), -1.0, 1.0)
    with pytest.raises(ZeroDivisionError):
        o.analyze(lambda x: 1 / 0, -1.0, 1.0)                         # user exceptions propagate


def test_python_density_mirrors_the_c_density(rng):
    lib = dlrand.load_library()
    cases = [(dlrand.Proposal.UNIFORM, -1.0, 3.0), (dlrand.Proposal.EXPONENTIAL, 1.7, 0.0),
             (dlrand.Proposal.CAUCHY, 0.3, 0.8), (dlrand.Proposal.LAPLACE, -0.4, 1.2),
             (dlrand.Proposal.DL_BOX_MULLER, 0.5, 0.9), (dlrand.Proposal.DL_ZIGGURAT, 0.5, 0.9)]
    for ptype, p1, p2 in cases:
        for x in (-5.0, -1.0, -0.2, 0.0, 0.3, 1.0, 2.5, 7.0, 60.0):
            c = lib.rl_proposal_density(int(ptype), x, p1, p2)
            assert abs(_density(ptype, x, p1, p2) - c) <= 1e-12 * max(1.0, abs(c))


def test_semicircle_plans_are_valid_and_sample_correctly(rng):
    # The old optimizer returned M = 0 for Uniform/Exponential and mis-sized envelopes elsewhere.
    o = dlrand.RejectionOptimizer(rng)
    plans = o.plan(semicircle, domain=(-1.0, 1.0))
    names = {p.name for p in plans}
    assert {"UNIFORM", "CAUCHY", "LAPLACE", "DL_BOX_MULLER", "DL_ZIGGURAT"} <= names
    assert "EXPONENTIAL" in o.skipped and "x >= 0" in o.skipped["EXPONENTIAL"]
    assert [p.efficiency for p in plans] == sorted((p.efficiency for p in plans), reverse=True)
    for p in plans:
        assert 0 < p.efficiency <= 1.0
        assert p.M >= true_sup_ratio(semicircle, p.proposal, p.p1, p.p2, -1.5, 1.5)
        xs = p.sample(rng, semicircle, size=8000)
        assert_cdf(xs, semicircle_cdf)
        assert abs(mean_var(xs)[1] - 0.25) < 0.015
    uni = next(p for p in plans if p.name == "UNIFORM")
    assert (uni.p1, uni.p2) == (-1.0, 1.0) and uni.M >= 2.0
    assert "restricted" in uni.notes


def test_gamma_target_includes_a_working_exponential_plan(rng):
    o = dlrand.RejectionOptimizer(rng)
    plans = o.plan(gamma2)                                           # automatic domain
    assert "EXPONENTIAL" in {p.name for p in plans}
    for p in plans:
        if p.name in ("UNIFORM",):
            continue                                                  # uniform samples the trimmed domain only
        xs = p.sample(rng, gamma2, size=6000)
        assert_cdf(xs, gamma2_cdf)
        assert abs(mean_var(xs)[0] - 2.0) < 0.15


def test_best_report_and_benchmark(rng):
    o = dlrand.RejectionOptimizer(rng)
    best = o.best(semicircle, domain=(-1, 1))
    assert best.efficiency == max(p.efficiency for p in o.plan(semicircle, domain=(-1, 1)))
    stats = o.benchmark(semicircle, best, n=6000)
    assert abs(stats["acceptance"] - stats["expected"]) < 0.06 and stats["seconds_per_sample"] > 0
    text = o.report(o.plan(semicircle, domain=(-1, 1)))
    assert "UNIFORM" in text and "skipped" in text


def test_compare_table(rng):
    text = dlrand.RejectionOptimizer(rng).compare(semicircle, domain=(-1, 1), n=1500)
    assert "UNIFORM" in text and "measured" in text and "skipped" in text


def test_plan_input_validation(rng):
    o = dlrand.RejectionOptimizer(rng)
    with pytest.raises(ValueError):
        o.plan(semicircle, domain=(1.0, -1.0))
    with pytest.raises(ValueError):
        o.plan(lambda x: 0.0, domain=(-1.0, 1.0))
    with pytest.raises(ValueError):
        o.plan(lambda x: -1.0 if x > 0 else 1.0, domain=(-1.0, 1.0))
    with pytest.raises(ValueError):
        o.best(gamma2, domain=(-5.0, 5.0), proposals=["exponential"])  # a < 0 -> every candidate skipped
