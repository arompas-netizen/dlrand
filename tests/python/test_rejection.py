import math

import pytest

import dlrand
from helpers import assert_cdf, assert_pmf, mean_var, poisson_pmf


def f_target(x):                      # x (2-x)^2 on [0, 2], unnormalised; area 4/3, mean 0.8, var 0.16
    return x * (2 - x) ** 2 if 0 <= x <= 2 else 0.0


def f_cdf(x):
    x = min(max(x, 0.0), 2.0)
    return (2 * x ** 2 - 4 * x ** 3 / 3 + x ** 4 / 4) / (4 / 3)


def test_valid_envelope_reproduces_the_target(rng):
    xs = rng.rejection_sampling(f_target, "uniform", 2.4, 0.0, 2.0, size=20000)
    assert_cdf(xs, f_cdf)
    m, v = mean_var(xs)
    assert abs(m - 0.8) < 0.02 and abs(v - 0.16) < 0.01


@pytest.mark.parametrize("proposal,M,p1,p2", [
    ("exponential", 3.1, 1.5, 0.0), ("cauchy", 2.6, 0.7, 0.5), ("laplace", 2.5, 0.7, 0.8),
    ("dl_box_muller", 1.8, 0.7, 0.5), ("dl_ziggurat", 1.8, 0.7, 0.5)])
def test_every_proposal_type(rng, proposal, M, p1, p2):
    xs = rng.rejection_sampling(f_target, proposal, M, p1, p2, size=15000)
    assert_cdf(xs, f_cdf)


@pytest.mark.parametrize("proposal,M,p1,p2", [          # the four examples shipped with the old library
    ("uniform", 1.2, 0.0, 2.0), ("exponential", 2.5, 1.5, 0.0), ("cauchy", 2.0, 0.7, 0.5),
    ("laplace", 1.8, 0.7, 0.8)])
def test_old_example_constants_are_now_detected(rng, proposal, M, p1, p2):
    with pytest.raises(dlrand.EnvelopeError):
        rng.rejection_sampling(f_target, proposal, M, p1, p2, size=5000)
    assert issubclass(dlrand.EnvelopeError, ValueError)


def test_envelope_check_can_be_disabled(rng):
    xs = rng.rejection_sampling(f_target, "uniform", 1.2, 0.0, 2.0, size=500, check_envelope=False)
    assert len(xs) == 500


def test_acceptance_rate_matches_theory(rng):
    n = 30000
    _, attempts = rng.rejection_sampling_stats(f_target, "uniform", 2.4, 0.0, 2.0, size=n)
    expected = (4 / 3) / 2.4
    sd = math.sqrt(expected * (1 - expected) / attempts)
    assert abs(n / attempts - expected) < 5 * sd + 0.005


def test_timeout_and_error_types(rng):
    with pytest.raises(dlrand.RejectionTimeoutError):
        rng.rejection_sampling(f_target, "uniform", 1e9, 0.0, 2.0, max_attempts=5, check_envelope=False)
    with pytest.raises(TimeoutError):
        rng.rejection_sampling(f_target, "uniform", 1e9, 0.0, 2.0, max_attempts=5, check_envelope=False)


def test_callback_exceptions_are_not_swallowed(rng):
    class Boom(RuntimeError):
        pass

    def bad(x):
        raise Boom("pdf failed")
    with pytest.raises(Boom):
        rng.rejection_sampling(bad, "uniform", 1.0, 0.0, 1.0)
    for value in (float("nan"), -1.0):
        with pytest.raises(ValueError):
            rng.rejection_sampling(lambda x, v=value: v, "uniform", 1.0, 0.0, 1.0)


@pytest.mark.parametrize("args", [("uniform", 2.4, 2.0, 0.0), ("uniform", 2.4, 1.0, 1.0),
                                  ("exponential", 2.4, -1.0, 0.0), ("cauchy", 2.4, 0.0, -1.0),
                                  ("laplace", 2.4, 0.0, 0.0), ("dl_box_muller", 2.4, 0.0, -2.0),
                                  ("uniform", 0.0, 0.0, 1.0), ("uniform", float("nan"), 0.0, 1.0)])
def test_invalid_proposal_parameters(rng, args):
    with pytest.raises(ValueError):
        rng.rejection_sampling(f_target, args[0], args[1], args[2], args[3])


def test_get_proposal_density_and_removed_generic_ziggurat(rng):
    assert abs(rng.get_proposal_density("uniform", 0.5, 0.0, 2.0) - 0.5) < 1e-15
    assert math.isnan(rng.get_proposal_density("cauchy", 0.0, 0.0, -1.0))
    with pytest.raises(NotImplementedError):
        rng.ziggurat(lambda x: x)


def test_positional_signature_is_backwards_compatible(rng):
    x = rng.rejection_sampling(f_target, 0, 2.4, 0.0, 2.0, 10000)           # (.., p2, max_attempts)
    assert 0.0 <= x <= 2.0
    assert rng.sample_custom_func(lambda k: 0.5 ** (k + 1), 1.0 + 1e-6) >= 0        # p_geom defaults to 0.5


def test_unknown_proposal(rng):
    for bad in ("zipf", 99):
        with pytest.raises(ValueError):
            rng.rejection_sampling(f_target, bad, 1.0, 0.0, 1.0)


def test_dl_target_with_ziggurat_proposal_needs_no_params_hack(rng):
    # type 5 used to receive the Ziggurat table through the target's params argument
    xs = rng.rejection_sampling(lambda x: math.exp(-0.5 * x * x), "dl_ziggurat", math.sqrt(2 * math.pi), 0.0, 1.0,
                                size=20000)
    m, v = mean_var(xs)
    assert abs(m) < 0.03 and abs(v - 1) < 0.05


# ------------------------------------------------------------------ discrete targets
def test_custom_array_default_M_and_distribution(rng):
    pmf = [0.1, 0.2, 0.3, 0.0, 0.4]
    xs = rng.sample_custom_array(pmf, size=100000)
    assert 3 not in xs
    assert_pmf(xs, lambda k: pmf[k] if 0 <= k < 5 else 0.0, 0, 4)
    unnormalised = rng.sample_custom_array([1, 2, 3, 4], size=50000)
    assert_pmf(unnormalised, lambda k: (k + 1) / 10, 0, 3)
    assert rng.sample_custom_array([0, 0, 5]) == 2


def test_custom_array_errors(rng):
    with pytest.raises(dlrand.EnvelopeError):
        rng.sample_custom_array([0.1, 0.2, 0.3, 0.4], M=1.0, size=2000)
    for bad in ([-1.0, 2.0], [], [float("nan")], [0.0, 0.0]):
        with pytest.raises(ValueError):
            rng.sample_custom_array(bad, size=3)
    with pytest.raises(ValueError):
        rng.sample_custom_array([0.0, 0.0], M=1.0, max_attempts=3)         # no mass: invalid, not a timeout
    with pytest.raises(ValueError):
        rng.sample_custom_array([0.5, 0.5], M=0.0)


def test_custom_func_matches_poisson(rng):
    lam = 4.0
    p = 0.2           # Geometric(0.2): g(x) = 0.2*0.8^x ; M >= sup f/g, f = Poisson(4)
    M = 1.2 * max(poisson_pmf(lam, k) / (p * (1 - p) ** k) for k in range(0, 400))
    xs = rng.sample_custom_func(lambda x: poisson_pmf(lam, x), M, p, size=60000)
    assert_pmf(xs, lambda k: poisson_pmf(lam, k), 0, 20)


def test_custom_func_errors(rng):
    with pytest.raises(dlrand.EnvelopeError):
        rng.sample_custom_func(lambda x: poisson_pmf(4.0, x), 0.5, 0.2, size=3000)
    with pytest.raises(KeyError):
        rng.sample_custom_func(lambda x: {}[x], 10.0, 0.5)
    with pytest.raises(ValueError):
        rng.sample_custom_func(lambda x: -1.0, 10.0, 0.5)
    with pytest.raises(ValueError):
        rng.sample_custom_func(lambda x: 1.0, 10.0, 1.5)
