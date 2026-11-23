"""RandomGenerator: the high-level Python interface to the dlrand C engines."""
from __future__ import annotations

import ctypes
import enum
import operator
import os
from ctypes import byref, c_double, c_int64, c_uint64

from . import _lib
from ._lib import PDF, PMF, DlrandError, RejectionTimeoutError, check, load_library

__all__ = ["RandomGenerator", "Proposal"]

_U64 = (1 << 64) - 1
_I64_MIN, _I64_MAX = -(1 << 63), (1 << 63) - 1


class Proposal(enum.IntEnum):
    """Proposal distributions of the continuous rejection engine."""
    UNIFORM = 0        # p1 = a, p2 = b
    EXPONENTIAL = 1    # p1 = scale (support x >= 0)
    CAUCHY = 2         # p1 = location, p2 = scale
    LAPLACE = 3        # p1 = mu, p2 = b
    DL_BOX_MULLER = 4  # p1 = mu, p2 = sigma
    DL_ZIGGURAT = 5    # p1 = mu, p2 = sigma


def _proposal(value) -> Proposal:
    if isinstance(value, str):
        try:
            return Proposal[value.upper()]
        except KeyError:
            raise ValueError(f"unknown proposal {value!r}; choose from {[p.name for p in Proposal]}") from None
    try:
        return Proposal(int(value))
    except ValueError:
        raise ValueError(f"invalid proposal type {value!r}") from None


def _size(size):
    if size is None:
        return None
    n = operator.index(size)
    if n < 0:
        raise ValueError("size must be >= 0")
    return n


def _i64(v, name="value"):
    v = operator.index(v)
    if not (_I64_MIN <= v <= _I64_MAX):
        raise ValueError(f"{name} does not fit in a signed 64-bit integer")
    return v


class _Callback:
    """Wraps a Python callable as a C callback. Exceptions are captured (never lost in the C frame)
    and re-raised by the caller; the callback returns NaN so the C engine stops with RL_ERR_CALLBACK."""

    def __init__(self, fn, ctype):
        self.exc = None

        def trampoline(x, _params):
            try:
                return float(fn(x))
            except BaseException as exc:          # noqa: BLE001 - must not propagate through C
                if self.exc is None:
                    self.exc = exc
                return float("nan")

        self.c = ctype(trampoline)


class RandomGenerator:
    """Seeded generator backed by xoshiro256++ (period 2^256-1) with 53-bit doubles.

    Same seed -> same stream, on every instance, regardless of what other generators exist
    (the C library has no shared mutable state). Not cryptographically secure. Methods taking
    ``size`` return a scalar when ``size is None`` and a list (or NumPy array with
    ``numpy=True``) otherwise.
    """

    def __init__(self, seed: int | None = None, lib_path: str | None = None, numpy: bool = False):
        self._lib = load_library(lib_path)
        self._lib_path = lib_path
        self._st = None
        self._zig = {}
        self._np = None
        if numpy:
            try:
                import numpy as _np
            except ImportError as exc:
                raise DlrandError("numpy=True requires NumPy to be installed") from exc
            self._np = _np
        if seed is None:
            seed = int.from_bytes(os.urandom(8), "little")
        seed = operator.index(seed)
        if not 0 <= seed <= _U64:
            raise ValueError("seed must satisfy 0 <= seed < 2**64")
        self._st = self._lib.rl_state_new(seed)
        if not self._st:
            raise MemoryError("could not allocate generator state")
        self.seed_value = seed

    # ------------------------------------------------------------------ lifecycle / state
    def close(self) -> None:
        lib = getattr(self, "_lib", None)
        if lib is None:
            return
        for t in getattr(self, "_zig", {}).values():
            lib.rl_dl_ziggurat_free(t)
        self._zig = {}
        if getattr(self, "_st", None):
            lib.rl_state_free(self._st)
            self._st = None

    def __del__(self):
        try:
            self.close()
        except Exception:                      # interpreter shutdown
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _live(self):
        if not self._st:
            raise DlrandError("generator is closed")
        return self._st

    def reseed(self, seed: int) -> None:
        seed = operator.index(seed)
        if not 0 <= seed <= _U64:
            raise ValueError("seed must satisfy 0 <= seed < 2**64")
        self._lib.rl_seed(self._live(), seed)
        self.seed_value = seed

    def getstate(self) -> tuple:
        """Snapshot of the full generator state (6 integers); restore with setstate()."""
        buf = (c_uint64 * 6)()
        self._lib.rl_state_get(self._live(), buf)
        return tuple(buf)

    def setstate(self, state) -> None:
        state = tuple(state)
        if len(state) != 6:
            raise ValueError("state must hold 6 integers")
        buf = (c_uint64 * 6)(*[operator.index(s) & _U64 for s in state])
        check(self._lib, self._lib.rl_state_set(self._live(), buf), "setstate")

    def spawn(self, n: int) -> list["RandomGenerator"]:
        """Return n generators on non-overlapping streams (child k is k * 2^128 draws ahead of
        this generator's current position; this generator is not advanced)."""
        n = operator.index(n)
        if n < 0:
            raise ValueError("n must be >= 0")
        snap = self.getstate()
        children = []
        tmp = RandomGenerator(0, lib_path=self._lib_path, numpy=self._np is not None)
        tmp.setstate(snap)
        for _ in range(n):
            self._lib.rl_jump(tmp._st)
            child = RandomGenerator(0, lib_path=self._lib_path, numpy=self._np is not None)
            child.setstate(tmp.getstate())
            children.append(child)
        tmp.close()
        return children

    # ------------------------------------------------------------------ helpers
    def _wrap(self, arr):
        if self._np is not None:
            return self._np.ctypeslib.as_array(arr).copy() if len(arr) else self._np.array([], dtype=arr._type_)
        return list(arr)

    def _fill(self, fn, ctype, size, *args, context=""):
        n = 1 if size is None else size
        arr = (ctype * n)()
        check(self._lib, fn(self._live(), *args, n, arr), context)
        return arr[0] if size is None else self._wrap(arr)

    def _raw_u01(self):
        return self._lib.rl_uniform01(self._live())

    # ------------------------------------------------------------------ uniform
    def uniform(self, low=0.0, high=1.0, size=None):
        """U[low, high) with 53 random bits."""
        return self._fill(self._lib.rl_uniform_fill, c_double, _size(size), float(low), float(high), context="uniform")

    def integers(self, low, high, size=None):
        """Unbiased integers in the INCLUSIVE range [low, high]."""
        return self._fill(self._lib.rl_integers_fill, c_int64, _size(size), _i64(low, "low"), _i64(high, "high"),
                          context="integers")

    # ------------------------------------------------------------------ discrete engines
    _GEO = {"success": 0, "trials": 0, "failure": 1, "failures": 1}

    def geometric(self, p, size=None, method="success"):
        """Geometric(p). method='success': trial of first success, support {1,2,...};
        method='failure': failures before first success, support {0,1,...}."""
        try:
            variant = self._GEO[method]
        except KeyError:
            raise ValueError(f"unknown geometric method {method!r}; use 'success' or 'failure'") from None
        return self._fill(self._lib.rl_geometric_fill, c_int64, _size(size), float(p), variant, context="geometric")

    _POISSON = {"auto": 0, "sequential": 1, "ptrs": 2}

    def poisson(self, lam, size=None, method="auto"):
        """Poisson(lam), exact for 0 <= lam <= 1e12. method: auto | sequential (lam <= 700) | ptrs (lam >= 10)."""
        if method == "centered":
            raise ValueError("method 'centered' was removed (it failed for lam > ~745); use 'ptrs' or 'auto'")
        try:
            m = self._POISSON[method]
        except KeyError:
            raise ValueError(f"unknown poisson method {method!r}; choose from {list(self._POISSON)}") from None
        return self._fill(self._lib.rl_poisson_fill, c_int64, _size(size), float(lam), m, context="poisson")

    _BINOMIAL = {"auto": 0, "summation": 1, "inversion": 2, "binv": 2, "btrs": 3, "poisson_approx": 4, "poisson": 4}

    def binomial(self, n, p, size=None, method="auto"):
        """Binomial(n, p). auto = exact inversion (n*min(p,1-p) < 10) or exact BTRS. 'poisson_approx' is an
        APPROXIMATION and is never chosen automatically."""
        try:
            m = self._BINOMIAL[method]
        except KeyError:
            raise ValueError(f"unknown binomial method {method!r}; choose from {list(self._BINOMIAL)}") from None
        s = _size(size)
        cnt = 1 if s is None else s
        arr = (c_int64 * cnt)()
        check(self._lib, self._lib.rl_binomial_fill(self._live(), _i64(n, "n"), float(p), m, cnt, arr), "binomial")
        return arr[0] if s is None else self._wrap(arr)

    def multinomial(self, n, p_vector, method="auto"):
        """Counts of n trials over len(p_vector) categories (p must sum to 1 within 1e-8)."""
        mm = {"auto": 0, "alias": 1, "sequencing": 2}
        if method not in mm:
            raise ValueError(f"unknown multinomial method {method!r}; choose from {list(mm)}")
        p = [float(x) for x in p_vector]
        r = len(p)
        parr = (c_double * max(r, 1))(*p)
        out = (c_int64 * max(r, 1))()
        check(self._lib, self._lib.rl_multinomial(self._live(), _i64(n, "n"), r, parr, mm[method], out), "multinomial")
        return list(out)[:r]

    def bernoulli_sampling(self, dataset, p, method="auto"):
        """Keep each element of dataset independently with probability p (original order preserved)."""
        mm = {"standard": 0, "jump": 1}
        n = len(dataset)
        if method == "auto":
            method = "jump" if float(p) < 0.25 else "standard"
        if method not in mm:
            raise ValueError("method must be 'auto', 'standard' or 'jump'")
        out = (c_int64 * max(n, 1))()
        count = c_int64(0)
        check(self._lib, self._lib.rl_bernoulli_select(self._live(), n, float(p), mm[method], out, byref(count)),
              "bernoulli_sampling")
        return [dataset[i] for i in out[:count.value]]

    def _permuted_indices(self, n, r):
        arr = (c_int64 * max(n, 1))(*range(n))
        check(self._lib, self._lib.rl_permutation_partial(self._live(), arr, n, r), "permutation")
        return arr[:r]

    def permutation(self, dataset, r=None):
        """Uniform random r-permutation (sampling without replacement) of any sequence; dataset is not modified."""
        n = len(dataset)
        r = n if r is None else operator.index(r)
        if r < 0 or r > n:
            raise ValueError(f"r must satisfy 0 <= r <= len(dataset) ({n}), got {r}")
        return [dataset[i] for i in self._permuted_indices(n, r)]

    def choice(self, data, size=1, distribution="uniform", as_index=False, overflow="clamp", **params):
        """Pick elements of ``data``.

        distribution: 'uniform' (with replacement) | 'permutation' (without replacement, size <= len) |
        'geometric' (p=...) | 'poisson' (lam=...); the last two draw a 0-based index, and draws >= len(data)
        are clamped onto the last element (overflow='clamp') or redrawn (overflow='reject')."""
        n = len(data)
        if n == 0:
            raise ValueError("cannot choose from an empty sequence")
        size = 1 if size is None else _size(size)
        if overflow not in ("clamp", "reject"):
            raise ValueError("overflow must be 'clamp' or 'reject'")
        if distribution == "uniform":
            idx = self.integers(0, n - 1, size)
        elif distribution == "permutation":
            if size > n:
                raise ValueError(f"cannot take {size} elements without replacement from {n}")
            idx = self._permuted_indices(n, size)
        elif distribution in ("geometric", "poisson"):
            if distribution == "geometric":
                draw = lambda k: self.geometric(params["p"], k, method="failure")   # noqa: E731
            else:
                draw = lambda k: self.poisson(params["lam"], k)                      # noqa: E731
            try:
                idx = draw(size)
            except KeyError as exc:
                raise ValueError(f"distribution {distribution!r} needs the keyword {exc.args[0]!r}") from None
            if overflow == "clamp":
                idx = [min(i, n - 1) for i in idx]
            else:
                for _ in range(10000):
                    bad = [k for k, i in enumerate(idx) if i >= n]
                    if not bad:
                        break
                    for k, i in zip(bad, draw(len(bad))):
                        idx[k] = i
                else:
                    raise RejectionTimeoutError("choice(overflow='reject') could not draw valid indices")
        else:
            raise ValueError("distribution must be 'uniform', 'permutation', 'geometric' or 'poisson'")
        return list(idx) if as_index else [data[i] for i in idx]

    # ------------------------------------------------------------------ custom discrete targets (rejection)
    def sample_custom_array(self, pmf, M=None, size=None, max_attempts=1_000_000):
        """Sample {0..len(pmf)-1} with weights pmf by rejection from the uniform proposal.
        M defaults to the smallest valid constant, len(pmf)*max(pmf)."""
        vals = [float(x) for x in pmf]
        arr = (c_double * max(len(vals), 1))(*vals)
        if M is None:
            M = self._lib.rl_rejection_array_min_M(arr, len(vals))
            if M != M:
                raise ValueError("pmf must be non-empty, finite and non-negative")
        s = _size(size)
        cnt = 1 if s is None else s
        out = (c_int64 * cnt)()
        check(self._lib, self._lib.rl_rejection_array_fill(self._live(), arr, len(vals), float(M),
                                                           int(max_attempts), cnt, out), "sample_custom_array")
        return out[0] if s is None else self._wrap(out)

    def sample_custom_func(self, pmf_func, M, p_geom=0.5, size=None, max_attempts=100_000):
        """Sample {0,1,2,...} with weights pmf_func(x) by rejection from Geometric(p_geom) (failures variant).
        Needs pmf_func(x) <= M * (1-p)^x * p for all x; a violation raises EnvelopeError."""
        cb = _Callback(pmf_func, PMF)
        s = _size(size)
        out = []
        for _ in range(1 if s is None else s):
            v = c_int64()
            rc = self._lib.rl_rejection_func(self._live(), cb.c, None, float(p_geom), float(M),
                                             int(max_attempts), byref(v))
            check(self._lib, rc, "sample_custom_func", cb.exc)
            out.append(v.value)
        return out[0] if s is None else self._np.array(out) if self._np is not None else out

    # ------------------------------------------------------------------ continuous engines
    def exponential(self, scale=1.0, size=None):
        return self._fill(self._lib.rl_exponential_fill, c_double, _size(size), float(scale), context="exponential")

    def cauchy(self, location=0.0, scale=1.0, size=None):
        return self._fill(self._lib.rl_cauchy_fill, c_double, _size(size), float(location), float(scale),
                          context="cauchy")

    def laplace(self, location=0.0, scale=1.0, size=None):
        """Laplace with the given location (mu) and scale (b)."""
        return self._fill(self._lib.rl_laplace_fill, c_double, _size(size), float(location), float(scale),
                          context="laplace")

    def dl_box_muller(self, mu=0.0, sigma=1.0, size=None):
        """DL (De Moivre-Laplace) distribution with mean mu and standard deviation sigma, by Box-Muller."""
        return self._fill(self._lib.rl_dl_box_muller_fill, c_double, _size(size), float(mu), float(sigma),
                          context="dl_box_muller")

    normal_bm = dl_box_muller      # name used by earlier versions of the library

    def _table(self, layers):
        layers = operator.index(layers)
        t = self._zig.get(layers)
        if t is None:
            t = self._lib.rl_dl_ziggurat_new(layers)
            if not t:
                raise ValueError("layers must be an integer in [8, 65536]")
            self._zig[layers] = t
        return t

    def dl_ziggurat(self, size=None, layers=256, mu=0.0, sigma=1.0):
        """DL(mu, sigma^2) by the Ziggurat method; the table for `layers` strips is solved on first use."""
        mu, sigma = float(mu), float(sigma)
        if not (sigma >= 0.0 and sigma != float("inf") and mu == mu and abs(mu) != float("inf")):
            raise ValueError("dl_ziggurat: mu must be finite and sigma finite and >= 0")
        t = self._table(layers)
        s = _size(size)
        n = 1 if s is None else s
        arr = (c_double * n)()
        check(self._lib, self._lib.rl_dl_ziggurat_fill(self._live(), t, n, arr), "dl_ziggurat")
        if mu != 0.0 or sigma != 1.0:
            arr[:] = [mu + sigma * v for v in arr]
        return arr[0] if s is None else self._wrap(arr)

    # ------------------------------------------------------------------ continuous rejection
    def rejection_sampling_stats(self, target_pdf, proposal_type, M, p1, p2=1.0, max_attempts=100_000,
                                 size=None, check_envelope=True, layers=256):
        """Like rejection_sampling but returns (samples, total_candidates_drawn)."""
        ptype = _proposal(proposal_type)
        cb = _Callback(target_pdf, PDF)
        table = self._table(layers) if ptype == Proposal.DL_ZIGGURAT else None
        s = _size(size)
        out, attempts_total = [], 0
        x, att = c_double(), c_int64()
        for _ in range(1 if s is None else s):
            rc = self._lib.rl_rejection_continuous(self._live(), cb.c, None, int(ptype), float(M), float(p1),
                                                   float(p2), table, int(max_attempts), int(bool(check_envelope)),
                                                   byref(x), byref(att))
            check(self._lib, rc, "rejection_sampling", cb.exc)
            out.append(x.value)
            attempts_total += att.value
        if s is None:
            return out[0], attempts_total
        return (self._np.array(out) if self._np is not None else out), attempts_total

    def rejection_sampling(self, target_pdf, proposal_type, M, p1, p2=1.0, max_attempts=100_000,
                           size=None, check_envelope=True, layers=256):
        """Von Neumann rejection for a continuous target f (need not be normalised).

        Requires f(x) <= M*g(x) wherever f(x) > 0, with g the proposal density (see ``Proposal`` for p1/p2).
        With check_envelope=True a violation seen during sampling raises EnvelopeError instead of silently
        returning biased samples."""
        return self.rejection_sampling_stats(target_pdf, proposal_type, M, p1, p2, max_attempts=max_attempts,
                                             size=size, check_envelope=check_envelope, layers=layers)[0]

    def get_proposal_density(self, proposal_type, x, p1, p2=1.0):
        """Analytical density g(x) of a proposal (NaN for invalid parameters)."""
        return self._lib.rl_proposal_density(int(_proposal(proposal_type)), float(x), float(p1), float(p2))

    def ziggurat(self, *args, **kwargs):
        raise NotImplementedError(
            "the generic ziggurat() of earlier versions never worked (its tables were never built) and was "
            "removed. Use dl_ziggurat() for the DL distribution, or rejection_sampling() for other densities.")

    # ------------------------------------------------------------------ composition / alias
    def mixture(self, components, mode="alias"):
        """Build a reusable discrete mixture; see dlrand.mixture.MixtureBuilder."""
        from .mixture import MixtureBuilder
        return MixtureBuilder.from_specs(self, components).build(mode)

    def sample_mixture(self, components, size=None, mode="alias"):
        """Convenience: build the mixture once, draw `size` values, free it."""
        mix = self.mixture(components, mode)
        try:
            return mix() if size is None else mix.sample(size)
        finally:
            mix.close()

    def alias_sampler(self, weights):
        from .mixture import AliasSampler
        return AliasSampler(self, weights)
