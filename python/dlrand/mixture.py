"""Composition (mixture) and alias-table samplers. Engines are built once and reused."""
from __future__ import annotations

import ctypes
import operator
from ctypes import POINTER, Structure, byref, c_double, c_int64, c_void_p

from ._lib import WORKER, check

__all__ = ["MixtureBuilder", "Mixture", "AliasSampler"]


class _UniformParams(Structure):
    _fields_ = [("start", c_int64), ("range", c_int64)]


class _PoissonParams(Structure):
    _fields_ = [("lam", c_double)]


class _GeometricParams(Structure):
    _fields_ = [("p", c_double)]


class _BinomialParams(Structure):
    _fields_ = [("n", c_int64), ("p", c_double)]


_MODES = {"manual": 0, "residual": 1, "alias": 2}


class AliasSampler:
    """O(1) sampling of category indices 0..n-1 with the given (unnormalised) weights."""

    def __init__(self, rng, weights):
        w = [float(x) for x in weights]
        self._rng, self._lib, self._t = rng, rng._lib, c_void_p()
        arr = (c_double * max(len(w), 1))(*w)
        check(self._lib, self._lib.rl_alias_new(arr, len(w), byref(self._t)), "alias_sampler")
        self.n = len(w)

    def sample(self, size=None):
        if self._t is None:
            raise ValueError("sampler is closed")
        st = self._rng._live()
        if size is None:
            return self._lib.rl_alias_draw(st, self._t)
        n = operator.index(size)
        if n < 0:
            raise ValueError("size must be >= 0")
        return [self._lib.rl_alias_draw(st, self._t) for _ in range(n)]

    __call__ = sample

    def close(self):
        if getattr(self, "_t", None) is not None and self._t:
            self._lib.rl_alias_free(self._t)
        self._t = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class Mixture:
    """A finalized composition engine. Call it (or .sample(n)) to draw integers."""

    def __init__(self, rng, handle, keepalive):
        self._rng, self._lib, self._h, self._keep = rng, rng._lib, handle, keepalive

    def __call__(self):
        if not self._h:
            raise ValueError("mixture is closed")
        out = c_int64()
        check(self._lib, self._lib.rl_composition_sample(self._rng._live(), self._h, byref(out)),
              "mixture", self._keep.error())
        return out.value

    def sample(self, size):
        n = operator.index(size)
        if n < 0:
            raise ValueError("size must be >= 0")
        return [self() for _ in range(n)]

    def close(self):
        if getattr(self, "_h", None):
            self._lib.rl_composition_free(self._h)
        self._h = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class _Keep:
    """Holds parameter structs / callbacks alive for the lifetime of the engine."""

    def __init__(self):
        self.items, self.exc = [], None

    def error(self):
        return self.exc


class MixtureBuilder:
    """f = sum_i weight_i * F_i. Weights must be >= 0 and sum to 1 (within 1e-8)."""

    def __init__(self, rng):
        self._rng = rng
        self._specs = []   # (weight, kind, payload)

    def add_uniform(self, weight, start, range_val):
        """Integers start .. start+range_val-1, equally likely."""
        self._specs.append((weight, "uniform", (operator.index(start), operator.index(range_val))))
        return self

    def add_poisson(self, weight, lam):
        self._specs.append((weight, "poisson", (float(lam),)))
        return self

    def add_geometric(self, weight, p):
        """Geometric(p), support {1,2,...}."""
        self._specs.append((weight, "geometric", (float(p),)))
        return self

    def add_binomial(self, weight, n, p):
        self._specs.append((weight, "binomial", (operator.index(n), float(p))))
        return self

    def add_callable(self, weight, fn):
        """A Python worker: fn() -> int. Slower than the native components."""
        self._specs.append((weight, "callable", (fn,)))
        return self

    @classmethod
    def from_specs(cls, rng, components):
        b = cls(rng)
        for c in components:
            c = dict(c)
            w, kind = c.pop("weight"), c.pop("kind")
            try:
                {"uniform": lambda: b.add_uniform(w, c["start"], c["range"]),
                 "poisson": lambda: b.add_poisson(w, c["lam"]),
                 "geometric": lambda: b.add_geometric(w, c["p"]),
                 "binomial": lambda: b.add_binomial(w, c["n"], c["p"]),
                 "callable": lambda: b.add_callable(w, c["fn"])}[kind]()
            except KeyError as exc:
                raise ValueError(f"bad component spec (missing/unknown key {exc.args[0]!r}): {c!r}") from None
        return b

    def build(self, mode="alias") -> Mixture:
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {list(_MODES)}")
        if not self._specs:
            raise ValueError("a mixture needs at least one component")
        lib = self._rng._lib
        h = lib.rl_composition_new(_MODES[mode], len(self._specs))
        if not h:
            raise MemoryError("could not allocate composition engine")
        keep = _Keep()
        try:
            for i, (w, kind, payload) in enumerate(self._specs):
                if kind == "uniform":
                    params = _UniformParams(*payload)
                    worker = lib.rl_worker_discrete_uniform
                elif kind == "poisson":
                    params, worker = _PoissonParams(*payload), lib.rl_worker_poisson
                elif kind == "geometric":
                    params, worker = _GeometricParams(*payload), lib.rl_worker_geometric
                elif kind == "binomial":
                    params, worker = _BinomialParams(*payload), lib.rl_worker_binomial
                else:
                    fn = payload[0]

                    def tramp(_st, _p, out, fn=fn):
                        try:
                            out[0] = int(fn())
                            return 0
                        except BaseException as exc:      # noqa: BLE001
                            keep.exc = keep.exc or exc
                            return -4
                    cb = WORKER(tramp)
                    keep.items.append(cb)
                    check(lib, lib.rl_composition_set(h, i, float(w), ctypes.cast(cb, c_void_p), None), "mixture")
                    continue
                keep.items.append(params)
                addr = ctypes.cast(worker, c_void_p)
                check(lib, lib.rl_composition_set(h, i, float(w), addr, ctypes.addressof(params)), "mixture")
            check(lib, lib.rl_composition_finalize(h), "mixture (weights must be >= 0 and sum to 1)")
        except BaseException:
            lib.rl_composition_free(h)
            raise
        return Mixture(self._rng, h, keep)
