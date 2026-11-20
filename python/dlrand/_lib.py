"""Low-level ctypes binding: library discovery, C signatures, and status handling."""
from __future__ import annotations

import ctypes
import glob
import os
import sys
from ctypes import CFUNCTYPE, POINTER, Structure, c_double, c_int, c_int64, c_uint64, c_void_p

__all__ = ["DlrandError", "RejectionTimeoutError", "EnvelopeError", "load_library", "check",
           "PDF", "PMF", "WORKER", "Analysis", "STATUS"]

# Status codes (mirror rl_common.h)
STATUS = {"OK": 0, "ARG": -1, "TIMEOUT": -2, "ALLOC": -3, "CALLBACK": -4, "ENVELOPE": -5}


class DlrandError(Exception):
    """Base class of every error raised by dlrand itself."""


class RejectionTimeoutError(DlrandError, TimeoutError):
    """A rejection sampler exceeded ``max_attempts`` without accepting a sample."""


class EnvelopeError(DlrandError, ValueError):
    """Detected f(x) > M*g(x): the supplied envelope constant M is too small."""


PDF = CFUNCTYPE(c_double, c_double, c_void_p)            # double f(double x, void* params)
PMF = CFUNCTYPE(c_double, c_int64, c_void_p)             # double f(int64 x, void* params)
WORKER = CFUNCTYPE(c_int, c_void_p, c_void_p, POINTER(c_int64))


class Analysis(Structure):
    _fields_ = [("a", c_double), ("b", c_double), ("f_min", c_double), ("f_max", c_double),
                ("area", c_double), ("total_area", c_double)]


def _candidates():
    here = os.path.dirname(os.path.abspath(__file__))
    env = os.environ.get("DLRAND_PATH")
    if env:
        yield env
    for pat in ("_dlrand*.so", "_dlrand*.pyd", "_dlrand*.dylib", "_dlrand*.dll",
                "libdlrand*.so", "libdlrand*.dylib", "dlrand*.dll"):
        yield from sorted(glob.glob(os.path.join(here, pat)))
    root = os.path.dirname(os.path.dirname(here))        # source checkout: <repo>/python/dlrand
    for sub in ("build", "build/Release", "build/Debug", "."):
        for pat in ("libdlrand.so", "libdlrand.dylib", "dlrand.dll", "libdlrand.dll"):
            yield os.path.join(root, sub, pat)


_cache: dict = {}


def load_library(path: str | None = None):
    """Load (once) and declare the shared library. ``path`` overrides discovery."""
    key = os.path.abspath(path) if path else None
    if key in _cache:
        return _cache[key]
    tried = []
    for cand in ([path] if path else _candidates()):
        if cand and os.path.exists(cand):
            try:
                lib = ctypes.CDLL(os.path.abspath(cand))
            except OSError as exc:
                tried.append(f"{cand}: {exc}")
                continue
            _declare(lib)
            _cache[key] = lib
            return lib
        tried.append(f"{cand}: not found")
    raise DlrandError(
        "could not locate the dlrand shared library. Install the package (pip install .), build it with "
        "CMake, or set DLRAND_PATH. Tried:\n  " + "\n  ".join(tried[:8]))


def _declare(lib) -> None:
    P = POINTER
    sig = {
        # name: (restype, argtypes)
        "rl_version": (ctypes.c_char_p, []),
        "rl_status_string": (ctypes.c_char_p, [c_int]),
        "rl_state_new": (c_void_p, [c_uint64]),
        "rl_state_free": (None, [c_void_p]),
        "rl_seed": (None, [c_void_p, c_uint64]),
        "rl_next_u64": (c_uint64, [c_void_p]),
        "rl_uniform01": (c_double, [c_void_p]),
        "rl_jump": (None, [c_void_p]),
        "rl_long_jump": (None, [c_void_p]),
        "rl_state_get": (None, [c_void_p, P(c_uint64)]),
        "rl_state_set": (c_int, [c_void_p, P(c_uint64)]),
        "rl_loggam": (c_double, [c_double]),
        "rl_uniform_fill": (c_int, [c_void_p, c_double, c_double, c_int64, P(c_double)]),
        "rl_integers_fill": (c_int, [c_void_p, c_int64, c_int64, c_int64, P(c_int64)]),
        "rl_permutation_partial": (c_int, [c_void_p, P(c_int64), c_int64, c_int64]),
        "rl_bernoulli_select": (c_int, [c_void_p, c_int64, c_double, c_int, P(c_int64), P(c_int64)]),
        "rl_geometric_fill": (c_int, [c_void_p, c_double, c_int, c_int64, P(c_int64)]),
        "rl_poisson_fill": (c_int, [c_void_p, c_double, c_int, c_int64, P(c_int64)]),
        "rl_binomial_fill": (c_int, [c_void_p, c_int64, c_double, c_int, c_int64, P(c_int64)]),
        "rl_multinomial": (c_int, [c_void_p, c_int64, c_int64, P(c_double), c_int, P(c_int64)]),
        "rl_alias_new": (c_int, [P(c_double), c_int64, P(c_void_p)]),
        "rl_alias_draw": (c_int64, [c_void_p, c_void_p]),
        "rl_alias_free": (None, [c_void_p]),
        "rl_composition_new": (c_void_p, [c_int, c_int]),
        "rl_composition_set": (c_int, [c_void_p, c_int, c_double, c_void_p, c_void_p]),
        "rl_composition_finalize": (c_int, [c_void_p]),
        "rl_composition_sample": (c_int, [c_void_p, c_void_p, P(c_int64)]),
        "rl_composition_free": (None, [c_void_p]),
        "rl_rejection_array": (c_int, [c_void_p, P(c_double), c_int64, c_double, c_int64, P(c_int64)]),
        "rl_rejection_array_fill": (c_int, [c_void_p, P(c_double), c_int64, c_double, c_int64, c_int64, P(c_int64)]),
        "rl_rejection_array_min_M": (c_double, [P(c_double), c_int64]),
        "rl_rejection_func": (c_int, [c_void_p, PMF, c_void_p, c_double, c_double, c_int64, P(c_int64)]),
        "rl_exponential_fill": (c_int, [c_void_p, c_double, c_int64, P(c_double)]),
        "rl_cauchy_fill": (c_int, [c_void_p, c_double, c_double, c_int64, P(c_double)]),
        "rl_laplace_fill": (c_int, [c_void_p, c_double, c_double, c_int64, P(c_double)]),
        "rl_dl_box_muller_fill": (c_int, [c_void_p, c_double, c_double, c_int64, P(c_double)]),
        "rl_dl_ziggurat_new": (c_void_p, [c_int]),
        "rl_dl_ziggurat_free": (None, [c_void_p]),
        "rl_dl_ziggurat_fill": (c_int, [c_void_p, c_void_p, c_int64, P(c_double)]),
        "rl_dl_pdf": (c_double, [c_double, c_void_p]),
        "rl_proposal_density": (c_double, [c_int, c_double, c_double, c_double]),
        "rl_rejection_continuous": (c_int, [c_void_p, PDF, c_void_p, c_int, c_double, c_double, c_double,
                                            c_void_p, c_int64, c_int, P(c_double), P(c_int64)]),
        "rl_analyze_and_normalize": (c_int, [PDF, c_void_p, c_double, c_double, c_double, c_double,
                                             c_int, P(Analysis)]),
    }
    for name, (res, args) in sig.items():
        fn = getattr(lib, name)
        fn.restype = res
        fn.argtypes = args
    # native worker entry points are passed to the composition engine as raw addresses
    for name in ("rl_worker_discrete_uniform", "rl_worker_poisson", "rl_worker_geometric", "rl_worker_binomial"):
        getattr(lib, name)


def check(lib, rc: int, context: str = "", callback_error: BaseException | None = None) -> None:
    """Translate a C status code into the matching Python exception."""
    if rc == 0:
        return
    if rc == STATUS["CALLBACK"] and callback_error is not None:
        raise callback_error
    msg = lib.rl_status_string(rc).decode()
    if context:
        msg = f"{context}: {msg}"
    if rc == STATUS["ARG"] or rc == STATUS["CALLBACK"]:
        raise ValueError(msg)
    if rc == STATUS["TIMEOUT"]:
        raise RejectionTimeoutError(msg)
    if rc == STATUS["ALLOC"]:
        raise MemoryError(msg)
    if rc == STATUS["ENVELOPE"]:
        raise EnvelopeError(msg)
    raise DlrandError(f"{msg} (code {rc})")
