"""RejectionOptimizer: choose a proposal distribution and a *valid* envelope constant M for a target density."""
from __future__ import annotations

import math
import time
from ctypes import byref
from dataclasses import dataclass, field

from ._lib import PDF, Analysis as _CAnalysis, check
from .generator import Proposal, RandomGenerator, _Callback, _proposal

__all__ = ["RejectionOptimizer", "ProposalPlan", "Analysis"]

_SQRT_2PI = math.sqrt(2.0 * math.pi)
_INF = float("inf")


@dataclass(frozen=True)
class Analysis:
    """Numerical summary of a target density over [a, b]."""
    a: float
    b: float
    f_min: float
    f_max: float
    area: float          # integral of f over [a, b]
    total_area: float    # integral over the whole scanned window (equals `area` for an explicit domain)


@dataclass
class ProposalPlan:
    """A proposal with tuned parameters and a verified envelope constant."""
    name: str
    proposal: Proposal
    p1: float
    p2: float
    M: float
    efficiency: float                 # expected acceptance rate = area / M
    domain: tuple
    notes: str = ""

    def sample(self, rng: RandomGenerator, target_pdf, size=None, **kwargs):
        return rng.rejection_sampling(target_pdf, self.proposal, self.M, self.p1, self.p2, size=size, **kwargs)


def _density(ptype: int, x: float, p1: float, p2: float) -> float:
    """Python mirror of rl_proposal_density (kept identical; tests compare the two)."""
    if ptype == Proposal.UNIFORM:
        return 1.0 / (p2 - p1) if p1 <= x <= p2 else 0.0
    if ptype == Proposal.EXPONENTIAL:
        return math.exp(-x / p1) / p1 if x >= 0.0 else 0.0
    if ptype == Proposal.CAUCHY:
        z = (x - p1) / p2
        return 1.0 / (math.pi * p2 * (1.0 + z * z))
    if ptype == Proposal.LAPLACE:
        return math.exp(-abs(x - p1) / p2) / (2.0 * p2)
    z = (x - p1) / p2
    return math.exp(-0.5 * z * z) / (p2 * _SQRT_2PI) if abs(z) < 38.0 else 0.0


def _max_ratio(ptype, p1, p2, xs, fx):
    """max f/g over the grid; +inf if g vanishes where f is positive."""
    best, arg = 0.0, 0
    for i, (x, f) in enumerate(zip(xs, fx)):
        if f <= 0.0:
            continue
        g = _density(ptype, x, p1, p2)
        if g <= 0.0:
            return _INF, i
        r = f / g
        if r > best:
            best, arg = r, i
    return best, arg


def _golden_max(fn, lo, hi, iters=60):
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    c, d = hi - gr * (hi - lo), lo + gr * (hi - lo)
    fc, fd = fn(c), fn(d)
    for _ in range(iters):
        if fc > fd:
            hi, d, fd = d, c, fc
            c = hi - gr * (hi - lo)
            fc = fn(c)
        else:
            lo, c, fc = c, d, fd
            d = lo + gr * (hi - lo)
            fd = fn(d)
    return max(fc, fd)


@dataclass
class RejectionOptimizer:
    """Finds, for a target density f, the proposal parameters minimising M = sup f/g and reports the
    resulting efficiency. M is evaluated on a dense grid, refined by golden-section search around the
    maximum and inflated by ``safety``; the sampler additionally verifies f <= M*g on every candidate
    (check_envelope=True), so an undersized M is reported rather than silently biasing the output.

    Scope: M is estimated on the analysed domain (extended by 50% for infinite-support proposals).
    If f has non-negligible mass further out, pass a wider ``domain``.
    """
    rng: RandomGenerator
    skipped: dict = field(default_factory=dict)

    # ---------------------------------------------------------------- analysis
    def analyze(self, target_pdf, a=0.0, b=0.0, center=0.0, tail_mass=1e-4, steps=20001) -> Analysis:
        """Support / extrema / area. With a < b the domain is used as is; otherwise the support is found
        automatically around `center` and trimmed to the central (1 - tail_mass) of the mass."""
        cb = _Callback(target_pdf, PDF)
        res = _CAnalysis()
        lib = self.rng._lib
        rc = lib.rl_analyze_and_normalize(cb.c, None, float(a), float(b), float(center), float(tail_mass),
                                          int(steps), byref(res))
        if rc == -1 and a >= b:
            raise ValueError("could not find a decaying support automatically; pass an explicit domain=(a, b)")
        check(lib, rc, "analyze", cb.exc)
        return Analysis(res.a, res.b, res.f_min, res.f_max, res.area, res.total_area)

    # ---------------------------------------------------------------- planning
    def plan(self, target_pdf, domain=None, proposals=None, center=0.0, grid=4001, safety=1.02,
             tail_mass=1e-4) -> list:
        """Return ProposalPlans sorted by efficiency (best first). Unsuitable proposals are listed in
        ``self.skipped`` with the reason."""
        if domain is None:
            an = self.analyze(target_pdf, center=center, tail_mass=tail_mass)
        else:
            lo, hi = float(domain[0]), float(domain[1])
            if not lo < hi:
                raise ValueError("domain must satisfy a < b")
            an = self.analyze(target_pdf, lo, hi)
        a, b = an.a, an.b
        if an.f_max <= 0.0 or an.area <= 0.0:
            raise ValueError("target density is zero on the analysed domain")
        width = b - a

        wanted = [_proposal(p) for p in (proposals if proposals is not None else list(Proposal))]
        self.skipped = {}
        plans = []

        def evalgrid(lo, hi, n):
            step = (hi - lo) / (n - 1)
            xs = [lo + i * step for i in range(n)]
            fx = []
            for x in xs:
                v = float(target_pdf(x))
                if not (v >= 0.0 and v < _INF):
                    raise ValueError(f"target_pdf({x!r}) returned {v!r}; must be finite and >= 0")
                fx.append(v)
            return xs, fx

        ext_lo, ext_hi = a - 0.5 * width, b + 0.5 * width
        xs_full, fx_full = evalgrid(ext_lo, ext_hi, grid)
        sub = max(1, grid // 400)
        xs_s, fx_s = xs_full[::sub], fx_full[::sub]
        xs_in, fx_in = evalgrid(a, b, grid)

        def finalize(ptype, p1, p2, xs, fx):
            m, arg = _max_ratio(ptype, p1, p2, xs, fx)
            if m == _INF or m <= 0.0:
                return None
            lo_i, hi_i = max(arg - 1, 0), min(arg + 1, len(xs) - 1)
            refined = _golden_max(lambda x: float(target_pdf(x)) / max(_density(ptype, x, p1, p2), 1e-300),
                                  xs[lo_i], xs[hi_i])
            return max(m, refined) * safety

        for ptype in wanted:
            name = ptype.name
            if ptype == Proposal.UNIFORM:
                M = finalize(ptype, a, b, xs_in, fx_in)
                if M is not None:
                    plans.append(ProposalPlan(name, ptype, a, b, M, an.area / M, (a, b),
                                              "samples the target restricted to [a, b]"))
                continue
            if ptype == Proposal.EXPONENTIAL:
                if a < 0.0:
                    self.skipped[name] = f"support x >= 0 does not cover the domain (a = {a:g} < 0)"
                    continue
                best = None
                for k in range(60):
                    scale = width * 10 ** (-2.0 + 3.5 * k / 59.0) + 1e-300
                    m, _ = _max_ratio(ptype, scale, 0.0, xs_s, fx_s)
                    if best is None or m < best[0]:
                        best = (m, scale)
                if best[0] == _INF:
                    self.skipped[name] = "no scale gives a finite M"
                    continue
                scale = _golden_min_log(lambda s: _max_ratio(ptype, s, 0.0, xs_s, fx_s)[0], best[1])
                M = finalize(ptype, scale, 0.0, xs_full, fx_full)
                if M is not None:
                    plans.append(ProposalPlan(name, ptype, scale, 0.0, M, an.area / M, (a, b)))
                continue
            # location-scale family: Cauchy, Laplace, DL
            loc_lo, loc_hi = a, b
            sc_lo, sc_hi = width / 200.0, width * 2.0
            best = (_INF, 0.0, 1.0)
            n_loc, n_sc = 17, 17
            for rnd in range(5):
                for i in range(n_loc):
                    loc = loc_lo + (loc_hi - loc_lo) * i / (n_loc - 1)
                    for j in range(n_sc):
                        sc = sc_lo * (sc_hi / sc_lo) ** (j / (n_sc - 1))
                        m, _ = _max_ratio(ptype, loc, sc, xs_s, fx_s)
                        if m < best[0]:
                            best = (m, loc, sc)
                if best[0] == _INF:
                    break
                dl, ds = (loc_hi - loc_lo) / (n_loc - 1), (sc_hi / sc_lo) ** (1.0 / (n_sc - 1))
                loc_lo, loc_hi = best[1] - dl, best[1] + dl
                sc_lo, sc_hi = best[2] / ds, best[2] * ds
                n_loc, n_sc = 9, 9
            if best[0] == _INF:
                self.skipped[name] = "no (location, scale) gives a finite M"
                continue
            M = finalize(ptype, best[1], best[2], xs_full, fx_full)
            if M is not None:
                plans.append(ProposalPlan(name, ptype, best[1], best[2], M, an.area / M, (a, b)))
            else:
                self.skipped[name] = "envelope check failed on the refined grid"
        plans.sort(key=lambda p: -p.efficiency)
        return plans

    def best(self, target_pdf, **kwargs) -> ProposalPlan:
        plans = self.plan(target_pdf, **kwargs)
        if not plans:
            raise ValueError(f"no usable proposal found; skipped: {self.skipped}")
        return plans[0]

    # ---------------------------------------------------------------- benchmark / report
    def benchmark(self, target_pdf, plan: ProposalPlan, n=5000) -> dict:
        """Measure the real acceptance rate and time per sample. The time includes the cost of calling
        back into Python for every candidate, so it is NOT the speed of the C engine."""
        t0 = time.perf_counter()
        _, attempts = self.rng.rejection_sampling_stats(target_pdf, plan.proposal, plan.M, plan.p1, plan.p2, size=n)
        dt = time.perf_counter() - t0
        return {"acceptance": n / attempts, "expected": plan.efficiency, "seconds_per_sample": dt / n}

    def compare(self, target_pdf, domain=None, n=3000, **plan_kwargs) -> str:
        """Plan every proposal, measure each one and return a text table (replaces the old benchmark())."""
        plans = self.plan(target_pdf, domain=domain, **plan_kwargs)
        rows = [f"{'proposal':<14}{'M':>9}{'expected':>10}{'measured':>10}{'us/sample*':>12}"]
        for p in plans:
            b = self.benchmark(target_pdf, p, n=n)
            rows.append(f"{p.name:<14}{p.M:>9.3f}{b['expected']:>10.1%}{b['acceptance']:>10.1%}"
                        f"{b['seconds_per_sample'] * 1e6:>12.1f}")
        for name, why in self.skipped.items():
            rows.append(f"{name:<14}skipped: {why}")
        rows.append("* includes the cost of calling the Python target for every candidate")
        return "\n".join(rows)

    def report(self, plans) -> str:
        rows = [f"{'proposal':<14}{'p1':>12}{'p2':>12}{'M':>10}{'accept':>9}"]
        for p in plans:
            rows.append(f"{p.name:<14}{p.p1:>12.4g}{p.p2:>12.4g}{p.M:>10.4f}{p.efficiency:>9.1%}")
        for name, why in self.skipped.items():
            rows.append(f"{name:<14}skipped: {why}")
        return "\n".join(rows)


def _golden_min_log(fn, start, iters=50):
    """Minimise fn over a log-scale bracket around `start` (golden-section)."""
    lo, hi = math.log(start) - 0.7, math.log(start) + 0.7
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    c, d = hi - gr * (hi - lo), lo + gr * (hi - lo)
    fc, fd = fn(math.exp(c)), fn(math.exp(d))
    for _ in range(iters):
        if fc < fd:
            hi, d, fd = d, c, fc
            c = hi - gr * (hi - lo)
            fc = fn(math.exp(c))
        else:
            lo, c, fc = c, d, fd
            d = lo + gr * (hi - lo)
            fd = fn(math.exp(d))
    return math.exp(c if fc < fd else d)
