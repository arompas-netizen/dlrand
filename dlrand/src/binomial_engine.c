#include "rl_internal.h"

#define BINV_MAX_NP 500.0   /* keeps (1-p)^n above the double underflow threshold */

static int64_t binomial_summation(RL_State* st, int64_t n, double p) {
    int64_t k = 0;
    for (int64_t i = 0; i < n; i++)
        if (rl_u01(st) < p) k++;
    return k;
}

/* BINV (Kachitvichyanukul & Schmeiser): CDF inversion by the pmf recurrence. p <= 0.5. */
static int64_t binomial_inversion(RL_State* st, int64_t n, double p) {
    const double q = 1.0 - p;
    const double s = p / q;
    const double a = ((double)n + 1.0) * s;
    double r = exp((double)n * log1p(-p));    /* P(X = 0) */
    double u = rl_u01(st);
    int64_t x = 0;
    while (u > r) {
        u -= r;
        x++;
        if (x > n) return n;
        r *= (a / (double)x - s);
        if (r <= 0.0) return x;               /* probability ~1e-16: avoid an endless loop */
    }
    return x;
}

/**
 * BTRS (Hormann 1993): transformed rejection with squeeze. Exact, O(1), for
 * p <= 0.5 and n*p >= 10.
 */
static int64_t binomial_btrs(RL_State* st, int64_t n, double p) {
    const double nd = (double)n;
    const double q = 1.0 - p;
    const double spq = sqrt(nd * p * q);
    const double b = 1.15 + 2.53 * spq;
    const double a = -0.0873 + 0.0248 * b + 0.01 * p;
    const double c = nd * p + 0.5;
    const double vr = 0.92 - 4.2 / b;
    const double alpha = (2.83 + 5.1 / b) * spq;
    const double m = floor((nd + 1.0) * p);
    /* k-independent part of ln(f(k)/f(m)); each log argument is within O(1/sqrt(n)) of 1, so it is
     * evaluated as log1p of an exactly computed difference (fma keeps m - n*p exact). */
    const double term_m = (m + 0.5) * log1p((fma(-nd, p, m) + (1.0 - 2.0 * p)) / ((nd - m + 1.0) * p));
    const double tail_m = rl_stirling_tail(m) + rl_stirling_tail(nd - m);
    for (;;) {
        const double u = rl_u01(st) - 0.5;
        double v = rl_uopen(st);
        const double us = 0.5 - fabs(u);
        const double kf = floor((2.0 * a / us + b) * u + c);
        if (kf < 0.0 || kf > nd) continue;
        if (us >= 0.07 && v <= vr) return (int64_t)kf;
        v = log(v * alpha / (a / (us * us) + b));
        /* ln(f(k)/f(m)) via the Stirling decomposition of ln(x!) (linear and constant terms cancel). */
        const double t2 = (nd + 1.0) * log1p((kf - m) / (nd - kf + 1.0));
        const double t3 = (kf + 0.5) * log1p((fma(nd, p, -(kf + 1.0)) + 2.0 * p) / ((kf + 1.0) * q));
        const double ub = term_m + t2 + t3 + tail_m - rl_stirling_tail(kf) - rl_stirling_tail(nd - kf);
        if (v <= ub) return (int64_t)kf;
    }
}

/* Symmetry: Bin(n,p) = n - Bin(n,1-p), so the kernels only ever see p <= 0.5. */
int64_t rl_binomial_unchecked(RL_State* st, int64_t n, double p, int method) {
    if (n == 0 || p == 0.0) return 0;
    if (p == 1.0) return n;
    if (method == RL_BINOMIAL_SUMMATION) return binomial_summation(st, n, p);
    if (method == RL_BINOMIAL_POISSON_APPROX) {
        int64_t x = rl_poisson_unchecked(st, (double)n * p, RL_POISSON_AUTO);
        return x > n ? n : x;
    }
    const int flip = p > 0.5;
    const double pp = flip ? 1.0 - p : p;
    const double np = (double)n * pp;
    int use_inversion;
    if (method == RL_BINOMIAL_INVERSION) use_inversion = 1;
    else if (method == RL_BINOMIAL_BTRS) use_inversion = 0;
    else use_inversion = (np < 10.0);
    const int64_t x = use_inversion ? binomial_inversion(st, n, pp) : binomial_btrs(st, n, pp);
    return flip ? n - x : x;
}

static int binomial_args_ok(int64_t n, double p, int method) {
    if (n < 0 || !(p >= 0.0 && p <= 1.0)) return 0;
    if (method < RL_BINOMIAL_AUTO || method > RL_BINOMIAL_POISSON_APPROX) return 0;
    if (method == RL_BINOMIAL_SUMMATION) return 1;
    if (n > RL_TWO_53) return 0;
    const double pp = p > 0.5 ? 1.0 - p : p;
    const double np = (double)n * pp;
    if (method == RL_BINOMIAL_INVERSION) return np <= BINV_MAX_NP;
    if (method == RL_BINOMIAL_BTRS) return np >= 10.0 || n == 0 || p == 0.0 || p == 1.0;
    if (method == RL_BINOMIAL_POISSON_APPROX) return (double)n * p <= 1e12;
    return 1;
}

int rl_binomial(RL_State* st, int64_t n, double p, int method, int64_t* out) {
    if (!st || !out || !binomial_args_ok(n, p, method)) return RL_ERR_ARG;
    *out = rl_binomial_unchecked(st, n, p, method);
    return RL_OK;
}

int rl_binomial_fill(RL_State* st, int64_t n, double p, int method, int64_t count, int64_t* out) {
    if (!st || !binomial_args_ok(n, p, method)) return RL_ERR_ARG;
    RL_CHECK_FILL(count, out);
    for (int64_t i = 0; i < count; i++) out[i] = rl_binomial_unchecked(st, n, p, method);
    return RL_OK;
}
