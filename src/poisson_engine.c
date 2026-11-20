#include "rl_internal.h"

#define POISSON_MAX_LAMBDA     1e15   /* k must stay an exactly representable integer (< 2^53) */
#define POISSON_SEQ_MAX_LAMBDA 700.0   /* exp(-lambda) underflows to 0 beyond ~745 */

/* CDF inversion by sequential search from 0 (small lambda). */
static int64_t poisson_sequential(RL_State* st, double lambda) {
    const double u = rl_u01(st);
    double p = exp(-lambda), F = p;
    int64_t k = 0;
    while (u > F && k < 4000) {      /* the cap only guards against F stalling just below u */
        k++;
        p *= lambda / (double)k;
        F += p;
    }
    return k;
}

/**
 * PTRS (Hormann 1993): transformed rejection with squeeze. Exact, O(1) expected
 * time, needs lambda >= 10. Constants are Hormann's published regression fits.
 */
static int64_t poisson_ptrs(RL_State* st, double lambda) {
    const double slam = sqrt(lambda);
    const double b = 0.931 + 2.53 * slam;
    const double a = -0.059 + 0.02483 * b;
    const double inv_alpha = 1.1239 + 1.1328 / (b - 3.4);
    const double vr = 0.9277 - 3.6224 / (b - 2.0);
    for (;;) {
        const double U = rl_u01(st) - 0.5;
        const double V = rl_uopen(st);
        const double us = 0.5 - fabs(U);
        const double kf = floor((2.0 * a / us + b) * U + lambda + 0.43);
        if (us >= 0.07 && V <= vr) return (int64_t)kf;
        if (kf < 0.0 || (us < 0.013 && V > us)) continue;
        /* ln f(k) = -lambda + k ln(lambda) - ln(k!) rewritten with the Stirling remainder:
         *   (k+1-lambda) + k*log1p((lambda-k-1)/(k+1)) - ln(k+1)/2 - ln(2pi)/2 - T(k)
         * every term is O(1)..O(sqrt(lambda)), so nothing cancels catastrophically. */
        const double k1 = kf + 1.0;
        const double rhs = (k1 - lambda) + kf * log1p((lambda - k1) / k1)
                           - 0.5 * log(k1) - 0.5 * log(2.0 * RL_PI) - rl_stirling_tail(kf);
        if ((log(V) + log(inv_alpha) - log(a / (us * us) + b)) <= rhs)
            return (int64_t)kf;
    }
}

int64_t rl_poisson_unchecked(RL_State* st, double lambda, int method) {
    if (lambda == 0.0) return 0;
    if (method == RL_POISSON_AUTO) method = (lambda < 10.0) ? RL_POISSON_SEQUENTIAL : RL_POISSON_PTRS;
    return (method == RL_POISSON_SEQUENTIAL) ? poisson_sequential(st, lambda) : poisson_ptrs(st, lambda);
}

static int poisson_args_ok(double lambda, int method) {
    if (!RL_FINITE(lambda) || lambda < 0.0 || lambda > POISSON_MAX_LAMBDA) return 0;
    if (method == RL_POISSON_AUTO) return 1;
    if (method == RL_POISSON_SEQUENTIAL) return lambda <= POISSON_SEQ_MAX_LAMBDA;
    if (method == RL_POISSON_PTRS) return lambda >= 10.0;
    return 0;
}

int rl_poisson(RL_State* st, double lambda, int method, int64_t* out) {
    if (!st || !out || !poisson_args_ok(lambda, method)) return RL_ERR_ARG;
    *out = rl_poisson_unchecked(st, lambda, method);
    return RL_OK;
}

int rl_poisson_fill(RL_State* st, double lambda, int method, int64_t n, int64_t* out) {
    if (!st || !poisson_args_ok(lambda, method)) return RL_ERR_ARG;
    RL_CHECK_FILL(n, out);
    for (int64_t i = 0; i < n; i++) out[i] = rl_poisson_unchecked(st, lambda, method);
    return RL_OK;
}
