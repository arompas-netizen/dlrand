#include "rl_internal.h"

#define ENVELOPE_TOL 1e-9   /* relative slack before f > M*g is reported */

double rl_rejection_array_min_M(const double* pmf, int64_t size) {
    if (!pmf || size <= 0) return NAN;
    double mx = 0.0;
    for (int64_t i = 0; i < size; i++) {
        if (!RL_FINITE(pmf[i]) || pmf[i] < 0.0) return NAN;
        if (pmf[i] > mx) mx = pmf[i];
    }
    return mx * (double)size;
}

static int array_args_ok(const double* pmf, int64_t size, double M, int64_t max_attempts) {
    if (!pmf || size <= 0 || max_attempts <= 0 || !RL_FINITE(M) || M <= 0.0) return 0;
    int any_positive = 0;
    for (int64_t i = 0; i < size; i++) {
        if (!RL_FINITE(pmf[i]) || pmf[i] < 0.0) return 0;
        if (pmf[i] > 0.0) any_positive = 1;
    }
    return any_positive;               /* an all-zero pmf has no mass: fail fast, do not time out */
}

/* Proposal Y ~ U{0..size-1} (g = 1/size). Accept when U*M*g <= f(Y), i.e. U*M <= size*f(Y). */
static int array_one(RL_State* st, const double* pmf, int64_t size, double M, int64_t max_attempts, int64_t* out) {
    const double dsize = (double)size;
    for (int64_t a = 0; a < max_attempts; a++) {
        const int64_t y = (int64_t)rl_below(st, (uint64_t)size);
        const double fy = pmf[y] * dsize;                       /* f(y) / g(y) */
        if (fy > M * (1.0 + ENVELOPE_TOL)) return RL_ERR_ENVELOPE;
        if (rl_u01(st) * M <= fy) { *out = y; return RL_OK; }
    }
    return RL_ERR_TIMEOUT;
}

int rl_rejection_array(RL_State* st, const double* pmf, int64_t size, double M,
                       int64_t max_attempts, int64_t* out) {
    if (!st || !out || !array_args_ok(pmf, size, M, max_attempts)) return RL_ERR_ARG;
    return array_one(st, pmf, size, M, max_attempts, out);
}

int rl_rejection_array_fill(RL_State* st, const double* pmf, int64_t size, double M,
                            int64_t max_attempts, int64_t n, int64_t* out) {
    if (!st || !array_args_ok(pmf, size, M, max_attempts)) return RL_ERR_ARG;
    RL_CHECK_FILL(n, out);
    for (int64_t i = 0; i < n; i++) {
        const int rc = array_one(st, pmf, size, M, max_attempts, &out[i]);
        if (rc != RL_OK) return rc;
    }
    return RL_OK;
}

/* Proposal Y ~ Geometric(p) on {0,1,...}; ln g(y) = y ln(1-p) + ln p (log space: no underflow). */
int rl_rejection_func(RL_State* st, RL_PMF f, void* params, double p_geom, double M,
                      int64_t max_attempts, int64_t* out) {
    if (!st || !f || !out || max_attempts <= 0 || !RL_FINITE(M) || M <= 0.0) return RL_ERR_ARG;
    if (!(p_geom > 0.0 && p_geom < 1.0)) return RL_ERR_ARG;
    const double log_q = log1p(-p_geom), log_p = log(p_geom), log_M = log(M);
    for (int64_t a = 0; a < max_attempts; a++) {
        const int64_t y = rl_geometric_failures_unchecked(st, p_geom);
        if (y == INT64_MAX) continue;
        const double fy = f(y, params);
        if (isnan(fy) || fy < 0.0) return RL_ERR_CALLBACK;
        const double u = rl_uopen(st);
        if (fy == 0.0) continue;
        const double log_ratio = log(fy) - log_M - ((double)y * log_q + log_p);   /* ln( f / (M g) ) */
        if (log_ratio > ENVELOPE_TOL) return RL_ERR_ENVELOPE;
        if (log(u) <= log_ratio) { *out = y; return RL_OK; }
    }
    return RL_ERR_TIMEOUT;
}
