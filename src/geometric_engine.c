#include "rl_internal.h"

#define GEO_OK(p) (RL_FINITE(p) && (p) > 0.0 && (p) <= 1.0)

/* Inversion: failures = floor(ln U / ln(1-p)), U in (0,1). log1p keeps tiny p exact. */
int64_t rl_geometric_failures_unchecked(RL_State* st, double p) {
    if (p == 1.0) return 0;
    const double x = floor(log(rl_uopen(st)) / log1p(-p));
    if (!(x < 9.2233720368547758e18)) return INT64_MAX;   /* saturate */
    return (int64_t)x;
}

static int64_t trials_unchecked(RL_State* st, double p) {
    const int64_t f = rl_geometric_failures_unchecked(st, p);
    return f == INT64_MAX ? INT64_MAX : f + 1;
}

int rl_geometric_failures(RL_State* st, double p, int64_t* out) {
    if (!st || !out || !GEO_OK(p)) return RL_ERR_ARG;
    *out = rl_geometric_failures_unchecked(st, p);
    return RL_OK;
}
int rl_geometric_trials(RL_State* st, double p, int64_t* out) {
    if (!st || !out || !GEO_OK(p)) return RL_ERR_ARG;
    *out = trials_unchecked(st, p);
    return RL_OK;
}
int rl_geometric_fill(RL_State* st, double p, int variant, int64_t n, int64_t* out) {
    if (!st || !GEO_OK(p) || (variant != RL_GEOMETRIC_TRIALS && variant != RL_GEOMETRIC_FAILURES)) return RL_ERR_ARG;
    RL_CHECK_FILL(n, out);
    for (int64_t i = 0; i < n; i++)
        out[i] = (variant == RL_GEOMETRIC_TRIALS) ? trials_unchecked(st, p)
                                                  : rl_geometric_failures_unchecked(st, p);
    return RL_OK;
}
