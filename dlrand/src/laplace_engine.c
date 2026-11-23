#include "rl_internal.h"

static int ok(double mu, double b) { return RL_FINITE(mu) && RL_FINITE(b) && b > 0.0; }

/* v = U - 1/2 in (-1/2, 1/2): X = mu - b*sgn(v)*ln(1 - 2|v|). 1-2|v| lies in (0,1]: always finite. */
static double laplace_unchecked(RL_State* st, double mu, double b) {
    const double v = rl_uopen(st) - 0.5;
    const double l = log(1.0 - 2.0 * fabs(v));
    return (v < 0.0) ? mu + b * l : mu - b * l;
}

int rl_laplace(RL_State* st, double mu, double b, double* out) {
    if (!st || !out || !ok(mu, b)) return RL_ERR_ARG;
    *out = laplace_unchecked(st, mu, b);
    return RL_OK;
}
int rl_laplace_fill(RL_State* st, double mu, double b, int64_t n, double* out) {
    if (!st || !ok(mu, b)) return RL_ERR_ARG;
    RL_CHECK_FILL(n, out);
    for (int64_t i = 0; i < n; i++) out[i] = laplace_unchecked(st, mu, b);
    return RL_OK;
}
