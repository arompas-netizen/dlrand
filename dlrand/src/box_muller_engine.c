#include "rl_internal.h"

static int ok(double mu, double sigma) { return RL_FINITE(mu) && RL_FINITE(sigma) && sigma >= 0.0; }

static double dl_unchecked(RL_State* st, double mu, double sigma) {
    if (st->has_spare) {
        st->has_spare = 0;
        return mu + sigma * st->spare;
    }
    const double u1 = rl_uopen(st);
    const double u2 = rl_u01(st);
    const double mag = sqrt(-2.0 * log(u1));
    st->spare = mag * sin(2.0 * RL_PI * u2);
    st->has_spare = 1;
    return mu + sigma * (mag * cos(2.0 * RL_PI * u2));
}

int rl_dl_box_muller(RL_State* st, double mu, double sigma, double* out) {
    if (!st || !out || !ok(mu, sigma)) return RL_ERR_ARG;
    *out = dl_unchecked(st, mu, sigma);
    return RL_OK;
}
int rl_dl_box_muller_fill(RL_State* st, double mu, double sigma, int64_t n, double* out) {
    if (!st || !ok(mu, sigma)) return RL_ERR_ARG;
    RL_CHECK_FILL(n, out);
    for (int64_t i = 0; i < n; i++) out[i] = dl_unchecked(st, mu, sigma);
    return RL_OK;
}
