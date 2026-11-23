#include "rl_internal.h"

static int ok(double scale) { return RL_FINITE(scale) && scale > 0.0; }

int rl_exponential(RL_State* st, double scale, double* out) {
    if (!st || !out || !ok(scale)) return RL_ERR_ARG;
    *out = -log(rl_uopen(st)) * scale;      /* U in (0,1): finite, no clamp, no tail truncation */
    return RL_OK;
}
int rl_exponential_fill(RL_State* st, double scale, int64_t n, double* out) {
    if (!st || !ok(scale)) return RL_ERR_ARG;
    RL_CHECK_FILL(n, out);
    for (int64_t i = 0; i < n; i++) out[i] = -log(rl_uopen(st)) * scale;
    return RL_OK;
}
