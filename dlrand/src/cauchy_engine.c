#include "rl_internal.h"

static int ok(double loc, double scale) { return RL_FINITE(loc) && RL_FINITE(scale) && scale > 0.0; }

int rl_cauchy(RL_State* st, double location, double scale, double* out) {
    if (!st || !out || !ok(location, scale)) return RL_ERR_ARG;
    *out = location + scale * tan(RL_PI * (rl_uopen(st) - 0.5));
    return RL_OK;
}
int rl_cauchy_fill(RL_State* st, double location, double scale, int64_t n, double* out) {
    if (!st || !ok(location, scale)) return RL_ERR_ARG;
    RL_CHECK_FILL(n, out);
    for (int64_t i = 0; i < n; i++) out[i] = location + scale * tan(RL_PI * (rl_uopen(st) - 0.5));
    return RL_OK;
}
