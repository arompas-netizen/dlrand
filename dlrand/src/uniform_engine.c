#include "rl_internal.h"

static int range_ok(double a, double b) { return RL_FINITE(a) && RL_FINITE(b) && a <= b; }

static double uniform_unchecked(RL_State* st, double a, double b) {
    double x = a + (b - a) * rl_u01(st);
    if (b > a && x >= b) x = nextafter(b, a);   /* rounding must never reach b */
    return x;
}

int rl_uniform_range(RL_State* st, double a, double b, double* out) {
    if (!st || !out || !range_ok(a, b)) return RL_ERR_ARG;
    *out = uniform_unchecked(st, a, b);
    return RL_OK;
}

int rl_uniform_fill(RL_State* st, double a, double b, int64_t n, double* out) {
    if (!st || !range_ok(a, b)) return RL_ERR_ARG;
    RL_CHECK_FILL(n, out);
    for (int64_t i = 0; i < n; i++) out[i] = uniform_unchecked(st, a, b);
    return RL_OK;
}

static int64_t integers_unchecked(RL_State* st, int64_t low, int64_t high) {
    const uint64_t range = (uint64_t)high - (uint64_t)low + 1u;   /* 0 means the full 2^64 span */
    const uint64_t off = (range == 0) ? rl_u64(st) : rl_below(st, range);
    return (int64_t)((uint64_t)low + off);
}

int rl_integers(RL_State* st, int64_t low, int64_t high, int64_t* out) {
    if (!st || !out || high < low) return RL_ERR_ARG;
    *out = integers_unchecked(st, low, high);
    return RL_OK;
}

int rl_integers_fill(RL_State* st, int64_t low, int64_t high, int64_t n, int64_t* out) {
    if (!st || high < low) return RL_ERR_ARG;
    RL_CHECK_FILL(n, out);
    for (int64_t i = 0; i < n; i++) out[i] = integers_unchecked(st, low, high);
    return RL_OK;
}
