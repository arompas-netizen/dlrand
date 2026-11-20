/* rl_internal.h - private helpers shared by the engines (not installed). */
#ifndef RL_INTERNAL_H
#define RL_INTERNAL_H

#include <math.h>
#include "dlrand/dlrand.h"

#define RL_INV_2_53 (1.0 / 9007199254740992.0)  /* 2^-53 */
#define RL_INV_2_52 (1.0 / 4503599627370496.0)  /* 2^-52 */
#define RL_TWO_53   9007199254740992LL          /* 2^53  */

/* ---- inline generator (xoshiro256++); the exported wrappers live in rl_core.c ---- */
static inline uint64_t rl_rotl64(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

static inline uint64_t rl_u64(RL_State* st) {
    uint64_t* s = st->s;
    const uint64_t result = rl_rotl64(s[0] + s[3], 23) + s[0];
    const uint64_t t = s[1] << 17;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rl_rotl64(s[3], 45);
    return result;
}
/* [0, 1) */
static inline double rl_u01(RL_State* st)   { return (double)(rl_u64(st) >> 11) * RL_INV_2_53; }
/* (0, 1) */
static inline double rl_uopen(RL_State* st) { return ((double)(rl_u64(st) >> 12) + 0.5) * RL_INV_2_52; }
/* unbiased [0, n), n > 0 */
static inline uint64_t rl_below(RL_State* st, uint64_t n) {
    const uint64_t threshold = (0 - n) % n;   /* 2^64 mod n */
    for (;;) {
        const uint64_t r = rl_u64(st);
        if (r >= threshold) return r % n;
    }
}

/* ---- validation helpers ---- */
#define RL_FINITE(x) isfinite(x)
#define RL_CHECK_FILL(n, out) do { if ((n) < 0 || ((n) > 0 && (out) == NULL)) return RL_ERR_ARG; } while (0)

/* ---- cross-engine unchecked kernels (arguments already validated) ---- */
double  rl_stirling_tail(double x);
int64_t rl_geometric_failures_unchecked(RL_State* st, double p);
int64_t rl_poisson_unchecked(RL_State* st, double lambda, int method);
int64_t rl_binomial_unchecked(RL_State* st, int64_t n, double p, int method);

#endif
