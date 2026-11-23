/**
 * rl_core.h - the entropy source: xoshiro256++ (period 2^256 - 1) seeded
 * through SplitMix64. Provides 53-bit doubles, unbiased bounded integers,
 * and 2^128 / 2^192 jump-ahead for independent parallel streams.
 */
#ifndef DLRAND_CORE_H
#define DLRAND_CORE_H
#include "rl_common.h"
#ifdef __cplusplus
extern "C" {
#endif

RL_API RL_State* rl_state_new(uint64_t seed);
RL_API void      rl_state_free(RL_State* st);
RL_API void      rl_seed(RL_State* st, uint64_t seed);

RL_API uint64_t  rl_next_u64(RL_State* st);
/* U in [0, 1) with 53 random bits (multiples of 2^-53). */
RL_API double    rl_uniform01(RL_State* st);
/* U in (0, 1) strictly: safe for log(U) and log(1-U) without clamping. */
RL_API double    rl_uniform_open(RL_State* st);
/* Unbiased integer in [0, n). n must be > 0. (Rejection on the modulo.) */
RL_API uint64_t  rl_bounded(RL_State* st, uint64_t n);

/* Advance the stream by 2^128 (jump) or 2^192 (long_jump) draws. Calling
 * rl_jump on copies of a state yields 2^128 non-overlapping substreams. */
RL_API void      rl_jump(RL_State* st);
RL_API void      rl_long_jump(RL_State* st);

/* Snapshot / restore: 6 words (4 state words, has_spare, bit pattern of spare). */
RL_API void      rl_state_get(const RL_State* st, uint64_t out[6]);
RL_API int       rl_state_set(RL_State* st, const uint64_t in[6]);

/* Thread-safe log-gamma (Stirling series), x >= 1. Used instead of libm's
 * lgamma, which writes a global (signgam) and differs across platforms. */
RL_API double    rl_loggam(double x);

#ifdef __cplusplus
}
#endif
#endif
