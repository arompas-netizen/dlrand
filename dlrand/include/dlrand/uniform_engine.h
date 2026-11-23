/* uniform_engine.h - continuous U[a,b) and discrete uniform integers. */
#ifndef DLRAND_UNIFORM_H
#define DLRAND_UNIFORM_H
#include "rl_core.h"
#ifdef __cplusplus
extern "C" {
#endif

/* X = a + (b-a)*U, result in [a, b) (never exactly b when a < b). Needs finite a <= b. */
RL_API int rl_uniform_range(RL_State* st, double a, double b, double* out);
RL_API int rl_uniform_fill(RL_State* st, double a, double b, int64_t n, double* out);

/* Unbiased integer in the INCLUSIVE range [low, high]. */
RL_API int rl_integers(RL_State* st, int64_t low, int64_t high, int64_t* out);
RL_API int rl_integers_fill(RL_State* st, int64_t low, int64_t high, int64_t n, int64_t* out);

#ifdef __cplusplus
}
#endif
#endif
