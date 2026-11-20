/* cauchy_engine.h - Cauchy by inversion: X = x0 + gamma*tan(pi*(U-1/2)). */
#ifndef DLRAND_CAUCHY_H
#define DLRAND_CAUCHY_H
#include "rl_core.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Needs finite location and finite scale > 0. */
RL_API int rl_cauchy(RL_State* st, double location, double scale, double* out);
RL_API int rl_cauchy_fill(RL_State* st, double location, double scale, int64_t n, double* out);
#ifdef __cplusplus
}
#endif
#endif
