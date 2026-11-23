/* exponential_engine.h - Exponential by inversion: X = -scale*ln(U), U in (0,1). */
#ifndef DLRAND_EXPONENTIAL_H
#define DLRAND_EXPONENTIAL_H
#include "rl_core.h"
#ifdef __cplusplus
extern "C" {
#endif
/* f(x) = (1/scale) e^(-x/scale), x >= 0. Needs finite scale > 0. */
RL_API int rl_exponential(RL_State* st, double scale, double* out);
RL_API int rl_exponential_fill(RL_State* st, double scale, int64_t n, double* out);
#ifdef __cplusplus
}
#endif
#endif
