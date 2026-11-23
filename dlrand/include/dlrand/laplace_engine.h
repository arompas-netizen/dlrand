/* laplace_engine.h - Laplace (double exponential) by inversion. */
#ifndef DLRAND_LAPLACE_H
#define DLRAND_LAPLACE_H
#include "rl_core.h"
#ifdef __cplusplus
extern "C" {
#endif
/* f(x) = 1/(2b) exp(-|x-mu|/b). Needs finite mu and finite b > 0. Always finite output. */
RL_API int rl_laplace(RL_State* st, double mu, double b, double* out);
RL_API int rl_laplace_fill(RL_State* st, double mu, double b, int64_t n, double* out);
#ifdef __cplusplus
}
#endif
#endif
