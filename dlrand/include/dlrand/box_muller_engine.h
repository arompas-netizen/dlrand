/**
 * box_muller_engine.h - the DL (De Moivre-Laplace) distribution via Box-Muller.
 *
 * Two independent uniforms give two independent DL variates:
 *   Z0 = sqrt(-2 ln U1) cos(2 pi U2),  Z1 = sqrt(-2 ln U1) sin(2 pi U2).
 * Z1 is cached INSIDE the RL_State (never in a static), and rl_seed()/rl_jump()
 * clear the cache, so a given seed always reproduces the same stream.
 */
#ifndef DLRAND_BOX_MULLER_H
#define DLRAND_BOX_MULLER_H
#include "rl_core.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Sample from DL(mu, sigma^2). Needs finite mu and finite sigma >= 0. */
RL_API int rl_dl_box_muller(RL_State* st, double mu, double sigma, double* out);
RL_API int rl_dl_box_muller_fill(RL_State* st, double mu, double sigma, int64_t n, double* out);
#ifdef __cplusplus
}
#endif
#endif
