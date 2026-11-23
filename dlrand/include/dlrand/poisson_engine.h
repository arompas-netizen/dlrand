/* poisson_engine.h - Poisson(lambda), exact for every supported lambda. */
#ifndef DLRAND_POISSON_H
#define DLRAND_POISSON_H
#include "rl_core.h"
#ifdef __cplusplus
extern "C" {
#endif

enum { RL_POISSON_AUTO = 0, RL_POISSON_SEQUENTIAL = 1, RL_POISSON_PTRS = 2 };

/**
 * AUTO       : sequential inversion for lambda < 10, PTRS otherwise.
 * SEQUENTIAL : CDF inversion from 0; lambda <= 700 (exp(-lambda) underflows beyond).
 * PTRS       : Hormann's transformed rejection with squeeze; lambda >= 10.
 * Domain: 0 <= lambda <= 1e15.
 */
RL_API int rl_poisson(RL_State* st, double lambda, int method, int64_t* out);
RL_API int rl_poisson_fill(RL_State* st, double lambda, int method, int64_t n, int64_t* out);

#ifdef __cplusplus
}
#endif
#endif
