/* geometric_engine.h - Geometric(p) by inversion, 0 < p <= 1. */
#ifndef DLRAND_GEOMETRIC_H
#define DLRAND_GEOMETRIC_H
#include "rl_core.h"
#ifdef __cplusplus
extern "C" {
#endif

enum { RL_GEOMETRIC_TRIALS = 0, RL_GEOMETRIC_FAILURES = 1 };

/* Trial number of the first success, support {1,2,3,...}. Saturates at INT64_MAX. */
RL_API int rl_geometric_trials(RL_State* st, double p, int64_t* out);
/* Number of failures before the first success, support {0,1,2,...}. */
RL_API int rl_geometric_failures(RL_State* st, double p, int64_t* out);
RL_API int rl_geometric_fill(RL_State* st, double p, int variant, int64_t n, int64_t* out);

#ifdef __cplusplus
}
#endif
#endif
