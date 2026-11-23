/**
 * composition_engine.h - the Composition (mixture) method.
 *   f(x) = sum_i alpha_i * F_i(x),  sum alpha_i = 1.
 * A component is chosen with probability alpha_i, then its worker draws x.
 *
 * Usage: new -> set(0..count-1) -> finalize -> sample (repeatedly) -> free.
 * The engine is built ONCE and reused; sampling is O(1) (alias) or O(count).
 */
#ifndef DLRAND_COMPOSITION_H
#define DLRAND_COMPOSITION_H
#include "rl_core.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    RL_COMP_MANUAL   = 0, /* linear scan in the given order                    */
    RL_COMP_RESIDUAL = 1, /* linear scan, largest weights first (fewer tests)  */
    RL_COMP_ALIAS    = 2  /* Vose alias table, O(1)                            */
} RL_CompositionMode;

/* A worker draws one integer from its own component using the shared state. */
typedef int (*RL_Worker)(RL_State* st, void* params, int64_t* out);

typedef struct RL_Composition RL_Composition;

RL_API RL_Composition* rl_composition_new(int mode, int count);
RL_API int  rl_composition_set(RL_Composition* c, int index, double alpha, RL_Worker worker, void* params);
/* Validates (alphas finite, >= 0, sum within 1e-8 of 1; all slots set) and builds tables. */
RL_API int  rl_composition_finalize(RL_Composition* c);
RL_API int  rl_composition_sample(RL_State* st, const RL_Composition* c, int64_t* out);
RL_API void rl_composition_free(RL_Composition* c);

/* Native workers; params are plain structs that do NOT hold the RNG state. */
typedef struct { int64_t start; int64_t range; } RL_UniformWorkerParams;   /* start .. start+range-1 */
typedef struct { double lam; }                   RL_PoissonWorkerParams;
typedef struct { double p; }                     RL_GeometricWorkerParams; /* trials: {1,2,...}     */
typedef struct { int64_t n; double p; }          RL_BinomialWorkerParams;

RL_API int rl_worker_discrete_uniform(RL_State* st, void* params, int64_t* out);
RL_API int rl_worker_poisson(RL_State* st, void* params, int64_t* out);
RL_API int rl_worker_geometric(RL_State* st, void* params, int64_t* out);
RL_API int rl_worker_binomial(RL_State* st, void* params, int64_t* out);

#ifdef __cplusplus
}
#endif
#endif
