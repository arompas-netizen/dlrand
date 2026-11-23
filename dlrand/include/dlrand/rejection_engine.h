/* rejection_engine.h - von Neumann rejection sampling for DISCRETE targets. */
#ifndef DLRAND_REJECTION_H
#define DLRAND_REJECTION_H
#include "rl_core.h"
#ifdef __cplusplus
extern "C" {
#endif

/* User PMF f(x). Return NaN or a negative value to signal an error. */
typedef double (*RL_PMF)(int64_t x, void* params);

/**
 * PATH A - finite support {0..size-1}, uniform proposal g(x) = 1/size.
 * Requires f(x) <= M*g(x), i.e. M >= size*max(pmf). Violations met while
 * sampling are reported as RL_ERR_ENVELOPE instead of silently biasing output.
 * pmf need not be normalised (any positive scale works for the same M rule).
 */
RL_API int    rl_rejection_array(RL_State* st, const double* pmf, int64_t size, double M,
                                 int64_t max_attempts, int64_t* out);
/* Same as rl_rejection_array but draws n samples, validating pmf only once. */
RL_API int    rl_rejection_array_fill(RL_State* st, const double* pmf, int64_t size, double M,
                                      int64_t max_attempts, int64_t n, int64_t* out);
/* Smallest valid M for rl_rejection_array: size * max(pmf). Returns NaN for bad input. */
RL_API double rl_rejection_array_min_M(const double* pmf, int64_t size);

/**
 * PATH B - infinite support {0,1,2,...}, Geometric(p_geom) proposal
 * g(x) = (1-p)^x * p  (0 < p_geom < 1). Requires f(x) <= M*g(x).
 */
RL_API int    rl_rejection_func(RL_State* st, RL_PMF f, void* params, double p_geom, double M,
                                int64_t max_attempts, int64_t* out);

#ifdef __cplusplus
}
#endif
#endif
