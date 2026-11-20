/* bernoulli_engine.h - inclusion sampling: keep each of n items with prob p. */
#ifndef DLRAND_BERNOULLI_H
#define DLRAND_BERNOULLI_H
#include "rl_core.h"
#ifdef __cplusplus
extern "C" {
#endif

enum { RL_BERNOULLI_STANDARD = 0, RL_BERNOULLI_JUMP = 1 };

/**
 * Writes the (sorted) indices of the selected items into out[] (capacity n)
 * and the number selected into *count.
 *  STANDARD: n independent trials, O(n).
 *  JUMP:     skips failures with Geometric jumps, O(p*n) expected; needs n <= 2^53.
 * p must satisfy 0 <= p <= 1 (p == 0 and p == 1 are handled exactly).
 */
RL_API int rl_bernoulli_select(RL_State* st, int64_t n, double p, int method,
                               int64_t* out, int64_t* count);

#ifdef __cplusplus
}
#endif
#endif
