/* permutation_engine.h - (partial) Fisher-Yates shuffle. */
#ifndef DLRAND_PERMUTATION_H
#define DLRAND_PERMUTATION_H
#include "rl_core.h"
#ifdef __cplusplus
extern "C" {
#endif

/**
 * Shuffles the first r positions of arr[0..n-1]: afterwards arr[0..r-1] is a
 * uniformly random r-permutation of the original elements (sampling without
 * replacement). Requires 0 <= r <= n; otherwise RL_ERR_ARG and arr is untouched.
 */
RL_API int rl_permutation_partial(RL_State* st, int64_t* arr, int64_t n, int64_t r);

#ifdef __cplusplus
}
#endif
#endif
