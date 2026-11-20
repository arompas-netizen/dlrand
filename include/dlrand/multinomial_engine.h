/* multinomial_engine.h - Multinomial(n, p) random vectors. */
#ifndef DLRAND_MULTINOMIAL_H
#define DLRAND_MULTINOMIAL_H
#include "rl_core.h"
#ifdef __cplusplus
extern "C" {
#endif

enum { RL_MULTINOMIAL_AUTO = 0, RL_MULTINOMIAL_ALIAS = 1, RL_MULTINOMIAL_SEQUENCING = 2 };

/**
 * counts_out[0..r-1] receives the category counts (they sum to n).
 *  ALIAS      : n trials through an alias table, O(r + n)   (AUTO uses it when r > n).
 *  SEQUENCING : chain of conditional Binomials, O(r)        (AUTO uses it when n >= r).
 * p_vector must be finite, >= 0 and sum to 1 within 1e-8.
 */
RL_API int rl_multinomial(RL_State* st, int64_t n, int64_t r, const double* p_vector,
                          int method, int64_t* counts_out);

#ifdef __cplusplus
}
#endif
#endif
