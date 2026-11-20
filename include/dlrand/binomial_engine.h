/* binomial_engine.h - Binomial(n, p). */
#ifndef DLRAND_BINOMIAL_H
#define DLRAND_BINOMIAL_H
#include "rl_core.h"
#ifdef __cplusplus
extern "C" {
#endif

enum {
    RL_BINOMIAL_AUTO          = 0, /* inversion if n*min(p,1-p) < 10, else BTRS (both exact)  */
    RL_BINOMIAL_SUMMATION     = 1, /* O(n) sum of Bernoulli trials (exact)                    */
    RL_BINOMIAL_INVERSION     = 2, /* BINV; exact; needs n*min(p,1-p) <= 500                  */
    RL_BINOMIAL_BTRS          = 3, /* Hormann BTRS; exact; needs n*min(p,1-p) >= 10           */
    RL_BINOMIAL_POISSON_APPROX = 4 /* Poisson(n*p) clamped to n: an APPROXIMATION, never auto */
};

/* Domain: n >= 0 (n <= 2^53 except for SUMMATION), 0 <= p <= 1. */
RL_API int rl_binomial(RL_State* st, int64_t n, double p, int method, int64_t* out);
RL_API int rl_binomial_fill(RL_State* st, int64_t n, double p, int method, int64_t count, int64_t* out);

#ifdef __cplusplus
}
#endif
#endif
