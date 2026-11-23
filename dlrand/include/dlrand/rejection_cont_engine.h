/**
 * rejection_cont_engine.h - rejection sampling for CONTINUOUS targets.
 *
 * Accept a proposal x ~ g when U <= f(x) / (M g(x)).  f may be unnormalised;
 * M must satisfy f(x) <= M g(x) for every x with f(x) > 0 (so g must be
 * positive wherever f is). With check_envelope != 0 every candidate is tested
 * and a violation returns RL_ERR_ENVELOPE rather than producing biased samples.
 */
#ifndef DLRAND_REJECTION_CONT_H
#define DLRAND_REJECTION_CONT_H
#include "rl_core.h"
#include "dl_ziggurat_engine.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef double (*RL_TargetPDF)(double x, void* params);

enum {
    RL_PROP_UNIFORM       = 0, /* p1 = a (lower), p2 = b (upper); a < b                  */
    RL_PROP_EXPONENTIAL   = 1, /* p1 = scale > 0; support x >= 0 (p2 unused)             */
    RL_PROP_CAUCHY        = 2, /* p1 = location, p2 = scale > 0                          */
    RL_PROP_LAPLACE       = 3, /* p1 = mu,       p2 = b > 0                              */
    RL_PROP_DL_BOX_MULLER = 4, /* p1 = mu,       p2 = sigma > 0                          */
    RL_PROP_DL_ZIGGURAT   = 5  /* same parameters; requires a DL Ziggurat table argument */
};

/* Analytical g(x) of a proposal; NaN if the parameters are invalid. */
RL_API double rl_proposal_density(int proposal_type, double x, double p1, double p2);

/* Draw one candidate from the proposal (table only used for RL_PROP_DL_ZIGGURAT). */
RL_API int rl_proposal_sample(RL_State* st, int proposal_type, double p1, double p2,
                              const RL_DLZiggurat* table, double* out);

/**
 * attempts_used (optional, may be NULL) receives the number of candidates drawn,
 * which lets callers measure the real acceptance rate.
 */
RL_API int rl_rejection_continuous(RL_State* st, RL_TargetPDF f, void* f_params,
                                   int proposal_type, double M, double p1, double p2,
                                   const RL_DLZiggurat* table, int64_t max_attempts,
                                   int check_envelope, double* out, int64_t* attempts_used);

#ifdef __cplusplus
}
#endif
#endif
