/**
 * dl_ziggurat_engine.h - Ziggurat sampler for the DL distribution.
 *
 * The table is SOLVED at creation time for any layer count N (the tail start R
 * and the common strip area V are found by bisection so that the N equal-area
 * strips exactly tile the density). No hard-coded constants are tied to one N.
 */
#ifndef DLRAND_DL_ZIGGURAT_H
#define DLRAND_DL_ZIGGURAT_H
#include "rl_core.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef struct RL_DLZiggurat {
    int     num_layers;
    double  r;        /* start of the tail                    */
    double  area_v;   /* common area of every strip           */
    double* x_table;  /* N+1 entries: x_0 = V/f(R), x_1 = R ... x_N = 0 */
    double* y_table;  /* N+1 entries: y_i = f(x_i), y_N = f(0) = 1      */
} RL_DLZiggurat;

/* 8 <= N <= 65536 (N = 128 or 256 recommended). Returns NULL on bad N or OOM. */
RL_API RL_DLZiggurat* rl_dl_ziggurat_new(int N);
RL_API void           rl_dl_ziggurat_free(RL_DLZiggurat* table);
/* One standard DL(0,1) variate. */
RL_API int            rl_dl_ziggurat_sample(RL_State* st, const RL_DLZiggurat* table, double* out);
RL_API int            rl_dl_ziggurat_fill(RL_State* st, const RL_DLZiggurat* table, int64_t n, double* out);

/* Unnormalised DL density exp(-x^2/2), usable as a target for the rejection engine. */
RL_API double         rl_dl_pdf(double x, void* params);

#ifdef __cplusplus
}
#endif
#endif
