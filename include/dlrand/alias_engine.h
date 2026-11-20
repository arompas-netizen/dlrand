/* alias_engine.h - Vose's alias method: O(n) build, O(1) draw. */
#ifndef DLRAND_ALIAS_H
#define DLRAND_ALIAS_H
#include "rl_core.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef struct RL_AliasTable {
    int64_t   n;
    double*   prob;
    int64_t*  alias;
} RL_AliasTable;

/* Weights must be finite, >= 0, with a positive sum (they are normalised internally). */
RL_API int  rl_alias_new(const double* weights, int64_t n, RL_AliasTable** out);
RL_API int64_t rl_alias_draw(RL_State* st, const RL_AliasTable* table);
RL_API void rl_alias_free(RL_AliasTable* table);

#ifdef __cplusplus
}
#endif
#endif
