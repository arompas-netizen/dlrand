#include <stdlib.h>
#include "rl_internal.h"

/**
 * Vose's alias method. Each of the n bins holds one primary category (kept with
 * probability prob[i]) and one alias; a draw is one bounded integer plus one
 * uniform, independent of each other (no reuse of the fractional part).
 */
int rl_alias_new(const double* w, int64_t n, RL_AliasTable** out) {
    if (!w || !out || n <= 0) return RL_ERR_ARG;
    double sum = 0.0;
    for (int64_t i = 0; i < n; i++) {
        if (!RL_FINITE(w[i]) || w[i] < 0.0) return RL_ERR_ARG;
        sum += w[i];
    }
    if (!(sum > 0.0) || !RL_FINITE(sum)) return RL_ERR_ARG;

    RL_AliasTable* t = (RL_AliasTable*)malloc(sizeof(RL_AliasTable));
    double*  scaled = (double*)malloc((size_t)n * sizeof(double));
    int64_t* small  = (int64_t*)malloc((size_t)n * sizeof(int64_t));
    int64_t* large  = (int64_t*)malloc((size_t)n * sizeof(int64_t));
    double*  prob   = (double*)malloc((size_t)n * sizeof(double));
    int64_t* alias  = (int64_t*)malloc((size_t)n * sizeof(int64_t));
    if (!t || !scaled || !small || !large || !prob || !alias) {
        free(t); free(scaled); free(small); free(large); free(prob); free(alias);
        return RL_ERR_ALLOC;
    }
    int64_t ns = 0, nl = 0;
    for (int64_t i = 0; i < n; i++) {
        scaled[i] = w[i] / sum * (double)n;
        alias[i] = i;
        prob[i] = 1.0;
        if (scaled[i] < 1.0) small[ns++] = i; else large[nl++] = i;
    }
    while (ns > 0 && nl > 0) {
        const int64_t s = small[--ns];
        const int64_t l = large[--nl];
        prob[s] = scaled[s];
        alias[s] = l;
        scaled[l] = (scaled[l] + scaled[s]) - 1.0;
        if (scaled[l] < 1.0) small[ns++] = l; else large[nl++] = l;
    }
    /* leftovers are 1.0 up to rounding error; prob was initialised to 1.0 */
    free(scaled); free(small); free(large);
    t->n = n; t->prob = prob; t->alias = alias;
    *out = t;
    return RL_OK;
}

int64_t rl_alias_draw(RL_State* st, const RL_AliasTable* t) {
    const int64_t col = (int64_t)rl_below(st, (uint64_t)t->n);
    return (rl_u01(st) < t->prob[col]) ? col : t->alias[col];
}

void rl_alias_free(RL_AliasTable* t) {
    if (!t) return;
    free(t->prob); free(t->alias); free(t);
}
