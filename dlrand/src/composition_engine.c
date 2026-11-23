#include <stdlib.h>
#include "rl_internal.h"

typedef struct { double alpha; RL_Worker worker; void* params; int set; } Component;

struct RL_Composition {
    int mode, count, finalized;
    Component* comp;
    int* order;              /* RESIDUAL: indices sorted by decreasing alpha */
    RL_AliasTable* alias;    /* ALIAS */
};

RL_Composition* rl_composition_new(int mode, int count) {
    if (count <= 0 || mode < RL_COMP_MANUAL || mode > RL_COMP_ALIAS) return NULL;
    RL_Composition* c = (RL_Composition*)calloc(1, sizeof(RL_Composition));
    if (!c) return NULL;
    c->comp = (Component*)calloc((size_t)count, sizeof(Component));
    c->order = (int*)calloc((size_t)count, sizeof(int));
    if (!c->comp || !c->order) { rl_composition_free(c); return NULL; }
    c->mode = mode; c->count = count;
    return c;
}

int rl_composition_set(RL_Composition* c, int index, double alpha, RL_Worker worker, void* params) {
    if (!c || index < 0 || index >= c->count || !worker || !RL_FINITE(alpha) || alpha < 0.0) return RL_ERR_ARG;
    c->comp[index].alpha = alpha;
    c->comp[index].worker = worker;
    c->comp[index].params = params;
    c->comp[index].set = 1;
    c->finalized = 0;
    return RL_OK;
}

/* stable insertion sort of indices by decreasing alpha (count is small) */
static void sort_order(RL_Composition* c) {
    for (int i = 0; i < c->count; i++) c->order[i] = i;
    for (int i = 1; i < c->count; i++) {
        const int v = c->order[i];
        int j = i - 1;
        while (j >= 0 && c->comp[c->order[j]].alpha < c->comp[v].alpha) { c->order[j + 1] = c->order[j]; j--; }
        c->order[j + 1] = v;
    }
}

int rl_composition_finalize(RL_Composition* c) {
    if (!c) return RL_ERR_ARG;
    double sum = 0.0;
    for (int i = 0; i < c->count; i++) {
        if (!c->comp[i].set) return RL_ERR_ARG;
        sum += c->comp[i].alpha;
    }
    if (fabs(sum - 1.0) > 1e-8) return RL_ERR_ARG;
    rl_alias_free(c->alias); c->alias = NULL;
    if (c->mode == RL_COMP_ALIAS) {
        double* w = (double*)malloc((size_t)c->count * sizeof(double));
        if (!w) return RL_ERR_ALLOC;
        for (int i = 0; i < c->count; i++) w[i] = c->comp[i].alpha;
        const int rc = rl_alias_new(w, c->count, &c->alias);
        free(w);
        if (rc != RL_OK) return rc;
    } else if (c->mode == RL_COMP_RESIDUAL) {
        sort_order(c);
    } else {
        for (int i = 0; i < c->count; i++) c->order[i] = i;
    }
    c->finalized = 1;
    return RL_OK;
}

int rl_composition_sample(RL_State* st, const RL_Composition* c, int64_t* out) {
    if (!st || !c || !out || !c->finalized) return RL_ERR_ARG;
    int idx;
    if (c->mode == RL_COMP_ALIAS) {
        idx = (int)rl_alias_draw(st, c->alias);
    } else {
        const double u = rl_u01(st);
        double cum = 0.0;
        idx = -1;
        int last_positive = c->order[0];
        for (int k = 0; k < c->count; k++) {
            const int i = c->order[k];
            if (c->comp[i].alpha <= 0.0) continue;
            last_positive = i;
            cum += c->comp[i].alpha;
            if (u < cum) { idx = i; break; }
        }
        if (idx < 0) idx = last_positive;     /* rounding: never bias towards component 0 */
    }
    return c->comp[idx].worker(st, c->comp[idx].params, out);
}

void rl_composition_free(RL_Composition* c) {
    if (!c) return;
    rl_alias_free(c->alias);
    free(c->comp);
    free(c->order);
    free(c);
}

int rl_worker_discrete_uniform(RL_State* st, void* params, int64_t* out) {
    const RL_UniformWorkerParams* p = (const RL_UniformWorkerParams*)params;
    if (!p || p->range <= 0) return RL_ERR_ARG;
    *out = p->start + (int64_t)rl_below(st, (uint64_t)p->range);
    return RL_OK;
}
int rl_worker_poisson(RL_State* st, void* params, int64_t* out) {
    const RL_PoissonWorkerParams* p = (const RL_PoissonWorkerParams*)params;
    return p ? rl_poisson(st, p->lam, RL_POISSON_AUTO, out) : RL_ERR_ARG;
}
int rl_worker_geometric(RL_State* st, void* params, int64_t* out) {
    const RL_GeometricWorkerParams* p = (const RL_GeometricWorkerParams*)params;
    return p ? rl_geometric_trials(st, p->p, out) : RL_ERR_ARG;
}
int rl_worker_binomial(RL_State* st, void* params, int64_t* out) {
    const RL_BinomialWorkerParams* p = (const RL_BinomialWorkerParams*)params;
    return p ? rl_binomial(st, p->n, p->p, RL_BINOMIAL_AUTO, out) : RL_ERR_ARG;
}
