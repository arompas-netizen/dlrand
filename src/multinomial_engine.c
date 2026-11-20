#include <stdlib.h>
#include <string.h>
#include "rl_internal.h"

static int pvec_ok(const double* p, int64_t r) {
    double sum = 0.0;
    for (int64_t i = 0; i < r; i++) {
        if (!RL_FINITE(p[i]) || p[i] < 0.0) return 0;
        sum += p[i];
    }
    return fabs(sum - 1.0) <= 1e-8;
}

/* Conditional-binomial chain: X_i | X_<i ~ Binomial(n_rem, p_i / sum_{j>=i} p_j).
 * Suffix sums (not a running subtraction) keep the conditional probabilities accurate. */
static int multinomial_sequencing(RL_State* st, int64_t n, int64_t r, const double* p, int64_t* counts) {
    double* suffix = (double*)malloc((size_t)r * sizeof(double));
    if (!suffix) return RL_ERR_ALLOC;
    double acc = 0.0;
    for (int64_t i = r - 1; i >= 0; i--) { acc += p[i]; suffix[i] = acc; }
    int64_t n_rem = n;
    for (int64_t i = 0; i < r - 1 && n_rem > 0; i++) {
        double pi = (suffix[i] > 0.0) ? p[i] / suffix[i] : 0.0;
        if (pi > 1.0) pi = 1.0;
        const int64_t x = rl_binomial_unchecked(st, n_rem, pi, RL_BINOMIAL_AUTO);
        counts[i] = x;
        n_rem -= x;
    }
    counts[r - 1] += n_rem;      /* counts[] was zeroed; last slot takes the remainder */
    free(suffix);
    return RL_OK;
}

static int multinomial_alias(RL_State* st, int64_t n, int64_t r, const double* p, int64_t* counts) {
    RL_AliasTable* t = NULL;
    const int rc = rl_alias_new(p, r, &t);
    if (rc != RL_OK) return rc;
    for (int64_t j = 0; j < n; j++) counts[rl_alias_draw(st, t)]++;
    rl_alias_free(t);
    return RL_OK;
}

int rl_multinomial(RL_State* st, int64_t n, int64_t r, const double* p, int method, int64_t* counts) {
    if (!st || !p || !counts || n < 0 || r < 1) return RL_ERR_ARG;
    if (method < RL_MULTINOMIAL_AUTO || method > RL_MULTINOMIAL_SEQUENCING) return RL_ERR_ARG;
    if (!pvec_ok(p, r)) return RL_ERR_ARG;
    memset(counts, 0, (size_t)r * sizeof(int64_t));
    if (n == 0) return RL_OK;
    if (method == RL_MULTINOMIAL_AUTO)
        method = (r > n) ? RL_MULTINOMIAL_ALIAS : RL_MULTINOMIAL_SEQUENCING;
    if (method == RL_MULTINOMIAL_SEQUENCING && n > RL_TWO_53) return RL_ERR_ARG;
    return (method == RL_MULTINOMIAL_ALIAS) ? multinomial_alias(st, n, r, p, counts)
                                            : multinomial_sequencing(st, n, r, p, counts);
}
