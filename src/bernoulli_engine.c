#include "rl_internal.h"

/**
 * Jump method. The number of failures before the next success is Geometric(p):
 * floor(ln U / ln(1-p)) with U in (0,1). Jumping over failures costs O(p*n).
 * Positions are tracked as doubles (exact below 2^53) so huge skips can never
 * overflow an integer.
 */
int rl_bernoulli_select(RL_State* st, int64_t n, double p, int method, int64_t* out, int64_t* count) {
    if (!st || !count || n < 0 || !(p >= 0.0 && p <= 1.0)) return RL_ERR_ARG;
    if (n > 0 && !out) return RL_ERR_ARG;
    if (method != RL_BERNOULLI_STANDARD && method != RL_BERNOULLI_JUMP) return RL_ERR_ARG;

    int64_t c = 0;
    if (p == 0.0 || n == 0) { *count = 0; return RL_OK; }
    if (p == 1.0) {
        for (int64_t i = 0; i < n; i++) out[i] = i;
        *count = n;
        return RL_OK;
    }
    if (method == RL_BERNOULLI_STANDARD) {
        for (int64_t i = 0; i < n; i++)
            if (rl_u01(st) < p) out[c++] = i;
    } else {
        if (n > RL_TWO_53) return RL_ERR_ARG;
        const double log_q = log1p(-p);
        const double limit = (double)n;
        double pos = -1.0;
        for (;;) {
            pos += floor(log(rl_uopen(st)) / log_q) + 1.0;
            if (!(pos < limit)) break;       /* also stops on inf */
            out[c++] = (int64_t)pos;
        }
    }
    *count = c;
    return RL_OK;
}
