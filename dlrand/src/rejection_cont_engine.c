#include "rl_internal.h"

#define ENVELOPE_TOL 1e-9

/* g(x) for each proposal; NaN when the parameters are invalid. */
double rl_proposal_density(int type, double x, double p1, double p2) {
    switch (type) {
        case RL_PROP_UNIFORM:
            if (!(RL_FINITE(p1) && RL_FINITE(p2) && p1 < p2)) return NAN;
            return (x >= p1 && x <= p2) ? 1.0 / (p2 - p1) : 0.0;
        case RL_PROP_EXPONENTIAL:
            if (!(RL_FINITE(p1) && p1 > 0.0)) return NAN;
            return (x >= 0.0) ? exp(-x / p1) / p1 : 0.0;
        case RL_PROP_CAUCHY: {
            if (!(RL_FINITE(p1) && RL_FINITE(p2) && p2 > 0.0)) return NAN;
            const double z = (x - p1) / p2;
            return 1.0 / (RL_PI * p2 * (1.0 + z * z));
        }
        case RL_PROP_LAPLACE:
            if (!(RL_FINITE(p1) && RL_FINITE(p2) && p2 > 0.0)) return NAN;
            return exp(-fabs(x - p1) / p2) / (2.0 * p2);
        case RL_PROP_DL_BOX_MULLER:
        case RL_PROP_DL_ZIGGURAT: {
            if (!(RL_FINITE(p1) && RL_FINITE(p2) && p2 > 0.0)) return NAN;
            const double z = (x - p1) / p2;
            return exp(-0.5 * z * z) / (p2 * sqrt(2.0 * RL_PI));
        }
        default: return NAN;
    }
}

int rl_proposal_sample(RL_State* st, int type, double p1, double p2, const RL_DLZiggurat* table, double* out) {
    if (!st || !out) return RL_ERR_ARG;
    if (isnan(rl_proposal_density(type, p1, p1, p2))) return RL_ERR_ARG;   /* validates parameters */
    switch (type) {
        case RL_PROP_UNIFORM:       return rl_uniform_range(st, p1, p2, out);
        case RL_PROP_EXPONENTIAL:   return rl_exponential(st, p1, out);
        case RL_PROP_CAUCHY:        return rl_cauchy(st, p1, p2, out);
        case RL_PROP_LAPLACE:       return rl_laplace(st, p1, p2, out);
        case RL_PROP_DL_BOX_MULLER: return rl_dl_box_muller(st, p1, p2, out);
        case RL_PROP_DL_ZIGGURAT: {
            if (!table) return RL_ERR_ARG;
            double z;
            const int rc = rl_dl_ziggurat_sample(st, table, &z);
            if (rc == RL_OK) *out = p1 + p2 * z;
            return rc;
        }
        default: return RL_ERR_ARG;
    }
}

int rl_rejection_continuous(RL_State* st, RL_TargetPDF f, void* f_params, int type, double M,
                            double p1, double p2, const RL_DLZiggurat* table, int64_t max_attempts,
                            int check_envelope, double* out, int64_t* attempts_used) {
    if (!st || !f || !out || max_attempts <= 0 || !RL_FINITE(M) || M <= 0.0) return RL_ERR_ARG;
    if (isnan(rl_proposal_density(type, p1, p1, p2))) return RL_ERR_ARG;
    if (type == RL_PROP_DL_ZIGGURAT && !table) return RL_ERR_ARG;

    for (int64_t a = 1; a <= max_attempts; a++) {
        double x;
        int rc = rl_proposal_sample(st, type, p1, p2, table, &x);
        if (rc != RL_OK) return rc;
        const double g = rl_proposal_density(type, x, p1, p2);
        const double fx = f(x, f_params);
        if (isnan(fx) || fx < 0.0) return RL_ERR_CALLBACK;
        const double u = rl_u01(st);
        if (check_envelope && fx > M * g * (1.0 + ENVELOPE_TOL)) return RL_ERR_ENVELOPE;
        if (u * M * g <= fx) {
            *out = x;
            if (attempts_used) *attempts_used = a;
            return RL_OK;
        }
    }
    if (attempts_used) *attempts_used = max_attempts;
    return RL_ERR_TIMEOUT;
}
