#include <stdlib.h>
#include "rl_internal.h"

double rl_dl_pdf(double x, void* params) { (void)params; return exp(-0.5 * x * x); }

/* One-sided tail area  integral_R^inf exp(-x^2/2) dx = sqrt(pi/2) * erfc(R/sqrt2). */
static double tail_area(double R) { return sqrt(RL_PI / 2.0) * erfc(R / sqrt(2.0)); }

/**
 * Builds x[0..N], y[0..N] for a given R, returning +1 if the strips overshoot
 * f(0)=1 before the last layer (R too small) or at the end, -1 if they fall short.
 * Layer i >= 1 is the rectangle [0, x_i] x [f(x_i), f(x_{i+1})] of area V;
 * layer 0 is the base rectangle plus the tail, also of area V.
 */
static int build(double R, int N, double* x, double* y, double* V_out) {
    const double fR = exp(-0.5 * R * R);
    const double V = R * fR + tail_area(R);
    x[0] = V / fR;
    x[1] = R;
    for (int i = 2; i < N; i++) {
        const double arg = V / x[i - 1] + exp(-0.5 * x[i - 1] * x[i - 1]);
        if (arg >= 1.0) return +1;
        x[i] = sqrt(-2.0 * log(arg));
    }
    x[N] = 0.0;
    const double last = V / x[N - 1] + exp(-0.5 * x[N - 1] * x[N - 1]);
    *V_out = V;
    if (y) {
        for (int i = 0; i < N; i++) y[i] = exp(-0.5 * x[i] * x[i]);
        y[N] = 1.0;                       /* f(0): the top of the last wedge */
    }
    return last >= 1.0 ? +1 : -1;
}

RL_DLZiggurat* rl_dl_ziggurat_new(int N) {
    if (N < 8 || N > 65536) return NULL;
    RL_DLZiggurat* t = (RL_DLZiggurat*)malloc(sizeof(RL_DLZiggurat));
    double* x = (double*)malloc((size_t)(N + 1) * sizeof(double));
    double* y = (double*)malloc((size_t)(N + 1) * sizeof(double));
    if (!t || !x || !y) { free(t); free(x); free(y); return NULL; }

    double lo = 1.0, hi = 12.0, V = 0.0;
    for (int it = 0; it < 200; it++) {          /* bisection on R */
        const double mid = 0.5 * (lo + hi);
        if (build(mid, N, x, NULL, &V) > 0) lo = mid; else hi = mid;
    }
    const double R = 0.5 * (lo + hi);
    build(R, N, x, y, &V);
    t->num_layers = N; t->r = R; t->area_v = V; t->x_table = x; t->y_table = y;
    return t;
}

void rl_dl_ziggurat_free(RL_DLZiggurat* t) {
    if (!t) return;
    free(t->x_table); free(t->y_table); free(t);
}

static double sample_one(RL_State* st, const RL_DLZiggurat* t) {
    const double* x = t->x_table;
    const double* y = t->y_table;
    for (;;) {
        const uint64_t bits = rl_u64(st);
        const int i = (int)(((bits >> 11) & 0xFFFFFFFFu) % (uint64_t)t->num_layers);   /* layer */
        const double sign = (bits & 1u) ? 1.0 : -1.0;
        const double cand = rl_u01(st) * x[i];
        if (cand < x[i + 1]) return sign * cand;             /* inside the inner rectangle */
        if (i == 0) {                                          /* tail (Marsaglia's exact method) */
            double xt, yt;
            do {
                xt = -log(rl_uopen(st)) / t->r;
                yt = -log(rl_uopen(st));
            } while (2.0 * yt < xt * xt);
            return sign * (t->r + xt);
        }
        const double yy = y[i] + rl_u01(st) * (y[i + 1] - y[i]);   /* wedge */
        if (yy < exp(-0.5 * cand * cand)) return sign * cand;
    }
}

int rl_dl_ziggurat_sample(RL_State* st, const RL_DLZiggurat* t, double* out) {
    if (!st || !t || !out) return RL_ERR_ARG;
    *out = sample_one(st, t);
    return RL_OK;
}
int rl_dl_ziggurat_fill(RL_State* st, const RL_DLZiggurat* t, int64_t n, double* out) {
    if (!st || !t) return RL_ERR_ARG;
    RL_CHECK_FILL(n, out);
    for (int64_t i = 0; i < n; i++) out[i] = sample_one(st, t);
    return RL_OK;
}
