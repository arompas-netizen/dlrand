#include <stdlib.h>
#include "rl_internal.h"

#define DECAY_REL 1e-9
#define MAX_RANGE 1e9

/* Evaluates f on n+1 equispaced points of [lo, hi] into vals. Returns RL_ERR_CALLBACK on NaN/negative. */
static int scan(RL_TargetPDF f, void* params, double lo, double hi, int n, double* vals) {
    const double h = (hi - lo) / n;
    for (int i = 0; i <= n; i++) {
        const double v = f(lo + i * h, params);
        if (isnan(v) || v < 0.0) return RL_ERR_CALLBACK;
        vals[i] = v;
    }
    return RL_OK;
}

static double trapezoid(const double* v, int i0, int i1, double h) {
    double s = 0.0;
    for (int i = i0; i < i1; i++) s += 0.5 * (v[i] + v[i + 1]) * h;
    return s;
}

int rl_analyze_and_normalize(RL_TargetPDF f, void* params, double a, double b, double center,
                             double tail_mass, int steps, RL_Analysis* out) {
    if (!f || !out || steps < 100 || !RL_FINITE(center) || !(tail_mass >= 0.0 && tail_mass < 1.0)) return RL_ERR_ARG;
    double* v = (double*)malloc((size_t)(steps + 1) * sizeof(double));
    if (!v) return RL_ERR_ALLOC;
    int rc;
    double lo, hi;

    if (a < b) {                                   /* explicit domain */
        if (!RL_FINITE(a) || !RL_FINITE(b)) { free(v); return RL_ERR_ARG; }
        lo = a; hi = b;
        rc = scan(f, params, lo, hi, steps, v);
        if (rc != RL_OK) { free(v); return rc; }
        const double h = (hi - lo) / steps;
        double mn = v[0], mx = v[0];
        for (int i = 1; i <= steps; i++) { if (v[i] < mn) mn = v[i]; if (v[i] > mx) mx = v[i]; }
        out->a = lo; out->b = hi; out->f_min = mn; out->f_max = mx;
        out->area = out->total_area = trapezoid(v, 0, steps, h);
        free(v);
        return RL_OK;
    }

    /* auto mode: double the window until the density has decayed at both ends */
    int found = 0;
    for (double R = 1.0; R <= MAX_RANGE; R *= 2.0) {
        lo = center - R; hi = center + R;
        rc = scan(f, params, lo, hi, steps, v);
        if (rc != RL_OK) { free(v); return rc; }
        double mx = 0.0;
        for (int i = 0; i <= steps; i++) if (v[i] > mx) mx = v[i];
        if (mx > 0.0 && v[0] <= DECAY_REL * mx && v[steps] <= DECAY_REL * mx) { found = 1; break; }
    }
    if (!found) { free(v); return RL_ERR_ARG; }

    const double h = (hi - lo) / steps;
    const double total = trapezoid(v, 0, steps, h);
    /* trim to the central (1 - tail_mass) of the mass via the cumulative trapezoid */
    int i0 = 0, i1 = steps;
    double cum = 0.0;
    int got0 = 0;
    for (int i = 0; i < steps; i++) {
        cum += 0.5 * (v[i] + v[i + 1]) * h;
        if (!got0 && cum >= 0.5 * tail_mass * total) { i0 = i; got0 = 1; }
        if (cum >= (1.0 - 0.5 * tail_mass) * total) { i1 = i + 1; break; }
    }
    if (i1 <= i0) i1 = i0 + 1;
    double mn = v[i0], mx = v[i0];
    for (int i = i0; i <= i1; i++) { if (v[i] < mn) mn = v[i]; if (v[i] > mx) mx = v[i]; }
    out->a = lo + i0 * h; out->b = lo + i1 * h;
    out->f_min = mn; out->f_max = mx;
    out->area = trapezoid(v, i0, i1, h);
    out->total_area = total;
    free(v);
    return RL_OK;
}
