/**
 * optimizer_engine.h - numerical analysis of a target density: effective
 * support, extrema and normalising area (used by the Python RejectionOptimizer).
 */
#ifndef DLRAND_OPTIMIZER_H
#define DLRAND_OPTIMIZER_H
#include "rejection_cont_engine.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef struct RL_Analysis {
    double a, b;        /* analysed domain [a, b]                                   */
    double f_min, f_max;/* extrema of f on [a, b] (grid estimate)                   */
    double area;        /* integral of f over [a, b] (trapezoid rule)               */
    double total_area;  /* integral over the whole scanned window (auto mode)       */
} RL_Analysis;

/**
 * If a < b the given domain is analysed as is.
 * If a >= b (auto mode) the window [center-R, center+R] is doubled from R = 1
 * until f has decayed to <= 1e-9*max(f) at both ends; the reported domain is
 * then trimmed to the central (1 - tail_mass) of the mass (e.g. 0.01 keeps 99%).
 * Returns RL_ERR_ARG when no decaying support is found (e.g. f(x) = x^2) and
 * RL_ERR_CALLBACK when f returns NaN. steps is the grid resolution (>= 100).
 */
RL_API int rl_analyze_and_normalize(RL_TargetPDF f, void* params, double a, double b,
                                    double center, double tail_mass, int steps,
                                    RL_Analysis* out);

#ifdef __cplusplus
}
#endif
#endif
