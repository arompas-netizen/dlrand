#include "rl_internal.h"

/**
 * Stirling remainder  T(x) = ln(x!) - [ (x+1/2) ln(x+1) - (x+1) + 1/2 ln(2 pi) ],  x >= 0.
 * T(x) ~ 1/(12(x+1)) is tiny, which lets the acceptance tests of PTRS/BTRS be written as
 * differences of *small* quantities instead of differences of huge log-gamma values
 * (those lose every digit once n or lambda reaches ~1e15).
 */
double rl_stirling_tail(double x) {
    if (x < 20.0)
        return rl_loggam(x + 1.0) - ((x + 0.5) * log(x + 1.0) - (x + 1.0) + 0.5 * log(2.0 * RL_PI));
    const double z = 1.0 / (x + 1.0), z2 = z * z;
    return z * (1.0 / 12.0 - z2 * (1.0 / 360.0 - z2 * (1.0 / 1260.0 - z2 * (1.0 / 1680.0))));
}
