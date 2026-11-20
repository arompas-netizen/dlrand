#include <string.h>
#include <stdlib.h>
#include "rl_internal.h"

const char* rl_version(void) { return "0.1.0"; }

const char* rl_status_string(int status) {
    switch (status) {
        case RL_OK:           return "ok";
        case RL_ERR_ARG:      return "invalid argument";
        case RL_ERR_TIMEOUT:  return "rejection sampling exceeded max_attempts";
        case RL_ERR_ALLOC:    return "out of memory";
        case RL_ERR_CALLBACK: return "user callback returned NaN or a negative density";
        case RL_ERR_ENVELOPE: return "envelope violated: f(x) > M*g(x), M is too small";
        default:              return "unknown status";
    }
}

/* SplitMix64: expands one 64-bit seed into well-mixed state words. */
static uint64_t splitmix64(uint64_t* x) {
    uint64_t z = (*x += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}

void rl_seed(RL_State* st, uint64_t seed) {
    uint64_t x = seed;
    for (int i = 0; i < 4; i++) st->s[i] = splitmix64(&x);
    st->spare = 0.0;
    st->has_spare = 0;
}

RL_State* rl_state_new(uint64_t seed) {
    RL_State* st = (RL_State*)malloc(sizeof(RL_State));
    if (st) rl_seed(st, seed);
    return st;
}
void rl_state_free(RL_State* st) { free(st); }

uint64_t rl_next_u64(RL_State* st)     { return rl_u64(st); }
double   rl_uniform01(RL_State* st)    { return rl_u01(st); }
double   rl_uniform_open(RL_State* st) { return rl_uopen(st); }
uint64_t rl_bounded(RL_State* st, uint64_t n) { return n == 0 ? 0 : rl_below(st, n); }

/* Jump polynomials from the xoshiro256 reference implementation. */
static void jump_with(RL_State* st, const uint64_t poly[4]) {
    uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    for (int i = 0; i < 4; i++) {
        for (int b = 0; b < 64; b++) {
            if (poly[i] & (UINT64_C(1) << b)) {
                s0 ^= st->s[0]; s1 ^= st->s[1]; s2 ^= st->s[2]; s3 ^= st->s[3];
            }
            (void)rl_u64(st);
        }
    }
    st->s[0] = s0; st->s[1] = s1; st->s[2] = s2; st->s[3] = s3;
    st->has_spare = 0;
    st->spare = 0.0;
}
void rl_jump(RL_State* st) {
    static const uint64_t J[4] = { UINT64_C(0x180ec6d33cfd0aba), UINT64_C(0xd5a61266f0c9392c),
                                   UINT64_C(0xa9582618e03fc9aa), UINT64_C(0x39abdc4529b1661c) };
    jump_with(st, J);
}
void rl_long_jump(RL_State* st) {
    static const uint64_t J[4] = { UINT64_C(0x76e15d3efefdcbbf), UINT64_C(0xc5004e441c522fb3),
                                   UINT64_C(0x77710069854ee241), UINT64_C(0x39109bb02acbe635) };
    jump_with(st, J);
}

void rl_state_get(const RL_State* st, uint64_t out[6]) {
    for (int i = 0; i < 4; i++) out[i] = st->s[i];
    out[4] = (uint64_t)(st->has_spare != 0);
    memcpy(&out[5], &st->spare, sizeof(uint64_t));
}
int rl_state_set(RL_State* st, const uint64_t in[6]) {
    if ((in[0] | in[1] | in[2] | in[3]) == 0 || in[4] > 1) return RL_ERR_ARG;
    for (int i = 0; i < 4; i++) st->s[i] = in[i];
    st->has_spare = (int)in[4];
    memcpy(&st->spare, &in[5], sizeof(uint64_t));
    return RL_OK;
}

/* log(Gamma(x)) for x >= 1: shift to x >= 7, then the Stirling series. */
double rl_loggam(double x) {
    static const double a[10] = {
        8.333333333333333e-02, -2.777777777777778e-03, 7.936507936507937e-04,
        -5.952380952380952e-04, 8.417508417508418e-04, -1.917526917526918e-03,
        6.410256410256410e-03, -2.955065359477124e-02, 1.796443723688307e-01,
        -1.39243221690590e+00 };
    if (x == 1.0 || x == 2.0) return 0.0;
    double x0 = x;
    int n = 0;
    if (x < 7.0) { n = (int)(7.0 - x); x0 = x + n; }
    const double x2 = 1.0 / (x0 * x0);
    double gl0 = a[9];
    for (int k = 8; k >= 0; k--) { gl0 *= x2; gl0 += a[k]; }
    double gl = gl0 / x0 + 0.5 * log(2.0 * RL_PI) + (x0 - 0.5) * log(x0) - x0;
    if (x < 7.0) {
        for (int k = 1; k <= n; k++) { gl -= log(x0 - 1.0); x0 -= 1.0; }
    }
    return gl;
}
