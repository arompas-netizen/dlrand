/* test_core.c - C-level unit tests (public API only). Exit code 0 = all passed. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "dlrand/dlrand.h"

static int g_fail = 0, g_checks = 0;
#define CHECK(cond) do { g_checks++; if (!(cond)) { g_fail++; \
    fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); } } while (0)

static double unit_pdf(double x, void* p) { (void)p; return (x >= 0 && x <= 2) ? x * (2 - x) * (2 - x) : 0.0; }
static double nan_pdf(double x, void* p)  { (void)x; (void)p; return NAN; }

static void test_core(void) {
    RL_State *a = rl_state_new(123), *b = rl_state_new(123), *c = rl_state_new(124);
    int same = 1, diff = 0;
    for (int i = 0; i < 1000; i++) {
        uint64_t x = rl_next_u64(a), y = rl_next_u64(b), z = rl_next_u64(c);
        if (x != y) same = 0;
        if (x != z) diff = 1;
    }
    CHECK(same); CHECK(diff);

    uint64_t snap[6];
    rl_state_get(a, snap);
    uint64_t n1 = rl_next_u64(a);
    CHECK(rl_state_set(b, snap) == RL_OK);
    CHECK(rl_next_u64(b) == n1);
    uint64_t zero[6] = {0, 0, 0, 0, 0, 0};
    CHECK(rl_state_set(b, zero) == RL_ERR_ARG);

    int ok = 1;
    for (int i = 0; i < 200000; i++) {
        double u = rl_uniform01(a), o = rl_uniform_open(a);
        if (!(u >= 0.0 && u < 1.0) || !(o > 0.0 && o < 1.0)) ok = 0;
    }
    CHECK(ok);

    /* bounded: unbiased over 6 faces */
    long cnt[6] = {0};
    for (int i = 0; i < 600000; i++) cnt[rl_bounded(a, 6)]++;
    double chi = 0;
    for (int i = 0; i < 6; i++) chi += (cnt[i] - 100000.0) * (cnt[i] - 100000.0) / 100000.0;
    CHECK(chi < 5 + 6 * sqrt(10.0));

    RL_State j1 = *a, j2 = *a;
    rl_jump(&j1); rl_long_jump(&j2);
    CHECK(j1.s[0] != j2.s[0] || j1.s[1] != j2.s[1]);
    CHECK(j1.s[0] != a->s[0] || j1.s[1] != a->s[1]);

    int64_t v;
    CHECK(rl_integers(a, 5, 4, &v) == RL_ERR_ARG);
    CHECK(rl_integers(a, INT64_MIN, INT64_MAX, &v) == RL_OK);   /* full 2^64 span */
    CHECK(rl_integers(a, -3, 3, &v) == RL_OK && v >= -3 && v <= 3);
    double d;
    CHECK(rl_uniform_range(a, 2.0, 1.0, &d) == RL_ERR_ARG);
    CHECK(rl_uniform_range(a, NAN, 1.0, &d) == RL_ERR_ARG);
    CHECK(rl_uniform_range(a, 1.0, 1.0, &d) == RL_OK && d == 1.0);
    CHECK(fabs(rl_loggam(10.0) - 12.801827480081469) < 1e-12);   /* ln(9!) */
    CHECK(fabs(rl_loggam(1.5) - (-0.12078223763524522)) < 1e-12);
    rl_state_free(a); rl_state_free(b); rl_state_free(c);
}

static void test_dl_independence(void) {
    /* The old library kept the Box-Muller spare in a static: same seed, different stream. */
    RL_State *a = rl_state_new(7), *b = rl_state_new(7), *c = rl_state_new(99);
    double sa[4], sb[4], tmp;
    rl_dl_box_muller_fill(a, 0, 1, 4, sa);
    rl_dl_box_muller(b, 0, 1, &sb[0]);
    rl_dl_box_muller(c, 0, 1, &tmp);                  /* another generator in between */
    rl_dl_box_muller_fill(b, 0, 1, 3, &sb[1]);
    CHECK(memcmp(sa, sb, sizeof sa) == 0);
    /* re-seeding clears the cache */
    rl_dl_box_muller(a, 0, 1, &tmp);
    rl_seed(a, 7);
    double again;
    rl_dl_box_muller(a, 0, 1, &again);
    CHECK(again == sa[0]);
    CHECK(rl_dl_box_muller(a, 0, -1, &tmp) == RL_ERR_ARG);
    rl_state_free(a); rl_state_free(b); rl_state_free(c);
}

static void test_discrete(void) {
    RL_State* s = rl_state_new(1);
    int64_t arr[3] = {1, 2, 3};
    CHECK(rl_permutation_partial(s, arr, 3, 10) == RL_ERR_ARG);      /* was a stack overflow */
    CHECK(arr[0] == 1 && arr[1] == 2 && arr[2] == 3);
    CHECK(rl_permutation_partial(s, arr, 3, 0) == RL_OK);
    CHECK(rl_permutation_partial(s, arr, 3, -1) == RL_ERR_ARG);

    int64_t sel[1000], cnt;
    CHECK(rl_bernoulli_select(s, 1000, 0.0, RL_BERNOULLI_JUMP, sel, &cnt) == RL_OK && cnt == 0);   /* was a segfault */
    CHECK(rl_bernoulli_select(s, 1000, 1.0, RL_BERNOULLI_JUMP, sel, &cnt) == RL_OK && cnt == 1000);
    CHECK(rl_bernoulli_select(s, 1000, 1.5, RL_BERNOULLI_STANDARD, sel, &cnt) == RL_ERR_ARG);
    CHECK(rl_bernoulli_select(s, 1000, -0.1, RL_BERNOULLI_STANDARD, sel, &cnt) == RL_ERR_ARG);
    CHECK(rl_bernoulli_select(s, 1000, NAN, RL_BERNOULLI_STANDARD, sel, &cnt) == RL_ERR_ARG);
    CHECK(rl_bernoulli_select(s, 1000, 1e-300, RL_BERNOULLI_JUMP, sel, &cnt) == RL_OK && cnt == 0);
    int sorted = 1;
    CHECK(rl_bernoulli_select(s, 1000, 0.3, RL_BERNOULLI_JUMP, sel, &cnt) == RL_OK);
    for (int64_t i = 1; i < cnt; i++) if (sel[i] <= sel[i - 1] || sel[i] >= 1000) sorted = 0;
    CHECK(sorted && cnt > 200 && cnt < 400);

    int64_t g;
    CHECK(rl_geometric_trials(s, 0.0, &g) == RL_ERR_ARG);
    CHECK(rl_geometric_trials(s, 2.0, &g) == RL_ERR_ARG);
    CHECK(rl_geometric_trials(s, 1.0, &g) == RL_OK && g == 1);
    CHECK(rl_geometric_failures(s, 1.0, &g) == RL_OK && g == 0);
    CHECK(rl_geometric_trials(s, 1e-20, &g) == RL_OK && g >= 1);       /* log1p keeps this finite */

    int64_t k;
    double mean = 0;
    for (int i = 0; i < 20000; i++) { rl_poisson(s, 800.0, RL_POISSON_PTRS, &k); mean += (double)k; }
    mean /= 20000;
    CHECK(fabs(mean - 800) < 3 * sqrt(800.0 / 20000) + 0.5);            /* sequential/centered were broken here */
    CHECK(rl_poisson(s, 800.0, RL_POISSON_SEQUENTIAL, &k) == RL_ERR_ARG);
    CHECK(rl_poisson(s, 1e15, RL_POISSON_AUTO, &k) == RL_OK && k > 0);
    CHECK(rl_poisson(s, -1.0, RL_POISSON_AUTO, &k) == RL_ERR_ARG);
    CHECK(rl_poisson(s, NAN, RL_POISSON_AUTO, &k) == RL_ERR_ARG);
    CHECK(rl_poisson(s, 5.0, RL_POISSON_PTRS, &k) == RL_ERR_ARG);

    mean = 0;
    for (int i = 0; i < 20000; i++) { rl_binomial(s, 2000, 0.5, RL_BINOMIAL_AUTO, &k); mean += (double)k; }
    mean /= 20000;
    CHECK(fabs(mean - 1000) < 3 * sqrt(500.0 / 20000) + 0.5);           /* BINV returned n here */
    CHECK(rl_binomial(s, 20, 1.0, RL_BINOMIAL_INVERSION, &k) == RL_OK && k == 20);
    CHECK(rl_binomial(s, 20, 0.0, RL_BINOMIAL_AUTO, &k) == RL_OK && k == 0);
    CHECK(rl_binomial(s, 20, 1.5, RL_BINOMIAL_AUTO, &k) == RL_ERR_ARG);
    CHECK(rl_binomial(s, -1, 0.5, RL_BINOMIAL_AUTO, &k) == RL_ERR_ARG);
    CHECK(rl_binomial(s, 19, 0.5, RL_BINOMIAL_BTRS, &k) == RL_ERR_ARG);     /* n*p < 10 */
    CHECK(rl_binomial(s, 5000, 0.5, RL_BINOMIAL_INVERSION, &k) == RL_ERR_ARG);

    int64_t counts[2];
    double p2[2] = {0.5, 0.5};
    CHECK(rl_multinomial(s, 5000, 2, p2, RL_MULTINOMIAL_AUTO, counts) == RL_OK);
    CHECK(counts[0] + counts[1] == 5000 && counts[0] > 2300 && counts[0] < 2700);
    double bad[2] = {0.5, 0.6}, neg[2] = {1.5, -0.5};
    CHECK(rl_multinomial(s, 10, 2, bad, RL_MULTINOMIAL_AUTO, counts) == RL_ERR_ARG);
    CHECK(rl_multinomial(s, 10, 2, neg, RL_MULTINOMIAL_AUTO, counts) == RL_ERR_ARG);
    CHECK(rl_multinomial(s, 10, 0, p2, RL_MULTINOMIAL_AUTO, counts) == RL_ERR_ARG);   /* r = 0: was counts[-1] */
    int64_t c5[5]; double p5[5] = {0.1, 0.2, 0.3, 0.0, 0.4};
    for (int m = 1; m <= 2; m++) {
        CHECK(rl_multinomial(s, 1000, 5, p5, m, c5) == RL_OK);
        CHECK(c5[0] + c5[1] + c5[2] + c5[3] + c5[4] == 1000 && c5[3] == 0);
    }
    rl_state_free(s);
}

static void test_continuous(void) {
    RL_State* s = rl_state_new(5);
    enum { N = 200000 };
    double* x = (double*)malloc(N * sizeof(double));
    int allfinite = 1;
    rl_laplace_fill(s, 0, 1, N, x);       for (int i = 0; i < N; i++) if (!isfinite(x[i])) allfinite = 0;
    rl_exponential_fill(s, 1, N, x);      for (int i = 0; i < N; i++) if (!isfinite(x[i]) || x[i] < 0) allfinite = 0;
    rl_cauchy_fill(s, 0, 1, N, x);        for (int i = 0; i < N; i++) if (!isfinite(x[i])) allfinite = 0;
    CHECK(allfinite);
    double d;
    CHECK(rl_exponential(s, 0.0, &d) == RL_ERR_ARG);
    CHECK(rl_laplace(s, 0.0, -1.0, &d) == RL_ERR_ARG);
    CHECK(rl_cauchy(s, 0.0, 0.0, &d) == RL_ERR_ARG);

    /* DL Ziggurat tables: published Marsaglia-Tsang constants for 128 and 256 layers */
    RL_DLZiggurat *z128 = rl_dl_ziggurat_new(128), *z256 = rl_dl_ziggurat_new(256);
    CHECK(z128 && z256);
    CHECK(fabs(z128->r - 3.442619855899) < 1e-8 && fabs(z128->area_v - 9.91256303526217e-3) < 1e-11);
    CHECK(fabs(z256->r - 3.6541528853610088) < 1e-8 && fabs(z256->area_v - 4.92867323399e-3) < 1e-11);
    CHECK(rl_dl_ziggurat_new(7) == NULL && rl_dl_ziggurat_new(70000) == NULL);
    int Ns[] = {16, 64, 128, 256, 512, 1000, 4096};
    for (size_t t = 0; t < sizeof Ns / sizeof Ns[0]; t++) {
        RL_DLZiggurat* z = rl_dl_ziggurat_new(Ns[t]);
        CHECK(z != NULL);
        int mono = 1;
        for (int i = 1; i < Ns[t]; i++) if (!(z->x_table[i] > z->x_table[i + 1])) mono = 0;
        CHECK(mono && z->x_table[Ns[t]] == 0.0 && z->y_table[Ns[t]] == 1.0);
        rl_dl_ziggurat_fill(s, z, N, x);
        double m = 0, v = 0, k4 = 0;
        for (int i = 0; i < N; i++) m += x[i];
        m /= N;
        for (int i = 0; i < N; i++) { double dd = x[i] - m; v += dd * dd; k4 += dd * dd * dd * dd; }
        v /= N; k4 /= N * v * v;
        CHECK(fabs(m) < 0.012 && fabs(v - 1) < 0.02 && fabs(k4 - 3) < 0.1);
        rl_dl_ziggurat_free(z);
    }
    rl_dl_box_muller_fill(s, 0, 1, N, x);
    double m = 0, v = 0;
    for (int i = 0; i < N; i++) m += x[i];
    m /= N;
    for (int i = 0; i < N; i++) v += (x[i] - m) * (x[i] - m);
    v /= N;
    CHECK(fabs(m) < 0.012 && fabs(v - 1) < 0.02);

    /* rejection: invalid M is DETECTED, valid M reproduces the true mean 0.8 */
    int64_t att = 0;
    double mean = 0; int rc = RL_OK; int viol = 0;
    for (int i = 0; i < 2000 && rc == RL_OK; i++) {
        rc = rl_rejection_continuous(s, unit_pdf, NULL, RL_PROP_UNIFORM, 1.2, 0.0, 2.0, NULL, 10000, 1, &d, &att);
        if (rc == RL_ERR_ENVELOPE) viol = 1;
    }
    CHECK(viol);                              /* sup f/g is 2.37: M = 1.2 must be rejected */
    for (int i = 0; i < 40000; i++) {
        CHECK(rl_rejection_continuous(s, unit_pdf, NULL, RL_PROP_UNIFORM, 2.4, 0.0, 2.0, NULL, 10000, 1, &d, &att) == RL_OK);
        mean += d;
    }
    mean /= 40000;
    CHECK(fabs(mean - 0.8) < 0.02);
    CHECK(rl_rejection_continuous(s, nan_pdf, NULL, RL_PROP_UNIFORM, 2.4, 0.0, 2.0, NULL, 10, 1, &d, &att) == RL_ERR_CALLBACK);
    CHECK(rl_rejection_continuous(s, unit_pdf, NULL, RL_PROP_UNIFORM, 2.4, 2.0, 0.0, NULL, 10, 1, &d, &att) == RL_ERR_ARG);
    CHECK(rl_rejection_continuous(s, unit_pdf, NULL, RL_PROP_DL_ZIGGURAT, 2.4, 0.0, 1.0, NULL, 10, 1, &d, &att) == RL_ERR_ARG);
    CHECK(rl_rejection_continuous(s, unit_pdf, NULL, RL_PROP_UNIFORM, 1e9, 0.0, 2.0, NULL, 5, 0, &d, &att) == RL_ERR_TIMEOUT);
    CHECK(rl_rejection_continuous(s, rl_dl_pdf, NULL, RL_PROP_DL_ZIGGURAT, 2.5066282746310002, 0.0, 1.0, z256, 1000, 1, &d, &att) == RL_OK);
    CHECK(isnan(rl_proposal_density(99, 0, 0, 1)) && isnan(rl_proposal_density(RL_PROP_LAPLACE, 0, 0, -1)));

    /* optimizer analysis */
    RL_Analysis an;
    CHECK(rl_analyze_and_normalize(rl_dl_pdf, NULL, 0, 0, 0.0, 0.01, 20001, &an) == RL_OK);
    CHECK(fabs(an.total_area - sqrt(2 * RL_PI)) < 1e-6 && an.a < -2.4 && an.a > -2.7 && an.b > 2.4 && an.b < 2.7);
    CHECK(fabs(an.area / an.total_area - 0.99) < 0.003);
    CHECK(rl_analyze_and_normalize(rl_dl_pdf, NULL, 1.0, -1.0 + 5.0, 0, 0.01, 1000, &an) == RL_OK);   /* explicit domain [1,4] */
    CHECK(rl_analyze_and_normalize(unit_pdf, NULL, 0, 0, 0.0, 0.01, 1000, &an) == RL_OK);   /* finite support, decays to 0 */
    CHECK(rl_analyze_and_normalize(nan_pdf, NULL, 0, 0, 0.0, 0.01, 1000, &an) == RL_ERR_CALLBACK);
    rl_dl_ziggurat_free(z128); rl_dl_ziggurat_free(z256);
    free(x); rl_state_free(s);
}

static int64_t w_called = 0;
static int fixed_worker(RL_State* st, void* p, int64_t* out) { (void)st; w_called++; *out = *(int64_t*)p; return RL_OK; }

static void test_composition_and_rejection_discrete(void) {
    RL_State* s = rl_state_new(11);
    int64_t v0 = 0, v1 = 1, v2 = 2;
    double alpha[3] = {0.2, 0.5, 0.3};
    int64_t* vals[3] = {&v0, &v1, &v2};
    for (int mode = 0; mode < 3; mode++) {
        RL_Composition* c = rl_composition_new(mode, 3);
        CHECK(c != NULL);
        CHECK(rl_composition_sample(s, c, &v0) == RL_ERR_ARG);              /* not finalized */
        for (int i = 0; i < 3; i++) CHECK(rl_composition_set(c, i, alpha[i], fixed_worker, vals[i]) == RL_OK);
        CHECK(rl_composition_finalize(c) == RL_OK);
        long cnt[3] = {0, 0, 0};
        int64_t out;
        for (int i = 0; i < 100000; i++) { CHECK(rl_composition_sample(s, c, &out) == RL_OK); cnt[out]++; }
        for (int i = 0; i < 3; i++) CHECK(fabs(cnt[i] / 100000.0 - alpha[i]) < 0.01);
        rl_composition_free(c);
    }
    RL_Composition* c = rl_composition_new(RL_COMP_ALIAS, 2);
    CHECK(rl_composition_set(c, 0, 0.7, fixed_worker, &v0) == RL_OK);
    CHECK(rl_composition_finalize(c) == RL_ERR_ARG);                         /* slot 1 never set */
    CHECK(rl_composition_set(c, 1, 0.7, fixed_worker, &v1) == RL_OK);
    CHECK(rl_composition_finalize(c) == RL_ERR_ARG);                         /* sums to 1.4 */
    CHECK(rl_composition_set(c, 5, 0.1, fixed_worker, &v1) == RL_ERR_ARG);
    rl_composition_free(c);
    CHECK(rl_composition_new(7, 3) == NULL && rl_composition_new(0, 0) == NULL);

    /* discrete rejection */
    double pmf[4] = {0.1, 0.2, 0.3, 0.4};
    double minM = rl_rejection_array_min_M(pmf, 4);
    CHECK(fabs(minM - 1.6) < 1e-12);
    long cnt[4] = {0};
    int64_t out;
    for (int i = 0; i < 100000; i++) { CHECK(rl_rejection_array(s, pmf, 4, minM, 1000, &out) == RL_OK); cnt[out]++; }
    for (int i = 0; i < 4; i++) CHECK(fabs(cnt[i] / 100000.0 - pmf[i]) < 0.01);
    int rc = RL_OK;
    for (int i = 0; i < 200 && rc == RL_OK; i++) rc = rl_rejection_array(s, pmf, 4, 1.0, 1000, &out);
    CHECK(rc == RL_ERR_ENVELOPE);                                             /* M below size*max(pmf) */
    pmf[1] = -1.0;
    CHECK(rl_rejection_array(s, pmf, 4, 2.0, 10, &out) == RL_ERR_ARG);
    rl_state_free(s);
}

int main(void) {
    test_core();
    test_dl_independence();
    test_discrete();
    test_continuous();
    test_composition_and_rejection_discrete();
    printf("%d checks, %d failed\n", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
