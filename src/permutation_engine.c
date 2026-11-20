#include "rl_internal.h"

/* Fisher-Yates: position i swaps with a uniform element of the not-yet-fixed pool [i, n). */
int rl_permutation_partial(RL_State* st, int64_t* arr, int64_t n, int64_t r) {
    if (!st || n < 0 || r < 0 || r > n || (n > 0 && !arr)) return RL_ERR_ARG;
    for (int64_t i = 0; i < r; i++) {
        const int64_t j = i + (int64_t)rl_below(st, (uint64_t)(n - i));
        const int64_t tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
    return RL_OK;
}
