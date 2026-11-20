/**
 * rl_common.h - shared types for dlrand.
 *
 * Every engine takes an explicit RL_State*. There is NO global or static
 * mutable state anywhere in the library, so two generators never influence
 * each other and a generator can be used from one thread at a time without
 * locks (use rl_jump()/spawn() to give each thread its own stream).
 *
 * Error handling: functions that can fail return an rl_status (0 = success)
 * and write their result through an output pointer. Results are never
 * encoded as in-band sentinel values.
 */
#ifndef DLRAND_COMMON_H
#define DLRAND_COMMON_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) || defined(__CYGWIN__)
  #if defined(RL_STATIC)
    #define RL_API
  #elif defined(RL_BUILDING)
    #define RL_API __declspec(dllexport)
  #else
    #define RL_API __declspec(dllimport)
  #endif
#elif defined(__GNUC__) && __GNUC__ >= 4
  #define RL_API __attribute__((visibility("default")))
#else
  #define RL_API
#endif

#define RL_PI 3.14159265358979323846

typedef enum {
    RL_OK            =  0,
    RL_ERR_ARG       = -1,  /* invalid argument (out of domain, NULL, NaN...) */
    RL_ERR_TIMEOUT   = -2,  /* rejection sampler exceeded max_attempts        */
    RL_ERR_ALLOC     = -3,  /* out of memory                                  */
    RL_ERR_CALLBACK  = -4,  /* user callback returned NaN / negative density  */
    RL_ERR_ENVELOPE  = -5   /* detected f(x) > M*g(x): M is too small         */
} rl_status;

/* xoshiro256++ state plus the cached second DL (Box-Muller) variate.
 * Kept inside the state (never static) so that reproducibility holds. */
typedef struct RL_State {
    uint64_t s[4];
    double   spare;
    int      has_spare;
} RL_State;

RL_API const char* rl_version(void);
RL_API const char* rl_status_string(int status);

#ifdef __cplusplus
}
#endif
#endif /* DLRAND_COMMON_H */
