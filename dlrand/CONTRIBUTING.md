# Contributing

## Build and test

```bash
pip install -e ".[test]"             # builds the C library next to the package
pytest                                # 150+ statistical and behavioural tests

cmake -S . -B build -DRL_SANITIZE=ON  # C unit tests under AddressSanitizer + UBSan
cmake --build build && ctest --test-dir build --output-on-failure
```

## Ground rules

- **No global or static mutable state in C.** Every engine takes an `RL_State*`; reproducibility depends on it.
- **No in-band error values.** Return an `rl_status`, write results through an output pointer, validate every argument
  (NaN, infinities, ranges, NULL) before touching memory.
- **Never clamp a uniform to avoid log(0).** Use `rl_uopen()` (open interval) instead; clamping truncates tails.
- **New or changed samplers need a statistical test**: chi-square against the exact PMF (discrete) or Kolmogorov-Smirnov
  against the exact CDF (continuous), with a fixed seed. Add edge cases (p = 0, p = 1, huge parameters).
- Cancellation matters: write acceptance tests with `log1p`, `fma` and `rl_stirling_tail` instead of subtracting large log-gammas.
- Keep the public C API under `include/dlrand/`, prefix symbols `rl_`, mark exports `RL_API`.
- Code must compile warning-free with `-std=c99 -Wall -Wextra -Wpedantic -Wshadow` and MSVC `/W4`.
- Naming: the normal distribution is called the **DL (De Moivre-Laplace)** distribution throughout this project.
