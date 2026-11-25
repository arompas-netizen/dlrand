# dlrand

[![PyPI](https://img.shields.io/pypi/v/dlrand)](https://pypi.org/project/dlrand/)
[![Python](https://img.shields.io/pypi/pyversions/dlrand)](https://pypi.org/project/dlrand/)
[![CI](https://github.com/arompas-netizen/dlrand/actions/workflows/ci.yml/badge.svg)](https://github.com/arompas-netizen/dlrand/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/pypi/l/dlrand)](https://github.com/arompas-netizen/dlrand/blob/main/LICENSE)

Exact, reproducible random variate generation: a portable **C99** core with a clean **Python** interface.

- **Exact samplers.** Poisson (PTRS), Binomial (inversion / BTRS), Geometric, Multinomial, mixtures, alias tables, rejection
  sampling, DL Ziggurat. Every distribution is tested against its exact PMF/CDF.
- **Reproducible.** A seed fully determines a stream. The C library has *no* global or static mutable state, so generators
  never interfere. `getstate()/setstate()` and `spawn()` (jump-ahead, 2^128 apart) are built in.
- **Safe.** Every argument is validated; failures are status codes in C and exceptions in Python - never sentinel values
  and never silent bias. Rejection sampling detects an undersized envelope constant `M` (`EnvelopeError`).
- **Portable.** Plain C99 targeting Linux, macOS and Windows (MSVC, GCC, Clang); `pip install dlrand` or CMake. The included
  GitHub Actions workflow builds and tests all three.

> **Naming.** The distribution that is also called the *normal* distribution is called the **DL (De Moivre-Laplace)
> distribution** throughout this library, after Abraham de Moivre (1733) and Pierre-Simon Laplace, who introduced and
> developed it: `dl_box_muller`, `dl_ziggurat`, `rl_dl_pdf`, ...

> **Not for cryptography.** xoshiro256++ is a fast statistical generator, not a CSPRNG. Do not use it for keys, tokens or passwords.

## Install

```bash
pip install dlrand              # prebuilt wheels for Windows, macOS and Linux (no compiler needed)
pip install "dlrand[numpy]"     # optional: RandomGenerator(numpy=True) returns arrays
```

From source (needs a C99 compiler): `pip install .` in a checkout. You can also build the C library alone with
`cmake -S . -B build && cmake --build build`; the Python package then finds `build/libdlrand.*` automatically
(or set `DLRAND_PATH`).

## Quick start

```python
import dlrand

rng = dlrand.RandomGenerator(seed=2024)           # same seed -> same stream, always

rng.uniform(size=3)                                # U[0,1), 53 random bits
rng.integers(1, 6, size=10)                        # unbiased, inclusive bounds
rng.dl_box_muller(mu=0.0, sigma=1.0, size=3)       # DL distribution (Box-Muller)
rng.dl_ziggurat(size=3)                            # DL distribution (Ziggurat, any layer count)
rng.poisson(800, size=5)                           # exact for every lambda up to 1e15
rng.binomial(2000, 0.5, size=5)                    # exact for n up to 2^53
rng.multinomial(1000, [0.5, 0.3, 0.2])
rng.permutation(["A", "K", "Q", "J"], r=3)         # sampling without replacement, any objects
rng.bernoulli_sampling(list(range(20)), 0.25)      # keep each item with probability p

mix = dlrand.MixtureBuilder(rng).add_uniform(0.5, 0, 10).add_poisson(0.5, 100.0).build("alias")
mix.sample(8)                                      # engine built once, O(1) per draw

kids = rng.spawn(4)                                # independent parallel streams
```

Rejection sampling with an automatically found, *verified* envelope:

```python
f = lambda x: x * (2 - x) ** 2 if 0 <= x <= 2 else 0.0       # need not be normalised

opt = dlrand.RejectionOptimizer(rng)
print(opt.compare(f, domain=(0, 2)))                          # every proposal: M, expected vs measured acceptance
best = opt.best(f, domain=(0, 2))
xs = best.sample(rng, f, size=10_000)

rng.rejection_sampling(f, "uniform", 1.2, 0.0, 2.0)           # raises EnvelopeError: M = 1.2 < sup f/g = 2.37
```

More in [`examples/`](https://github.com/arompas-netizen/dlrand/tree/main/examples). Algorithms, domains and references: [`docs/ALGORITHMS.md`](https://github.com/arompas-netizen/dlrand/blob/main/docs/ALGORITHMS.md).
Coming from the earlier prototypes? See [`docs/MIGRATION.md`](https://github.com/arompas-netizen/dlrand/blob/main/docs/MIGRATION.md).

## Python API at a glance

| Area | Methods |
|---|---|
| Uniform | `uniform`, `integers` |
| DL distribution | `dl_box_muller` (alias `normal_bm`), `dl_ziggurat` |
| Other continuous | `exponential`, `cauchy`, `laplace`, `rejection_sampling`, `rejection_sampling_stats`, `get_proposal_density` |
| Discrete | `geometric`, `poisson`, `binomial`, `multinomial`, `bernoulli_sampling`, `permutation`, `choice` |
| Custom discrete | `sample_custom_array`, `sample_custom_func` |
| Composition | `mixture`, `sample_mixture`, `MixtureBuilder`, `alias_sampler` |
| Tuning | `RejectionOptimizer`: `analyze`, `plan`, `best`, `benchmark`, `compare`, `report` |
| State | `reseed`, `getstate`, `setstate`, `spawn`, `close` |

Methods taking `size` return a scalar for `size=None`, otherwise a list (NumPy array with `numpy=True`).

## C API

```c
#include <dlrand/dlrand.h>

RL_State* st = rl_state_new(2024);
double x;
int rc = rl_dl_ziggurat_sample(st, table, &x);       /* 0 = RL_OK */
if (rc != RL_OK) fprintf(stderr, "%s\n", rl_status_string(rc));
```

Results are written through output pointers; the return value is an `rl_status`
(`RL_ERR_ARG`, `RL_ERR_TIMEOUT`, `RL_ERR_ALLOC`, `RL_ERR_CALLBACK`, `RL_ERR_ENVELOPE`). Headers live in
[`include/dlrand/`](https://github.com/arompas-netizen/dlrand/tree/main/include/dlrand), one per engine plus the umbrella `dlrand.h`.

## Supported domains

| Distribution | Domain | Notes |
|---|---|---|
| Poisson | 0 <= lambda <= 1e15 | `sequential` needs lambda <= 700; `ptrs` needs lambda >= 10; `auto` picks |
| Binomial | 0 <= n <= 2^53, 0 <= p <= 1 | `auto` is exact; `poisson_approx` is an explicit approximation |
| Geometric | 0 < p <= 1 | `success` {1,2,...} or `failure` {0,1,...}; saturates at INT64_MAX |
| Multinomial | p >= 0, sum within 1e-8 of 1 | alias if r > n, else sequencing |
| Exponential / Cauchy / Laplace | finite parameters, scale > 0 | outputs are always finite |
| DL | finite mu, finite sigma >= 0 | Ziggurat: 8..65536 layers, table solved at creation |

Precision note: for n or lambda beyond ~1e12 the acceptance tests are accurate to floating-point resolution, not beyond.
Results are bit-reproducible for a given platform and build; `libm` differences between platforms can change the last
bit of `log`/`exp`, which on rare occasions flips a rejection decision.

## Thread safety

A generator is not synchronised: use one per thread (`spawn()` gives non-overlapping streams). Separate generators,
tables and alias/composition engines share no mutable state and may be used concurrently. The Python callbacks used by
rejection sampling run under the GIL.

## Tests

```bash
pip install -e ".[test]" && pytest                   # chi-square / KS tests against exact PMFs and CDFs
cmake -S . -B build -DRL_SANITIZE=ON && cmake --build build && ctest --test-dir build
```

All tests use fixed seeds (deterministic) with ~5-sigma thresholds. The CI workflow covers Linux, macOS and Windows with
Python 3.9, 3.12 and 3.13.

## License

MIT - see [LICENSE](https://github.com/arompas-netizen/dlrand/blob/main/LICENSE).
