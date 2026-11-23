# Migrating from the v1-v9 prototypes

The old layout (two library folders, `random_engine__v1..v9.py`, committed binaries) is replaced by one tree:
`include/dlrand/`, `src/`, `python/dlrand/`, `tests/`, `examples/`. Old versions live in git history.

**Streams changed.** The generator is now xoshiro256++ instead of a 31-bit LCG, so a given seed produces different numbers.

## C API
All symbols carry the `rl_` prefix, take `RL_State*` (was `LCG_State*`), return an `rl_status` and write results through a
pointer. Counts and sizes are `int64_t`. `rl_*_fill` batch variants exist for the common samplers.

| Old | New |
|---|---|
| `init_lcg(gen, seed)`, `next_uniform` | `rl_state_new` / `rl_seed`, `rl_uniform01`, `rl_uniform_open` |
| `generate_uniform_continuous` | `rl_uniform_range` |
| `fisher_yates_shuffle(gen, arr, n, r)` | `rl_permutation_partial` (rejects `r > n`) |
| `bernoulli_standard`, `bernoulli_jump` | `rl_bernoulli_select` (returns selected **indices**) |
| `generate_geometric_success` / `_failure` | `rl_geometric_trials` / `rl_geometric_failures` |
| `generate_poisson_sequential` / `_centered` | `rl_poisson(..., RL_POISSON_SEQUENTIAL / RL_POISSON_PTRS / RL_POISSON_AUTO)` |
| `generate_binomial_summation` / `_binv` / `_poisson_approx` | `rl_binomial(..., RL_BINOMIAL_SUMMATION / _INVERSION / _BTRS / _POISSON_APPROX / _AUTO)` |
| `generate_multinomial_alias` / `_sequencing` | `rl_multinomial(..., method, ...)` |
| `create_composition_engine`, `add_composite_component`, `sample_composition`, `free_composition_engine` | `rl_composition_new`, `rl_composition_set`, **`rl_composition_finalize`**, `rl_composition_sample`, `rl_composition_free` |
| `worker_*(void* params)` (params held the RNG) | `rl_worker_*(RL_State*, void* params, int64_t* out)` |
| `sample_rejection_array`, `sample_rejection_func` | `rl_rejection_array(_fill)`, `rl_rejection_func` |
| `sample_rejection_continuous` (returned -1.0 / -2.0) | `rl_rejection_continuous` (status code, optional `attempts_used`, Ziggurat table passed explicitly) |
| `get_proposal_density` | `rl_proposal_density` |
| `generate_exponential`, `generate_cauchy`, `generate_laplace` | `rl_exponential`, `rl_cauchy`, `rl_laplace` |
| `generate_normal_box_muller` | `rl_dl_box_muller` |
| `create_dl_ziggurat`, `sample_dl_ziggurat`, `sample_dl_batch`, `free_dl_ziggurat` | `rl_dl_ziggurat_new`, `_sample`, `_fill`, `_free` |
| `dl_pdf` | `rl_dl_pdf` |
| `create_ziggurat` / `sample_ziggurat` (generic) | removed (never functional) |
| `analyze_and_normalize` (returned a struct) | `rl_analyze_and_normalize` (status code + `RL_Analysis*`) |

## Python
| Old | New |
|---|---|
| `RandomGenerator(seed, lib_path="./random_engine.so")` | `RandomGenerator(seed=None, lib_path=None)`; library found automatically; `seed=None` uses OS entropy |
| `normal_bm` | `dl_box_muller` (`normal_bm` kept as an alias) |
| `dl_ziggurat(size=100, layers=256)` | `dl_ziggurat(size=None, layers=256, mu=0.0, sigma=1.0)`; default is now a scalar; any layer count works |
| `ziggurat(...)` | raises `NotImplementedError` with guidance |
| `poisson(method="centered")` | `method="ptrs"` or `"auto"`; `"centered"` raises with a message |
| `binomial(method="poisson")` | `"poisson_approx"` (still accepted as `"poisson"`); never chosen by `auto` |
| `bernoulli_sampling(list_of_ints, p, method)` | any sequence; `auto` selects `jump` for p < 0.25 |
| `permutation(list_of_ints, r)` | any sequence of any objects; not modified |
| `choice(..., "geometric")` (index 0 unreachable) | 0-based draw; `overflow="clamp"` (default) or `"reject"` |
| `rejection_sampling(f, type, M, p1, p2, max_attempts)` | same positional order, plus `size=`, `check_envelope=True`, `layers=`; raises `EnvelopeError` on an undersized M |
| `sample_custom_array(pmf, M, ...)` | `M=None` means the smallest valid constant |
| `sample_custom_func(f, M, p_geom=0.5)` | same; Python exceptions in `f` propagate |
| `sample_mixture(components_data, mode)` (ctypes workers, engine rebuilt per call) | `MixtureBuilder` / `rng.mixture(specs)`: build once, `mix()` / `mix.sample(n)`; `rng.sample_mixture(specs, size)` for one-shot use |
| `RejectionOptimizer.benchmark(f, domain, max_attempts)` | `plan()` / `best()` / `compare()`; plans carry a verified `M` |
| error values `-1`, `-2`, `-1.0` | `ValueError`, `RejectionTimeoutError`, `EnvelopeError`, `MemoryError` |
