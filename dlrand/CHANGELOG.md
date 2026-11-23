# Changelog

## 0.1.0 - first public release

Complete rewrite of the earlier `random_engine` v1-v9 prototypes. See `docs/MIGRATION.md` for the mapping.

### Fixed (each has a regression test)
- Binomial: BINV underflowed (`(1-p)^n = 0`) and returned `n` for n >= ~1100; also broke `multinomial` for large n.
  Now exact inversion / BTRS with the p > 1/2 symmetry; p = 1 and invalid p handled.
- Poisson: `exp(-lambda)` underflow made `sequential` return 1,000,001 and `centered` loop forever for
  lambda > ~745. Now exact PTRS for every lambda up to 1e15 (the "centered" method was removed).
- Memory safety: `permutation` with r > n (stack overflow), `bernoulli_jump` with p = 0 (segfault),
  `multinomial` with 0 categories (write to `counts[-1]`), unchecked `malloc`, a Ziggurat skeleton that crashed.
- Box-Muller kept its spare variate in a `static`, so generators interfered and seeds were not reproducible.
  The cache now lives in the generator state.
- DL Ziggurat only worked for 128 layers (256 worked through NaN rejection; 64 gave a wrong distribution) and read an
  uninitialised table entry. Tables are now solved for any layer count and verified against the published constants.
- Laplace returned `-inf` whenever the generator produced exactly 0.0, and Exponential/Geometric truncated their
  tails with a 1e-10 clamp. All inversions now use a uniform on the open interval (0, 1).
- `choice(..., "geometric")` could never select element 0.
- Rejection sampling silently produced biased samples when M was too small (all four shipped examples did).
  `check_envelope` now raises `EnvelopeError`.
- `RejectionOptimizer` picked M = 0 for Uniform/Exponential proposals and mis-parametrised the others.
- Composition: out-of-order `add` crashed (alias table never built), `RESIDUAL` mode was identical to `MANUAL`,
  rounding biased component 0, and `sample_mixture` rebuilt the engine for every sample.
- Binomial `auto` silently used a Poisson approximation; it is now an explicit, never-automatic method.
- Sentinel return values (`-1`, `-2`, `-1.0`) replaced by status codes and Python exceptions; input validation everywhere.

### Changed
- New generator: xoshiro256++ (2^256 - 1 period, 53-bit doubles, SplitMix64 seeding, 2^128/2^192 jump-ahead) instead of a
  31-bit LCG. **Streams for a given seed differ from earlier versions.**
- Batch (`*_fill`) entry points; optional NumPy output; `spawn()` for independent streams; `getstate()/setstate()`.
- Build: CMake and `pip install .` (setuptools) for Linux, macOS and Windows; sanitizer-clean unit tests; CI.

### Removed
- The generic `ziggurat()` (never functional). Use `dl_ziggurat()` or `rejection_sampling()`.
- Poisson method `centered`.
