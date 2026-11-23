# Algorithms, domains and references

All engines draw from one `RL_State` and use only three primitives: `rl_u01` ([0,1), 53 bits), `rl_uopen` ((0,1), strictly
inside, so `log(U)` and `log(1-U)` are always finite) and `rl_below` (unbiased integer in [0,n)).

## Core
| Item | Method |
|---|---|
| Generator | xoshiro256++ (Blackman & Vigna 2021), period 2^256 - 1 |
| Seeding | SplitMix64 expands a 64-bit seed into 4 state words (Steele, Lea & Flood 2014) |
| Jump-ahead | `rl_jump` = 2^128 steps, `rl_long_jump` = 2^192; verified against T^(2^k) over GF(2) |
| Bounded integers | rejection on the modulo threshold `2^64 mod n` (unbiased; see O'Neill, *Efficiently Generating a Number in a Range*) |
| log-gamma | `rl_loggam`: Stirling series after shifting to x >= 7 (thread-safe, platform-independent, unlike libm `lgamma`) |
| Stirling remainder | `rl_stirling_tail(x) = ln x! - [(x+1/2)ln(x+1) - (x+1) + ln(2 pi)/2]` keeps PTRS/BTRS acceptance tests free of cancellation |

## Discrete engines
| Engine | Algorithm | Cost | Reference |
|---|---|---|---|
| Permutation | partial Fisher-Yates; requires 0 <= r <= n | O(r) | Durstenfeld 1964 |
| Bernoulli selection | standard trials, or *jump*: skip `floor(ln U / ln(1-p))` failures, positions kept as doubles | O(n) / O(pn) | Devroye 1986 |
| Geometric | inversion with `log1p(-p)` | O(1) | Devroye 1986 |
| Poisson | `sequential`: CDF inversion (lambda <= 700). `ptrs`: transformed rejection with squeeze (lambda >= 10), Stirling-remainder acceptance test | O(lambda) / O(1) | Hormann 1993 |
| Binomial | `inversion` (BINV, n*min(p,1-p) <= 500), `btrs` (n*min(p,1-p) >= 10), `summation`; symmetry `Bin(n,p) = n - Bin(n,1-p)`; `poisson_approx` explicit only | O(np) / O(1) / O(n) | Kachitvichyanukul & Schmeiser 1988; Hormann 1993 |
| Multinomial | conditional-binomial *sequencing* using suffix sums, or *alias* draws | O(r) / O(r+n) | Devroye 1986; Vose 1991 |
| Alias table | Vose's method; column and coin use independent draws | build O(n), draw O(1) | Vose 1991 |
| Composition | choose component by manual scan, residual (largest first) scan, or alias; workers share the caller's state | O(1) / O(k) | Devroye 1986 |
| Discrete rejection | uniform proposal on {0..size-1} (`M >= size*max pmf`) or Geometric proposal in log space | | von Neumann 1951 |

## Continuous engines
| Engine | Algorithm |
|---|---|
| Uniform | `a + (b-a)U`, never returns exactly `b` |
| Exponential | `-scale ln(U)`, U in (0,1) |
| Cauchy | `x0 + gamma tan(pi (U - 1/2))`, U in (0,1) |
| Laplace | `mu - b sgn(v) ln(1 - 2|v|)`, `v = U - 1/2` |
| DL (Box-Muller) | `sqrt(-2 ln U1) (cos, sin)(2 pi U2)`; the second variate is cached in the generator state |
| DL (Ziggurat) | Marsaglia-Tsang with the tail start R and strip area V **solved by bisection** for any layer count (equals the published R = 3.442619855899 for 128 and R = 3.6541528853610088 for 256 layers); exact Marsaglia tail |
| Rejection | proposals Uniform, Exponential, Cauchy, Laplace, DL (Box-Muller or Ziggurat); `check_envelope` tests `f <= M g` on every candidate |
| Analysis | doubling window until `f` decays to 1e-9 of its maximum, then trapezoid CDF to trim the central `1 - tail_mass` |

## Rejection constants
`M` must satisfy `f(x) <= M g(x)` wherever `f(x) > 0`. `RejectionOptimizer` minimises `M = sup f/g` for each proposal
(grid + golden-section refinement, 2% safety margin) and reports `efficiency = area / M`. `M` is estimated on the analysed
domain (extended by 50% for infinite-support proposals); targets with relevant mass further out need a wider `domain`.
The Uniform proposal samples the target restricted to the domain. At sampling time `check_envelope=True` raises
`EnvelopeError` if any candidate violates the envelope.

## References
- D. Blackman, S. Vigna, *Scrambled linear pseudorandom number generators*, ACM TOMS 47(4), 2021.
- G. Steele, D. Lea, C. Flood, *Fast splittable pseudorandom number generators*, OOPSLA 2014.
- G. Marsaglia, W. W. Tsang, *The Ziggurat method for generating random variables*, J. Stat. Software 5(8), 2000.
- G. E. P. Box, M. E. Muller, *A note on the generation of random normal deviates*, Ann. Math. Statist. 29, 1958.
- W. Hormann, *The transformed rejection method for generating Poisson random variables*, Insurance: Math. & Econ. 12, 1993.
- W. Hormann, *The generation of binomial random variates*, J. Statist. Comput. Simul. 46, 1993.
- V. Kachitvichyanukul, B. W. Schmeiser, *Binomial random variate generation*, CACM 31(2), 1988.
- M. D. Vose, *A linear algorithm for generating random numbers with a given distribution*, IEEE TSE 17(9), 1991.
- R. Durstenfeld, *Algorithm 235: Random permutation*, CACM 7(7), 1964.
- J. von Neumann, *Various techniques used in connection with random digits*, 1951.
- L. Devroye, *Non-Uniform Random Variate Generation*, Springer, 1986.
- A. de Moivre, *Approximatio ad summam terminorum binomii (a+b)^n in seriem expansi*, 1733; P.-S. Laplace, *Theorie analytique des probabilites*, 1812.
