"""Throughput of the batch interfaces (values per second). Numbers depend on your machine."""
import time

import dlrand

N = 2_000_000
rng = dlrand.RandomGenerator(1)


def rate(label, fn):
    t = time.perf_counter()
    fn()
    dt = time.perf_counter() - t
    print(f"{label:<28}{N / dt / 1e6:>8.2f} M values/s")


rate("uniform", lambda: rng.uniform(size=N))
rng.dl_ziggurat(size=10)                                # builds the table outside the timing
rate("DL Ziggurat (256 layers)", lambda: rng.dl_ziggurat(size=N))
rate("DL Box-Muller", lambda: rng.dl_box_muller(size=N))
rate("exponential", lambda: rng.exponential(size=N))
rate("Poisson(1000) PTRS", lambda: rng.poisson(1000, size=N))
rate("Binomial(10^6, 0.3) BTRS", lambda: rng.binomial(10 ** 6, 0.3, size=N))
rate("geometric(0.1)", lambda: rng.geometric(0.1, size=N))
