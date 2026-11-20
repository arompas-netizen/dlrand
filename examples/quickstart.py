"""A short tour of dlrand. Run:  python examples/quickstart.py"""
import dlrand

rng = dlrand.RandomGenerator(seed=2024)            # same seed -> same stream, always

print("uniform         ", [round(x, 4) for x in rng.uniform(size=3)])
print("integers (1..6) ", rng.integers(1, 6, size=10))
print("DL (Box-Muller) ", [round(x, 4) for x in rng.dl_box_muller(mu=0.0, sigma=1.0, size=3)])
print("DL (Ziggurat)   ", [round(x, 4) for x in rng.dl_ziggurat(size=3)])
print("Poisson(800)    ", rng.poisson(800, size=5))              # exact for every lambda up to 1e15
print("Binomial(2000,.5)", rng.binomial(2000, 0.5, size=5))
print("Geometric(0.3)  ", rng.geometric(0.3, size=8), "(trials, support 1,2,..)")
print("multinomial     ", rng.multinomial(1000, [0.5, 0.3, 0.2]))

deck = ["A", "K", "Q", "J", "10"]
print("permutation     ", rng.permutation(deck), "| 3 without replacement:", rng.permutation(deck, r=3))
print("bernoulli keep  ", rng.bernoulli_sampling(list(range(20)), 0.25))

# Mixture: 50% uniform on 0..9, 50% Poisson(100) -- the engine is built once and reused
mix = dlrand.MixtureBuilder(rng).add_uniform(0.5, 0, 10).add_poisson(0.5, 100.0).build("alias")
print("mixture         ", mix.sample(8))

# Independent parallel streams (child k is k * 2^128 draws ahead of the parent)
streams = rng.spawn(3)
print("spawned streams ", [round(s.uniform(), 4) for s in streams])

# Save and restore the full state
snapshot = rng.getstate()
a = rng.uniform(size=3)
rng.setstate(snapshot)
assert a == rng.uniform(size=3)
print("state restore    ok")
