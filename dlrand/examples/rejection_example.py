"""Rejection sampling with a *valid* envelope constant, found automatically.

target f(x) = x (2-x)^2 on [0, 2]  (unnormalised; true mean 0.8)
"""
import dlrand


def f(x):
    return x * (2.0 - x) ** 2 if 0.0 <= x <= 2.0 else 0.0


rng = dlrand.RandomGenerator(7)
opt = dlrand.RejectionOptimizer(rng)

print(opt.compare(f, domain=(0.0, 2.0), n=3000))
best = opt.best(f, domain=(0.0, 2.0))
samples = best.sample(rng, f, size=20000)
print(f"\nbest proposal: {best.name} (M = {best.M:.3f}, expected acceptance {best.efficiency:.1%})")
print(f"sample mean = {sum(samples) / len(samples):.4f}   (exact 0.8)")

# A hand-picked M that is too small is now detected instead of silently producing biased samples:
try:
    rng.rejection_sampling(f, "uniform", 1.2, 0.0, 2.0, size=5000)      # needs M >= 2.37
except dlrand.EnvelopeError as exc:
    print("\nEnvelopeError:", exc)
