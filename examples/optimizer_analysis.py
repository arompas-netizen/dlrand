"""Support / normalisation analysis of a target density, with an optional plot.

    python examples/optimizer_analysis.py            # opens plot windows (needs matplotlib)
    python examples/optimizer_analysis.py --save     # writes analysis_*.png instead
    python examples/optimizer_analysis.py --no-plot  # numbers only
"""
import math
import sys

import dlrand


def show(title, res):
    print(f"\n{title}")
    print(f"{'PROPERTY':<25} | {'VALUE':>20}")
    print("-" * 48)
    for label, value in (("Domain Start (a)", res.a), ("Domain End (b)", res.b), ("Global Max (f_max)", res.f_max),
                         ("Global Min (f_min)", res.f_min), ("Norm Constant (Area)", res.area),
                         ("Total area scanned", res.total_area)):
        print(f"{label:<25} | {value:>20.6f}")


def plot_normalized(f, res, name, save):
    import matplotlib
    if save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    margin = (res.b - res.a) * 0.2
    n = 1000
    xs = [res.a - margin + (res.b - res.a + 2 * margin) * i / (n - 1) for i in range(n)]
    ys = [f(x) / res.area for x in xs]
    inside = [res.a <= x <= res.b for x in xs]
    plt.figure(figsize=(10, 6))
    plt.fill_between(xs, ys, where=inside, color="royalblue", alpha=0.6, label="central mass domain")
    plt.fill_between(xs, ys, where=[not v for v in inside], color="crimson", alpha=0.3, label="tails")
    plt.plot(xs, ys, color="black", lw=1.5)
    plt.title("Normalized target function support analysis")
    plt.xlabel("x")
    plt.ylabel("normalized density f(x)/k")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    if save:
        plt.savefig(f"analysis_{name}.png", dpi=120)
        print(f"saved analysis_{name}.png")
    else:
        plt.show()
    plt.close()


def main():
    save, plot = "--save" in sys.argv, "--no-plot" not in sys.argv
    opt = dlrand.RejectionOptimizer(dlrand.RandomGenerator(0))

    def dl_pdf(x):                                   # the DL (De Moivre-Laplace) density, unnormalised
        return math.exp(-0.5 * x * x)

    # domain left open: the support is detected automatically and trimmed to the central 99% of the mass
    res = opt.analyze(dl_pdf, tail_mass=0.01)
    show("DL density, automatic domain (central 99% of the mass)", res)
    if plot:
        plot_normalized(dl_pdf, res, "dl", save)

    # explicit domain: the analysis reports f_max, f_min and the normalising area on [0, 100000]
    res = opt.analyze(lambda x: x * x, 0.0, 100000.0)
    show("f(x) = x^2 on [0, 100000]", res)
    if plot:
        plot_normalized(lambda x: x * x, res, "x_squared", save)

    # without a domain this target has no decaying support, which is reported as an error
    try:
        opt.analyze(lambda x: x * x)
    except ValueError as exc:
        print("\nf(x) = x^2 without a domain ->", exc)


if __name__ == "__main__":
    main()
