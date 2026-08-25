import numpy as np

rng = np.random.default_rng(0)
nM, nS, NCELL = 15, 5, 400


def run(sigma_S, sigma_N, ncell=NCELL, seed=0):
    r = np.random.default_rng(seed)
    sig, nui = [], []
    for _ in range(ncell):
        a = r.normal(0, sigma_S, nS)
        b = r.normal(0, sigma_N, (nM, nS))
        D = np.zeros((nM, nS, nS))
        for m in range(nM):
            for i in range(nS):
                D[m, :, i] = a[i] + b[m, i]
        for ref in range(nS):
            others = [x for x in range(nS) if x != ref]
            permic = np.array([[D[m, ref, i] for i in others] for m in range(nM)])
            sig.append(float(permic.mean(0).std(ddof=1)))
            nui.append(float(np.mean([permic[:, k].std(ddof=1) for k in range(len(others))])))
    return float(np.median(sig)), float(np.median(nui))


print("TEST: feed the estimator TRULY EQUAL signal and nuisance (both sigma=1).")
print("An unbiased estimator must report ratio 1.000\n")
s, n = run(1.0, 1.0)
print("  reported signal   %.5f" % s)
print("  reported nuisance %.5f" % n)
print("  reported ratio    %.4f   <-- estimator's answer when truth is 1.0" % (s / n))
print()

print("Bias factors from chi-square theory:")
from scipy.stats import chi2
for df, lbl in [(3, "signal  (std over 4 strikes)"), (14, "nuisance(std over 15 mics)")]:
    print("  df=%2d %s  median-of-s factor %.4f" % (df, lbl, np.sqrt(chi2.median(df) / df)))
print()

print("Now invert: what TRUE ratio would produce the paper's reported 0.969?")
lo, hi = 0.5, 2.0
for _ in range(40):
    mid = (lo + hi) / 2
    s, n = run(mid, 1.0, ncell=400, seed=1)
    if s / n < 0.969:
        lo = mid
    else:
        hi = mid
true_ratio = (lo + hi) / 2
print("  true signal/nuisance = %.4f  reproduces a reported 0.969" % true_ratio)
print()

print("Applying that correction to the paper's floored numbers:")
print("  paper reports signal 0.01660  nuisance 0.01713  ratio 0.969")
print("  implied TRUE ratio  ~%.3f  -> signal %s nuisance" % (true_ratio, "EXCEEDS" if true_ratio > 1 else "below"))
print()
print("Same check for the unfloored convention (paper reports 1.155):")
lo, hi = 0.5, 3.0
for _ in range(40):
    mid = (lo + hi) / 2
    s, n = run(mid, 1.0, ncell=400, seed=2)
    if s / n < 1.155:
        lo = mid
    else:
        hi = mid
print("  true signal/nuisance = %.4f reproduces a reported 1.155" % ((lo + hi) / 2))
