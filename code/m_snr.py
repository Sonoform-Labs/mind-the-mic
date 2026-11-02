import numpy as np, os, json, warnings, itertools

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTD = os.environ.get("MTM_OUT_DIR", os.path.join(PKG, "outputs"))
os.makedirs(OUTD, exist_ok=True)
warnings.filterwarnings("ignore")
from scipy.signal import stft

ROOT = os.environ.get("MTM_REALIMPACT_DIR", os.path.join(PKG, "data", "realimpact"))
SR = 48000
OBJS = [
    d
    for d in sorted(os.listdir(ROOT))
    if os.path.isdir(f"{ROOT}/{d}")
    and os.path.exists(f"{ROOT}/{d}/preprocessed/deconvolved_0db.npy")
]
NOBJ = 16
NANG = 4


def cut(x, n=24000):
    e = np.convolve(x**2, np.ones(64) / 64, mode="same")
    i = int(np.argmax(e))
    s = max(0, i - 200)
    y = x[s : s + n]
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    return y / (np.abs(y).max() + 1e-9)


def mag(x, nper=512):
    _, _, Z = stft(x, fs=SR, nperseg=nper, noverlap=nper * 3 // 4)
    return np.abs(Z)


def dist(a, b, eps):
    A, B = np.log(mag(a) + eps), np.log(mag(b) + eps)
    n = min(A.shape[1], B.shape[1])
    return float(np.mean(np.abs(A[:, :n] - B[:, :n])))


print(f"loading {NOBJ} objects ...", flush=True)
CELLS = []
for o in OBJS[:NOBJ]:
    B = f"{ROOT}/{o}/preprocessed/"
    A = np.load(B + "deconvolved_0db.npy", mmap_mode="r")
    V = np.load(B + "vertexXYZ.npy")
    ANG = np.load(B + "angle.npy")
    DIST = np.load(B + "distance.npy")
    MIC = np.load(B + "micID.npy")
    uv = np.unique(V, axis=0)
    vid = np.array([int(np.argmin(((uv - v) ** 2).sum(1))) for v in V])
    angs = sorted(np.unique(ANG))[:: max(1, len(np.unique(ANG)) // NANG)][:NANG]
    mics = sorted(np.unique(MIC))
    for a in angs:
        for dd in sorted(np.unique(DIST)):
            grid, ok = [], True
            for s in range(len(uv)):
                row = []
                for m in mics:
                    w = np.where((vid == s) & (ANG == a) & (DIST == dd) & (MIC == m))[0]
                    if not len(w):
                        ok = False
                        break
                    row.append(cut(np.array(A[int(w[0])])))
                if not ok:
                    break
                grid.append(row)
            if ok:
                CELLS.append(grid)
print(f"{len(CELLS)} cells\n", flush=True)

EPS = [("unfloored, eps 1e-6", 1e-6), ("floored, eps 1e-3 (the field)", 1e-3)]
OUT = {}

print("=" * 104)
print(
    "THE MAKE-OR-BREAK TEST: does the 1e-3 floor suppress SIGNAL as much as NUISANCE?"
)
print(
    "signal   = spread of distance across different strike positions (what a model must capture)"
)
print(
    "nuisance = spread of the same distance across the 15 microphones (what we measure)"
)
print(
    "if signal/nuisance is unchanged by the floor, the floor is cosmetic and the finding dies."
)
print("=" * 104)
print(
    f"{'convention':<34}{'signal':>12}{'nuisance':>12}{'signal/nuisance':>18}{'sigma_paired':>14}"
)
print("-" * 104)
for name, eps in EPS:
    sig, nui, paired = [], [], []
    for grid in CELLS:
        nS, nM = len(grid), len(grid[0])
        D = np.zeros((nM, nS, nS))
        for m in range(nM):
            for i in range(nS):
                for j in range(i + 1, nS):
                    d = dist(grid[i][m], grid[j][m], eps)
                    D[m, i, j] = d
                    D[m, j, i] = d
        for r in range(nS):
            others = [x for x in range(nS) if x != r]
            permic = np.array([[D[m, r, i] for i in others] for m in range(nM)])
            sig.append(float(permic.mean(0).std(ddof=1)))
            nui.append(
                float(np.mean([permic[:, k].std(ddof=1) for k in range(len(others))]))
            )
            for i, j in itertools.combinations(others, 2):
                paired.append(float((D[:, r, i] - D[:, r, j]).std(ddof=1)))
    s, n_, p = float(np.median(sig)), float(np.median(nui)), float(np.median(paired))
    OUT[name] = dict(signal=s, nuisance=n_, snr=s / n_, sigma_paired=p)
    print(f"{name:<34}{s:>12.5f}{n_:>12.5f}{s/n_:>18.3f}{p:>14.5f}")
print("-" * 104)

a, b = OUT[EPS[0][0]], OUT[EPS[1][0]]
rs = a["signal"] / b["signal"]
rn = a["nuisance"] / b["nuisance"]
rr = a["snr"] / b["snr"]
print(f"\n  floor shrinks SIGNAL   by {rs:.2f}x")
print(f"  floor shrinks NUISANCE by {rn:.2f}x")
print(f"  ratio of ratios (signal/nuisance change): {rr:.3f}")
print()
if rr > 1.3:
    print("  RESULT: the floor suppresses NUISANCE MORE than signal.")
    print(
        "  signal-to-nuisance IMPROVES under the floor, so the floor is arguably GOOD"
    )
    print("  and the '15x collapse' framing is misleading. reframe or drop it.")
elif rr < 0.77:
    print("  RESULT: the floor suppresses SIGNAL MORE than nuisance.")
    print(
        "  the field's convention actively degrades discriminability. STRONG finding, keep it."
    )
else:
    print("  RESULT: the floor scales signal and nuisance BY THE SAME FACTOR.")
    print("  signal-to-nuisance is invariant, so the floor is COSMETIC.")
    print(
        "  strike the 'factor of 15' framing. fall back to the weaker claim that four"
    )
    print("  incompatible conventions circulate and no paper documents which it uses.")

with open(os.path.join(OUTD, "out_snr.json"), "w") as f:
    json.dump(OUT, f, indent=2)
print("\nwrote out_snr.json")
print("\ndone")
