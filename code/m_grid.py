import numpy as np, os, json, warnings, itertools

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTD = os.environ.get("MTM_OUT_DIR", os.path.join(PKG, "outputs"))
os.makedirs(OUTD, exist_ok=True)
warnings.filterwarnings("ignore")
from scipy.signal import stft
import torchaudio.functional as AF
import torch

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
rng = np.random.default_rng(0)


def cut(x, n=24000):
    e = np.convolve(x**2, np.ones(64) / 64, mode="same")
    i = int(np.argmax(e))
    s = max(0, i - 200)
    y = x[s : s + n]
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    return y / (np.abs(y).max() + 1e-9)


def rs(x, src, dst):
    if src == dst:
        return np.asarray(x, np.float32)
    return (
        AF.resample(torch.from_numpy(np.ascontiguousarray(x)).float(), src, dst)
        .numpy()
        .astype(np.float32)
    )


def spec(x, sr, nper=512):
    _, _, Z = stft(x, fs=sr, nperseg=nper, noverlap=nper * 3 // 4)
    return np.abs(Z)


def d_rms_nat(a, b, sr):
    A, B = np.log(spec(a, sr) + 1e-6), np.log(spec(b, sr) + 1e-6)
    n = min(A.shape[1], B.shape[1])
    return float(np.sqrt(np.mean((A[:, :n] - B[:, :n]) ** 2)))


def d_l1_nat(a, b, sr):
    A, B = np.log(spec(a, sr) + 1e-6), np.log(spec(b, sr) + 1e-6)
    n = min(A.shape[1], B.shape[1])
    return float(np.mean(np.abs(A[:, :n] - B[:, :n])))


def d_l1_eps3(a, b, sr):
    A, B = np.log(spec(a, sr) + 1e-3), np.log(spec(b, sr) + 1e-3)
    n = min(A.shape[1], B.shape[1])
    return float(np.mean(np.abs(A[:, :n] - B[:, :n])))


def d_multi(a, b, sr):
    tot = []
    for nper in (64, 256, 1024, 4096):
        A, B = np.log(spec(a, sr, nper) + 1e-6), np.log(spec(b, sr, nper) + 1e-6)
        n = min(A.shape[1], B.shape[1])
        tot.append(float(np.mean(np.abs(A[:, :n] - B[:, :n]))))
    return float(np.mean(tot))


VARIANTS = [
    ("RMS natural-log (ours)", d_rms_nat),
    ("L1-mean natural-log", d_l1_nat),
    ("L1-mean, +1e-3 floor (NAF/INRAS)", d_l1_eps3),
    ("multiscale L1, eps 1e-6 (DiffRIR)", d_multi),
]
RATES = [48000, 22050]

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
    print(f"  {o}: {len(CELLS)} cells so far", flush=True)

print(f"\n{len(CELLS)} cells total\n", flush=True)

OUT = {}
print("=" * 98)
print("A4 SIGMA UNDER EVERY CONVENTION THE TARGET COMMUNITY USES")
print(
    "all values are the receiver-position sigma of the paired difference, in that convention's units"
)
print("=" * 98)
print(
    f"{'convention':<38}{'48 kHz sigma':>15}{'1.96 sigma':>13}{'22.05k sigma':>15}{'1.96 sigma':>13}"
)
print("-" * 98)
for name, fn in VARIANTS:
    row = {}
    for sr in RATES:
        vals = []
        for grid in CELLS:
            nS, nM = len(grid), len(grid[0])
            g = [[rs(grid[s][m], SR, sr) for m in range(nM)] for s in range(nS)]
            D = np.zeros((nM, nS, nS))
            for m in range(nM):
                for i in range(nS):
                    for j in range(i + 1, nS):
                        d = fn(g[i][m], g[j][m], sr)
                        D[m, i, j] = d
                        D[m, j, i] = d
            for r in range(nS):
                for i, j in itertools.combinations([x for x in range(nS) if x != r], 2):
                    vals.append(float((D[:, r, i] - D[:, r, j]).std(ddof=1)))
        row[sr] = float(np.median(vals))
    OUT[name] = row
    print(
        f"{name:<38}{row[48000]:>15.5f}{1.96*row[48000]:>13.5f}{row[22050]:>15.5f}{1.96*row[22050]:>13.5f}"
    )
print("-" * 98)

with open(os.path.join(OUTD, "out_grid.json"), "w") as f:
    json.dump(OUT, f, indent=2)

print(
    "\nRAF / NAF / INRAS / NeRAF deltas to compare against (STFT error, held-out receivers):"
)
print("  0.00, 0.00, 0.02, 0.03, 0.03, 0.03, 0.03, 0.10, 0.19, 0.28, 0.88")
for name, _ in VARIANTS:
    t = 1.96 * OUT[name][22050]
    deltas = [0.00, 0.00, 0.02, 0.03, 0.03, 0.03, 0.03, 0.10, 0.19, 0.28, 0.88]
    below = sum(1 for d in deltas if d < t)
    print(f"  {name:<38} 22.05k threshold {t:.4f} -> {below}/11 published deltas below")
print("\nwrote out_grid.json")
print("\ndone")
