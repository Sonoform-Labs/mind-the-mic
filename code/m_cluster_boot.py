import os, json, itertools, warnings

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTD = os.environ.get("MTM_OUT_DIR", os.path.join(PKG, "outputs"))
os.makedirs(OUTD, exist_ok=True)
warnings.filterwarnings("ignore")

import numpy as np
from scipy.signal import stft

ROOT = os.environ.get("MTM_REALIMPACT_DIR", os.path.join(PKG, "data", "realimpact"))
SR = 48000
ANG_KEEP = 5
rng = np.random.default_rng(0)

OBJS = [
    d
    for d in sorted(os.listdir(ROOT))
    if os.path.isdir(f"{ROOT}/{d}")
    and os.path.exists(f"{ROOT}/{d}/preprocessed/deconvolved_0db.npy")
]


def cut(x, n=24000):
    e = np.convolve(x**2, np.ones(64) / 64, mode="same")
    i = int(np.argmax(e))
    s = max(0, i - 200)
    y = x[s : s + n]
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    return y / (np.abs(y).max() + 1e-9)


def f_logspec(x):
    _, _, Z = stft(x, fs=SR, nperseg=512, noverlap=384)
    return np.log(np.abs(Z) + 1e-6)


def d_frob(A, B):
    n = min(A.shape[-1], B.shape[-1])
    return float(np.sqrt(np.mean((A[..., :n] - B[..., :n]) ** 2)))


vals, objid, ncell = [], [], 0
for oi, o in enumerate(OBJS):
    B = f"{ROOT}/{o}/preprocessed/"
    A = np.load(B + "deconvolved_0db.npy", mmap_mode="r")
    V = np.load(B + "vertexXYZ.npy")
    ANG = np.load(B + "angle.npy")
    DIST = np.load(B + "distance.npy")
    MIC = np.load(B + "micID.npy")
    uv = np.unique(V, axis=0)
    vid = np.array([int(np.argmin(((uv - v) ** 2).sum(1))) for v in V])
    angs = sorted(np.unique(ANG))[:: max(1, len(np.unique(ANG)) // ANG_KEEP)][:ANG_KEEP]
    dsts = sorted(np.unique(DIST))
    mics = sorted(np.unique(MIC))

    cells = {}
    for a in angs:
        for dd in dsts:
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
            if ok and len(grid) >= 3:
                cells[(int(a), int(dd))] = grid
    if not cells:
        continue

    n0 = len(vals)
    for key, grid in cells.items():
        nS, nM = len(grid), len(grid[0])
        F = [[f_logspec(grid[s][m]) for s in range(nS)] for m in range(nM)]
        D = np.zeros((nM, nS, nS))
        for m in range(nM):
            for i in range(nS):
                for j in range(i + 1, nS):
                    d = d_frob(F[m][i], F[m][j])
                    D[m, i, j] = d
                    D[m, j, i] = d
        for r in range(nS):
            for i, j in itertools.combinations([x for x in range(nS) if x != r], 2):
                delta = D[:, r, i] - D[:, r, j]
                vals.append(float(delta.std(ddof=1)))
                objid.append(oi)
    ncell += len(cells)
    print(
        "  [%2d/%d] %-24s %2d cells, %4d values"
        % (oi + 1, len(OBJS), o, len(cells), len(vals) - n0),
        flush=True,
    )

vals = np.asarray(vals)
objid = np.asarray(objid)
print(
    "\ntotal: %d values, %d cells, %d objects"
    % (len(vals), ncell, len(np.unique(objid)))
)


def boot_iid(v, R=4000):
    n = len(v)
    b = [np.median(v[rng.integers(0, n, n)]) for _ in range(R)]
    return np.percentile(b, [2.5, 97.5])


def boot_cluster(v, g, R=4000):
    groups = np.unique(g)
    idx = {k: np.where(g == k)[0] for k in groups}
    K = len(groups)
    out = []
    for _ in range(R):
        pick = rng.integers(0, K, K)
        sel = np.concatenate([idx[groups[p]] for p in pick])
        out.append(np.median(v[sel]))
    return np.percentile(out, [2.5, 97.5])


point = float(np.median(vals))
ci_i = boot_iid(vals)
ci_c = boot_cluster(vals, objid)
per_obj = [float(np.median(vals[objid == k])) for k in np.unique(objid)]

print()
print("=" * 68)
print("LSD within-column sigma, 95 percent intervals")
print("=" * 68)
print("  point estimate            %.4f" % point)
print(
    "  i.i.d. bootstrap          [%.4f, %.4f]   half-width %.4f"
    % (ci_i[0], ci_i[1], (ci_i[1] - ci_i[0]) / 2)
)
print(
    "  cluster bootstrap (obj)   [%.4f, %.4f]   half-width %.4f"
    % (ci_c[0], ci_c[1], (ci_c[1] - ci_c[0]) / 2)
)
print(
    "  widening factor           %.1fx"
    % (
        ((ci_c[1] - ci_c[0]) / (ci_i[1] - ci_i[0]))
        if ci_i[1] > ci_i[0]
        else float("nan")
    )
)
print()
print("  per-object medians: %.3f to %.3f" % (min(per_obj), max(per_obj)))

json.dump(
    dict(
        point=point,
        ci_iid=list(map(float, ci_i)),
        ci_cluster=list(map(float, ci_c)),
        n=len(vals),
        ncell=ncell,
        nobj=int(len(np.unique(objid))),
        per_object_medians=per_obj,
    ),
    open(os.path.join(OUTD, "out_cluster_boot.json"), "w"),
    indent=1,
)
print("\nwrote out_cluster_boot.json")
