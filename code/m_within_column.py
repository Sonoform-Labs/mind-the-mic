import numpy as np, os, sys, json, warnings, torch, itertools

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTD = os.environ.get("MTM_OUT_DIR", os.path.join(PKG, "outputs"))
os.makedirs(OUTD, exist_ok=True)
warnings.filterwarnings("ignore")
from scipy.signal import stft
import librosa

ROOT = os.environ.get("MTM_REALIMPACT_DIR", os.path.join(PKG, "data", "realimpact"))
SR = 48000
OBJS = [
    d
    for d in sorted(os.listdir(ROOT))
    if os.path.isdir(f"{ROOT}/{d}")
    and os.path.exists(f"{ROOT}/{d}/preprocessed/deconvolved_0db.npy")
]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
ANG_KEEP = 5
rng = np.random.default_rng(0)


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


def f_mel(x):
    return np.log(
        librosa.feature.melspectrogram(y=x, sr=SR, n_mels=64, n_fft=512, hop_length=128)
        + 1e-8
    )


def f_mfcc(x):
    return librosa.feature.mfcc(y=x, sr=SR, n_mfcc=20, n_fft=512, hop_length=128)


def d_frob(A, B):
    n = min(A.shape[-1], B.shape[-1])
    return float(np.sqrt(np.mean((A[..., :n] - B[..., :n]) ** 2)))


_c = {}


def clap(ws):
    if "m" not in _c:
        from transformers import ClapModel, ClapProcessor

        _c["m"] = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(DEV).eval()
        _c["p"] = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
    cm, cp = _c["m"], _c["p"]
    out = []
    for i in range(0, len(ws), 32):
        f = cp(
            audio=[np.asarray(x, np.float32) for x in ws[i : i + 32]],
            sampling_rate=48000,
            return_tensors="pt",
            padding=True,
        )
        with torch.no_grad():
            e = cm.get_audio_features(**{k: v.to(DEV) for k, v in f.items()})
        if not torch.is_tensor(e):
            e = getattr(e, "pooler_output", e[0])
        if e.ndim > 2:
            e = e.mean(dim=tuple(range(1, e.ndim - 1)))
        out.append(e.float().cpu())
    E = torch.cat(out).numpy()
    return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)


METRICS = ["LSD", "mel", "MFCC", "CLAP"]
FEAT = {"LSD": f_logspec, "mel": f_mel, "MFCC": f_mfcc}


def dists_for_cell(ws, metric):
    nS = len(ws)
    nM = len(ws[0])
    D = np.zeros((nM, nS, nS))
    if metric == "CLAP":
        flat = [ws[s][m] for m in range(nM) for s in range(nS)]
        E = clap(flat).reshape(nM, nS, -1)
        for m in range(nM):
            D[m] = 1.0 - (E[m] @ E[m].T)
    else:
        fn = FEAT[metric]
        F = [[fn(ws[s][m]) for s in range(nS)] for m in range(nM)]
        for m in range(nM):
            for i in range(nS):
                for j in range(i + 1, nS):
                    d = d_frob(F[m][i], F[m][j])
                    D[m, i, j] = d
                    D[m, j, i] = d
    return D


print(f"objects: {len(OBJS)}  device: {DEV}", flush=True)
print(
    "design: WITHIN-COLUMN. one gantry position = one set of strikes, heard by 15 simultaneous mics."
)
print(
    "only the receiver height varies. strike, object, and instant are all held fixed.\n",
    flush=True,
)

WITHIN = {m: [] for m in METRICS}
ACROSS = {m: [] for m in METRICS}
ncell = 0

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
            grid = []
            ok = True
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
    for metric in METRICS:
        percell = {}
        for key, grid in cells.items():
            D = dists_for_cell(grid, metric)
            percell[key] = D
            nS = D.shape[1]
            for r in range(nS):
                for i, j in itertools.combinations([x for x in range(nS) if x != r], 2):
                    delta = D[:, r, i] - D[:, r, j]
                    WITHIN[metric].append(float(delta.std(ddof=1)))
        keys = list(percell.keys())
        if len(keys) >= 2:
            nS = percell[keys[0]].shape[1]
            for r in range(nS):
                for i, j in itertools.combinations([x for x in range(nS) if x != r], 2):
                    vals = []
                    for key in keys:
                        Dk = percell[key]
                        vals.extend((Dk[:, r, i] - Dk[:, r, j]).tolist())
                    ACROSS[metric].append(float(np.std(vals, ddof=1)))
    ncell += len(cells)
    print(f"  [{oi+1}/{len(OBJS)}] {o}: {len(cells)} gantry cells", flush=True)


def boot_med(v, R=4000):
    v = np.asarray(v)
    n = len(v)
    b = [np.median(v[rng.integers(0, n, n)]) for _ in range(R)]
    return np.percentile(b, [2.5, 97.5])


print("\n" + "=" * 104)
print("RECEIVER-POSITION-INDUCED SIGMA, in each metric's own units")
print(
    "within-column: same strike, same instant, only the microphone moves (15 receivers)"
)
print(
    "across-gantry: the old, confounded estimate (object re-struck between positions)"
)
print("=" * 104)
print(
    f"{'metric':>8}{'n cells':>9}{'sigma_within':>15}{'95% CI':>24}{'1.96*sigma':>13}{'sigma_across':>15}{'inflation':>11}"
)
print("-" * 104)
OUT = {}
for m in METRICS:
    w = np.array(WITHIN[m])
    a = np.array(ACROSS[m]) if ACROSS[m] else np.array([np.nan])
    mw = float(np.median(w))
    ma = float(np.median(a))
    ci = boot_med(w)
    infl = ma / mw if mw > 0 else float("nan")
    OUT[m] = dict(
        sigma_within=mw,
        ci=[float(ci[0]), float(ci[1])],
        thr=1.96 * mw,
        sigma_across=ma,
        inflation=float(infl),
        n=len(w),
    )
    print(
        f"{m:>8}{len(w):>9}{mw:>15.5f}{f'[{ci[0]:.5f}, {ci[1]:.5f}]':>24}{1.96*mw:>13.5f}{ma:>15.5f}{infl:>10.2f}x"
    )
print("-" * 104)
print(f"\ntotal gantry cells across all objects: {ncell}")
print(
    "\nsigma_within is the honest quantity: it holds the strike fixed and moves only the receiver."
)
print(
    "sigma_across mixes receiver movement with strike-to-strike variation, because RealImpact"
)
print(
    "re-strikes the object when the gantry moves. the inflation column is how much of the"
)
print("earlier estimate was NOT the microphone.")
print(
    "\n1.96*sigma is the detectability threshold: a per-item difference between two systems"
)
print("smaller than this is not distinguishable from moving the listening position.")

with open(os.path.join(OUTD, "out_within_column.json"), "w") as f:
    json.dump(OUT, f, indent=2)
print("\nwrote out_within_column.json")
print("\ndone")
