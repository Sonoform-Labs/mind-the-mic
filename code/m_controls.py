import numpy as np, os, json, warnings, torch, itertools

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTD = os.environ.get("MTM_OUT_DIR", os.path.join(PKG, "outputs"))
os.makedirs(OUTD, exist_ok=True)
warnings.filterwarnings("ignore")
from scipy.signal import stft, correlate
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
NOBJ = 12
NANG = 4
rng = np.random.default_rng(0)

MATERIAL = {}
for o in OBJS:
    n = o.lower()
    MATERIAL[o] = (
        "ceramic"
        if "ceramic" in n or "planter" in n or "koi" in n
        else (
            "wood"
            if "wood" in n
            else (
                "metal"
                if any(k in n for k in ("iron", "metal", "pan", "skillet", "spoon"))
                else (
                    "plastic"
                    if "plastic" in n
                    else "glass" if "glass" in n else "other"
                )
            )
        )
    )


def cut(x, n=24000, norm="peak", pre=200):
    e = np.convolve(x**2, np.ones(64) / 64, mode="same")
    i = int(np.argmax(e))
    s = max(0, i - pre)
    y = x[s : s + n]
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    if norm == "peak":
        return y / (np.abs(y).max() + 1e-9)
    return y / (np.sqrt(np.mean(y**2)) + 1e-9)


def align_to(ref, y, maxlag=512):
    c = correlate(y[:8192], ref[:8192], mode="full")
    lag = int(np.argmax(np.abs(c))) - (len(ref[:8192]) - 1)
    lag = int(np.clip(lag, -maxlag, maxlag))
    if lag > 0:
        y = np.concatenate([y[lag:], np.zeros(lag, np.float32)])
    elif lag < 0:
        y = np.concatenate([np.zeros(-lag, np.float32), y[:lag]])
    return y


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


FEAT = {"LSD": f_logspec, "mel": f_mel, "MFCC": f_mfcc}

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


def sigmas(grid, metric):
    nS = len(grid)
    nV = len(grid[0])
    if metric == "CLAP":
        flat = [grid[s][v] for v in range(nV) for s in range(nS)]
        E = clap(flat).reshape(nV, nS, -1)
        D = np.stack([1.0 - (E[v] @ E[v].T) for v in range(nV)])
    else:
        fn = FEAT[metric]
        F = [[fn(grid[s][v]) for s in range(nS)] for v in range(nV)]
        D = np.zeros((nV, nS, nS))
        for v in range(nV):
            for i in range(nS):
                for j in range(i + 1, nS):
                    d = d_frob(F[v][i], F[v][j])
                    D[v, i, j] = d
                    D[v, j, i] = d
    out = []
    for r in range(nS):
        for i, j in itertools.combinations([x for x in range(nS) if x != r], 2):
            out.append(float((D[:, r, i] - D[:, r, j]).std(ddof=1)))
    return out


RAW = {}
print(f"loading {NOBJ} objects x {NANG} angles x 4 distances ...", flush=True)
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
    cells = {}
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
                    row.append(np.array(A[int(w[0])], dtype=np.float32))
                if not ok:
                    break
                grid.append(row)
            if ok:
                cells[(int(a), int(dd))] = grid
    RAW[o] = cells
    print(f"  {o} ({MATERIAL[o]}): {len(cells)} cells", flush=True)

METRICS = ["LSD", "mel", "MFCC", "CLAP"]
RES = {}


def run(tag, builder, metrics=METRICS, per_object=False):
    acc = {m: [] for m in metrics}
    byobj = {m: {} for m in metrics}
    for o, cells in RAW.items():
        for key, grid in cells.items():
            g = builder(grid)
            for m in metrics:
                s = sigmas(g, m)
                acc[m].extend(s)
                byobj[m].setdefault(o, []).extend(s)
    row = {}
    for m in metrics:
        v = np.array(acc[m])
        med = float(np.median(v))
        row[m] = med
        if per_object:
            RES.setdefault("per_object", {})[m] = {
                o: float(np.median(byobj[m][o])) for o in byobj[m]
            }
    RES[tag] = row
    print(f"{tag:<28}" + "".join(f"{row[m]:>14.5f}" for m in metrics), flush=True)
    return row


def b_base(grid):
    return [[cut(x) for x in row] for row in grid]


def b_harness(grid):
    out = []
    for row in grid:
        x = row[7]
        vs = []
        for k in range(15):
            r = np.random.default_rng(1000 + k)
            pre = 200 + int(r.integers(-100, 101))
            nm = "peak" if k % 2 == 0 else "rms"
            vs.append(cut(x, norm=nm, pre=pre))
        out.append(vs)
    return out


def b_aligned(grid):
    out = []
    for row in grid:
        ws = [cut(x) for x in row]
        out.append([ws[0]] + [align_to(ws[0], y) for y in ws[1:]])
    return out


def b_rms(grid):
    return [[cut(x, norm="rms") for x in row] for row in grid]


def b_long(grid):
    return [[cut(x, n=48000) for x in row] for row in grid]


print("\n" + "=" * 90)
print("CONTROLS: median sigma of the paired difference, each metric's own units")
print("=" * 90)
print(f"{'condition':<28}" + "".join(f"{m:>14}" for m in METRICS))
print("-" * 90)
base = run("A0 baseline (15 mics)", b_base, per_object=True)
harn = run("A1 HARNESS NULL (mic fixed)", b_harness)
alig = run("B1 onset-aligned", b_aligned)
rmsn = run("B2 rms-normalised", b_rms)
lng = run("B3 1.0s window", b_long)
print("-" * 90)

print("\n" + "=" * 90)
print("A1 VERDICT: is the effect my own preprocessing?")
print("=" * 90)
print(f"{'metric':>8}{'receiver sigma':>17}{'harness sigma':>16}{'ratio':>9}   reading")
print("-" * 90)
dead = []
for m in METRICS:
    r = harn[m] / base[m] if base[m] > 0 else float("nan")
    rd = (
        "FATAL, effect is the pipeline"
        if r > 0.5
        else ("caution" if r > 0.25 else "clean, harness is minor")
    )
    if r > 0.5:
        dead.append(m)
    print(f"{m:>8}{base[m]:>17.5f}{harn[m]:>16.5f}{r:>8.2f}x   {rd}")
print("-" * 90)
print("\nPAPER LIVES" if not dead else f"\nPAPER DIES for: {', '.join(dead)}")

print("\n" + "=" * 90)
print("B4 per-object and per-material spread (baseline, LSD and CLAP)")
print("=" * 90)
po = RES.get("per_object", {})
for m in ["LSD", "CLAP"]:
    if m not in po:
        continue
    vals = po[m]
    v = np.array(list(vals.values()))
    print(
        f"\n  {m}: median {np.median(v):.5f}  range [{v.min():.5f}, {v.max():.5f}]  "
        f"IQR [{np.percentile(v,25):.5f}, {np.percentile(v,75):.5f}]"
    )
    bymat = {}
    for o, s in vals.items():
        bymat.setdefault(MATERIAL[o], []).append(s)
    for mat in sorted(bymat):
        a = np.array(bymat[mat])
        print(f"    {mat:<10} n={len(a):<3} median {np.median(a):.5f}")

with open(os.path.join(OUTD, "out_controls.json"), "w") as f:
    json.dump(RES, f, indent=2)
print("\nwrote out_controls.json")
print("\ndone")
