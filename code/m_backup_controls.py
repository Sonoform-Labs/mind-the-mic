import numpy as np, json, torch, warnings, os

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTD = os.environ.get("MTM_OUT_DIR", os.path.join(PKG, "outputs"))
os.makedirs(OUTD, exist_ok=True)
warnings.filterwarnings("ignore")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SCALES = [64, 128, 256, 512, 1024, 2048, 4096]
EPS = 1e-6
L = 96000
D = os.path.join(
    os.environ.get("MTM_DIFFRIR_DIR", os.path.join(PKG, "data", "diffrir")),
    "classroomBase",
)


def ms(x, y):
    t = 0.0
    for n in SCALES:
        w = torch.hann_window(n, device=DEV)
        a = torch.stft(x, n_fft=n, hop_length=n // 4, window=w, return_complex=True)
        b = torch.stft(y, n_fft=n, hop_length=n // 4, window=w, return_complex=True)
        ea = torch.sqrt(a.real**2 + a.imag**2 + EPS)
        eb = torch.sqrt(b.real**2 + b.imag**2 + EPS)
        t += float(torch.mean(torch.abs(torch.log(ea) - torch.log(eb))))
    return t


R = np.load(f"{D}/RIRs.npy", mmap_mode="r")
XYZ = np.load(f"{D}/xyzs.npy")
n = R.shape[0]
tr = np.arange(12) * 57
s1 = set(list(tr) + list(np.arange(315) * 2))
va = np.array([i for i in range(n) if i not in s1])[::2]
s2 = set(list(tr) + list(va))
te = np.array([i for i in range(n) if i not in s2])


def get(i, norm=None):
    x = np.array(R[i][:L], dtype=np.float32)
    if norm == "rms":
        x = x / (np.sqrt(np.mean(x**2)) + 1e-12)
    return torch.tensor(x, device=DEV)


OUT = {}
print("=" * 96)
print("BACKUP CONTROLS: CIs, seed robustness, level normalisation. classroom, L=96000.")
print("=" * 96, flush=True)

print("\n[1] NN reproduction with bootstrap CI over ALL test positions", flush=True)
nn_errs = []
for t in te:
    j = tr[int(np.argmin(((XYZ[tr] - XYZ[t]) ** 2).sum(1)))]
    nn_errs.append(ms(get(j), get(t)) * 10)
NN = np.array(nn_errs)
rng = np.random.default_rng(0)
boot = np.array([NN[rng.integers(0, len(NN), len(NN))].mean() for _ in range(8000)])
lo, hi = np.percentile(boot, [2.5, 97.5])
print(
    f"    our NN over all {len(NN)} test positions: mean {NN.mean():.3f}  "
    f"95% CI [{lo:.3f}, {hi:.3f}]  sd {NN.std(ddof=1):.3f}"
)
print(f"    their reported NN: 5.99  -> inside our CI: {lo <= 5.99 <= hi}")
OUT["nn_full"] = dict(
    mean=float(NN.mean()),
    ci=[float(lo), float(hi)],
    sd=float(NN.std(ddof=1)),
    n=len(NN),
    theirs=5.99,
)

print("\n[2] nuisance median with CI, three seeds", flush=True)
med_by_seed = {}
for seed in range(3):
    r2 = np.random.default_rng(seed)
    vals = []
    for _ in range(300):
        i, j = r2.integers(0, n, 2)
        if i == j:
            continue
        vals.append(ms(get(i), get(j)) * 10)
    V = np.array(vals)
    b2 = np.array([np.median(V[r2.integers(0, len(V), len(V))]) for _ in range(4000)])
    l2, h2 = np.percentile(b2, [2.5, 97.5])
    med_by_seed[seed] = dict(
        median=float(np.median(V)), ci=[float(l2), float(h2)], n=len(V)
    )
    print(
        f"    seed {seed}: median {np.median(V):.3f}  95% CI [{l2:.3f}, {h2:.3f}]  (n={len(V)})"
    )
OUT["nuisance_seeds"] = med_by_seed
meds = [v["median"] for v in med_by_seed.values()]
print(f"    across seeds: {min(meds):.3f} to {max(meds):.3f}")

print("\n[3] level control: per-RIR RMS normalisation before the metric", flush=True)
r3 = np.random.default_rng(0)
raw, nrm = [], []
for _ in range(250):
    i, j = r3.integers(0, n, 2)
    if i == j:
        continue
    raw.append(ms(get(i), get(j)) * 10)
    nrm.append(ms(get(i, "rms"), get(j, "rms")) * 10)
raw, nrm = np.array(raw), np.array(nrm)
print(f"    unnormalised nuisance median : {np.median(raw):.3f}")
print(
    f"    RMS-normalised nuisance      : {np.median(nrm):.3f}  "
    f"({np.median(nrm)/np.median(raw)*100:.0f}% of unnormalised)"
)
print(f"    vs their reported gain 0.77  : still {np.median(nrm)/0.77:.1f}x the gain")
OUT["level_control"] = dict(
    raw_median=float(np.median(raw)),
    rms_median=float(np.median(nrm)),
    ratio_vs_gain=float(np.median(nrm) / 0.77),
)

print(
    "\n[4] displacement-matched consistency: NN error IS the sub-1m nuisance",
    flush=True,
)
d_nn = np.array(
    [
        np.linalg.norm(
            XYZ[t] - XYZ[tr[int(np.argmin(((XYZ[tr] - XYZ[t]) ** 2).sum(1)))]]
        )
        for t in te
    ]
)
print(f"    median test-to-train displacement: {np.median(d_nn):.3f} m")
print(f"    their NN baseline (= GT at that displacement): 5.99")
print(
    f"    our independent random-pair median under 1 m must roughly agree; see fig2 (5.86)"
)
OUT["nn_displacement"] = dict(median_m=float(np.median(d_nn)))

json.dump(OUT, open(os.path.join(OUTD, "out_backup_controls.json"), "w"), indent=2)
print("\nwrote out_backup_controls.json\ndone")
