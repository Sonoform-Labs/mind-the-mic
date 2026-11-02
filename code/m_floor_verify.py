import numpy as np, torch, json, warnings, os

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTD = os.environ.get("MTM_OUT_DIR", os.path.join(PKG, "outputs"))
os.makedirs(OUTD, exist_ok=True)
warnings.filterwarnings("ignore")

D = os.path.join(
    os.environ.get("MTM_DIFFRIR_DIR", os.path.join(PKG, "data", "diffrir")),
    "classroomBase",
)
SCALES = [64, 128, 256, 512, 1024, 2048, 4096]
EPS = 1e-6
FLOOR = 1e-3
RIRLEN = 96000
DEV = "cuda" if torch.cuda.is_available() else "cpu"

R = np.load(f"{D}/RIRs.npy", mmap_mode="r")
N = R.shape[0]
print("n rirs in classroomBase:", N, " device:", DEV, flush=True)

res = {}
for n in SCALES:
    win = torch.hann_window(n, device=DEV)
    meds, fr10, tot = [], 0, 0
    allv = []
    for s in range(0, N, 32):
        blk = np.array(R[s : s + 32][:, :RIRLEN])
        X = torch.tensor(blk, dtype=torch.float32, device=DEV)
        Z = torch.stft(
            X,
            n_fft=n,
            hop_length=n // 4,
            win_length=n,
            window=win,
            center=True,
            return_complex=True,
        )
        m = torch.sqrt(Z.real**2 + Z.imag**2 + EPS).flatten()
        fr10 += int((m < 10 * FLOOR).sum().item())
        tot += m.numel()
        q = torch.quantile(
            m[torch.randperm(m.numel(), device=DEV)[:200000]].float(),
            torch.tensor([0.5], device=DEV),
        )
        allv.append(float(q.item()))
        del X, Z, m
    med = float(np.median(allv))
    res[n] = dict(
        median=med,
        median_over_floor=med / FLOOR,
        pct_above_floor=100 * (med / FLOOR - 1),
        frac_within_10x=fr10 / tot,
        n_bins=tot,
    )
    print(
        "win %5d  median %.6e  (%+.4f%% of floor)  within10x %.4f  bins %d"
        % (n, med, 100 * (med / FLOOR - 1), fr10 / tot, tot),
        flush=True,
    )

meds = [res[n]["median"] for n in SCALES]
w10 = [res[n]["frac_within_10x"] for n in SCALES]
print()
print("MEDIAN RANGE  : %.6e to %.6e" % (min(meds), max(meds)))
print("MAX %% ABOVE   : %.4f%%" % max(100 * (m / FLOOR - 1) for m in meds))
print("WITHIN 10x    : %.1f%% to %.1f%%" % (100 * min(w10), 100 * max(w10)))
json.dump(
    {str(k): v for k, v in res.items()},
    open(os.path.join(OUTD, "out_floor_verify.json"), "w"),
    indent=2,
)
