import numpy as np, json, torch, warnings, os

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTD = os.environ.get("MTM_OUT_DIR", os.path.join(PKG, "outputs"))
os.makedirs(OUTD, exist_ok=True)
warnings.filterwarnings("ignore")

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SCALES = [64, 128, 256, 512, 1024, 2048, 4096]
EPS = 1e-6
ROOT = os.environ.get("MTM_DIFFRIR_DIR", os.path.join(PKG, "data", "diffrir"))


def complement(idx, n):
    s = set(int(i) for i in idx)
    return np.array([i for i in range(n) if i not in s])


def log_L1_STFT(x, y, n_fft):
    w = torch.hann_window(n_fft, device=DEV)
    a = torch.stft(x, n_fft=n_fft, hop_length=n_fft // 4, window=w, return_complex=True)
    b = torch.stft(y, n_fft=n_fft, hop_length=n_fft // 4, window=w, return_complex=True)
    ea = torch.sqrt(a.real**2 + a.imag**2 + EPS)
    eb = torch.sqrt(b.real**2 + b.imag**2 + EPS)
    return torch.mean(torch.abs(torch.log(ea) - torch.log(eb)))


def multiscale(x, y):
    return float(sum(log_L1_STFT(x, y, n) for n in SCALES))


REPORTED = {
    "classroomBase": dict(nn=5.99, diffrir=5.22),
    "dampenedBase": dict(nn=1.36, diffrir=1.21),
    "hallwayBase": dict(nn=10.14, diffrir=9.13),
    "complexBase": dict(nn=5.52, diffrir=4.86),
}

print("=" * 94)
print("REPRODUCING DiffRIR's NEAREST-NEIGHBOUR BASELINE")
print(
    "their split: train_indices = arange(12)*57 out of 630. NN predicts a held-out receiver's"
)
print(
    "RIR by copying the nearest TRAINING receiver's RIR. metric is their multiscale_log_l1,"
)
print("their eps, their windows. table numbers are x10.")
print("=" * 94)
print(
    f"{'room':>16}{'n_test':>8}{'our NN x10':>13}{'their NN':>11}{'delta':>9}{'their DiffRIR':>15}{'verdict':>12}"
)
print("-" * 94)

ALL = {}
for room, rep in REPORTED.items():
    d = f"{ROOT}/{room}"
    if not os.path.exists(f"{d}/RIRs.npy"):
        print(f"{room:>16}{'':>8}{'not downloaded yet':>13}")
        continue
    R = np.load(f"{d}/RIRs.npy", mmap_mode="r")
    XYZ = np.load(f"{d}/xyzs.npy")
    n = R.shape[0]
    tr = np.arange(12) * (n // 12) if n != 630 else np.arange(12) * 57
    tr = tr[tr < n]
    va = complement(list(tr) + list(np.arange(n // 2) * 2), n)[::2]
    te = complement(list(tr) + list(va), n)
    errs = []
    for t in te:
        j = tr[int(np.argmin(((XYZ[tr] - XYZ[t]) ** 2).sum(1)))]
        a = torch.tensor(np.array(R[j]), dtype=torch.float32, device=DEV)
        b = torch.tensor(np.array(R[t]), dtype=torch.float32, device=DEV)
        errs.append(multiscale(a, b) * 10)
        del a, b
    E = np.array(errs)
    ours = float(E.mean())
    delta = ours - rep["nn"]
    ok = "MATCH" if abs(delta) < 0.35 else ("close" if abs(delta) < 1.0 else "MISMATCH")
    ALL[room] = dict(
        n_test=len(te),
        ours=ours,
        reported_nn=rep["nn"],
        reported_diffrir=rep["diffrir"],
        per_position=E.tolist(),
    )
    print(
        f"{room:>16}{len(te):>8}{ours:>13.2f}{rep['nn']:>11.2f}{delta:>+9.2f}{rep['diffrir']:>15.2f}{ok:>12}"
    )

print("-" * 94)
if ALL:
    print("\n" + "=" * 94)
    print("PER-POSITION DISTRIBUTION, the figure that sidesteps the sqrt(N) objection")
    print("=" * 94)
    print(
        f"{'room':>16}{'NN mean':>10}{'NN median':>11}{'NN sd':>9}{'5th':>8}{'95th':>8}"
        f"{'DiffRIR mean':>14}{'positions where':>17}"
    )
    print(
        f"{'':>16}{'':>10}{'':>11}{'':>9}{'':>8}{'':>8}{'':>14}{'NN would beat it':>17}"
    )
    print("-" * 94)
    for room, v in ALL.items():
        E = np.array(v["per_position"])
        frac = float((E < v["reported_diffrir"]).mean())
        print(
            f"{room:>16}{E.mean():>10.2f}{np.median(E):>11.2f}{E.std(ddof=1):>9.2f}"
            f"{np.percentile(E,5):>8.2f}{np.percentile(E,95):>8.2f}{v['reported_diffrir']:>14.2f}{frac:>16.1%}"
        )
    print("-" * 94)
    print(
        "\nthe last column is the fraction of individual receiver positions at which the"
    )
    print(
        "nearest-neighbour baseline's error is already below DiffRIR's reported MEAN error."
    )
    print(
        "a large value means the headline mean hides positions where the baseline wins."
    )

json.dump(
    {
        k: {kk: vv for kk, vv in v.items() if kk != "per_position"}
        for k, v in ALL.items()
    },
    open(os.path.join(OUTD, "out_reproduce.json"), "w"),
    indent=2,
)
np.save(
    os.path.join(OUTD, "out_reproduce_perpos.npy"),
    {k: np.array(v["per_position"]) for k, v in ALL.items()},
    allow_pickle=True,
)
print("\nwrote out_reproduce.json and out_reproduce_perpos.npy\ndone")
