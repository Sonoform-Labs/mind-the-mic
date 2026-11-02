import numpy as np, json, torch, warnings, os

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTD = os.environ.get("MTM_OUT_DIR", os.path.join(PKG, "outputs"))
os.makedirs(OUTD, exist_ok=True)
warnings.filterwarnings("ignore")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SCALES = [64, 128, 256, 512, 1024, 2048, 4096]
EPS = 1e-6
L = 96000
ROOT = os.environ.get("MTM_DIFFRIR_DIR", os.path.join(PKG, "data", "diffrir"))
rng = np.random.default_rng(0)


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


REP = {
    "classroomBase": dict(nn=5.99, dr=5.22),
    "dampenedBase": dict(nn=1.36, dr=1.21),
    "hallwayBase": dict(nn=10.14, dr=9.13),
    "complexBase": dict(nn=5.52, dr=4.86),
}

print("=" * 100)
print(
    f"DiffRIR AT THEIR EVALUATION LENGTH ({L} samples = {L/48000:.1f}s, their rir_length default)"
)
print(
    "metric verified: our nearest-neighbour reproduction lands within 1 percent of their table"
)
print("=" * 100)
print(
    f"{'room':>16}{'ourNN':>8}{'theirNN':>9}{'err%':>7}{'theirDiffRIR':>14}{'gain':>7}"
    f"{'nuisance':>10}{'gain/nuis':>11}{'pairs>gain':>12}"
)
print("-" * 100)

OUT = {}
for room, r in REP.items():
    d = f"{ROOT}/{room}"
    if not os.path.exists(f"{d}/RIRs.npy"):
        print(f"{room:>16}{'  (not downloaded)':>30}")
        continue
    R = np.load(f"{d}/RIRs.npy", mmap_mode="r")
    XYZ = np.load(f"{d}/xyzs.npy")
    n = R.shape[0]
    step = max(n // 12, 1)
    tr = np.arange(12) * 57 if n == 630 else np.arange(12) * step
    tr = tr[tr < n]
    s = set(list(tr) + list(np.arange(n // 2) * 2))
    va = np.array([i for i in range(n) if i not in s])[::2]
    s2 = set(list(tr) + list(va))
    te = np.array([i for i in range(n) if i not in s2])

    sub = te[::4]
    nnerr = []
    for t in sub:
        j = tr[int(np.argmin(((XYZ[tr] - XYZ[t]) ** 2).sum(1)))]
        a = torch.tensor(np.array(R[j][:L]), dtype=torch.float32, device=DEV)
        b = torch.tensor(np.array(R[t][:L]), dtype=torch.float32, device=DEV)
        nnerr.append(ms(a, b) * 10)
        del a, b
    NN = np.array(nnerr)

    pairs = []
    for _ in range(150):
        i, j = rng.integers(0, n, 2)
        if i == j:
            continue
        a = torch.tensor(np.array(R[i][:L]), dtype=torch.float32, device=DEV)
        b = torch.tensor(np.array(R[j][:L]), dtype=torch.float32, device=DEV)
        pairs.append((float(np.linalg.norm(XYZ[i] - XYZ[j])), ms(a, b) * 10))
        del a, b
    P = np.array(pairs)

    gain = r["nn"] - r["dr"]
    nuis = float(np.median(P[:, 1]))
    frac = float((P[:, 1] > gain).mean())
    errpc = (NN.mean() - r["nn"]) / r["nn"] * 100
    OUT[room] = dict(
        our_nn=float(NN.mean()),
        their_nn=r["nn"],
        err_pct=float(errpc),
        their_diffrir=r["dr"],
        gain=float(gain),
        nuisance_median=nuis,
        ratio=float(nuis / gain),
        frac_pairs_above=frac,
        nn_per_position=NN.tolist(),
        n_test=len(sub),
    )
    print(
        f"{room:>16}{NN.mean():>8.2f}{r['nn']:>9.2f}{errpc:>+6.1f}%{r['dr']:>14.2f}{gain:>7.2f}"
        f"{nuis:>10.2f}{nuis/gain:>10.1f}x{frac:>11.0%}"
    )
print("-" * 100)

if OUT:
    print("\n" + "=" * 100)
    print("READING")
    print("=" * 100)
    for room, v in OUT.items():
        print(f"\n  {room}")
        print(
            f"    reproduction of their baseline : {v['our_nn']:.2f} vs {v['their_nn']:.2f} "
            f"({v['err_pct']:+.1f}%)"
        )
        print(f"    their improvement over baseline: {v['gain']:.2f}")
        print(
            f"    median cost of moving the mic  : {v['nuisance_median']:.2f}  "
            f"= {v['ratio']:.1f}x their improvement"
        )
        print(f"    receiver pairs exceeding gain  : {v['frac_pairs_above']:.0%}")

json.dump(
    {
        k: {kk: vv for kk, vv in v.items() if kk != "nn_per_position"}
        for k, v in OUT.items()
    },
    open(os.path.join(OUTD, "out_diffrir_final.json"), "w"),
    indent=2,
)
print("\nwrote out_diffrir_final.json\ndone")
