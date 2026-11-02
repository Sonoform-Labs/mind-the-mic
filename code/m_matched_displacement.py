import os, sys, json, warnings

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTD = os.environ.get("MTM_OUT_DIR", os.path.join(PKG, "outputs"))
os.makedirs(OUTD, exist_ok=True)
import numpy as np, torch

warnings.filterwarnings("ignore")

HAA = os.environ.get(
    "MTM_DIFFRIR_REPO", os.path.join(PKG, "data", "hearinganythinganywhere")
)
D = os.environ.get("MTM_DIFFRIR_DIR", os.path.join(PKG, "data", "diffrir"))
sys.path.insert(0, HAA)
os.chdir(HAA)
import rooms.dataset as rdata

SCALES = [64, 128, 256, 512, 1024, 2048, 4096]
EPS = 1e-6
RIRLEN = 96000
DEV = "cuda" if torch.cuda.is_available() else "cpu"
WIN = {n: torch.hann_window(n, device=DEV) for n in SCALES}
GAIN = {
    "classroomBase": 0.77,
    "dampenedBase": 0.15,
    "hallwayBase": 1.01,
    "complexBase": 0.66,
}
UNRESTRICTED = {
    "classroomBase": 6.65,
    "dampenedBase": 1.91,
    "hallwayBase": 12.17,
    "complexBase": 6.47,
}


def ms(a, b):
    t = 0.0
    for n in SCALES:
        za = torch.stft(
            a,
            n_fft=n,
            hop_length=n // 4,
            win_length=n,
            window=WIN[n],
            center=True,
            return_complex=True,
        )
        zb = torch.stft(
            b,
            n_fft=n,
            hop_length=n // 4,
            win_length=n,
            window=WIN[n],
            center=True,
            return_complex=True,
        )
        ma = torch.sqrt(za.real**2 + za.imag**2 + EPS)
        mb = torch.sqrt(zb.real**2 + zb.imag**2 + EPS)
        k = min(ma.shape[-1], mb.shape[-1])
        t += float(
            torch.mean(torch.abs(torch.log(ma[..., :k]) - torch.log(mb[..., :k])))
        )
    return t


OUT = {}
print(
    f"{'room':>16}{'band lo':>9}{'band hi':>9}{'held med':>10}"
    f"{'pairs':>7}{'pair med':>10}{'matched':>10}{'unrestr':>10}"
)

for room in GAIN:
    rng = np.random.default_rng(0)
    try:
        ds = rdata.dataLoader(room)
        ds.load_data()
        tr = np.array(sorted(ds.train_indices))
        te = np.array(sorted(ds.test_indices))
    except Exception as e:
        print(f"{room:>16}  dataLoader failed: {repr(e)[:60]}")
        continue

    XYZ = np.load(f"{D}/{room}/xyzs.npy")
    R = np.load(f"{D}/{room}/RIRs.npy", mmap_mode="r")

    disp = np.array([np.min(np.linalg.norm(XYZ[tr] - XYZ[t], axis=1)) for t in te])
    lo, hi = float(disp.min()), float(disp.max())
    held_med = float(np.median(disp))

    n = R.shape[0]
    pairs = []
    tries = 0
    while len(pairs) < 150 and tries < 40000:
        tries += 1
        i, j = rng.integers(0, n, 2)
        if i == j:
            continue
        sep = float(np.linalg.norm(XYZ[i] - XYZ[j]))
        if not (lo <= sep <= hi):
            continue
        a = torch.tensor(np.array(R[i][:RIRLEN]), dtype=torch.float32, device=DEV)
        b = torch.tensor(np.array(R[j][:RIRLEN]), dtype=torch.float32, device=DEV)
        pairs.append((sep, ms(a, b) * 10))
        del a, b

    P = np.array(pairs)
    med = float(np.median(P[:, 1]))
    g = GAIN[room]
    OUT[room] = dict(
        band_lo=lo,
        band_hi=hi,
        heldout_median_disp=held_med,
        n_pairs=len(P),
        pair_median_sep=float(np.median(P[:, 0])),
        matched_nuisance_median=med,
        gain=g,
        matched_ratio=med / g,
        unrestricted_ratio=UNRESTRICTED[room] / g,
        frac_pairs_above_gain=float((P[:, 1] > g).mean()),
    )
    print(
        f"{room:>16}{lo:>9.3f}{hi:>9.3f}{held_med:>10.3f}"
        f"{len(P):>7}{np.median(P[:, 0]):>10.3f}{med / g:>9.2f}x{UNRESTRICTED[room] / g:>9.1f}x"
    )

json.dump(OUT, open(os.path.join(OUTD, "out_matched_displacement.json"), "w"), indent=2)
rs = [v["matched_ratio"] for v in OUT.values()]
fr = [v["frac_pairs_above_gain"] for v in OUT.values()]
print()
print("MATCHED ratio range : %.1fx to %.1fx" % (min(rs), max(rs)))
print("pairs above gain    : %.0f%% to %.0f%%" % (100 * min(fr), 100 * max(fr)))
