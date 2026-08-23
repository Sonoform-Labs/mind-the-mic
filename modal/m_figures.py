import numpy as np, json, torch, warnings, os
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "paper/figures"
os.makedirs(OUT, exist_ok=True)
D = "data/diffrir/classroomBase"
FS = 48000
SCALES = [64, 128, 256, 512, 1024, 2048, 4096]
EPS = 1e-6
DEV = "cuda" if torch.cuda.is_available() else "cpu"
rng = np.random.default_rng(0)

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200, "savefig.bbox": "tight", "axes.grid": True,
                     "grid.alpha": 0.25, "grid.linewidth": 0.5})
BLUE, RED, GREY = "#2166ac", "#b2182b", "#666666"

wc = json.load(open("modal/out_within_column.json"))
ct = json.load(open("modal/out_controls.json"))
sn = json.load(open("modal/out_snr.json"))

# ---------- E1 ----------
mets = ["LSD", "mel", "MFCC", "CLAP"]
sig = [wc[m]["sigma_within"] for m in mets]
lo = [wc[m]["sigma_within"]-wc[m]["ci"][0] for m in mets]
hi = [wc[m]["ci"][1]-wc[m]["sigma_within"] for m in mets]
harn = [ct["A1 HARNESS NULL (mic fixed)"][m] for m in mets]
base12 = [ct["A0 baseline (15 mics)"][m] for m in mets]

fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.1))
for k, m in enumerate(mets):
    ax = axes[k]
    ax.bar([0], [sig[k]], yerr=[[lo[k]], [hi[k]]], width=.6, color=BLUE, capsize=3, label="receiver moves")
    ax.bar([1], [harn[k]], width=.6, color=GREY, label="pipeline only")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["recv", "null"], fontsize=8)
    ax.set_title(m, fontsize=9)
    ax.text(1, harn[k], f"  {harn[k]/base12[k]*100:.0f}%", va="bottom", ha="center", fontsize=7, color=GREY)
    if k == 0: ax.set_ylabel("sigma (metric units)")
axes[0].legend(fontsize=7, frameon=False, loc="upper right")
fig.suptitle("Receiver-position sigma vs an analysis-pipeline null", fontsize=10, y=1.06)
fig.savefig(f"{OUT}/fig1_sigma_vs_null.pdf"); fig.savefig(f"{OUT}/fig1_sigma_vs_null.png"); plt.close(fig)
print("fig1 ok", flush=True)

# ---------- E3 ----------
fig, ax = plt.subplots(figsize=(3.2, 2.4))
keys = list(sn.keys())
lab = ["unfloored\n(eps 1e-6)", "the field's\nconvention\n(eps 1e-3)"]
x = np.arange(2); w = .36
ax.bar(x-w/2, [sn[k]["signal"] for k in keys], w, color=BLUE, label="source signal")
ax.bar(x+w/2, [sn[k]["nuisance"] for k in keys], w, color=RED, label="receiver nuisance")
for i, k in enumerate(keys):
    ax.text(i, max(sn[k]["signal"], sn[k]["nuisance"])*1.08,
            f"S/N={sn[k]['snr']:.2f}", ha="center", fontsize=8,
            fontweight="bold" if sn[k]["snr"] < 1 else "normal")
ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=8)
ax.set_yscale("log"); ax.set_ylabel("sigma (natural-log units)")
ax.legend(fontsize=7, frameon=False)
ax.set_title("Under the field's convention,\nnuisance exceeds signal", fontsize=9)
fig.savefig(f"{OUT}/fig3_signal_vs_nuisance.pdf"); fig.savefig(f"{OUT}/fig3_signal_vs_nuisance.png"); plt.close(fig)
print("fig3 ok", flush=True)

# ---------- DiffRIR recompute for E2 and E4 ----------
R = np.load(f"{D}/RIRs.npy", mmap_mode="r"); XYZ = np.load(f"{D}/xyzs.npy"); N = R.shape[0]
WIN = {n: torch.hann_window(n, device=DEV) for n in SCALES}
def tmag(x, n):
    Z = torch.stft(x, n_fft=n, hop_length=n//4, win_length=n, window=WIN[n], center=True, return_complex=True)
    return torch.sqrt(Z.real**2+Z.imag**2+EPS)
def msl1(a, b):
    t = 0.0
    for n in SCALES:
        ma, mb = tmag(a, n), tmag(b, n)
        k = min(ma.shape[-1], mb.shape[-1])
        t += float(torch.mean(torch.abs(torch.log(ma[..., :k])-torch.log(mb[..., :k]))))
    return t

print("computing fig2 pairs ...", flush=True)
pts = []
for _ in range(400):
    i, j = rng.integers(0, N, 2)
    if i == j: continue
    a = torch.tensor(np.array(R[i]), dtype=torch.float32, device=DEV)
    b = torch.tensor(np.array(R[j]), dtype=torch.float32, device=DEV)
    pts.append((float(np.linalg.norm(XYZ[i]-XYZ[j])), msl1(a, b)*10))
    del a, b
P = np.array(pts)

fig, ax = plt.subplots(figsize=(4.2, 2.8))
ax.scatter(P[:, 0], P[:, 1], s=6, alpha=.28, color=BLUE, edgecolors="none", label="GT vs GT receiver pair")
edges = np.array([0, .5, 1, 2, 4, 8, 100])
cx, cy = [], []
for a, b in zip(edges[:-1], edges[1:]):
    m = (P[:, 0] >= a) & (P[:, 0] < b)
    if m.sum() >= 3: cx.append(np.median(P[m, 0])); cy.append(np.median(P[m, 1]))
ax.plot(cx, cy, "o-", color=BLUE, lw=1.6, ms=4, label="median")
ax.axhline(0.77, color=RED, lw=1.6, ls="--", label="DiffRIR reported gain (0.77)")
ax.axvspan(0.255, 1.891, color="orange", alpha=.10)
_f = float((P[:, 1] > 0.77).mean())
ax.text(0.17, 0.60, "their held-out\ndisplacement\nrange", transform=ax.transAxes,
        fontsize=6.5, color="#aa6600", ha="center")
ax.text(0.98, 0.10, f"all {len(P)} pairs exceed it (min {P[:,1].min():.2f})",
        transform=ax.transAxes, fontsize=7.5, ha="right", color=RED)
ax.set_xscale("log"); ax.set_xlabel("inter-receiver distance (m)")
ax.set_ylabel("multiscale log-L1 (x10, table units)")
ax.set_title("DiffRIR classroom: moving the microphone vs their reported gain", fontsize=9)
ax.legend(fontsize=7, frameon=False, loc="upper left")
fig.savefig(f"{OUT}/fig2_displacement.pdf"); fig.savefig(f"{OUT}/fig2_displacement.png"); plt.close(fig)
print("fig2 ok", flush=True)

print("computing fig4 magnitudes ...", flush=True)
idx = rng.choice(N, size=16, replace=False)
X = torch.tensor(np.array(R[np.sort(idx)]), dtype=torch.float32, device=DEV)
fig, ax = plt.subplots(figsize=(4.2, 2.6))
for n, c in zip([64, 512, 4096], [BLUE, "#66a61e", RED]):
    m = tmag(X, n).flatten().float().cpu().numpy()
    m = m[rng.choice(len(m), size=min(400000, len(m)), replace=False)]
    ax.hist(np.log10(m), bins=180, histtype="step", lw=1.2, color=c, density=True, label=f"nperseg {n}")
ax.axvline(np.log10(1e-3), color="k", lw=1.4, ls="--")
ax.text(np.log10(1e-3), ax.get_ylim()[1]*.92, "  effective floor\n  sqrt(1e-6) = 1e-3", fontsize=7, va="top")
ax.set_xlabel("log10 STFT magnitude"); ax.set_ylabel("density")
ax.set_title("The median spectral bin is pinned at the epsilon floor", fontsize=9)
ax.legend(fontsize=7, frameon=False)
fig.savefig(f"{OUT}/fig4_floor.pdf"); fig.savefig(f"{OUT}/fig4_floor.png"); plt.close(fig)
print("fig4 ok", flush=True)

json.dump(dict(fig2_frac_above_gain=float((P[:, 1] > 0.77).mean()),
               fig2_min=float(P[:, 1].min()),
               fig2_median_table=float(np.median(P[:, 1])),
               fig2_median_under1m=float(np.median(P[P[:, 0] < 1, 1])),
               diffrir_reported_gain=0.77),
          open("modal/out_figures.json", "w"), indent=2)
print("\nall four figures written to paper/figures/")
