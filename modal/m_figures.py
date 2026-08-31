import numpy as np, json, torch, warnings, os
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "paper/tex/figures"
CACHE = "modal/.figcache"
os.makedirs(OUT, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)
D = "data/diffrir/classroomBase"
FS = 48000
SCALES = [64, 128, 256, 512, 1024, 2048, 4096]
EPS = 1e-6
RIRLEN = 96000
DEV = "cuda" if torch.cuda.is_available() else "cpu"
rng = np.random.default_rng(0)

INK = "#22223b"
SLATE = "#4a4e69"
MAUVE = "#9a8c98"
ROSE = "#c9ada7"
CREAM = "#f2e9e4"
NEUTRAL = MAUVE
RAMP = [SLATE, MAUVE, ROSE]
RULE = INK

TW = 5.5

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.linewidth": 0.5,
    "axes.edgecolor": "#3a3a4a",
    "axes.labelpad": 4.0,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.minor.width": 0.35,
    "ytick.minor.width": 0.35,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.minor.size": 2.0,
    "ytick.minor.size": 2.0,
    "xtick.major.pad": 4.0,
    "ytick.major.pad": 4.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.color": "#3a3a4a",
    "ytick.color": "#3a3a4a",
    "axes.grid": False,
    "axes.axisbelow": True,
    "grid.color": "#6a6478",
    "lines.solid_capstyle": "round",
    "figure.dpi": 300,
    "savefig.pad_inches": 0.01,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})


def ygrid(ax, xminor=True):
    ax.grid(True, which="major", linestyle="-", linewidth=0.5, alpha=0.25)
    ax.grid(True, which="minor", linestyle="-", linewidth=0.25, alpha=0.15)
    ax.set_axisbelow(True)
    if not xminor:
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.grid(False, axis="x")


def legend_above(ax, ncol, y=1.02):
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, y, 1.0, 0.14), mode="expand",
              ncol=ncol, frameon=False, borderaxespad=0.0, handlelength=1.5,
              handletextpad=0.5, columnspacing=1.4)


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf")
    fig.savefig(f"{OUT}/{name}.svg")
    fig.savefig(f"{OUT}/{name}.png")
    plt.close(fig)
    print(name, "ok", flush=True)


wc = json.load(open("modal/out_within_column.json"))
ct = json.load(open("modal/out_controls.json"))
sn = json.load(open("modal/out_snr.json"))

mets = ["LSD", "mel", "MFCC"]
LABEL = {"LSD": "log-spectral distance", "mel": "log-mel", "MFCC": "MFCC"}
sig = [wc[m]["sigma_within"] for m in mets]
lo = [wc[m]["sigma_within"] - wc[m]["ci"][0] for m in mets]
hi = [wc[m]["ci"][1] - wc[m]["sigma_within"] for m in mets]
harn = [ct["A1 HARNESS NULL (mic fixed)"][m] for m in mets]

fig, axes = plt.subplots(1, 3, figsize=(TW, 1.85), layout="constrained")
for k, m in enumerate(mets):
    ax = axes[k]
    ax.bar([0], [sig[k]], yerr=[[lo[k]], [hi[k]]], width=0.5, color=MAUVE,
           capsize=2.2, error_kw=dict(elinewidth=0.35, capthick=0.35, ecolor=INK))
    ax.bar([1], [harn[k]], width=0.5, color=SLATE)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["receiver", "null"])
    ax.set_xlim(-0.6, 1.6)
    ax.set_title(LABEL[m])
    ygrid(ax, xminor=False)
    if k == 0:
        ax.set_ylabel("standard deviation")
save(fig, "fig1_sigma_vs_null")

keys = list(sn.keys())
CONV = ["unfloored", "field convention"]
sg = [sn[k]["signal"] for k in keys]
nu = [sn[k]["nuisance"] for k in keys]
ratio = [sn[k]["snr"] for k in keys]

fig, axes = plt.subplots(1, 2, figsize=(TW, 2.45), layout="constrained")
x = np.arange(2)
w = 0.32

ax = axes[0]
ax.bar(x - w / 2, sg, w, color=ROSE, label="between-source signal")
ax.bar(x + w / 2, nu, w, color=SLATE, label="receiver nuisance")
ax.set_xticks(x)
ax.set_xticklabels(CONV)
ax.set_xlim(-0.55, 1.55)
ax.set_ylim(0, max(sg + nu) * 1.12)
ax.set_ylabel("standard deviation")
ygrid(ax, xminor=False)

ax = axes[1]
ax.bar(x, ratio, 0.5, color=[ROSE if r > 1 else SLATE for r in ratio])
ax.set_xticks(x)
ax.set_xticklabels(CONV)
ax.set_xlim(-0.55, 1.55)
ax.set_ylim(0, 1.32)
ax.set_ylabel("signal / nuisance")
ygrid(ax, xminor=False)

axes[0].legend(loc="lower left", bbox_to_anchor=(0.0, 1.02, 2.16, 0.14),
               mode="expand", ncol=2, frameon=False, borderaxespad=0.0,
               handlelength=1.5, handletextpad=0.5, columnspacing=1.4)
save(fig, "fig3_signal_vs_nuisance")

R = np.load(f"{D}/RIRs.npy", mmap_mode="r")
XYZ = np.load(f"{D}/xyzs.npy")
N = R.shape[0]
WIN = {n: torch.hann_window(n, device=DEV) for n in SCALES}


def tmag(x, n):
    Z = torch.stft(x, n_fft=n, hop_length=n // 4, win_length=n, window=WIN[n],
                   center=True, return_complex=True)
    return torch.sqrt(Z.real ** 2 + Z.imag ** 2 + EPS)


def msl1(a, b):
    t = 0.0
    for n in SCALES:
        ma, mb = tmag(a, n), tmag(b, n)
        k = min(ma.shape[-1], mb.shape[-1])
        t += float(torch.mean(torch.abs(torch.log(ma[..., :k]) - torch.log(mb[..., :k]))))
    return t


pair_cache = f"{CACHE}/fig2_pairs.npy"
if os.path.exists(pair_cache):
    P = np.load(pair_cache)
    print("fig2 pairs from cache", P.shape, flush=True)
else:
    print("computing fig2 pairs ...", flush=True)
    pts = []
    for _ in range(400):
        i, j = rng.integers(0, N, 2)
        if i == j:
            continue
        a = torch.tensor(np.array(R[i][:RIRLEN]), dtype=torch.float32, device=DEV)
        b = torch.tensor(np.array(R[j][:RIRLEN]), dtype=torch.float32, device=DEV)
        pts.append((float(np.linalg.norm(XYZ[i] - XYZ[j])), msl1(a, b) * 10))
        del a, b
    P = np.array(pts)
    np.save(pair_cache, P)

fig, ax = plt.subplots(figsize=(TW, 2.55), layout="constrained")
ax.axvspan(0.255, 1.891, color=CREAM, alpha=0.55, lw=0, label="held-out range")
ax.scatter(P[:, 0], P[:, 1], s=13, alpha=0.55, facecolors=MAUVE,
           edgecolors=SLATE, linewidths=0.3, zorder=3,
           label="ground-truth receiver pairs")
edges = np.array([0, .5, 1, 2, 4, 8, 100])
cx, cy = [], []
for a, b in zip(edges[:-1], edges[1:]):
    m = (P[:, 0] >= a) & (P[:, 0] < b)
    if m.sum() >= 3:
        cx.append(np.median(P[m, 0]))
        cy.append(np.median(P[m, 1]))
ax.plot(cx, cy, "o-", color=SLATE, lw=0.9, ms=3.0, zorder=4, label="median")
ax.axhline(0.77, color=RULE, lw=0.8, ls=(0, (2.8, 2.2)), zorder=2, label="DiffRIR gain")
ax.set_xscale("log")
ax.set_xticks([0.25, 0.5, 1, 2, 4, 8])
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
    lambda v, _: ("%g" % v)))
ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
ax.set_xlim(0.225, 9.8)
ax.set_xlabel("inter-receiver distance (m)")
ax.set_ylabel("multiscale log-L1 (table units)")
ax.set_ylim(0, P[:, 1].max() * 1.10)
ygrid(ax)
legend_above(ax, 4)
save(fig, "fig2_displacement")

mag_cache = f"{CACHE}/fig4_mags.npz"
if os.path.exists(mag_cache):
    z = np.load(mag_cache)
    MAGS = {int(k): z[k] for k in z.files}
    print("fig4 magnitudes from cache", flush=True)
else:
    print("computing fig4 magnitudes ...", flush=True)
    idx = rng.choice(N, size=16, replace=False)
    X = torch.tensor(np.array(R[np.sort(idx)][:, :RIRLEN]), dtype=torch.float32, device=DEV)
    MAGS = {}
    for n in [64, 512, 4096]:
        m = tmag(X, n).flatten().float().cpu().numpy()
        MAGS[n] = m[rng.choice(len(m), size=min(400000, len(m)), replace=False)]
    np.savez_compressed(mag_cache, **{str(k): v for k, v in MAGS.items()})

fig, ax = plt.subplots(figsize=(TW, 2.35), layout="constrained")
ax.axvline(np.log10(1e-3), color=RULE, lw=0.8, ls=(0, (2.8, 2.2)),
           label="effective floor", zorder=1)
for n, c in zip([64, 512, 4096], RAMP):
    v = np.sort(np.log10(MAGS[n]))
    y = np.arange(1, len(v) + 1) / len(v)
    step = max(1, len(v) // 3000)
    ax.plot(v[::step], y[::step], lw=0.9, color=c, label=f"window {n}", zorder=2)
ax.set_xlabel("$\\log_{10}$ STFT magnitude")
ax.set_ylabel("cumulative fraction of bins")
ax.set_xlim(-3.42, 0.32)
ax.set_ylim(0.70, 1.012)
ax.set_yticks([0.7, 0.8, 0.9, 1.0])
ygrid(ax)
legend_above(ax, 4)
save(fig, "fig4_floor")

json.dump(dict(fig2_frac_above_gain=float((P[:, 1] > 0.77).mean()),
               fig2_min=float(P[:, 1].min()),
               fig2_median_table=float(np.median(P[:, 1])),
               fig2_median_under1m=float(np.median(P[P[:, 0] < 1, 1])),
               fig2_n_pairs=int(len(P)),
               diffrir_reported_gain=0.77),
          open("modal/out_figures.json", "w"), indent=2)
print("\nall four figures written to paper/figures/")
