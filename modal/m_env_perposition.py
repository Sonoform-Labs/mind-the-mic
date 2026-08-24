import numpy as np, json, torch, warnings, os, sys
import scipy.signal as signal
warnings.filterwarnings("ignore")

HAA = "data/hearinganythinganywhere"
sys.path.insert(0, HAA)
os.chdir(HAA)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
L = 96000
FS = 48000

def env_loss(x, y, envelope_size=32, eps=1e-6):
    x = x.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    env1 = signal.convolve(x**2, np.ones((envelope_size)))[int(envelope_size/2):]+eps
    env2 = signal.convolve(y**2, np.ones((envelope_size)))[int(envelope_size/2):]+eps
    return float(np.mean(np.abs(np.log(env1) - np.log(env2))))

def c50_one(x, fs=FS):
    n_50 = int(0.05 * fs)
    et = torch.sum(x ** 2)
    ee = torch.sum(x[:n_50] ** 2)
    D = ee / (et + 1e-10)
    return float(10 * torch.log10(D / (1 - D + 1e-10)))

def drr_one(x, fs=FS, window_ms=2.5):
    n = x.shape[-1]
    w = int(fs * window_ms / 1000)
    p = torch.argmax(torch.abs(x)).item()
    direct = x[max(0, p - w): min(n, p + w + 1)]
    reverb = x[min(n, p + w + 1):]
    return float(10 * torch.log10(torch.sum(direct ** 2) / (torch.sum(reverb ** 2) + 1e-10)))

import rooms.dataset as rdata

ROOMS = ["classroomBase", "dampenedBase", "hallwayBase", "complexBase"]
OUT = {}
for room in ROOMS:
    D = rdata.dataLoader(room); D.load_data()
    te = np.array(sorted(D.test_indices)); tr = np.array(sorted(D.train_indices))
    XYZ = D.xyzs; R = D.RIRs; n = len(R)

    dr_env = np.load(os.path.join(HAA, "models", room, "errors", "errors_env_loss.npy"))[te]
    dr_c50 = np.load(os.path.join(HAA, "models", room, "errors", "errors_C50.npy"))[te]
    dr_drr = np.load(os.path.join(HAA, "models", room, "errors", "errors_DRR.npy"))[te]

    nn_env, nn_c50, nn_drr = [], [], []
    for t in te:
        j = tr[int(np.argmin(((XYZ[tr]-XYZ[t])**2).sum(1)))]
        a = torch.tensor(np.asarray(R[j][:L], dtype=np.float32))
        b = torch.tensor(np.asarray(R[t][:L], dtype=np.float32))
        nn_env.append(env_loss(a, b))
        nn_c50.append(abs(c50_one(a) - c50_one(b)))
        nn_drr.append(abs(drr_one(a) - drr_one(b)))
    nn_env = np.array(nn_env); nn_c50 = np.array(nn_c50); nn_drr = np.array(nn_drr)

    rng = np.random.default_rng(0)
    pe, pc = [], []
    for _ in range(150):
        i, j = rng.integers(0, n, 2)
        if i == j: continue
        a = torch.tensor(np.asarray(R[i][:L], dtype=np.float32))
        b = torch.tensor(np.asarray(R[j][:L], dtype=np.float32))
        pe.append(env_loss(a, b))
        pc.append(abs(c50_one(a) - c50_one(b)))
    pe = np.array(pe); pc = np.array(pc)

    def wins(nn, dr):
        d = nn - dr
        rngb = np.random.default_rng(1)
        frac = float((nn <= dr).mean())
        bf = np.array([(nn[idx] <= dr[idx]).mean() for idx in rngb.integers(0, len(d), (8000, len(d)))])
        lo, hi = np.percentile(bf, [2.5, 97.5])
        return dict(mean_gain=float(d.mean()), sd=float(d.std(ddof=1)),
                    frac_nn_wins=frac, ci=[float(lo), float(hi)])

    gain_env = float(nn_env.mean() - dr_env.mean())
    OUT[room] = dict(
        n_test=len(te),
        env=dict(nn_mean=float(nn_env.mean()), model_mean=float(dr_env.mean()),
                 **wins(nn_env, dr_env),
                 nuisance_median=float(np.median(pe)),
                 nuis_over_gain=float(np.median(pe)/gain_env) if gain_env != 0 else None,
                 frac_pairs_above_gain=float((pe > gain_env).mean())),
        c50=dict(nn_mean=float(nn_c50.mean()), model_mean=float(dr_c50.mean()),
                 **wins(nn_c50, dr_c50),
                 gtgt_median_dB=float(np.median(pc)),
                 nn_median_dB=float(np.median(nn_c50))),
        drr=dict(nn_mean=float(nn_drr.mean()), model_mean=float(dr_drr.mean()),
                 **wins(nn_drr, dr_drr)))
    v = OUT[room]
    print(f"\n{room} (n_test={len(te)})")
    print(f"  ENV  nn {v['env']['nn_mean']:.4f}  model {v['env']['model_mean']:.4f}  "
          f"gain {v['env']['mean_gain']:.4f}  nn_wins {v['env']['frac_nn_wins']:.1%} "
          f"CI [{v['env']['ci'][0]:.1%},{v['env']['ci'][1]:.1%}]  "
          f"nuis {v['env']['nuisance_median']:.4f} = {v['env']['nuis_over_gain']:.1f}x gain, "
          f"pairs>gain {v['env']['frac_pairs_above_gain']:.0%}")
    print(f"  C50  nn {v['c50']['nn_mean']:.3f}  model {v['c50']['model_mean']:.3f}  "
          f"nn_wins {v['c50']['frac_nn_wins']:.1%}  gt-gt median {v['c50']['gtgt_median_dB']:.2f} dB  "
          f"nn median {v['c50']['nn_median_dB']:.2f} dB")
    print(f"  DRR  nn {v['drr']['nn_mean']:.3f}  model {v['drr']['model_mean']:.3f}  "
          f"nn_wins {v['drr']['frac_nn_wins']:.1%}", flush=True)

json.dump(OUT, open("modal/out_env_perposition.json", "w"), indent=2)
print("\nwrote out_env_perposition.json\ndone")
