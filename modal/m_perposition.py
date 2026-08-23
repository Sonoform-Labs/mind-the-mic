import numpy as np, json, torch, warnings, os, sys
warnings.filterwarnings("ignore")

HAA = "data/hearinganythinganywhere"
sys.path.insert(0, HAA)
os.chdir(HAA)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SCALES = [64, 128, 256, 512, 1024, 2048, 4096]
EPS = 1e-6
L = 96000
REPORTED = {"classroomBase": dict(nn=5.99, dr=5.22), "dampenedBase": dict(nn=1.36, dr=1.21),
            "hallwayBase": dict(nn=10.14, dr=9.13), "complexBase": dict(nn=5.52, dr=4.86)}

def ms(x, y):
    t = 0.0
    for n in SCALES:
        w = torch.hann_window(n, device=DEV)
        a = torch.stft(x, n_fft=n, hop_length=n//4, window=w, return_complex=True)
        b = torch.stft(y, n_fft=n, hop_length=n//4, window=w, return_complex=True)
        ea = torch.sqrt(a.real**2+a.imag**2+EPS); eb = torch.sqrt(b.real**2+b.imag**2+EPS)
        t += float(torch.mean(torch.abs(torch.log(ea)-torch.log(eb))))
    return t

import rooms.dataset as rdata

print("="*100)
print("PER-POSITION: DiffRIR's own rendered errors vs nearest-neighbour, using THEIR split code")
print("="*100, flush=True)

OUT = {}
for room, rep in REPORTED.items():
    errf = os.path.join(HAA, "models", room, "errors", "errors_multiscale_log_l1.npy")
    if not os.path.exists(errf):
        print(f"\n{room}: no rendered errors yet, skipping"); continue
    try:
        D = rdata.dataLoader(room); D.load_data()
    except Exception as e:
        print(f"\n{room}: dataLoader failed: {repr(e)[:120]}"); continue
    dr_all = np.load(errf)*10
    te = np.array(sorted(D.test_indices)); tr = np.array(sorted(D.train_indices))
    XYZ = D.xyzs; R = D.RIRs
    dr = dr_all[te]

    nn = []
    disp = []
    for t in te:
        j = tr[int(np.argmin(((XYZ[tr]-XYZ[t])**2).sum(1)))]
        a = torch.tensor(np.asarray(R[j][:L], dtype=np.float32), device=DEV)
        b = torch.tensor(np.asarray(R[t][:L], dtype=np.float32), device=DEV)
        nn.append(ms(a, b)*10); del a, b
        disp.append(float(np.linalg.norm(XYZ[j]-XYZ[t])))
    nn = np.array(nn); disp = np.array(disp)

    d = nn-dr
    rng = np.random.default_rng(0)
    frac = float((nn <= dr).mean())
    bf = np.array([ (nn[idx] <= dr[idx]).mean() for idx in
                    (rng.integers(0, len(d), (8000, len(d)))) ])
    flo, fhi = np.percentile(bf, [2.5, 97.5])

    OUT[room] = dict(n_test=len(te),
                     diffrir_mean=float(dr.mean()), reported_dr=rep["dr"],
                     nn_mean=float(nn.mean()), reported_nn=rep["nn"],
                     mean_gain=float(d.mean()), reported_gain=rep["nn"]-rep["dr"],
                     frac_nn_wins=frac, frac_ci=[float(flo), float(fhi)],
                     gain_sd_per_position=float(d.std(ddof=1)),
                     median_displacement_m=float(np.median(disp)))
    print(f"\n{room}  (n_test={len(te)}, median NN displacement {np.median(disp):.2f} m)")
    print(f"  rendered DiffRIR mean error : {dr.mean():.3f}   (their table: {rep['dr']})")
    print(f"  our NN mean error           : {nn.mean():.3f}   (their table: {rep['nn']})")
    print(f"  per-position gain, mean(sd) : {d.mean():.3f} ({d.std(ddof=1):.3f})   "
          f"(their table gain: {rep['nn']-rep['dr']:.2f})")
    print(f"  positions where NN >= DiffRIR (model wins): {(d>=0).mean():.1%}")
    print(f"  positions where NN BEATS the model         : {frac:.1%}  95% CI [{flo:.1%}, {fhi:.1%}]")

json.dump(OUT, open("modal/out_perposition.json", "w"), indent=2)
np.save("modal/out_perposition_arrays.npy",
        OUT, allow_pickle=True)
print("\nwrote out_perposition.json\ndone")
