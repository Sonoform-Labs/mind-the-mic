import numpy as np, json
import scipy.signal as signal
D = "data/diffrir"
L = 96000
def env_loss(x, y, envelope_size=32, eps=1e-6):
    env1 = signal.convolve(x**2, np.ones((envelope_size)))[int(envelope_size/2):]+eps
    env2 = signal.convolve(y**2, np.ones((envelope_size)))[int(envelope_size/2):]+eps
    return float(np.mean(np.abs(np.log(env1) - np.log(env2))))
band = json.load(open("modal/out_matched_displacement.json"))
envj = json.load(open("modal/out_env_perposition.json"))
pub = {"classroomBase": 0.016, "dampenedBase": 0.005, "hallwayBase": 0.009, "complexBase": 0.007}
out = {}
for room in ["classroomBase", "dampenedBase", "hallwayBase", "complexBase"]:
    XYZ = np.load(f"{D}/{room}/xyzs.npy")
    R = np.load(f"{D}/{room}/RIRs.npy", mmap_mode="r")
    n = R.shape[0]
    lo, hi = band[room]["band_lo"], band[room]["band_hi"]
    rng = np.random.default_rng(0)
    P = []; tries = 0
    while len(P) < 150 and tries < 40000:
        tries += 1
        i, j = rng.integers(0, n, 2)
        if i == j: continue
        sep = float(np.linalg.norm(XYZ[i] - XYZ[j]))
        if not (lo <= sep <= hi): continue
        a = np.asarray(R[i][:L], dtype=np.float32); b = np.asarray(R[j][:L], dtype=np.float32)
        P.append((sep, env_loss(a, b)))
    P = np.array(P)
    rng = np.random.default_rng(0)
    U = []
    for _ in range(150):
        i, j = rng.integers(0, n, 2)
        if i == j: continue
        a = np.asarray(R[i][:L], dtype=np.float32); b = np.asarray(R[j][:L], dtype=np.float32)
        U.append((float(np.linalg.norm(XYZ[i] - XYZ[j])), env_loss(a, b)))
    U = np.array(U)
    rg = envj[room]["env"]["mean_gain"]; pg = pub[room]
    out[room] = dict(n=int(n), R_shape=list(R.shape), dtype=str(R.dtype),
        band=[lo, hi],
        inband_n=len(P), inband_med_sep=float(np.median(P[:,0])), inband_med=float(np.median(P[:,1])), inband_min=float(P[:,1].min()),
        inband_ratio_pub=float(np.median(P[:,1])/pg), inband_ratio_repro=float(np.median(P[:,1])/rg),
        inband_frac_above_pub=float((P[:,1] > pg).mean()),
        unr_n=len(U), unr_med_sep=float(np.median(U[:,0])), unr_max_sep=float(U[:,0].max()), unr_med=float(np.median(U[:,1])), unr_min=float(U[:,1].min()),
        stored_unr_med=envj[room]["env"]["nuisance_median"],
        unr_ratio_pub=float(np.median(U[:,1])/pg), unr_ratio_repro=float(np.median(U[:,1])/rg),
        unr_frac_above_pub=float((U[:,1] > pg).mean()))
    print(room, json.dumps(out[room]), flush=True)
json.dump(out, open("modal/out_env_band.json", "w"), indent=1)
