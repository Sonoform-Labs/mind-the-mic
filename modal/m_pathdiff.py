import os, sys, json, ctypes
import numpy as np

if sys.platform == "win32":
    try:
        ctypes.windll.kernel32.SetProcessAffinityMask(
            ctypes.windll.kernel32.GetCurrentProcess(), 0xFFF)
    except Exception:
        pass

ROOT = "data/realimpact"
C = 343.0  # m/s, speed of sound

OBJS = [d for d in sorted(os.listdir(ROOT))
        if os.path.isdir(f"{ROOT}/{d}")
        and os.path.exists(f"{ROOT}/{d}/preprocessed/listenerXYZ.npy")]

print("computing max within-column path-length difference")
print("(same gantry angle and distance, varying microphone only)\n")

worst = 0.0
worst_where = None
bar_lengths = []
radii = []
rows = []

for o in OBJS:
    B = f"{ROOT}/{o}/preprocessed/"
    L = np.load(B + "listenerXYZ.npy")
    V = np.load(B + "vertexXYZ.npy")
    ANG = np.load(B + "angle.npy")
    DIST = np.load(B + "distance.npy")
    MIC = np.load(B + "micID.npy")

    uv = np.unique(V, axis=0)
    best_o = 0.0
    for a in np.unique(ANG):
        for d in np.unique(DIST):
            sel = np.where((ANG == a) & (DIST == d))[0]
            if len(sel) < 2:
                continue
            mics = {}
            for i in sel:
                mics[int(MIC[i])] = L[i]
            P = np.array(list(mics.values()))
            if len(P) < 2:
                continue
            bar_lengths.append(float(np.linalg.norm(P.max(0) - P.min(0))))
            for src in uv:
                r = np.linalg.norm(P - src, axis=1)
                radii.append(float(r.min()))
                spread = float(r.max() - r.min())
                if spread > best_o:
                    best_o = spread
                if spread > worst:
                    worst = spread
                    worst_where = (o, int(a), int(d))
    rows.append((o, best_o))
    print("  %-24s max path difference %.3f m  (%.2f ms)" % (o, best_o, 1000*best_o/C))

print()
print("=" * 68)
print("  microphone bar span      %.3f m  (min %.3f, max %.3f)"
      % (np.median(bar_lengths), min(bar_lengths), max(bar_lengths)))
print("  source-receiver radius   %.3f to %.3f m" % (min(radii), max(radii)))
print()
print("  WORST-CASE path-length difference within a column")
print("     %.3f m  =  %.2f ms   at %s" % (worst, 1000*worst/C, worst_where))
print()
print("  the paper currently claims the bar 'spans up to 5.3 ms',")
print("  which assumes a path difference equal to the full bar length")
print("     bar length %.3f m would be %.2f ms" % (max(bar_lengths), 1000*max(bar_lengths)/C))

json.dump(dict(max_path_diff_m=worst, max_path_diff_ms=1000*worst/C,
               bar_span_m=float(np.median(bar_lengths)),
               radius_min=float(min(radii)), radius_max=float(max(radii))),
          open("modal/out_pathdiff.json", "w"), indent=1)
print("\nwrote modal/out_pathdiff.json")
