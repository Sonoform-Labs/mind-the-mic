import os, re, sys

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.abspath(
    os.environ.get(
        "MTM_DIFFRIR_REPO", os.path.join(PKG, "data", "hearinganythinganywhere")
    )
)
DATA = os.path.abspath(
    os.environ.get("MTM_DIFFRIR_DIR", os.path.join(PKG, "data", "diffrir"))
)
ROOMS = ["classroomBase", "dampenedBase", "hallwayBase", "complexBase"]

cfg = os.path.join(REPO, "config.py")
if not os.path.exists(cfg):
    sys.exit(f"config.py not found in {REPO}; clone the DiffRIR repository there first")
s = open(cfg, encoding="utf-8").read()
for room in ROOMS:
    target = os.path.join(DATA, room).replace("\\", "/")
    if not os.path.exists(os.path.join(target, "RIRs.npy")):
        print(f"warning: {target}/RIRs.npy not found yet")
    s, n = re.subn(rf"^{room}_path\s*=.*$", f'{room}_path = "{target}"', s, flags=re.M)
    if n != 1:
        sys.exit(f"expected one {room}_path line in config.py, found {n}")
    print(f"{room}_path -> {target}")
open(cfg, "w", encoding="utf-8").write(s)
print("updated", cfg)
