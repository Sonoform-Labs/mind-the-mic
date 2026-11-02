import os, sys, runpy

HERE = os.path.dirname(os.path.abspath(__file__))
THREADS = os.environ.get("MTM_THREADS", "4")
for v in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[v] = THREADS
if os.environ.get("MTM_CPU_ONLY") == "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
for k in list(os.environ):
    if k.startswith("MTM_") and (k.endswith("_DIR") or k.endswith("_REPO")):
        os.environ[k] = os.path.abspath(os.environ[k])
aff = os.environ.get("MTM_AFFINITY", "").strip()
if aff:
    import psutil

    psutil.Process().cpu_affinity([int(c) for c in aff.split(",") if c.strip()])

SCRIPTS = sorted(
    f[:-3] for f in os.listdir(HERE) if f.startswith("m_") and f.endswith(".py")
)
names = sys.argv[1:]
if not names:
    print("usage: python code/run.py <script> [<script> ...]")
    print("scripts: " + ", ".join(SCRIPTS))
    sys.exit(1)
for n in names:
    n = os.path.basename(n)
    n = n[:-3] if n.endswith(".py") else n
    if n not in SCRIPTS:
        sys.exit(f"unknown script {n}; choose from: " + ", ".join(SCRIPTS))
for n in names:
    n = os.path.basename(n)
    n = n[:-3] if n.endswith(".py") else n
    path = os.path.join(HERE, n + ".py")
    print(f"== {n}", flush=True)
    cwd = os.getcwd()
    sys.argv = [path]
    try:
        runpy.run_path(path, run_name="__main__")
    finally:
        os.chdir(cwd)
