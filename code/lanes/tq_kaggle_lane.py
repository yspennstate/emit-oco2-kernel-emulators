# One lane of the EMIT training-target sensitivity campaign on a Kaggle CPU session.
# Protocol: PREREGISTRATION_TQ_20260923.md (sha256 47e61ba922e19a53..) with Addendum 1 (seeds 106-110 on Kaggle).
# Same two commands and arguments as tq_lane.py on the DGX; code at commit 1f5d4df, numpy/scipy/torch at the DGX
# versions in a fresh virtual environment, data verified against the campaign records' SHA-256 before any fit.
import hashlib
import json
import os
import pathlib
import platform
import subprocess
import sys
import time

SEED = int("@SEED@")
POLICY = "@POLICY@"            # raw | admissible | matched-unfiltered
CONFIG = "@CONFIG@"            # w512 | w2000
TAG = "@TAG@"
COMMIT = "1f5d4df632c3bd6214757981c168a424ff1a0b56"
REPO_URL = "https://github.com/yspennstate/emit-oco2-kernel-emulators.git"
DATA_SHA = {
    "X": "3745bd59c5f8782db3021062bd4b18ef2d4efe24d9e029cd5593642619f10a1d",
    "Y1": "3bc568bd7d7aa5bbda866975e666e5882aa1f58e431bc39b829d424d43a33811",
    "Y2": "d7536cbce94e9908668333633e3a03bebcabdb28b36fb1d5f22d81eba51f7f15",
    "Y3": "eb118f3590c3fbaa38dc469a6c0fa2dea7fae771ad7533222c6bf61bd63ad511",
    "Y4": "43f315155486edd49c5c367b968471972962a10fb0b3f625f573020bb8241ec9",
}
CONFIGS = {
    "w512": ["--widths", "512,512,512", "--epochs", "150", "--members", "5"],
    "w2000": ["--widths", "2000,2000,2000", "--epochs", "500", "--members", "1"],
}
FAMILIES = "ridge3,krr,ard,dnn,dnn_corr,dkr,stack"
THRESHOLDS = ["1e-12", "1e-3", "1e-2"]

t0 = time.time()
OUT = pathlib.Path("/kaggle/working/results_tq")
(OUT / "preds").mkdir(parents=True, exist_ok=True)
LOG = open(OUT / (TAG + ".kaggle.log"), "a", encoding="utf-8")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(time.strftime("%H:%M:%S ") + s + "\n")
    LOG.flush()


def status(**kw):
    kw.update(tag=TAG, seed=SEED, policy=POLICY, config=CONFIG, elapsed_min=round((time.time() - t0) / 60, 1))
    (OUT / (TAG + ".kaggle_status.json")).write_text(json.dumps(kw, indent=1))


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


# 1. data: the five arrays of the private dataset, checked against the campaign records
xs = sorted(pathlib.Path("/kaggle/input").rglob("X.npy"))
if not xs:
    status(stage="data", ok=False, why="X.npy not found under /kaggle/input")
    sys.exit("X.npy not found")
emit_dir = xs[0].parent
for name, want in DATA_SHA.items():
    got = sha256(emit_dir / (name + ".npy"))
    say("data", name, got[:16], "ok" if got == want else "MISMATCH")
    if got != want:
        status(stage="data", ok=False, why=f"{name} sha256 {got} != {want}")
        sys.exit(f"{name}.npy does not match the campaign records")

# 2. code at the protocol's commit
repo = pathlib.Path("/tmp/repo")
subprocess.check_call(["git", "clone", "-q", REPO_URL, str(repo)])
subprocess.check_call(["git", "-C", str(repo), "checkout", "-q", COMMIT])
head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
say("code", head)
if head != COMMIT:
    status(stage="code", ok=False, why=f"HEAD {head}")
    sys.exit("wrong commit")

# 3. environment: the DGX versions of the three imported packages, in a fresh venv
venv = pathlib.Path("/tmp/venv")
# Kaggle's system Python has no ensurepip: make the venv without pip and bootstrap it
subprocess.check_call([sys.executable, "-m", "venv", "--without-pip", str(venv)])
vpy = str(venv / "bin" / "python")
import urllib.request  # noqa: E402
urllib.request.urlretrieve("https://bootstrap.pypa.io/get-pip.py", "/tmp/get-pip.py")
subprocess.check_call([vpy, "/tmp/get-pip.py", "-q"])
subprocess.check_call([vpy, "-m", "pip", "install", "-q", "numpy==2.2.6", "scipy==1.15.3"])
subprocess.check_call([vpy, "-m", "pip", "install", "-q", "torch==2.13.0", "--index-url",
                       "https://download.pytorch.org/whl/cpu"])
freeze = subprocess.check_output([vpy, "-m", "pip", "freeze"], text=True)
try:
    lscpu = subprocess.check_output(["lscpu"], text=True)
except Exception as e:  # noqa: BLE001
    lscpu = f"lscpu unavailable: {e}"
pyver = subprocess.check_output([vpy, "-c", "import sys; print(sys.version)"], text=True).strip()
(OUT / (TAG + ".env.txt")).write_text(f"python {pyver}\nplatform {platform.platform()}\nnproc {os.cpu_count()}\n\n"
                                      f"{freeze}\n{lscpu}")
say("env", pyver, "|", " ".join(l for l in freeze.split() if l.split("==")[0] in ("numpy", "scipy", "torch")))
status(stage="setup_done", ok=True, setup_min=round((time.time() - t0) / 60, 1))

# 4. train and score, exactly as tq_lane.py
env = dict(os.environ, P2_OUT=str(OUT), EMIT_DATA=str(emit_dir), NMKC_THREADS="4", PYTHONUNBUFFERED="1")
train = [vpy, "-u", str(repo / "code" / "emit_campaign.py"), "--seed", str(SEED), "--training-policy", POLICY] + \
    CONFIGS[CONFIG] + ["--pca_rank", "64", "--families", FAMILIES, "--tag", TAG]
say("=== train:", " ".join(train))
t1 = time.time()
with open(OUT / (TAG + ".train.log"), "w", encoding="utf-8") as tl:
    rc = subprocess.call(train, env=env, cwd=str(repo / "code"), stdout=tl, stderr=subprocess.STDOUT)
train_min = round((time.time() - t1) / 60, 1)
say("=== train rc", rc, "minutes", train_min)
if rc != 0:
    status(stage="train", ok=False, rc=rc, train_min=train_min)
    sys.exit(f"training failed rc={rc}")

score = [vpy, "-u", str(repo / "code" / "conditioned_reflectance.py"), "--data-dir", str(emit_dir),
         "--record", str(OUT / (TAG + ".json")), "--predictions", str(OUT / "preds" / (TAG + ".npz")),
         "--domain", "physical", "--rho", "0.7", "--q-min", "0.3", "--thresholds"] + THRESHOLDS + [
         "--output", str(OUT / (TAG + "_conditioned.json"))]
say("=== score:", " ".join(score))
with open(OUT / (TAG + ".score.log"), "w", encoding="utf-8") as sl:
    src = subprocess.call(score, env=env, cwd=str(repo / "code"), stdout=sl, stderr=subprocess.STDOUT)
say("=== score rc", src)
if src != 0:
    (OUT / ("SCORE_FAILED_" + TAG)).write_text("scorer rc %d\n" % src)
status(stage="done", ok=src == 0, train_min=train_min, score_rc=src)
say("done in", round((time.time() - t0) / 60, 1), "min")
