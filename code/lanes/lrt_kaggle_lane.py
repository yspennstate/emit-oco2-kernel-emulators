# One lane of the forward-inverse comparison on libRadtran, in a Kaggle session.
# Protocol: results/libradtran/PREREGISTRATION_LRT_20260923.md (sha256 bf105fb61877cc74..), fixed before any run.
# Code at commit 1f5d4df (unmodified emit_campaign.py and conditioned_reflectance.py), numpy/scipy/torch at the DGX
# versions in a fresh virtual environment, the five arrays verified against MANIFEST.json before any fit.
import hashlib
import json
import os
import pathlib
import platform
import subprocess
import sys
import time

SEED = int("@SEED@")
TAG = "@TAG@"
COMMIT = "1f5d4df632c3bd6214757981c168a424ff1a0b56"
REPO_URL = "https://github.com/yspennstate/emit-oco2-kernel-emulators.git"
DATA_SHA = {
    "X": "f02293c76ae8fc244c9ab2e83901f4033f3db19c0cfa436a333abe69f59af3f6",
    "Y1": "6789c57d0c88f1d336bc4576b93f5b4a53a933c142c62301e558eabcb3db481c",
    "Y2": "faf0af8ee5e5c6c56e34b830307714ebb4e85c6e23c54a8154cb2979b7cbc12b",
    "Y3": "15c24d29a063631479496924c5c053d76831446d302119787ef7171159add765",
    "Y4": "68701724a682b6b6e060b1f6cff59c99bc4f4283437202c812fedc1ef85c89fd",
}
CONFIG = ["--widths", "512,512,512", "--epochs", "150", "--members", "5", "--pca_rank", "13"]
FAMILIES = "ridge3,krr,ard,dnn,dnn_corr,dkr,stack"
THRESHOLDS = ["1e-12", "1e-3", "1e-2"]

t0 = time.time()
OUT = pathlib.Path("/kaggle/working/results_lrt")
(OUT / "preds").mkdir(parents=True, exist_ok=True)
LOG = open(OUT / (TAG + ".kaggle.log"), "a", encoding="utf-8")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(time.strftime("%H:%M:%S ") + s + "\n")
    LOG.flush()


def status(**kw):
    kw.update(tag=TAG, seed=SEED, elapsed_min=round((time.time() - t0) / 60, 1))
    (OUT / (TAG + ".kaggle_status.json")).write_text(json.dumps(kw, indent=1))


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


# 1. data: the five arrays of the private dataset, checked against the protocol's digests
mans = sorted(pathlib.Path("/kaggle/input").rglob("MANIFEST.json"))
if not mans:
    status(stage="data", ok=False, why="MANIFEST.json not found under /kaggle/input")
    sys.exit("MANIFEST.json not found")
data_dir = mans[0].parent
for name, want in DATA_SHA.items():
    got = sha256(data_dir / (name + ".npy"))
    say("data", name, got[:16], "ok" if got == want else "MISMATCH")
    if got != want:
        status(stage="data", ok=False, why=f"{name} sha256 {got} != {want}")
        sys.exit(f"{name}.npy does not match the protocol's digest")

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

# 4. fit and score, the protocol's two commands
env = dict(os.environ, P2_OUT=str(OUT), EMIT_DATA=str(data_dir), NMKC_THREADS="4", PYTHONUNBUFFERED="1")
train = [vpy, "-u", str(repo / "code" / "emit_campaign.py"), "--seed", str(SEED), "--training-policy", "raw"] + \
    CONFIG + ["--families", FAMILIES, "--tag", TAG]
say("=== train:", " ".join(train))
t1 = time.time()
with open(OUT / (TAG + ".train.log"), "w", encoding="utf-8") as tl:
    rc = subprocess.call(train, env=env, cwd=str(repo / "code"), stdout=tl, stderr=subprocess.STDOUT)
train_min = round((time.time() - t1) / 60, 1)
say("=== train rc", rc, "minutes", train_min)
if rc != 0:
    status(stage="train", ok=False, rc=rc, train_min=train_min)
    sys.exit(f"training failed rc={rc}")

score = [vpy, "-u", str(repo / "code" / "conditioned_reflectance.py"), "--data-dir", str(data_dir),
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
