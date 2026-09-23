"""One lane of the forward-inverse comparison on libRadtran (sixteen inputs), on the Caltech DGX.

Protocol: results/libradtran/PREREGISTRATION_LRT_20260923.md with addendum 1. The protocol allows whole seeds to be
split between Kaggle and the DGX with the assignment recorded; this lane is the DGX counterpart of lrtc_kaggle_lane.py
and runs the same two commands: the files of repository commit 1f5d4df (~/p23/tq_repo_1f5d4df, checked by digest), the
DGX environment
(numpy 2.2.6, scipy 1.15.3, torch 2.13.0 CPU, the versions the Kaggle lanes install), four threads, the five arrays
checked against the protocol's digests before any fit. Writes to ~/p23/results_lrtc; the runner's completion record
~/p23/results/<tag>.json points at the lane's record. A scoring failure leaves SCORE_FAILED_<tag>.

usage: lrtc_lane_dgx.py --seed S --tag lrtc_s<S>_w512
"""
import argparse
import hashlib
import os
import pathlib
import platform
import subprocess
import sys

HOME = pathlib.Path.home()
REPO = HOME / "p23" / "tq_repo_1f5d4df"
DATA = HOME / "p23" / "data_lrt13cats"
OUT = HOME / "p23" / "results_lrtc"
DATA_SHA = {
    "X": "c403bb22912296f4143d71adcb31fa793d9d9c6b3ec552dee03f4f43326f1258",
    "Y1": "6789c57d0c88f1d336bc4576b93f5b4a53a933c142c62301e558eabcb3db481c",
    "Y2": "faf0af8ee5e5c6c56e34b830307714ebb4e85c6e23c54a8154cb2979b7cbc12b",
    "Y3": "15c24d29a063631479496924c5c053d76831446d302119787ef7171159add765",
    "Y4": "68701724a682b6b6e060b1f6cff59c99bc4f4283437202c812fedc1ef85c89fd",
}
# the three files the lane runs, at commit 1f5d4df (SHA-256 with line endings as LF; the DGX copy of the commit was
# unpacked from an archive made on Windows and carries CRLF)
CODE_SHA = {
    "emit_campaign.py": "c81fb62de3386ec66556e844a8513fa21d216a3ef0fffc69744a85d2b350cc31",
    "conditioned_reflectance.py": "601d0ae6d9e5b29e2ad88b7bd5e3d4e5d57a7b7c5354e73f96063c5a9b8734fb",
    "emit_target_quality.py": "ff41865787467da5e979a5eb6c7ca9db7f38043ab5abbf1054ef1f21dfd77053",
}
CONFIG = ["--widths", "512,512,512", "--epochs", "150", "--members", "5", "--pca_rank", "13"]
FAMILIES = "ridge3,krr,ard,dnn,dnn_corr,dkr,stack"
THRESHOLDS = ["1e-12", "1e-3", "1e-2"]

a = argparse.ArgumentParser()
a.add_argument("--seed", type=int, required=True)
a.add_argument("--tag", required=True)
args = a.parse_args()
(OUT / "preds").mkdir(parents=True, exist_ok=True)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


for name, want in DATA_SHA.items():
    got = sha256(DATA / (name + ".npy"))
    print("data", name, got[:16], "ok" if got == want else "MISMATCH", flush=True)
    if got != want:
        sys.exit(f"{name}.npy does not match the protocol's digest")
for name, want in CODE_SHA.items():
    got = hashlib.sha256((REPO / "code" / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    print("code", name, got[:16], "ok" if got == want else "MISMATCH", flush=True)
    if got != want:
        sys.exit(f"{name} is not the file of commit 1f5d4df")
freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
(OUT / (args.tag + ".env.txt")).write_text(f"python {sys.version}\nplatform {platform.platform()}\nnproc {os.cpu_count()}\n"
                                          f"machine Caltech DGX\n\n{freeze}")

env = dict(os.environ, P2_OUT=str(OUT), EMIT_DATA=str(DATA), NMKC_THREADS="4", PYTHONUNBUFFERED="1")
train = [sys.executable, "-u", str(REPO / "code" / "emit_campaign.py"), "--seed", str(args.seed),
         "--training-policy", "raw"] + CONFIG + ["--families", FAMILIES, "--tag", args.tag]
print("=== train:", " ".join(train), flush=True)
rc = subprocess.call(train, env=env, cwd=str(REPO / "code"))
print("=== train rc", rc, flush=True)
if rc != 0:
    sys.exit(rc)
link = HOME / "p23" / "results" / (args.tag + ".json")
if (OUT / (args.tag + ".json")).is_file() and not os.path.lexists(link):
    os.symlink(os.path.join("..", "results_lrtc", args.tag + ".json"), link)
    print("=== linked", link, flush=True)

score = [sys.executable, "-u", str(REPO / "code" / "conditioned_reflectance.py"), "--data-dir", str(DATA),
         "--record", str(OUT / (args.tag + ".json")), "--predictions", str(OUT / "preds" / (args.tag + ".npz")),
         "--domain", "physical", "--rho", "0.7", "--q-min", "0.3", "--thresholds"] + THRESHOLDS + [
         "--output", str(OUT / (args.tag + "_conditioned.json"))]
print("=== score:", " ".join(score), flush=True)
src = subprocess.call(score, env=env, cwd=str(REPO / "code"))
print("=== score rc", src, flush=True)
if src != 0:
    (OUT / ("SCORE_FAILED_" + args.tag)).write_text("scorer rc %d\n" % src)
sys.exit(0)
