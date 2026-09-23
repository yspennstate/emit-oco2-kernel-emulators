"""One lane of the EMIT training-target sensitivity campaign: train with the preregistered settings, then score.

Runs repository commit 1f5d4df unmodified (~/p23/tq_repo_1f5d4df) and writes to ~/p23/results_tq, never to the
historical results. Exit status is the training run's; a scoring failure leaves SCORE_FAILED_<tag> for a rerun of the
scorer alone, so the runner never repeats a finished training run because the scorer failed.

usage: tq_lane.py --seed S --policy raw|admissible|matched-unfiltered --config w512|w2000 --tag TAG
"""
import argparse
import os
import pathlib
import subprocess
import sys

HOME = pathlib.Path.home()
REPO = HOME / "p23" / "tq_repo_1f5d4df"
OUT = HOME / "p23" / "results_tq"
CONFIG = {
    "w512": ["--widths", "512,512,512", "--epochs", "150", "--members", "5"],
    "w2000": ["--widths", "2000,2000,2000", "--epochs", "500", "--members", "1"],
}
FAMILIES = "ridge3,krr,ard,dnn,dnn_corr,dkr,stack"
THRESHOLDS = ["1e-12", "1e-3", "1e-2"]

a = argparse.ArgumentParser()
a.add_argument("--seed", type=int, required=True)
a.add_argument("--policy", choices=("raw", "admissible", "matched-unfiltered"), required=True)
a.add_argument("--config", choices=tuple(CONFIG), required=True)
a.add_argument("--tag", required=True)
args = a.parse_args()

(OUT / "preds").mkdir(parents=True, exist_ok=True)
env = dict(os.environ, P2_OUT=str(OUT))
data = env.get("EMIT_DATA", str(HOME / "p2" / "data" / "emit"))
env["EMIT_DATA"] = data

train = [sys.executable, "-u", str(REPO / "code" / "emit_campaign.py"), "--seed", str(args.seed),
         "--training-policy", args.policy] + CONFIG[args.config] + [
         "--pca_rank", "64", "--families", FAMILIES, "--tag", args.tag]
print("=== train:", " ".join(train), flush=True)
rc = subprocess.call(train, env=env, cwd=str(REPO / "code"))
print("=== train rc", rc, flush=True)
if rc != 0:
    sys.exit(rc)
# The runner counts a lane as finished only when ~/p23/results/<tag>.json exists. Point it at this record so a
# finished lane is never run a second time; the record itself stays in results_tq.
link = HOME / "p23" / "results" / (args.tag + ".json")
if (OUT / (args.tag + ".json")).is_file() and not os.path.lexists(link):
    os.symlink(os.path.join("..", "results_tq", args.tag + ".json"), link)
    print("=== linked", link, flush=True)

score = [sys.executable, "-u", str(REPO / "code" / "conditioned_reflectance.py"), "--data-dir", data,
         "--record", str(OUT / (args.tag + ".json")), "--predictions", str(OUT / "preds" / (args.tag + ".npz")),
         "--domain", "physical", "--rho", "0.7", "--q-min", "0.3", "--thresholds"] + THRESHOLDS + [
         "--output", str(OUT / (args.tag + "_conditioned.json"))]
print("=== score:", " ".join(score), flush=True)
src = subprocess.call(score, env=env, cwd=str(REPO / "code"))
print("=== score rc", src, flush=True)
if src != 0:
    (OUT / ("SCORE_FAILED_" + args.tag)).write_text("scorer rc %d\n" % src)
sys.exit(0)
