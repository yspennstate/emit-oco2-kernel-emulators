"""Conditional error profile t -> E[e_R^2 | t] in the three training arms of one split and width (exploratory).

For each family, e_R = |e_a| + R|e_t| + R^2 t|e_s| after the projections t_hat -> max(t_hat, 0) and s_hat -> [0, S], with
R = 0.9 and S the largest albedo not above one in the ORIGINAL training block (the same hypothesis for every arm).
Entries are those of the physical domain, t > 0 and 0 <= s < 1, on the common test block. They are binned by the true
transmission at its quantiles over those entries (0, 0.01, 0.05, 0.1, 0.25, 0.5, 1), and the script reports the mean of
e_R^2 and of min{R^2, e_R^2/t^2} in each bin for each arm, with the ratios admissible/raw and matched/raw.
This is not part of the preregistered protocol.

usage: EMIT_DATA=<dir> TQ_RES=<dir with preds/> [TQ_KAGGLE_OUT=<dir>] [TQ_OUT=<dir>] python profile_by_arm.py <seed> <config>
       (a prediction file is looked for in TQ_RES/preds/, then in TQ_KAGGLE_OUT/<tag>/results_tq/preds/)
"""
import json
import os
import sys

import numpy as np

DATA, RES = os.environ["EMIT_DATA"], os.environ["TQ_RES"]
KOUT = os.environ.get("TQ_KAGGLE_OUT")
OUT = os.environ.get("TQ_OUT", RES)
seed, cfg = int(sys.argv[1]), sys.argv[2]
ARMS = ("raw", "admissible", "matched")
C = ("Y1", "Y2", "Y3", "Y4")
R = 0.9
QS = (0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0)


def pred_path(tag):
    for p in [os.path.join(RES, "preds", tag + ".npz")] + (
            [os.path.join(KOUT, tag, "results_tq", "preds", tag + ".npz")] if KOUT else []):
        if os.path.isfile(p):
            return p
    raise SystemExit(f"no predictions for {tag}")


P = {a: np.load(pred_path(f"tq_s{seed}_{a}_{cfg}"), allow_pickle=False) for a in ARMS}
te = P["raw"]["idx_te"]
for a in ARMS:
    if not np.array_equal(P[a]["idx_te"], te):
        raise SystemExit(f"{a}: test indices differ from the raw arm")
full = {c: np.load(os.path.join(DATA, c + ".npy"), allow_pickle=False) for c in C}
s_tr = full["Y4"][P["raw"]["idx_tr"]].astype(float)
S = float(s_tr[np.isfinite(s_tr) & (s_tr <= 1.0)].max())
Y = {c: full[c][te].astype(float) for c in C}
a_, t, s = Y["Y1"], Y["Y2"] + Y["Y3"], Y["Y4"]
dom = (t > 0) & (s >= 0) & (s < 1)
edges = np.quantile(t[dom], QS)
edges[-1] = np.inf
bins = [(dom & (t >= lo) & (t < hi)) for lo, hi in zip(edges[:-1], edges[1:])]
out = {"seed": seed, "config": cfg, "R": R, "S": S, "quantiles": list(QS), "edges": [float(e) for e in edges[:-1]],
       "bin_entries": [int(b.sum()) for b in bins], "families": {}}
fams = [f.rsplit("_", 1)[0] for f in P["raw"].files if f.endswith("_Y1")]
for fam in fams:
    if not all(f"{fam}_Y1" in P[a].files for a in ARMS):
        continue
    rec = {}
    for a in ARMS:
        Q = {c: P[a][f"{fam}_{c}"].astype(float) for c in C}
        th, sh = np.maximum(Q["Y2"] + Q["Y3"], 0.0), np.clip(Q["Y4"], 0.0, S)
        eR = np.abs(Q["Y1"] - a_) + R * np.abs(th - t) + R * R * t * np.abs(sh - s)
        with np.errstate(divide="ignore", invalid="ignore"):
            capped = np.minimum(R * R, (eR / t) ** 2)
        rec[a] = {"mean_eR2": [float(np.mean(eR[b] ** 2)) for b in bins],
                  "mean_capped_ratio2": [float(np.mean(capped[b])) for b in bins]}
    for a in ("admissible", "matched"):
        rec[f"{a}_over_raw_eR2"] = [x / y if y > 0 else None for x, y in zip(rec[a]["mean_eR2"], rec["raw"]["mean_eR2"])]
        rec[f"{a}_over_raw_capped"] = [x / y if y > 0 else None
                                       for x, y in zip(rec[a]["mean_capped_ratio2"], rec["raw"]["mean_capped_ratio2"])]
    out["families"][fam] = rec
path = os.path.join(OUT, f"profile_by_arm_s{seed}_{cfg}.json")
with open(path, "w", encoding="utf-8", newline="\n") as f:
    json.dump(out, f, indent=1)
for fam, rec in out["families"].items():
    print(fam, "adm/raw eR2 by bin", [round(x, 2) if x else None for x in rec["admissible_over_raw_eR2"]],
          "| matched/raw", [round(x, 2) if x else None for x in rec["matched_over_raw_eR2"]])
