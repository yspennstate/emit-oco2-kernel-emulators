"""Second moments of the EMIT residuals behind the combination paragraph of Section 4.

For each of the ten main runs and each component, the test residuals of every family are divided by the norm of the
true spectrum of their state. The script forms their uncentered second-moment matrix S, the correlations
S_hk / (e_h e_k) with e_h^2 = S_hh, and, for every pair, the test rho < e_min / e_max under which a convex
combination of the two improves on the better one in this quadratic metric. It also writes, for every pair of
families, the paired per-split differences of the mean component error, read from the records.

usage: EMIT_DATA=<dir with Y1.npy ... Y4.npy> python code/residual_moments.py <dir with emit_s<seed>.npz>
Writes results/emit/emit_secmom.json.
"""
import glob
import json
import os
import re
import sys

import numpy as np

W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED = sys.argv[1]
DATA = os.environ["EMIT_DATA"]
COMPONENTS = ["Y1", "Y2", "Y3", "Y4"]
HEADS = ["ridge3", "krr4k", "krr", "ard", "dnn", "dnn_ens", "dnn_corr", "ens_corr", "dkr", "select", "stack"]
Ys = {c: np.load(os.path.join(DATA, c + ".npy")) for c in COMPONENTS}

out = {"lanes": {}, "pairs": {}}
for f in sorted(glob.glob(os.path.join(PRED, "emit_s*.npz"))):
    tag = os.path.basename(f)[:-4]
    if not re.fullmatch(r"emit_s\d+", tag):
        continue
    z = np.load(f)
    idx_te = z["idx_te"]
    lane = {}
    for c in COMPONENTS:
        Yt = Ys[c][idx_te]
        nrm = np.linalg.norm(Yt, axis=1, keepdims=True)
        heads = [h for h in HEADS if f"{h}_{c}" in z.files]
        R = np.stack([(z[f"{h}_{c}"].astype(np.float64) - Yt) / nrm for h in heads])   # (heads, states, bands)
        S = np.einsum("hnd,knd->hk", R, R) / R.shape[1]
        e = np.sqrt(np.diag(S))
        C = S / np.outer(e, e)
        adm = {}
        for i in range(len(heads)):
            for j in range(i + 1, len(heads)):
                lo, hi = (i, j) if e[i] <= e[j] else (j, i)
                adm[f"{heads[lo]}|{heads[hi]}"] = dict(rho=float(C[i, j]), ratio=float(e[lo] / e[hi]),
                                                       admitted=bool(C[i, j] < e[lo] / e[hi]))
        lane[c] = dict(heads=heads, rms=[float(x) for x in e], corr=[[round(float(x), 4) for x in row] for row in C],
                       admission=adm)
    out["lanes"][tag] = lane
    print(tag, "done", flush=True)

rows = []
for f in sorted(glob.glob(os.path.join(W, "results", "emit", "emit_s*.json"))):
    if re.fullmatch(r"emit_s\d+", os.path.basename(f)[:-5]):
        rows.append(json.load(open(f, encoding="utf-8")))
fams = [h for h in HEADS if all(h in r["families"] for r in rows)]
for i in range(len(fams)):
    for j in range(i + 1, len(fams)):
        d = np.array([100 * (r["families"][fams[i]]["mean_rel_l2_components"]
                             - r["families"][fams[j]]["mean_rel_l2_components"]) for r in rows])
        out["pairs"][f"{fams[i]}-{fams[j]}"] = dict(mean=float(d.mean()), sd=float(d.std(ddof=1)), n=len(d),
                                                   positive=int((d > 0).sum()))
json.dump(out, open(os.path.join(W, "results", "emit", "emit_secmom.json"), "w"), indent=1)
print("wrote results/emit/emit_secmom.json over", len(out["lanes"]), "runs")
