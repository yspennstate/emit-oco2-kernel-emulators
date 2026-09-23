"""Which training states the admissible filter removes, split by split.

For each seed the training block is rebuilt with the drivers' split (RandomState(seed) permutation, 10 percent test,
10 percent of the rest for validation) and checked against the split-index digest of the run record when one exists.
A state is admissible when every band has t = Y2 + Y3 > 0 and 0 <= Y4 < 1. For the removed and the kept states the
script reports the medians of the six inputs, the median over states of the smallest positive transmission, and the
share of training entries with 0 < t < u T_train held by the removed states, where T_train is the median positive
training transmission and u = 1e-3, 1e-2. It also reports the admissible fraction of the test states.

usage: EMIT_DATA=<dir> python code/admissible_filter_profile.py [seeds...]      (default 101-110)
"""
import hashlib
import json
import os
import sys

import numpy as np

W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(W, "results", "target_quality")
DATA = os.environ["EMIT_DATA"]
NAMES = ("aerosol_optical_depth", "elevation_km", "water_vapour_cm", "relative_azimuth_deg", "solar_zenith_deg",
         "view_zenith_deg")


def split(n, seed):
    perm = np.random.RandomState(seed).permutation(n)
    n_te = int(round(0.1 * n))
    idx_te, tr_full = perm[:n_te], perm[n_te:]
    vp = np.random.RandomState(seed + 10000).permutation(len(tr_full))
    n_val = int(round(0.1 * len(tr_full)))
    return idx_te, tr_full[vp[:n_val]], tr_full[vp[n_val:]]


X = np.load(os.path.join(DATA, "X.npy")).astype(float)
Y2, Y3, Y4 = (np.load(os.path.join(DATA, c + ".npy")).astype(float) for c in ("Y2", "Y3", "Y4"))
if X.shape[0] != Y2.shape[0]:
    X = X.T
t, s = Y2 + Y3, Y4
adm = np.all((t > 0) & (s >= 0) & (s < 1), axis=1)
tmin = np.where(t > 0, t, np.inf).min(axis=1)
seeds = [int(a) for a in sys.argv[1:]] or list(range(101, 111))
out = {"admissible_states_whole_table": int(adm.sum()), "states": int(len(adm)), "seeds": {}}
for seed in seeds:
    te, va, tr = split(len(adm), seed)
    rec = os.path.join(RES, f"tq_s{seed}_raw_w512.json")
    checked = None
    if os.path.exists(rec):
        with open(rec, encoding="utf-8") as f:
            want = json.load(f)["split_indices_sha256"]["train"]
        checked = hashlib.sha256(np.ascontiguousarray(tr, dtype=np.int64).tobytes()).hexdigest() == want   # the records hash int64
    kept, rem = tr[adm[tr]], tr[~adm[tr]]
    pos = t[tr][t[tr] > 0]
    T = float(np.median(pos))
    row = {"train": int(len(tr)), "kept": int(len(kept)), "removed": int(len(rem)), "T_train": T,
           "record_digest_matches": checked, "test_admissible_fraction": float(adm[te].mean()),
           "median_inputs": {n: {"kept": float(np.median(X[kept, j])), "removed": float(np.median(X[rem, j]))}
                             for j, n in enumerate(NAMES)},
           "median_smallest_transmission": {"kept": float(np.median(tmin[kept])), "removed": float(np.median(tmin[rem]))}}
    for u in (1e-3, 1e-2):
        small = (t[tr] > 0) & (t[tr] < u * T)
        row[f"share_of_entries_below_{u:g}_T_in_removed_states"] = float(small[~adm[tr]].sum() / small.sum())
    out["seeds"][str(seed)] = row
vals = lambda f: [f(r) for r in out["seeds"].values()]  # noqa: E731
out["pooled"] = {
    "removed_fraction": [float(np.min(vals(lambda r: r["removed"] / r["train"]))), float(np.max(vals(lambda r: r["removed"] / r["train"])))],
    "share_below_1e-3_T": [float(np.min(vals(lambda r: r["share_of_entries_below_0.001_T_in_removed_states"]))),
                           float(np.max(vals(lambda r: r["share_of_entries_below_0.001_T_in_removed_states"])))],
    "test_admissible_fraction": [float(np.min(vals(lambda r: r["test_admissible_fraction"]))),
                                 float(np.max(vals(lambda r: r["test_admissible_fraction"])))]}
with open(os.path.join(RES, "admissible_filter_profile.json"), "w", encoding="utf-8", newline="\n") as f:
    json.dump(out, f, indent=1)
print(json.dumps(out["pooled"]), {k: (v["removed"], v["record_digest_matches"]) for k, v in out["seeds"].items()})
