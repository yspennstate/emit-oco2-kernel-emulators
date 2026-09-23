"""Integrity check of the training-target experiment: its raw arm against the historical run of the same split.

For each seed with both a raw-arm record (results/target_quality/tq_s<seed>_raw_<config>.json) and a historical record
(results/emit/emit_s<seed>.json for w512, results/scaling/per_seed/emit_s<seed>_wide.json for w2000), prints the
relative difference of every shared family and metric. The kernel and cubic families are deterministic given the
split and should agree to rounding; the networks and everything built on them agree only to run-to-run variation.
Writes results/target_quality/reproduction_check.json.
"""
import glob
import json
import os
import re

W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(W, "results")
METRICS = ("rel_l2_Y1", "rel_l2_Y2", "rel_l2_Y3", "rel_l2_Y4", "rel_l2_radiance", "refl_mae_median", "refl_p95_abs")
HIST = {"w512": os.path.join(RES, "emit", "emit_s{seed}.json"),
        "w2000": os.path.join(RES, "scaling", "per_seed", "emit_s{seed}_wide.json")}
out = {}
for p in sorted(glob.glob(os.path.join(RES, "target_quality", "tq_s*_raw_w*.json"))):
    m = re.match(r"tq_s(\d+)_raw_(w512|w2000)\.json$", os.path.basename(p))
    if not m:
        continue
    seed, cfg = int(m.group(1)), m.group(2)
    h = HIST[cfg].format(seed=seed)
    if not os.path.exists(h):
        continue
    new, old = json.load(open(p, encoding="utf-8")), json.load(open(h, encoding="utf-8"))
    same_split = (new["ntrain"], new["n_val"], new["n_test"]) == (old["ntrain"], old["n_val"], old["n_test"])
    rows = {}
    for fam in new["families"]:
        if fam not in old["families"]:
            continue
        rows[fam] = {k: (new["families"][fam][k] - old["families"][fam][k]) / abs(old["families"][fam][k])
                     for k in METRICS if k in new["families"][fam] and k in old["families"][fam] and old["families"][fam][k]}
    out[f"s{seed}_{cfg}"] = {"same_split_sizes": same_split, "same_data": new["data_sha"] == old["data_sha"],
                             "relative_difference": rows}
    print(f"== seed {seed} {cfg}: split sizes equal {same_split}, data hashes equal {new['data_sha'] == old['data_sha']}")
    for fam, r in rows.items():
        worst = max(r.items(), key=lambda kv: abs(kv[1]))
        print(f"   {fam:9s} radiance {100 * r.get('rel_l2_radiance', float('nan')):+8.3f}%   largest |rel diff| "
              f"{100 * abs(worst[1]):8.3f}% ({worst[0]})")
with open(os.path.join(RES, "target_quality", "reproduction_check.json"), "w", encoding="utf-8", newline="\n") as f:
    json.dump(out, f, indent=1)
