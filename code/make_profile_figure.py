"""Where training on admissible targets moves the component error (exploratory, outside the protocol).

Reads results/target_quality/profile_by_arm_s<seed>_<config>.json (code/profile_by_arm.py) for one configuration and
draws, for each family, the median over splits of E[e_R^2 | t] in the admissible and matched arms divided by the raw arm,
in six bins of the true transmission (quantiles 0-1, 1-5, 5-10, 10-25, 25-50 and 50-100 percent over the physical
domain), with the interquartile range across splits. Writes figures/tq_profile_<config>.png and prints the medians.

usage: python code/make_profile_figure.py [w512|w2000]
"""
import glob
import json
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = sys.argv[1] if len(sys.argv) > 1 else "w512"
FAMILIES = [("krr", "isotropic kernel"), ("ard", "input-scaled kernel"), ("dnn", "network"),
            ("dnn_corr", "network + residual kernel"), ("dkr", "kernel on features"), ("stack", "convex stack")]
BINS = ["0-1", "1-5", "5-10", "10-25", "25-50", "50-100"]
files = sorted(glob.glob(os.path.join(W, "results", "target_quality", f"profile_by_arm_s*_{CFG}.json")))
runs = [json.load(open(p, encoding="utf-8")) for p in files]
seeds = [r["seed"] for r in runs]
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
out = {"config": CFG, "seeds": seeds, "median_ratio": {}}
x = np.arange(len(BINS))
for ax, arm in zip(axes, ("admissible", "matched")):
    for j, (k, lab) in enumerate(FAMILIES):
        R = np.array([[v if v else np.nan for v in r["families"][k][f"{arm}_over_raw_eR2"]] for r in runs
                      if k in r["families"]])
        if not len(R):
            continue
        med = np.nanmedian(R, axis=0)
        lo, hi = np.nanquantile(R, 0.25, axis=0), np.nanquantile(R, 0.75, axis=0)
        out["median_ratio"][f"{arm}/{k}"] = [float(v) for v in med]
        off = (j - len(FAMILIES) / 2) * 0.06
        ax.errorbar(x + off, med, yerr=[med - lo, hi - med], marker="o", ms=4, lw=1.2, capsize=2, label=lab)
    ax.axhline(1.0, color="k", lw=0.6, ls=":")
    ax.set_yscale("log")
    ax.set_xticks(x, BINS)
    ax.set_xlabel("transmission quantile bin [%]")
    ax.set_title(f"{arm} / raw")
axes[0].set_ylabel(r"$\mathbb{E}[e_R^2\mid t]$ ratio")
axes[1].legend(fontsize=9, loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(W, "figures", f"tq_profile_{CFG}.png"), dpi=170)
print(json.dumps({"seeds": seeds, "median_ratio": {k: [round(v, 2) for v in vals] for k, vals in out["median_ratio"].items()}}))
