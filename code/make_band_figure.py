"""Residual quantile bands, band by band, for the main pipeline at one split.

Reads the float64 test predictions saved by emit_campaign.py (results/target_quality/preds/<tag>.npz) and the
EMIT arrays (EMIT_DATA), and draws nested percentile envelopes of the residual at every wavelength:

    red 25-75, blue 5-25 and 75-95, green 2.5-5 and 95-97.5, grey 0.5-2.5 and 97.5-99.5, median in black.

Columns: the direct-flux residual in physical units, the radiance residual at rho = 0.7, the reflectance residual
over all bands, and the reflectance residual on the physical domain with the conditioning floor used by
conditioned_reflectance.py (t >= 1e-12 of the training flux scale, 0 <= Y4 < 1, q >= 0.3). Also counts the
"wall" bands, where the 95th percentile of the absolute reflectance error exceeds 0.02, per family.

usage: EMIT_DATA=<dir> python code/make_band_figure.py [tag]      (default tag tq_s101_raw_w512)
"""
import json
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAG = sys.argv[1] if len(sys.argv) > 1 else "tq_s101_raw_w512"
RES = os.path.join(W, "results", "target_quality")
DATA = os.environ["EMIT_DATA"]
C = ("Y1", "Y2", "Y3", "Y4")
RHO, Q_MIN, THRESH = 0.7, 0.3, 1e-12
QS = [.005, .025, .05, .25, .5, .75, .95, .975, .995]
COLORS = ["gray", "green", "blue", "red"]
FAMILIES = [("ridge3", "cubic\nridge"), ("dnn", "network\n$3\\times512$"), ("dnn_corr", "network +\nresidual kernel"),
            ("ard", "input-scaled\nkernel"), ("dkr", "kernel on\nfeatures"), ("stack", "convex\nstack")]

with open(os.path.join(W, "results", "emit_wavelengths.json"), encoding="utf-8") as f:
    wls = np.array(json.load(f)["nm"])
with open(os.path.join(RES, TAG + "_conditioned.json"), encoding="utf-8") as f:
    scale = json.load(f)["results"]["stack"][0]["training_flux_scale"]
pred = np.load(os.path.join(RES, "preds", TAG + ".npz"), allow_pickle=False)
te = pred["idx_te"]
Y = {c: np.load(os.path.join(DATA, c + ".npy"), allow_pickle=False)[te].astype(float) for c in C}
t, s = Y["Y2"] + Y["Y3"], Y["Y4"]
q = 1.0 - RHO * s
L = Y["Y1"] + RHO * t / q
DOMAIN = (t > 0) & (t >= THRESH * scale) & (s >= 0) & (s < 1) & (q >= Q_MIN)


def band(ax, res, title, ylim=None, mask=None):
    A = res.astype(float).copy()
    if mask is not None:
        A[~mask] = np.nan
    with np.errstate(invalid="ignore"):
        qq = np.nanquantile(A, QS, axis=0).T
    for i, col in enumerate(COLORS):
        ax.fill_between(wls, qq[:, i], qq[:, i + 1], color=col, alpha=0.2, linewidth=0)
        ax.fill_between(wls, qq[:, -i - 2], qq[:, -i - 1], color=col, alpha=0.2, linewidth=0)
    ax.plot(wls, qq[:, 4], color="k", lw=0.6)
    ax.axhline(0.0, color="k", lw=0.3, ls=":")
    ax.set_xlim(wls.min(), wls.max())
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_title(title)


plt.rcParams.update({"font.size": 20, "axes.titlesize": 19, "axes.labelsize": 20,
                     "xtick.labelsize": 17, "ytick.labelsize": 17})
fams = [(k, lab) for k, lab in FAMILIES if f"{k}_Y1" in pred.files]
fig, axes = plt.subplots(len(fams), 4, figsize=(18, 3.1 * len(fams)), squeeze=False)
walls, summary = {}, {"tag": TAG, "coverage_pct": float(100 * DOMAIN.mean()), "families": {}}
for r, (k, lab) in enumerate(fams):
    P = {c: pred[f"{k}_{c}"].astype(float) for c in C}
    u = L - P["Y1"]
    den = P["Y2"] + P["Y3"] + P["Y4"] * u
    with np.errstate(divide="ignore", invalid="ignore"):
        rho_hat = u / den
    err = np.where(np.isfinite(rho_hat), rho_hat, 0.0) - RHO
    radiance = P["Y1"] + RHO * (P["Y2"] + P["Y3"]) / (1.0 - RHO * P["Y4"])
    top = r == 0
    band(axes[r][0], Y["Y2"] - P["Y2"], "direct flux $Y_2$\n" if top else "")
    band(axes[r][1], L - radiance, "radiance\nat $\\rho=0.7$" if top else "")
    band(axes[r][2], err, "reflectance\nall bands" if top else "", ylim=(-0.02, 0.02))
    band(axes[r][3], err, "reflectance\nphysical domain" if top else "", ylim=(-0.02, 0.02), mask=DOMAIN)
    axes[r][0].set_ylabel(lab, fontsize=19)
    walls[k] = set(np.nonzero(np.quantile(np.abs(err), 0.95, axis=0) > 0.02)[0].tolist())
    summary["families"][k] = {"wall_bands": len(walls[k]),
                              "nonfinite_inversions": int((~np.isfinite(rho_hat)).sum())}
for a in axes[-1]:
    a.set_xlabel("wavelength [nm]")
fig.tight_layout()
fig.savefig(os.path.join(W, "figures", "emit_quantiles_top_models.png"), dpi=170)
common = sorted(set.intersection(*walls.values()))
union = sorted(set.union(*walls.values()))
runs, start = [], None
for i, b in enumerate(common):
    if start is None:
        start = b
    if i == len(common) - 1 or common[i + 1] != b + 1:
        runs.append(f"{wls[start]:.0f}-{wls[b]:.0f}")
        start = None
summary.update(common_wall_bands=len(common), union_wall_bands=len(union), common_wall_regions_nm=runs)
with open(os.path.join(RES, f"band_walls_{TAG}.json"), "w", encoding="utf-8", newline="\n") as f:
    json.dump(summary, f, indent=1)
print(json.dumps(summary))
