"""Per-band 95th percentiles of the constrained retrieval error on libRadtran, per family, pooled over splits.

For every lane lrt_s<seed>_w512 with predictions, the constrained retrieval of Section 9.3 (R = 0.9, S the largest
training albedo not above one, t_hat -> max(t_hat, 0), s_hat -> [0, S], a nonpositive denominator returning 0) is
evaluated at rho = 0.7 on the physical domain t > 0, 0 <= s < 1 of the test block. The 95th percentile of the absolute
error is taken band by band over the test states of all splits together, for each family, and drawn as a heatmap
(log10 of percentage points). Writes figures/lrt_band_p95.png and results/libradtran/lrt_band_p95.json.

The lanes are those with the given prefix: lrtc (the default, sixteen inputs) or lrt (the seven numeric inputs), whose
outputs are written with the suffix _numeric.

usage: EMIT_DATA=<libRadtran arrays> python code/make_lrt_band_figure.py <dir with <tag>/results_lrt/preds/<tag>.npz>
       [lrtc|lrt]
"""
import glob
import json
import os
import re
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA, KOUT = os.environ["EMIT_DATA"], sys.argv[1]
PREFIX = sys.argv[2] if len(sys.argv) > 2 else "lrtc"
SUFFIX = "" if PREFIX == "lrtc" else "_numeric"
C = ("Y1", "Y2", "Y3", "Y4")
RHO, R = 0.7, 0.9
FAMILIES = [("ridge3", "cubic ridge"), ("krr", "isotropic kernel"), ("ard", "input-scaled kernel"), ("dnn", "network"),
            ("dnn_corr", "network + residual kernel"), ("dkr", "kernel on features"), ("stack", "convex stack")]
with open(os.path.join(DATA, "MANIFEST.json"), encoding="utf-8") as f:
    bands = json.load(f)["bands"]
full = {c: np.load(os.path.join(DATA, c + ".npy")).astype(float) for c in C}
errs = {k: [] for k, _ in FAMILIES}
doms = []
seeds = []
for p in sorted(glob.glob(os.path.join(KOUT, f"{PREFIX}_s*_w512", "results_lrt", "preds", f"{PREFIX}_s*_w512.npz"))):
    P = np.load(p, allow_pickle=False)
    te, tr = P["idx_te"], P["idx_tr"]
    s_tr = full["Y4"][tr]
    S = float(s_tr[np.isfinite(s_tr) & (s_tr <= 1.0)].max())
    a, t, s = full["Y1"][te], full["Y2"][te] + full["Y3"][te], full["Y4"][te]
    L = a + RHO * t / (1.0 - RHO * s)
    doms.append((t > 0) & (s >= 0) & (s < 1))
    seeds.append(re.search(r"_s(\d+)_w512\.npz$", p).group(1))
    for k, _ in FAMILIES:
        Q = {c: P[f"{k}_{c}"].astype(float) for c in C}
        th, sh = np.maximum(Q["Y2"] + Q["Y3"], 0.0), np.clip(Q["Y4"], 0.0, S)
        u = L - Q["Y1"]
        den = th + sh * u
        with np.errstate(divide="ignore", invalid="ignore"):
            rb = np.where(den > 0, np.clip(u / den, 0.0, R), 0.0)
        errs[k].append(np.abs(rb - RHO))
dom = np.concatenate(doms)
p95 = {}
for k, _ in FAMILIES:
    e = np.where(dom, np.concatenate(errs[k]), np.nan)
    p95[k] = (100 * np.nanquantile(e, 0.95, axis=0)).tolist()
out = {"seeds": seeds, "bands": bands, "p95_pp": p95, "domain_fraction": float(dom.mean())}
with open(os.path.join(W, "results", "libradtran", f"lrt_band_p95{SUFFIX}.json"), "w", encoding="utf-8",
          newline="\n") as f:
    json.dump(out, f, indent=1)
M = np.log10(np.maximum(np.array([p95[k] for k, _ in FAMILIES]), 1e-4))
plt.rcParams.update({"font.size": 13})
fig, ax = plt.subplots(figsize=(11, 4.2))
im = ax.imshow(M, aspect="auto", cmap="viridis")
ax.set_xticks(range(len(bands)), bands)
ax.set_yticks(range(len(FAMILIES)), [lab for _, lab in FAMILIES])
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        v = p95[FAMILIES[i][0]][j]
        ax.text(j, i, f"{v:.2g}", ha="center", va="center", fontsize=8, color="white" if M[i, j] < M.max() - 1 else "black")
fig.colorbar(im, ax=ax, label=r"$\log_{10}$ 95th percentile [pp]")
ax.set_xlabel("Sentinel-2 band")
fig.tight_layout()
fig.savefig(os.path.join(W, "figures", f"lrt_band_p95{SUFFIX}.png"), dpi=170)
print(json.dumps({"seeds": seeds, "B10_index": bands.index("B10") if "B10" in bands else None,
                  "p95_B10": {k: round(p95[k][bands.index("B10")], 3) for k, _ in FAMILIES},
                  "max_other_band": {k: round(max(v for j, v in enumerate(p95[k]) if bands[j] != "B10"), 3)
                                     for k, _ in FAMILIES}}))
