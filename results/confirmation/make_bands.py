"""Reflectance tails of the frozen pipeline on its development split (seed 101), and residual percentile bands.

Writes development_split_tails.json: for each family of the fresh-partition protocol, the radiance error and the
median, 95th and 99th percentiles of the reflectance error over all bands and on the screen t >= 1e-12, q >= 0.3.
The development-split factors of Section 6.5 (stack over feature kernel, 6.13 over all bands and 4.99 on the screen)
are ratios of its 95th percentiles. The figures it draws are exploratory. The band convention is
nested quantile envelopes of the per-band residual across the test set,

    red    25-75          blue   5-25  and 75-95
    green  2.5-5 and 95-97.5     grey   0.5-2.5 and 97.5-99.5

with the median drawn as a thin black line.

Three columns per model:
  1. a component residual in physical units,
  2. the reflectance residual over all bands (what Table 3 scores),
  3. the reflectance residual restricted to the admissible domain of Proposition 1,
     t = Y2+Y3 >= tau and q = 1 - rho*Y4 >= kappa, with the retained coverage printed.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import h5py
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = r"C:/Users/owner/jpl_kernel_dnn_2026_07_16/data/jpl_reg_data"
PRED = r"C:/Users/owner/paper2_bands_20260921/preds"
FIG = r"C:/Users/owner/paper2_bands_20260921/figures"
os.makedirs(FIG, exist_ok=True)
COMPONENTS = ["Y1", "Y2", "Y3", "Y4"]
RHO = 0.7
SEED = int(os.environ.get("SEED", "101"))
TAU, KAPPA = 1e-12, 0.3

QS = [.005, .025, .05, .25, .5, .75, .95, .975, .995]
COLORS = ["gray", "green", "blue", "red"]

MODELS = [
    ("cubic_ridge", "cubic ridge"),
    ("fc_dnn_512", "network, $3\\times512$"),
    ("dnn_plus_residual_krr", "network + residual kernel"),
    ("krr_ard_matern", "input-scaled Matern kernel"),
    ("dkr_feature_kernel", "kernel on learned features"),
    ("convex_stack", "convex stack"),
]
SHOW_COMPONENT = "Y2"

Ys = {c: np.load(D + f"/{c}.npy") for c in COMPONENTS}
with h5py.File(D + "/data_EMIT_24k.jld2", "r") as f:
    wls = np.array(f["wls"])
te = np.load(PRED + f"/idx_test_seed{SEED}.npy")
Yt = {c: Ys[c][te] for c in COMPONENTS}

t_true = Yt["Y2"] + Yt["Y3"]
q_true = 1.0 - RHO * Yt["Y4"]
ADM = (t_true >= TAU) & (q_true >= KAPPA)
print(f"test block {Yt['Y1'].shape}, admissible coverage {100*ADM.mean():.4f}%")

L_true = Yt["Y1"] + RHO * t_true / q_true


def reflectance(P):
    u = L_true - P["Y1"]
    den = P["Y2"] + P["Y3"] + P["Y4"] * u
    with np.errstate(divide="ignore", invalid="ignore"):
        r = u / den
    nf = ~np.isfinite(r)
    return np.where(nf, 0.0, r), int(nf.sum())


def band(ax, res, x, title, ylim=None, mask=None):
    """res is (samples, bands); mask selects entries, NaN elsewhere."""
    A = res.astype(float).copy()
    if mask is not None:
        A[~mask] = np.nan
    with np.errstate(invalid="ignore"):
        q = np.nanquantile(A, QS, axis=0).T
    for i, col in enumerate(COLORS):
        ax.fill_between(x, q[:, i], q[:, i + 1], color=col, alpha=0.2, linewidth=0)
        ax.fill_between(x, q[:, -i - 2], q[:, -i - 1], color=col, alpha=0.2, linewidth=0)
    ax.plot(x, q[:, 4], color="k", lw=0.6)
    ax.axhline(0.0, color="k", lw=0.3, ls=":")
    ax.set_xlim(x.min(), x.max())
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=7.5)
    ax.tick_params(labelsize=7)


avail = [(k, lab) for k, lab in MODELS
         if os.path.exists(PRED + f"/{k}_Y1_te_seed{SEED}.npy")]
print("models found:", [k for k, _ in avail])
if not avail:
    raise SystemExit("no predictions yet")

def radiance_of(P):
    return P["Y1"] + RHO * (P["Y2"] + P["Y3"]) / (1.0 - RHO * P["Y4"])


# Fonts are set for the printed size: the figure is 18 in wide and is placed at a text
# width of about 6.3 in, so everything is scaled by roughly 0.35 on the page.
plt.rcParams.update({"font.size": 20, "axes.titlesize": 20, "axes.labelsize": 20,
                     "xtick.labelsize": 17, "ytick.labelsize": 17})

summary = {}
fig, axes = plt.subplots(len(avail), 4, figsize=(18, 3.1 * len(avail)), squeeze=False)
for r, (key, lab) in enumerate(avail):
    P = {c: np.load(PRED + f"/{key}_{c}_te_seed{SEED}.npy") for c in COMPONENTS}
    resid_c = Yt[SHOW_COMPONENT] - P[SHOW_COMPONENT]
    resid_L = L_true - radiance_of(P)
    rho_hat, n_nonfinite = reflectance(P)
    err = rho_hat - RHO

    band(axes[r][0], resid_c, wls, f"$Y_2$ residual, {lab}")
    band(axes[r][1], resid_L, wls,
         f"radiance residual at reflectance 0.7, {lab}")
    band(axes[r][2], err, wls, f"reflectance, all bands, {lab}",
         ylim=(-0.02, 0.02))
    band(axes[r][3], err, wls,
         f"reflectance on the admissible domain, {lab}",
         ylim=(-0.02, 0.02), mask=ADM)
    axes[r][0].set_ylabel("residual", fontsize=8)

    e_all = err.ravel()
    e_adm = err[ADM]
    rl = (np.linalg.norm(L_true - radiance_of(P), axis=1)
          / np.linalg.norm(L_true, axis=1)).mean()
    summary[key] = {
        "label": lab,
        "radiance_rel_l2_pct": float(100 * rl),
        "nonfinite_inversions": n_nonfinite,
        "all_bands": {
            "rmse": float(np.sqrt((e_all ** 2).mean())),
            "p95_abs_pct_points": float(100 * np.quantile(np.abs(e_all), 0.95)),
            "p99_abs_pct_points": float(100 * np.quantile(np.abs(e_all), 0.99)),
            "median_abs_pct_points": float(100 * np.median(np.abs(e_all))),
        },
        "admissible": {
            "coverage_pct": float(100 * ADM.mean()),
            "rmse": float(np.sqrt((e_adm ** 2).mean())),
            "p95_abs_pct_points": float(100 * np.quantile(np.abs(e_adm), 0.95)),
            "p99_abs_pct_points": float(100 * np.quantile(np.abs(e_adm), 0.99)),
            "median_abs_pct_points": float(100 * np.median(np.abs(e_adm))),
        },
    }
    print(f"{key:24s} all-band p95 {summary[key]['all_bands']['p95_abs_pct_points']:8.3f} pp"
          f"   admissible p95 {summary[key]['admissible']['p95_abs_pct_points']:8.4f} pp"
          f"   non-finite {n_nonfinite}")

for a in axes[-1]:
    a.set_xlabel("wavelength [nm]", fontsize=8)
fig.tight_layout()
fig.savefig(FIG + "/emit_quantiles_top_models.png", dpi=170)
plt.close(fig)

# a compact two-panel version for the manuscript body: the two extreme rows
best = min(summary, key=lambda k: summary[k]["all_bands"]["rmse"])
worst = max(summary, key=lambda k: summary[k]["all_bands"]["rmse"])
pair = [k for k in ["fc_dnn_512", "dnn_plus_residual_krr", "krr_ard_matern",
                    "dkr_feature_kernel"] if k in summary][:2]
if len(pair) == 2:
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.4), squeeze=False)
    for r, key in enumerate(pair):
        lab = dict(MODELS)[key]
        P = {c: np.load(PRED + f"/{key}_{c}_te_seed{SEED}.npy") for c in COMPONENTS}
        band(axes[r][0], Yt[SHOW_COMPONENT] - P[SHOW_COMPONENT], wls,
             f"${SHOW_COMPONENT[0]}_{SHOW_COMPONENT[1]}$ residual quantiles, {lab}")
        rho_hat, _ = reflectance(P)
        band(axes[r][1], rho_hat - RHO, wls,
             f"reflectance residual quantiles, {lab}", ylim=(-0.02, 0.02))
    for a in axes[-1]:
        a.set_xlabel("wavelength [nm]", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG + "/emit_quantiles_pair.png", dpi=170)
    plt.close(fig)

with open("development_split_tails.json", "w", encoding="utf-8") as f:
    json.dump({"seed": SEED, "tau": TAU, "kappa": KAPPA,
               "coverage_pct": float(100 * ADM.mean()), "models": summary}, f, indent=2)
print("wrote figures and development_split_tails.json")
