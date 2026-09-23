"""The figures drawn from the EMIT arrays themselves.

figures/emit_data_examples.png  rows 0, 40, 80, 120 and 160 of each component against wavelength;
figures/emit_structure.png      explained-variance spectra of the standardized outputs, and the drop in validation R^2
                                of a linear ridge fit when one standardized input is shuffled;
figures/emit_band_anatomy.png   band RMSE over band RMS of the test predictions of the isotropic Matern kernel, the
                                3x512 network and the network with its residual kernel, drawn when --predictions names
                                a folder with <family>_<component>_te.npy for krr_matern, mlp512 and
                                mlp512_plus_resid_krr.
Also writes results/emit_structure.json with the numbers behind the second figure and the structure paragraph of
Section 3: per component, the number of principal components for 99.99% and 99.9999% of the training variance, the
variance left after 64, and the median and smallest correlation between adjacent bands with the bands of the smallest.

The split is the one of the JPL notebook: scikit-learn's train_test_split(test_size=0.1, random_state=42) on row
order, then 10% of the training block, chosen by RandomState(0), for validation. Needs numpy, scikit-learn and
matplotlib, and the arrays X.npy and Y1.npy-Y4.npy in EMIT_DATA.

usage: EMIT_DATA=/path/to/emit python code/make_data_figures.py [--predictions DIR]
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(ROOT, "figures")
COMPONENTS = ["Y1", "Y2", "Y3", "Y4"]


class Standardizer:
    def __init__(self, A):
        self.mean = A.mean(axis=0)
        self.std = np.sqrt(A.var(axis=0))
        self.std[self.std == 0] = 1.0

    def fwd(self, A):
        return (A - self.mean) / self.std


def split_indices(n, test_size=0.1, seed=42, val_frac=0.1, val_seed=0):
    idx = np.arange(n)
    idx_tr_full, idx_te = train_test_split(idx, test_size=test_size, random_state=seed)
    perm = np.random.RandomState(val_seed).permutation(len(idx_tr_full))
    n_val = int(round(val_frac * len(idx_tr_full)))
    return idx_tr_full[perm[n_val:]], idx_tr_full[perm[:n_val]], idx_te, idx_tr_full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", help="folder of the three families' test predictions (band-anatomy figure)")
    args = ap.parse_args()
    data = os.environ["EMIT_DATA"]
    X = np.load(os.path.join(data, "X.npy"))
    Ys = {c: np.load(os.path.join(data, c + ".npy")) for c in COMPONENTS}
    with open(os.path.join(ROOT, "results", "emit_wavelengths.json"), encoding="utf-8") as f:
        wls = np.asarray(json.load(f)["nm"], float)
    n, d = X.shape
    idx_tr, idx_val, idx_te, idx_tr_full = split_indices(n)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, c in zip(axes.ravel(), COMPONENTS):
        for i in range(0, 200, 40):
            ax.plot(wls, Ys[c][i], lw=0.8)
        ax.set_title(c)
        ax.set_xlabel("wavelength [nm]")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "emit_data_examples.png"), dpi=140)
    plt.close(fig)

    out = {"output_pca": {}}
    for c in COMPONENTS:
        A = Standardizer(Ys[c][idx_tr_full]).fwd(Ys[c][idx_tr_full])
        S = np.linalg.svd(A - A.mean(axis=0), compute_uv=False)
        ev = S ** 2 / (S ** 2).sum()
        cum = np.cumsum(ev)
        # A has mean zero and unit variance per band, so these means are the Pearson correlations
        adj = (A[:, :-1] * A[:, 1:]).mean(axis=0)
        k = int(np.argmin(adj))
        out["output_pca"][c] = {"rank_9999": int(np.searchsorted(cum, 0.9999) + 1),
                                "rank_999999": int(np.searchsorted(cum, 0.999999) + 1),
                                "unexplained_after_64": float(1.0 - cum[63]),
                                "top10_evr": [float(v) for v in ev[:10]],
                                "adjacent_band_corr_median": float(np.median(adj)),
                                "adjacent_band_corr_min": float(adj[k]),
                                "adjacent_band_corr_min_nm": [float(wls[k]), float(wls[k + 1])]}
    # the shuffle test reads validation rows only
    Xs = Standardizer(X[idx_tr]).fwd(X)
    Ycat = np.concatenate([Standardizer(Ys[c][idx_tr]).fwd(Ys[c]) for c in COMPONENTS], axis=1)
    r = Ridge(alpha=1.0).fit(Xs[idx_tr], Ycat[idx_tr])
    base = r.score(Xs[idx_val], Ycat[idx_val])
    rng = np.random.RandomState(0)
    drop = {}
    for j in range(d):
        Xp = Xs[idx_val].copy()
        Xp[:, j] = Xp[rng.permutation(len(idx_val)), j]
        drop[j] = float(base - r.score(Xp, Ycat[idx_val]))
    out["linear_r2"] = float(base)
    out["linear_relevance_drop"] = drop
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.4))
    for c in COMPONENTS:
        a1.semilogy(np.arange(1, 11), out["output_pca"][c]["top10_evr"], marker="o", ms=3, label=c)
    a1.set_xlabel("principal component")
    a1.set_ylabel("explained variance fraction")
    a1.legend(fontsize=8)
    a1.grid(alpha=0.3)
    names = ["AOD", "elev.", "H$_2$O", "rel. azim.", "solar zen.", "view zen."]
    a2.bar(range(6), [drop[j] for j in range(6)])
    a2.set_xticks(range(6))
    a2.set_xticklabels(names, fontsize=8)
    a2.set_ylabel("linear $R^2$ drop when shuffled")
    a2.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "emit_structure.png"), dpi=170)
    plt.close(fig)
    with open(os.path.join(ROOT, "results", "emit_structure.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=2)

    if args.predictions:
        models = [("krr_matern", "Matérn KRR", "tab:blue"), ("mlp512", "FC-DNN", "tab:orange"),
                  ("mlp512_plus_resid_krr", "DNN + residual KRR", "tab:green")]
        fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True)
        for ax, c in zip(axes.ravel(), COMPONENTS):
            T = Ys[c][idx_te]
            scale = np.sqrt(np.mean(T ** 2, axis=0)) + 1e-12
            for key, lab, col in models:
                P = np.load(os.path.join(args.predictions, f"{key}_{c}_te.npy"))
                ax.semilogy(wls, np.sqrt(np.mean((T - P) ** 2, axis=0)) / scale, lw=0.9, label=lab, color=col)
            ax.set_title(c, fontsize=10)
            ax.grid(alpha=0.25)
        axes[1][0].set_xlabel("wavelength [nm]")
        axes[1][1].set_xlabel("wavelength [nm]")
        axes[0][0].set_ylabel("band RMSE / band RMS")
        axes[1][0].set_ylabel("band RMSE / band RMS")
        axes[0][0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGS, "emit_band_anatomy.png"), dpi=170)
        plt.close(fig)
    print(json.dumps({"rank_9999": {c: out["output_pca"][c]["rank_9999"] for c in COMPONENTS},
                      "linear_r2": round(out["linear_r2"], 4)}))


if __name__ == "__main__":
    main()
