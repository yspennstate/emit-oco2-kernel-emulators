"""Ridge-path and noise diagnostics for the residual correction (plan v4, block E6; Proposition 2 and the power function).

Reads a feature-and-prediction dump of the wide network at one seed (feats/<tag>_<component>.npz written by the
patched campaign driver: sub, idx_tr, idx_te, Ztr_sub, pca_Vt, pca_center, ystd_mean, ystd_std, xstd_mean,
xstd_std, m0_tr / m0_te features, m0_ptr / m0_pte network outputs in standardised band space) and the data
directory, and reports on the n-row training subsample, for the kernel on the state inputs and the kernel on the
network's features (Matern-5/2, median length scale, the record's scale multiplier):

  the residual r = z - h_X in the Gram eigenbasis: its energy by eigendirection and the cumulative fraction, against
  the target's own energy (the alignment picture of Proposition 1);
  along a ridge ladder lambda = n gamma: the effective degrees of freedom tr S_lambda, the bias term
  (1/n) sum_j (lambda/(mu_j + lambda))^2 r_j^2, the noise factor (1/n) sum_j (mu_j/(mu_j + lambda))^2 of
  Proposition 2(iii), and on the test rows the out-of-sample noise factor ||a_lambda(x)||^2 and the nugget-corrected
  power function P~_lambda(x) of the kernel paper (median and 95th percentile);
  the split-sample check: the correction fitted on half the subsample and read on the other half and on the test
  rows, beside the same-sample fit, along the ladder, in the PCA-space relative error and in physical relative L2;
  the test error of the network alone, of the corrected network and of the kernel alone along the ladder.
numpy only.
"""
import argparse, json, time
from pathlib import Path
import numpy as np

COMPONENTS = ("Y1", "Y2", "Y3", "Y4")


def sqd(A, B):
    D2 = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * (A @ B.T)
    return np.maximum(D2, 0.0)


def matern(D2, ls, nu=2.5):
    r = np.sqrt(D2) / ls
    if nu == 1.5:
        a = np.sqrt(3.0) * r; return (1.0 + a) * np.exp(-a)
    a = np.sqrt(5.0) * r; return (1.0 + a + a * a / 3.0) * np.exp(-a)


def standardise(Ftr, *others):
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-9
    return [(F - mu) / sd for F in (Ftr,) + others]


def rel_l2(Yt, Yp):
    return float(np.mean(np.linalg.norm(Yt - Yp, axis=1) / np.linalg.norm(Yt, axis=1)))


def run(args):
    D = np.load(args.dump, allow_pickle=False)
    data = Path(args.data_dir)
    c = args.component
    sub, idx_tr, idx_te = D["sub"], D["idx_tr"], D["idx_te"]
    Vt, center, ym, ysd = D["pca_Vt"].astype(np.float64), D["pca_center"].astype(np.float64), D["ystd_mean"].astype(np.float64), D["ystd_std"].astype(np.float64)
    xm, xs = D["xstd_mean"].astype(np.float64), D["xstd_std"].astype(np.float64)
    X = np.load(data / "X.npy"); Y = np.load(data / (c + ".npy"))
    Xtr = (X[idx_tr][sub] - xm) / xs; Xte = (X[idx_te] - xm) / xs
    if args.n_query and args.n_query < len(Xte):
        Xte = Xte[:args.n_query]; idx_q = idx_te[:args.n_query]
    else:
        idx_q = idx_te
    Ztr = D["Ztr_sub"].astype(np.float64)
    Zte = ((Y[idx_q] - ym) / ysd - center) @ Vt.T
    Ptr = (D["m0_ptr"].astype(np.float64) - center) @ Vt.T           # network outputs, projected to the PCA coordinates
    Pte = (D["m0_pte"].astype(np.float64)[:len(idx_q)] - center) @ Vt.T
    phys = lambda Z: (Z @ Vt + center) * ysd + ym
    Yte = Y[idx_q]
    n = len(Ztr); r_tr = Ztr - Ptr
    hp = json.loads(Path(args.record).read_text(encoding="utf-8"))["hyper"] if args.record else {}
    out = dict(dump=str(args.dump), component=c, n=n, n_query=len(idx_q), network_test_rel_l2_phys=rel_l2(Yte, phys(Pte)), kernels={})
    gammas = [float(g) for g in args.gammas.split(",")]
    for kname in ("input", "feature"):
        if kname == "input":
            Ftr, Fte = Xtr, Xte; scale = float(hp.get("dnn_corr", {}).get(c, {}).get("scale", 1.0)); nu = float(hp.get("dnn_corr", {}).get(c, {}).get("nu", 2.5))
        else:
            Ftr, Fte = standardise(D["m0_tr"].astype(np.float64), D["m0_te"].astype(np.float64)[:len(idx_q)])
            scale = float(hp.get("dkr", {}).get(c, {}).get("scale", 1.0)); nu = float(hp.get("dkr", {}).get(c, {}).get("nu", 2.5))
        D2 = sqd(Ftr, Ftr); med = float(np.sqrt(np.median(D2[np.triu_indices(n, 1)]))); ls = scale * med
        K = matern(D2, ls, nu); Kq = matern(sqd(Fte, Ftr), ls, nu); kxx = 1.0
        mu, U = np.linalg.eigh(K); order = np.argsort(mu)[::-1]; mu, U = np.maximum(mu[order], 0.0), U[:, order]
        Rj = np.sum((U.T @ r_tr) ** 2, axis=1); Zj = np.sum((U.T @ Ztr) ** 2, axis=1)      # energies by eigendirection
        cum_r = np.cumsum(Rj) / Rj.sum(); cum_z = np.cumsum(Zj) / Zj.sum()
        rec = dict(length_scale=ls, nu=nu, scale=scale, eig_top=[float(x) for x in mu[:10]], eig_median=float(np.median(mu)),
                   residual_energy_fraction_first_k={str(k): float(cum_r[k - 1]) for k in (10, 50, 100, 500, 1000, 2000) if k <= n},
                   target_energy_fraction_first_k={str(k): float(cum_z[k - 1]) for k in (10, 50, 100, 500, 1000, 2000) if k <= n},
                   residual_over_target_energy=float(Rj.sum() / Zj.sum()), ladder=[])
        half = np.random.RandomState(7).permutation(n); A, B = np.sort(half[: n // 2]), np.sort(half[n // 2:])
        KA = K[np.ix_(A, A)]; KBA = K[np.ix_(B, A)]; KqA = Kq[:, A]
        for g in gammas:
            lam = n * g
            shrink = mu / (mu + lam)
            dof = float(shrink.sum()); bias = float(np.sum((lam / (mu + lam)) ** 2 * Rj) / n); noise = float(np.sum(shrink ** 2) / n)
            Hinv_kq = np.linalg.solve(K + lam * np.eye(n), Kq.T)                   # n x q: a_lambda(x) columns
            a2 = np.sum(Hinv_kq ** 2, axis=0)
            pf2 = kxx - np.sum(Kq.T * Hinv_kq, axis=0) - lam * a2                   # nugget-corrected power function squared
            alpha = np.linalg.solve(K + lam * np.eye(n), r_tr)
            corr_te = Pte + Kq @ alpha
            kern_te = Kq @ np.linalg.solve(K + lam * np.eye(n), Ztr)
            # split sample: correction fitted on half A, read on half B and on test
            lamA = len(A) * g
            alphaA = np.linalg.solve(KA + lamA * np.eye(len(A)), r_tr[A])
            corrB = Ptr[B] + KBA @ alphaA; corr_teA = Pte + KqA @ alphaA
            alphaS = np.linalg.solve(K + lam * np.eye(n), r_tr)                    # same-sample fit read on the same half B
            corrB_same = Ptr[B] + K[np.ix_(B, np.arange(n))] @ alphaS
            rec["ladder"].append(dict(gamma=g, lam=lam, dof=dof, bias_term=bias, noise_factor=noise,
                                      a2_median=float(np.median(a2)), a2_p95=float(np.quantile(a2, 0.95)),
                                      power_fn_median=float(np.median(np.sqrt(np.maximum(pf2, 0.0)))), power_fn_p95=float(np.quantile(np.sqrt(np.maximum(pf2, 0.0)), 0.95)),
                                      corrected_test_rel_pca=rel_l2(Zte, corr_te), corrected_test_rel_phys=rel_l2(Yte, phys(corr_te)),
                                      kernel_alone_test_rel_phys=rel_l2(Yte, phys(kern_te)),
                                      split_fit_halfB_rel_pca=rel_l2(Ztr[B], corrB), same_fit_halfB_rel_pca=rel_l2(Ztr[B], corrB_same),
                                      split_fit_test_rel_phys=rel_l2(Yte, phys(corr_teA))))
            print(f"{c} {kname:7s} gamma={g:g} dof={dof:.0f} bias={bias:.3g} noise={noise:.3g} a2_med={np.median(a2):.3g} pf_med={np.median(np.sqrt(np.maximum(pf2,0))):.3g} "
                  f"corr_test_phys={rec['ladder'][-1]['corrected_test_rel_phys']:.5f} kernel_alone={rec['ladder'][-1]['kernel_alone_test_rel_phys']:.5f} "
                  f"halfB same/split {rec['ladder'][-1]['same_fit_halfB_rel_pca']:.4f}/{rec['ladder'][-1]['split_fit_halfB_rel_pca']:.4f}", flush=True)
        out["kernels"][kname] = rec
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump", required=True); ap.add_argument("--data-dir", required=True); ap.add_argument("--component", default="Y2")
    ap.add_argument("--record", default=None); ap.add_argument("--n-query", type=int, default=1000)
    ap.add_argument("--gammas", default="1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1")
    ap.add_argument("--output", required=True)
    a = ap.parse_args(); t0 = time.time()
    res = run(a); res["seconds"] = round(time.time() - t0, 1)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.output, "w"), indent=1); print("wrote", a.output, "in", res["seconds"], "s")
