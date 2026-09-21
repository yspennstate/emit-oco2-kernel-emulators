"""Kernel-approximation perturbation experiment (theory v4, Proposition 7) on EMIT state inputs.

Reference system: exact Matern-5/2 ridge regression on n standardized state inputs (the first rows of the seed's
training block) with the median-heuristic length scale and nugget gamma (lambda = n gamma), targets the PCA-64
coordinates of the standardized component, mean M = 0 (the kernel-alone branch) or M = a ridge readout on the same
inputs (the corrected branch). Perturbed systems: Nystrom approximations of the same kernel, the same kernel at a
changed nugget, the kernel on PCA-compressed inputs. For each: the signed spectrum of B (lambda_min, delta), the
exact identity (a) checked, the residual-action bound (b), the signed-spectrum bound (c), the ridge bound (d) where
it applies, the v3 bound where delta < 1, each against the actual movement on held-out rows. numpy only.
"""
import argparse, json, time, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from perturb_bounds import inv_sqrt_psd, perturbation_report, ridge_bound

COMPONENTS = ("Y1", "Y2", "Y3", "Y4")


def split(n, seed):
    perm = np.random.RandomState(seed).permutation(n)
    n_te = int(round(0.1 * n)); idx_te, tr_full = perm[:n_te], perm[n_te:]
    vp = np.random.RandomState(seed + 10000).permutation(len(tr_full))
    n_val = int(round(0.1 * len(tr_full)))
    return idx_te, tr_full[vp[:n_val]], tr_full[vp[n_val:]]


def sqd(A, B):
    D2 = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * (A @ B.T)
    return np.maximum(D2, 0.0)


def matern52(D2, ls):
    r = np.sqrt(D2) / ls; a = np.sqrt(5.0) * r
    return (1.0 + a + a * a / 3.0) * np.exp(-a)


def run(args):
    D = Path(args.data_dir)
    X = np.load(D / "X.npy"); Ys = {c: np.load(D / (c + ".npy")) for c in COMPONENTS}
    idx_te, idx_val, idx_tr = split(len(X), args.seed)
    idx_fit, idx_q = idx_tr[:args.n], idx_te[:args.n_query]
    mu, sd = X[idx_fit].mean(0), X[idx_fit].std(0) + 1e-9
    Xf, Xq = (X[idx_fit] - mu) / sd, (X[idx_q] - mu) / sd
    Y = Ys[args.component][idx_fit]; ym, ysd = Y.mean(0), np.sqrt(Y.var(0)); ysd[ysd == 0] = 1
    Ystd = (Y - ym) / ysd; center = Ystd.mean(0)
    U, Sv, Vt = np.linalg.svd(Ystd - center, full_matrices=False); Vt = Vt[:64]
    Z = (Ystd - center) @ Vt.T
    Zq = ((Ys[args.component][idx_q] - ym) / ysd - center) @ Vt.T
    n = len(Xf)
    D2 = sqd(Xf, Xf); med = float(np.sqrt(np.median(D2[np.triu_indices(n, 1)]))); ls = args.scale * med
    K = matern52(D2, ls); Kq = matern52(sqd(Xq, Xf), ls)
    out = dict(seed=args.seed, n=n, n_query=len(Xq), component=args.component, length_scale=ls, gamma=args.gamma, systems=[])
    relerr = lambda p: float(np.mean(np.linalg.norm(p - Zq, axis=1) / np.linalg.norm(Zq, axis=1)))
    for mean_kind in ("zero", "ridge"):
        if mean_kind == "zero":
            M, Mq = np.zeros_like(Z), np.zeros((len(Xq), Z.shape[1]))
        else:
            A = np.hstack([Xf, np.ones((n, 1))]); Aq = np.hstack([Xq, np.ones((len(Xq), 1))])
            Wr = np.linalg.solve(A.T @ A + 1e-3 * np.eye(A.shape[1]), A.T @ Z)
            M, Mq = A @ Wr, Aq @ Wr
        lam = n * args.gamma
        H = K + lam * np.eye(n); Hm12 = inv_sqrt_psd(H)
        p_ref = Mq + Kq @ np.linalg.solve(H, Z - M)
        systems = []
        rng = np.random.RandomState(args.seed + 7)
        for m in args.landmarks:
            lm = rng.permutation(n)[:m]; Knm = K[:, lm]; Kmm = K[np.ix_(lm, lm)] + 1e-10 * np.eye(m)
            systems.append((f"nystrom_m{m}", Knm @ np.linalg.solve(Kmm, Knm.T), Kq[:, lm] @ np.linalg.solve(Kmm, Knm.T), lam, None))
        for f in (0.5, 2.0, 10.0, 100.0):
            systems.append((f"gamma_x{f}", K, Kq, lam * f, lam * (f - 1.0)))
        Ux, Sx, Vtx = np.linalg.svd(Xf - Xf.mean(0), full_matrices=False)
        for dprime in (2, 3, 4, 5):
            P = Vtx[:dprime].T; Xc, Xqc = (Xf - Xf.mean(0)) @ P, (Xq - Xf.mean(0)) @ P
            systems.append((f"pca_inputs_d{dprime}", matern52(sqd(Xc, Xc), ls), matern52(sqd(Xqc, Xc), ls), lam, None))
        for name, Kt, Kqt, lam_t, s_ridge in systems:
            Ht = Kt + lam_t * np.eye(n)
            try:
                p_t = Mq + Kqt @ np.linalg.solve(Ht, Z - M)
            except np.linalg.LinAlgError:
                continue
            rec = dict(system=name, mean=mean_kind, ref_rel_err=relerr(p_ref), pert_rel_err=relerr(p_t))
            rec.update(perturbation_report(H, Hm12, Ht, Kq, Kqt, Z, M, M, p_ref, p_t))
            if s_ridge is not None and s_ridge > 0:
                rec.update(ridge_bound(H, Hm12, Kq, Z, M, s_ridge, p_ref, p_t))
            out["systems"].append(rec)
            print(f"{mean_kind:5s} {name:18s} lam_min(B)={rec['lambda_min_B']:+.3f} delta={rec['delta']:.3f} move_med={rec['median_move']:.3e} "
                  f"b_ratio={rec.get('bound_b_median_ratio')} c_ratio={rec.get('bound_c_median_ratio')} v3_ratio={rec.get('bound_v3_median_ratio')} "
                  f"d_ratio={rec.get('bound_d_median_ratio')} holds b/c {rec.get('bound_b_holds')}/{rec.get('bound_c_holds')} ident {rec.get('identity_max_rel_err')}", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True); ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--n", type=int, default=2000); ap.add_argument("--n-query", type=int, default=1000)
    ap.add_argument("--component", default="Y2"); ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=1e-4)
    ap.add_argument("--landmarks", type=int, nargs="+", default=[50, 100, 200, 400, 800, 1600])
    ap.add_argument("--output", required=True)
    a = ap.parse_args(); t0 = time.time()
    res = run(a); res["seconds"] = round(time.time() - t0, 1)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.output, "w"), indent=1); print("wrote", a.output, "in", res["seconds"], "s")
