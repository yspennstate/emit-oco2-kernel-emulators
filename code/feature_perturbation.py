"""Perturbation experiment on the dumped last-layer features of the wide networks (theory v4, Proposition 7).

Reads a feature dump written by the patched campaign driver (feats/<tag>_<component>.npz: sub, idx_tr, idx_te,
Ztr_sub, pca_Vt, pca_center, ystd_mean, ystd_std, m<k>_tr, m<k>_te for every member k), rebuilds the reference
system as the exact Matern kernel on member 0's standardised features over the 4,000-row training subsample (the
record's dkr hyperparameters nu, scale x median heuristic, nugget gamma; the median recomputed on the subsample),
with M = 0, and for each perturbed system reports the signed spectrum of B, the exact identity, the residual-action
bound (b), the signed-spectrum bound (c), the ridge bound (d) where it applies and the v3 bound where delta < 1,
against the actual prediction movement on the test rows: (a) PCA-compressed features (training-only PCA, d'
ladder), (b) the concatenated features of all members, (c) the features of another member alone, (d) Nystrom on the
feature kernel, (e) a changed nugget. numpy only.
"""
import argparse, json, time, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from perturb_bounds import inv_sqrt_psd, perturbation_report, ridge_bound


def sqd(A, B):
    D2 = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * (A @ B.T)
    return np.maximum(D2, 0.0)


def matern(D2, ls, nu):
    r = np.sqrt(D2) / ls
    if nu == 1.5:
        a = np.sqrt(3.0) * r; return (1.0 + a) * np.exp(-a)
    a = np.sqrt(5.0) * r; return (1.0 + a + a * a / 3.0) * np.exp(-a)


def standardise(Ftr, *others):
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-9
    return [(F - mu) / sd for F in (Ftr,) + others]


def run(args):
    D = np.load(args.dump, allow_pickle=False)
    members = sorted({int(k[1:].split("_")[0]) for k in D.files if k.startswith("m") and k.endswith("_tr")})
    Z = D["Ztr_sub"].astype(np.float64); n = len(Z)
    hp = json.loads(Path(args.record).read_text(encoding="utf-8"))["hyper"]["dkr"][args.component] if args.record else {}
    nu, scale, gamma = float(hp.get("nu", 2.5)), float(hp.get("scale", 1.0)), float(hp.get("nugget", args.gamma))
    F0tr, F0te = D["m0_tr"].astype(np.float64), D["m0_te"].astype(np.float64)
    if args.n_query and args.n_query < len(F0te):
        F0te = F0te[:args.n_query]
    Xf, Xq = standardise(F0tr, F0te)
    D2 = sqd(Xf, Xf); med = float(np.sqrt(np.median(D2[np.triu_indices(n, 1)]))); ls = scale * med
    K = matern(D2, ls, nu); Kq = matern(sqd(Xq, Xf), ls, nu); lam = n * gamma
    H = K + lam * np.eye(n); Hm12 = inv_sqrt_psd(H)
    M = np.zeros_like(Z); p_ref = Kq @ np.linalg.solve(H, Z)
    out = dict(dump=str(args.dump), component=args.component, n=n, n_query=len(Xq), members=members, nu=nu, scale=scale,
               gamma=gamma, length_scale=ls, systems=[])
    systems = []
    U, Sv, Vt = np.linalg.svd(Xf - Xf.mean(0), full_matrices=False)
    for dprime in args.pca_ladder:
        if dprime >= Xf.shape[1]:
            continue
        P = Vt[:dprime].T; c = Xf.mean(0); Xc, Xqc = (Xf - c) @ P, (Xq - c) @ P
        D2c = sqd(Xc, Xc); medc = float(np.sqrt(np.median(D2c[np.triu_indices(n, 1)])))
        systems.append((f"pca_features_d{dprime}", matern(D2c, scale * medc, nu), matern(sqd(Xqc, Xc), scale * medc, nu), lam, None))
    if len(members) > 1:
        Ftr_all = np.concatenate([D[f"m{m}_tr"].astype(np.float64) for m in members], axis=1)
        Fte_all = np.concatenate([D[f"m{m}_te"].astype(np.float64)[:len(Xq)] for m in members], axis=1)
        Xa, Xqa = standardise(Ftr_all, Fte_all)
        D2a = sqd(Xa, Xa); meda = float(np.sqrt(np.median(D2a[np.triu_indices(n, 1)])))
        systems.append(("concat_all_members", matern(D2a, scale * meda, nu), matern(sqd(Xqa, Xa), scale * meda, nu), lam, None))
        for m in members[1:]:
            Xm, Xqm = standardise(D[f"m{m}_tr"].astype(np.float64), D[f"m{m}_te"].astype(np.float64)[:len(Xq)])
            D2m = sqd(Xm, Xm); medm = float(np.sqrt(np.median(D2m[np.triu_indices(n, 1)])))
            systems.append((f"member{m}_alone", matern(D2m, scale * medm, nu), matern(sqd(Xqm, Xm), scale * medm, nu), lam, None))
    rng = np.random.RandomState(7)
    for m_ in args.landmarks:
        if m_ >= n:
            continue
        lm = rng.permutation(n)[:m_]; Knm = K[:, lm]; Kmm = K[np.ix_(lm, lm)] + 1e-10 * np.eye(m_)
        systems.append((f"nystrom_m{m_}", Knm @ np.linalg.solve(Kmm, Knm.T), Kq[:, lm] @ np.linalg.solve(Kmm, Knm.T), lam, None))
    for f in (0.5, 2.0, 10.0, 100.0):
        systems.append((f"gamma_x{f}", K, Kq, lam * f, lam * (f - 1.0)))
    for name, Kt, Kqt, lam_t, s_ridge in systems:
        Ht = Kt + lam_t * np.eye(n)
        try:
            p_t = Kqt @ np.linalg.solve(Ht, Z)
        except np.linalg.LinAlgError:
            continue
        rec = dict(system=name)
        rec.update(perturbation_report(H, Hm12, Ht, Kq, Kqt, Z, M, M, p_ref, p_t))
        if s_ridge is not None and s_ridge > 0:
            rec.update(ridge_bound(H, Hm12, Kq, Z, M, s_ridge, p_ref, p_t))
        out["systems"].append(rec)
        print(f"{name:22s} lam_min(B)={rec['lambda_min_B']:+.3f} delta={rec['delta']:.3f} move_med={rec['median_move']:.3e} "
              f"b_ratio={rec.get('bound_b_median_ratio')} c_ratio={rec.get('bound_c_median_ratio')} v3_ratio={rec.get('bound_v3_median_ratio')} "
              f"holds b/c {rec.get('bound_b_holds')}/{rec.get('bound_c_holds')}", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump", required=True); ap.add_argument("--record", default=None); ap.add_argument("--component", default="Y2")
    ap.add_argument("--gamma", type=float, default=1e-4); ap.add_argument("--n-query", type=int, default=1000)
    ap.add_argument("--pca-ladder", type=int, nargs="+", default=[1000, 500, 250, 125, 64, 32, 16])
    ap.add_argument("--landmarks", type=int, nargs="+", default=[100, 400, 1600, 3200])
    ap.add_argument("--output", required=True)
    a = ap.parse_args(); t0 = time.time()
    res = run(a); res["seconds"] = round(time.time() - t0, 1)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.output, "w"), indent=1); print("wrote", a.output, "in", res["seconds"], "s")
