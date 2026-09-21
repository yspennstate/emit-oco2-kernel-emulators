"""Retrieval-weighted combinations fitted on validation predictions, scored on test.

Historical fitting contract (see paper/sec_theory_new.tex):
* Clip each base Y4 to [0,S] before fitting; leave Y2 and Y3 unprojected.
* The finite-threshold ``theorem`` arm fits J'_tau, the separated raw-error
  quadratic, NOT the total-flux J_tau. Its deterministic transfer factor is four
  on the theorem's physical domain. Total-flux projection occurs only in scoring.
* ``theorem`` with tau=inf is a distinct unweighted-component baseline.
* ``paper`` fits row-relative squared component errors; ``paper_norm`` uses six
  IRLS sweeps for mean row-relative norms. Neither is an exact retrieval objective.
* simplex_weights uses equality-penalized NNLS then renormalization. Feasible
  weights do not establish exact optimality of the equality-constrained QP.

Raw and constrained retrieval statistics are reported separately. Legacy all-band
statistics may include entries outside the theorem's hypotheses. Reusing validation
rows for earlier model selection does not meet independent-selection assumptions.
The revision adds objective/precision metadata without changing the weight algorithm.
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import conditioned_reflectance as cr
from transmission_conditioned import split, digest, retrieval
from stacking_contract import component_weights, require_scoring_precision, physical_domain

COMPONENTS = ("Y1", "Y2", "Y3", "Y4")
HEADS = ("krr", "dnn", "dnn_ens", "dnn_corr", "ens_corr", "dkr", "dkr_cat")


def nnls(A, b, iters=3000):
    m, n = A.shape
    x = np.zeros(n); P = np.zeros(n, dtype=bool)
    w = A.T @ (b - A @ x)
    tol = 1e-12 * max(1.0, float(np.abs(w).max()))
    it = 0
    while (~P).any() and (w[~P] > tol).any() and it < iters:
        it += 1
        j = int(np.argmax(np.where(P, -np.inf, w))); P[j] = True
        while True:
            s = np.zeros(n)
            s[P] = np.linalg.lstsq(A[:, P], b, rcond=None)[0]
            if (s[P] > 0).all():
                x = s; break
            neg = P & (s <= 0)
            alpha = np.min(x[neg] / (x[neg] - s[neg]))
            x = x + alpha * (s - x)
            P &= x > 1e-15
            x[~P] = 0.0
        w = A.T @ (b - A @ x)
    return x


def simplex_weights(Ph, y, wts):
    """w >= 0, sum w = 1, approximately fitting weighted squared error; the equality constraint is a penalty row, then weights are normalized."""
    sq = np.sqrt(wts)[:, None]
    big = 1e3 * np.sqrt(np.mean(wts)) * np.sqrt(len(wts))
    A = np.vstack([Ph * sq, big * np.ones((1, Ph.shape[1]))])
    b = np.concatenate([y * sq[:, 0], [big]])
    w = nnls(A, b)
    return w / w.sum() if w.sum() > 0 else np.full(Ph.shape[1], 1.0 / Ph.shape[1])


def rel_l2(Yt, Yp):
    return float(np.mean(np.linalg.norm(Yt - Yp, axis=1) / np.linalg.norm(Yt, axis=1)))


def radiance(Ys, rho):
    return Ys["Y1"] + rho * (Ys["Y2"] + Ys["Y3"]) / (1.0 - rho * Ys["Y4"])


def score(truth, pred, scale, S, R, rhos, thresholds):
    a, t, s = cr.components(truth); ah, th, sh = cr.components(pred)
    eR = np.abs(ah - a) + R * np.abs(np.maximum(th, 0.0) - t) + R * R * t * np.abs(np.clip(sh, 0.0, S) - s)
    out = dict(rel_l2={c: rel_l2(truth[c], pred[c]) for c in COMPONENTS}, epsR=float(np.sqrt(np.mean(eR ** 2))), rhos={}, state_loss={})
    out["rel_l2_radiance_0.7"] = rel_l2(radiance(truth, 0.7), radiance(pred, 0.7))
    for rho in rhos:
        r = retrieval(a, t, s, ah, th, sh, rho, R, S)
        raw_err = np.where(r["raw_finite"], r["raw"], 0.0) - rho
        bar_err = r["bar"] - rho
        out["rhos"][str(rho)] = dict(
            raw_p95=float(np.quantile(np.abs(raw_err), 0.95)), raw_p99=float(np.quantile(np.abs(raw_err), 0.99)),
            raw_rmse=float(np.sqrt(np.mean(raw_err ** 2))), raw_nonpositive_denominator_fraction=float(np.mean(~r["raw_posden"])),
            constrained_median=float(np.median(np.abs(bar_err))), constrained_p95=float(np.quantile(np.abs(bar_err), 0.95)),
            constrained_p99=float(np.quantile(np.abs(bar_err), 0.99)), constrained_rmse=float(np.sqrt(np.mean(bar_err ** 2))),
            constrained_failure_fraction=float(np.mean(r["bar_fail"])), clipped_at_zero_fraction=float(np.mean(r["clip_lo"])),
            clipped_at_R_fraction=float(np.mean(r["clip_hi"])),
            paper_conditioned={str(th_): {k: v for k, v in cr.evaluate(truth, pred, flux_scale=scale, threshold=th_, rho=rho).items()
                                         if k in ("retained_fraction_all_entries", "failure_fraction_on_mask", "median_absolute_error", "p95_absolute_error", "rmse")}
                               for th_ in thresholds})
        out["state_loss"][str(rho)] = np.mean(bar_err ** 2, axis=1).tolist()      # per state, mean over bands: the unit of Proposition 3
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True, type=Path); ap.add_argument("--predictions", required=True, type=Path)
    ap.add_argument("--record", required=True, type=Path)
    ap.add_argument("--taus", type=float, nargs="+", default=[1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, float("inf")])
    ap.add_argument("--rhos", type=float, nargs="+", default=[0.3, 0.7, 0.9])
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.0, 0.01, 0.1])
    ap.add_argument("--heads", nargs="*", default=None); ap.add_argument("--allow-float32", action="store_true")
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    record = json.loads(args.record.read_text(encoding="utf-8"))
    for c in ("X",) + COMPONENTS:
        if digest(args.data_dir / f"{c}.npy") != record["data_sha"][c]:
            raise ValueError(f"data digest mismatch for {c}")
    ys = {c: np.load(args.data_dir / f"{c}.npy", allow_pickle=False) for c in COMPONENTS}
    idx_te, idx_val, idx_tr = split(len(ys["Y1"]), int(record["seed"]), int(record.get("ntrain") or 0))
    truth_va = {c: ys[c][idx_val] for c in COMPONENTS}; truth_te = {c: ys[c][idx_te] for c in COMPONENTS}
    scale = cr.training_flux_scale({c: ys[c][idx_tr] for c in COMPONENTS})
    _, t_va, s_va = cr.components(truth_va); _, t_te, s_te = cr.components(truth_te)
    s_tr = ys["Y4"][idx_tr]; S = float(s_tr[np.isfinite(s_tr) & (s_tr <= 1.0)].max()); S_used = S; R = float(max(args.rhos))   # physical albedo bound from training rows
    out = dict(record_sha256=digest(args.record), prediction_sha256=digest(args.predictions), seed=int(record["seed"]),
               ntrain=int(len(idx_tr)), n_val=int(len(idx_val)), n_test=int(len(idx_te)), flux_scale=scale, S_training_max=S, S_used=S_used, R=R,
               heads=[], singles={}, arms={})
    with np.load(args.predictions, allow_pickle=False) as P:
        if not (np.array_equal(P["idx_te"], idx_te) and np.array_equal(P["idx_val"], idx_val)):
            raise ValueError("prediction indices do not match the recorded split")
        heads = [h for h in (args.heads or HEADS) if all(f"val_{h}_{c}" in P.files and f"{h}_{c}" in P.files for c in COMPONENTS)]
        if len(heads) < 2:
            raise ValueError(f"need at least two heads with validation twins; found {heads}")
        require_scoring_precision(
            {key: P[key] for h in heads for c in COMPONENTS
             for key in (f"val_{h}_{c}", f"{h}_{c}")},
            allow_precision_altered=args.allow_float32)
        out["schema_version"] = 2
        out["precision_altered_opt_in"] = bool(args.allow_float32)
        out["prediction_dtypes"] = {
            key: str(P[key].dtype) for h in heads for c in COMPONENTS
            for key in (f"val_{h}_{c}", f"{h}_{c}")}
        out["fit_contract"] = {
            "finite_threshold": "separated raw-component quadratic J_prime_tau",
            "projection": "base albedo before fit; total flux only during retrieval scoring",
            "solver": "equality-penalized NNLS followed by simplex renormalization",
            "mean_norm_solver": "six fixed IRLS sweeps; no convergence certificate",
            "infinite_threshold": "unweighted component square; not a threshold limit",
            "theorem_scope": "requires full physical domain; all-band metrics are descriptive"}
        out["physical_domain_entries_by_rho"] = {
            str(rho): int(physical_domain(t_te, s_te, rho=rho, R=R, S=S_used).sum())
            for rho in args.rhos}
        out["heads"] = heads
        VA = {c: {h: np.asarray(P[f"val_{h}_{c}"], dtype=np.float64) for h in heads} for c in COMPONENTS}
        TE = {c: {h: np.asarray(P[f"{h}_{c}"], dtype=np.float64) for h in heads} for c in COMPONENTS}
    # Only albedo heads are projected before the affine quadratic fit. The total-flux floor is downstream scoring.
    for h in heads:
        VA["Y4"][h] = np.clip(VA["Y4"][h], 0.0, S_used); TE["Y4"][h] = np.clip(TE["Y4"][h], 0.0, S_used)
    for h in heads:
        out["singles"][h] = score(truth_te, {c: TE[c][h] for c in COMPONENTS}, scale, S_used, R, args.rhos, args.thresholds)
    tv = np.maximum(t_va, 0.0)
    rownorm = {c: 1.0 / np.maximum(np.linalg.norm(truth_va[c], axis=1, keepdims=True), 1e-300) ** 2 for c in COMPONENTS}
    for arm in ("theorem", "paper", "paper_norm"):
        out["arms"][arm] = {}
        taus = args.taus if arm == "theorem" else [float("inf")]
        for tau in taus:
            key = "inf" if np.isinf(tau) else str(tau)
            if arm == "theorem":
                wts = component_weights(tv, R, tau, scale)
                coverage = float(np.mean(tv <= tau * scale)) if not np.isinf(tau) else 0.0
            else:
                wts = {c: np.broadcast_to(rownorm[c], tv.shape).copy() for c in COMPONENTS}; coverage = None
            weights, pred, sel = {}, {}, {}
            for c in COMPONENTS:
                Ph = np.stack([VA[c][h].ravel() for h in heads], axis=1)
                if arm == "paper_norm":
                    # the campaign stack's own objective, the MEAN RELATIVE NORM per row (not its square), by iteratively
                    # reweighted least squares: row weight 1/(||Y_i|| ||r_i(w)||), six sweeps from the squared solution
                    w = simplex_weights(Ph, truth_va[c].ravel(), wts[c].ravel())
                    Yv = truth_va[c]; nrm = np.linalg.norm(Yv, axis=1)
                    for _ in range(6):
                        resid = np.linalg.norm(sum(w_h * VA[c][h] for h, w_h in zip(heads, w)) - Yv, axis=1)
                        rw = 1.0 / np.maximum(nrm * np.maximum(resid, 1e-3 * np.median(resid) + 1e-300), 1e-300)
                        w = simplex_weights(Ph, Yv.ravel(), np.repeat(rw, Yv.shape[1]))
                else:
                    w = simplex_weights(Ph, truth_va[c].ravel(), wts[c].ravel())
                weights[c] = {h: float(x) for h, x in zip(heads, w)}
                pred[c] = sum(w_h * TE[c][h] for h, w_h in zip(heads, w))
                # the best single head on the same validation objective
                if arm == "paper_norm":
                    obj = [rel_l2(truth_va[c], VA[c][h]) for h in heads]
                else:
                    obj = [float(np.sum(wts[c].ravel() * (VA[c][h].ravel() - truth_va[c].ravel()) ** 2)) for h in heads]
                sel[c] = heads[int(np.argmin(obj))]
            rec = score(truth_te, pred, scale, S_used, R, args.rhos, args.thresholds); rec["weights"] = weights
            rec["validation_coverage_below_tau"] = coverage
            rec["select"] = dict(heads=sel, **score(truth_te, {c: TE[c][sel[c]] for c in COMPONENTS}, scale, S_used, R, args.rhos, args.thresholds))
            out["arms"][arm][key] = rec
            print(f"{arm:7s} tau={key:>6}: rad {100*rec['rel_l2_radiance_0.7']:.4f}%  epsR {rec['epsR']:.4g}  constrained p95 @0.7 {rec['rhos']['0.7']['constrained_p95']:.4f}  "
                  f"raw p95 {rec['rhos']['0.7']['raw_p95']:.4f}  select {sel}  Y4 weights {{{', '.join(f'{h}: {w:.2f}' for h, w in weights['Y4'].items() if w > 0.005)}}}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    print("wrote", args.output)


if __name__ == "__main__":
    main()
