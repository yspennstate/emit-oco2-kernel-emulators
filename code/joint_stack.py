"""Joint radiance-reflectance combination (theory v4, E3): stacks fitted on the validation twins, read once on test.

The campaign NPZ carries, for every head with a validation twin (krr, dnn, dnn_ens, dnn_corr, ens_corr, dkr, dkr_cat),
val_<head>_<component> on idx_val and <head>_<component> on idx_te, both in physical units. Three arms:

  theorem   every base head is PROJECTED first (t_hat -> max(t_hat, 0) applied to the two flux components' sum
            through Y2 and Y3 kept as predicted but their sum floored at zero in scoring, s_hat -> [0, S]); then per
            component a simplex combination minimises the convex quadratic surrogate
              J_tau(w) = mean[ (e_a^2 + R^2 e_t^2 + R^4 t^2 e_s^2) / (t v tau)^2 ]
            on the validation rows (Theorem 4 gives E|rho_bar_w - rho|^2 <= R^2 F_t(tau) + 3 J_tau(w)); tau is a
            multiple of the training flux scale with its validation coverage F_t(tau) reported, and tau = inf is the
            unweighted physical-unit stack.
  paper     the same simplex combination under the campaign stack's own objective, a row-relative squared error
            per component (every validation row divided by its norm), for comparison with the published stack.
  select    the best single head per component on the validation objective of each arm.

Every arm is frozen on validation and read once on the test rows: component and radiance errors, eps_R, the raw
inverse and its failures, the constrained retrieval and its tails, the paper's conditioned masks; single heads
beside the stacks. State-level losses (mean over bands and prescribed reflectances within a state) are written so
that a finite-class selection on an independent state sample can be made later. numpy only.
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import conditioned_reflectance as cr
from transmission_conditioned import split, digest, retrieval

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
    """w >= 0, sum w = 1, minimising sum_i wts_i (sum_h w_h P_h,i - y_i)^2; the sum constraint as a heavy row."""
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
        if any(P[f"{h}_{c}"].dtype.itemsize < 8 for h in heads for c in COMPONENTS) and not args.allow_float32:
            raise ValueError("float32 prediction dump; pass --allow-float32 to score it as a precision experiment")
        out["heads"] = heads
        VA = {c: {h: np.asarray(P[f"val_{h}_{c}"], dtype=np.float64) for h in heads} for c in COMPONENTS}
        TE = {c: {h: np.asarray(P[f"{h}_{c}"], dtype=np.float64) for h in heads} for c in COMPONENTS}
    # project the base heads first: the albedo onto [0, S]; the flux floor acts on the SUM, so it is applied at scoring
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
                if np.isinf(tau):
                    cond_at, cond_s = np.ones_like(tv), np.ones_like(tv)
                else:
                    tt = np.maximum(tv, tau * scale); cond_at = 1.0 / tt ** 2; cond_s = (tv / tt) ** 2
                wts = {"Y1": cond_at, "Y2": R * R * cond_at, "Y3": R * R * cond_at, "Y4": R ** 4 * cond_s}
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
