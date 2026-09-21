"""Transmission-conditioned retrieval diagnostics on stored campaign predictions (EMIT), theory v4.

For every family in a campaign NPZ (idx_te and <family>_Y1..Y4, physical units, test rows) this reports, on the
test table: the empirical transmission distribution F_t (quantiles, the fraction below a threshold ladder, the
tail slope over the observed range, the count of entries where t is below 2^-52 of the path radiance and the
float64 forward model cannot carry the reflectance term); the retrieval-weighted component error
e_R = |e_a| + R|e_t| + R^2 t|e_s| (after the projections t_hat -> max(t_hat, 0), s_hat -> [0, S]) with its
component shares; the conditional error profile t -> E[e_R^2 | t] by transmission bin, with the exponent nu of
E[e_R^2 | t] ~ sigma^2 t^{2 nu} fitted over the observed range; the raw inverse (unclipped, non-finite and
non-positive-denominator failures counted); the constrained retrieval (failures, clips at 0 and at R counted); the
three levels of the theorem's bound, E min{R^2, e_R^2/t^2}, R^2 F_t(tau) + E[e_R^2/t^2; t > tau] and
R^2 F_t(tau) + eps_R^2/tau^2 on a tau ladder, each against the measured constrained error, together with the old
constant-C bound for comparison; the pointwise inequality checked at every entry; and the paper's conditioned masks
at declared thresholds. S is the largest spherical albedo on the TRAINING rows; held-out exceedances of S are
counted and reported, since a training maximum is not automatically a bound on held-out truth. Nothing here is used
to select heads or thresholds; thresholds are declared on the command line before the read.

Split convention (identical to the campaign drivers): RandomState(seed) permutation, 10 percent test; validation
carve RandomState(seed + 10000), 10 percent of the training block; --ntrain takes the first rows.
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import conditioned_reflectance as cr

COMPONENTS = ("Y1", "Y2", "Y3", "Y4")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def split(n, seed, ntrain):
    perm = np.random.RandomState(seed).permutation(n)
    n_te = int(round(0.1 * n)); idx_te, tr_full = perm[:n_te], perm[n_te:]
    vp = np.random.RandomState(seed + 10000).permutation(len(tr_full))
    n_val = int(round(0.1 * len(tr_full)))
    idx_val, idx_tr = tr_full[vp[:n_val]], tr_full[vp[n_val:]]
    if ntrain and ntrain < len(idx_tr):
        idx_tr = idx_tr[:ntrain]
    return idx_te, idx_val, idx_tr


def q(x, ps=(0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)):
    return {str(p): float(np.quantile(x, p)) for p in ps}


def retrieval(a, t, s, ah, th, sh, rho, R, S):
    """Raw inverse at the true radiance (paper convention) and the constrained retrieval; errors and counts."""
    qq = 1.0 - rho * s
    L = a + rho * t / qq
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        den_raw = th + sh * (L - ah)
        raw = (L - ah) / den_raw
    raw_finite = np.isfinite(raw)
    raw_posden = np.isfinite(den_raw) & (den_raw > 0)
    tb, sb = np.maximum(th, 0.0), np.clip(sh, 0.0, S)
    den = tb + sb * (L - ah)
    ok = np.isfinite(den) & (den > 0)
    unclipped = np.full_like(L, np.nan)
    unclipped[ok] = (L[ok] - ah[ok]) / den[ok]
    bar = np.where(ok, np.clip(np.nan_to_num(unclipped, nan=0.0), 0.0, R), 0.0)
    clip_lo = ok & (unclipped < 0.0); clip_hi = ok & (unclipped > R)
    return dict(L=L, raw=raw, raw_finite=raw_finite, raw_posden=raw_posden, bar=bar, bar_fail=~ok, clip_lo=clip_lo, clip_hi=clip_hi,
                e_t_bar=np.abs(tb - t), e_s_bar=np.abs(sb - s))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--predictions", required=True, type=Path)
    ap.add_argument("--record", required=True, type=Path)
    ap.add_argument("--rhos", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7, 0.9])
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.0, 0.001, 0.003, 0.01, 0.03, 0.1],
                    help="prespecified flux/scale thresholds for the paper's conditioned masks")
    ap.add_argument("--tau-ladder", type=float, nargs="+", default=[3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0],
                    help="tau / flux scale for the bound ladder")
    ap.add_argument("--allow-float32", action="store_true")
    ap.add_argument("--families", nargs="*", default=None)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    record = json.loads(args.record.read_text(encoding="utf-8"))
    for c in ("X",) + COMPONENTS:
        if digest(args.data_dir / f"{c}.npy") != record["data_sha"][c]:
            raise ValueError(f"data digest mismatch for {c}")
    ys = {c: np.load(args.data_dir / f"{c}.npy", allow_pickle=False) for c in COMPONENTS}
    n = len(ys["Y1"]); seed = int(record["seed"])
    idx_te, idx_val, idx_tr = split(n, seed, int(record.get("ntrain") or 0))
    truth = {c: ys[c][idx_te] for c in COMPONENTS}
    train = {c: ys[c][idx_tr] for c in COMPONENTS}
    a, t, s = cr.components(truth)
    scale = cr.training_flux_scale(train)
    _, t_tr, s_tr = cr.components(train)
    # the hypothesis 0 <= s <= S with RS < 1, set on the TRAINING rows: the spherical albedo is a physical quantity in
    # [0, 1], and the table carries a few hundred entries above one in deep absorption bands (table artefacts at
    # vanishing flux); they are counted and left out of the bound, and held-out entries above S are counted too
    s_phys = s_tr[np.isfinite(s_tr) & (s_tr <= 1.0)]
    S = float(s_phys.max()); n_train_above_one = int(np.sum(s_tr > 1.0))
    S_exceed = int(np.sum(s > S)); S_eff = S
    hyp_ok = s <= S                                              # entries where the theorem's hypothesis holds
    R = float(max(args.rhos))
    if R * S_eff >= 1:
        raise ValueError(f"R S = {R * S_eff} is not below one; lower the largest rho")
    C_old = 2.0 * max(1.0 + R * S_eff, R * R / (1.0 - R * S_eff))
    tpos = t[np.isfinite(t) & (t > 0)]
    tail = {str(u): float(np.mean(t <= u * scale)) for u in args.tau_ladder}
    pts = [(np.log(u * scale), np.log(f)) for u, f in ((u, tail[str(u)]) for u in args.tau_ladder) if 0 < f < 0.5]
    tail_slope = float(np.polyfit([p[0] for p in pts], [p[1] for p in pts], 1)[0]) if len(pts) >= 3 else None
    # entries where the float64 forward model cannot carry the reflectance term (rho t / q below 2^-52 of a)
    unident = {str(rho): int(np.sum(rho * t / np.maximum(1.0 - rho * s, 1e-300) < np.abs(a) * 2.0 ** -52)) for rho in args.rhos}
    out = dict(record_sha256=digest(args.record), prediction_sha256=digest(args.predictions), seed=seed,
               ntrain=int(len(idx_tr)), n_test=int(len(idx_te)), bands=int(a.shape[1]), entries=int(a.size), data_sha=record["data_sha"],
               flux_scale=scale, S_training_max=S, S_training_entries_above_one=n_train_above_one, S_heldout_exceedances=S_exceed, S_used=S_eff, R=R, C_old=C_old,
               transmission=dict(quantiles_over_scale=q(tpos / scale), fraction_nonpositive=float(np.mean(t <= 0)),
                                 F_t_at_tau_over_scale=tail, observed_range_tail_slope=tail_slope,
                                 float64_unidentifiable_entries_by_rho=unident,
                                 note="the slope describes the observed range only; no power law is claimed down to t = 0"),
               families={}, prediction_dtypes={})
    # transmission bins for the conditional profile: deciles of t, then finer bins on the lowest decile
    edges = np.quantile(tpos, np.linspace(0, 1, 11))
    with np.load(args.predictions, allow_pickle=False) as P:
        if not np.array_equal(P["idx_te"], idx_te):
            raise ValueError("prediction test indices do not match the recorded split")
        fams = args.families or [f for f in record["families"] if f"{f}_Y1" in P.files]
        for fam in fams:
            pred = {c: np.asarray(P[f"{fam}_{c}"], dtype=np.float64) for c in COMPONENTS}
            dt = sorted({str(P[f"{fam}_{c}"].dtype) for c in COMPONENTS})
            out["prediction_dtypes"][fam] = dt
            if any(P[f"{fam}_{c}"].dtype.itemsize < 8 for c in COMPONENTS) and not args.allow_float32:
                raise ValueError(f"{fam}: float32 prediction dump; pass --allow-float32 to score it as a precision experiment")
            ah, th, sh = cr.components(pred)
            e_a, e_t, e_s = np.abs(ah - a), np.abs(np.maximum(th, 0.0) - t), np.abs(np.clip(sh, 0.0, S_eff) - s)
            eR = e_a + R * e_t + R * R * t * e_s
            e1 = e_a + e_t + t * e_s
            epsR2, eps2 = float(np.mean(eR ** 2)), float(np.mean(e1 ** 2))
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio2 = (eR / t) ** 2
            fam_out = dict(epsR2=epsR2, epsR=float(np.sqrt(epsR2)), eps=float(np.sqrt(eps2)),
                           component_shares_of_epsR2=dict(a=float(np.mean(e_a ** 2) / epsR2), t=float(np.mean((R * e_t) ** 2) / epsR2),
                                                          s=float(np.mean((R * R * t * e_s) ** 2) / epsR2)),
                           rel_l2_components={c: float(np.mean(np.linalg.norm(pred[c] - truth[c], axis=1) / np.linalg.norm(truth[c], axis=1))) for c in COMPONENTS},
                           profile_by_transmission_decile=[], profile_low_decile=[], share_of_epsR2_below=[], rhos={})
            fam_out["component_shares_of_epsR2"]["cross"] = float(1.0 - sum(fam_out["component_shares_of_epsR2"].values()))
            for i in range(10):
                m = ((t > edges[i]) & (t <= edges[i + 1])) if i else ((t >= edges[0]) & (t <= edges[1]))
                fam_out["profile_by_transmission_decile"].append(dict(
                    t_lo_over_scale=float(edges[i] / scale), t_hi_over_scale=float(edges[i + 1] / scale), entries=int(m.sum()),
                    E_eR2=float(np.mean(eR[m] ** 2)), E_eR2_over_t2=float(np.mean(ratio2[m])), E_min_R2_eR2_over_t2=float(np.mean(np.minimum(R * R, ratio2[m]))),
                    share_of_epsR2=float(np.sum(eR[m] ** 2) / np.sum(eR ** 2))))
            # the lowest decile split into logarithmic bins of t/scale, where the profile decides the rate
            lo = t <= edges[1]
            if lo.any():
                lb = np.geomspace(max(tpos.min() / scale, 1e-30), edges[1] / scale, 9)
                for i in range(8):
                    m = lo & (t > lb[i] * scale) & (t <= lb[i + 1] * scale)
                    if m.sum() >= 20:
                        fam_out["profile_low_decile"].append(dict(t_lo_over_scale=float(lb[i]), t_hi_over_scale=float(lb[i + 1]), entries=int(m.sum()),
                                                                  E_eR2=float(np.mean(eR[m] ** 2)), E_min_R2_eR2_over_t2=float(np.mean(np.minimum(R * R, ratio2[m])))))
            # exponent nu of E[e_R^2 | t] ~ sigma^2 t^{2 nu} over the observed range: log E[e_R^2] against log t on the bins
            binsx, binsy = [], []
            for d in fam_out["profile_low_decile"] + fam_out["profile_by_transmission_decile"][1:]:
                if d["E_eR2"] > 0:
                    binsx.append(np.log(np.sqrt(d["t_lo_over_scale"] * d["t_hi_over_scale"]) * scale)); binsy.append(np.log(d["E_eR2"]))
            fam_out["profile_exponent_2nu_observed_range"] = float(np.polyfit(binsx, binsy, 1)[0]) if len(binsx) >= 3 else None
            for u in args.tau_ladder:
                m = t <= u * scale
                fam_out["share_of_epsR2_below"].append(dict(tau_over_scale=u, fraction=float(np.mean(m)),
                                                            share=float(np.sum(eR[m] ** 2) / np.sum(eR ** 2)) if m.any() else 0.0))
            for rho in args.rhos:
                r = retrieval(a, t, s, ah, th, sh, rho, R, S_eff)
                raw_err = np.where(r["raw_finite"], r["raw"], 0.0) - rho      # the paper's convention (non-finite -> 0)
                bar_err = r["bar"] - rho
                pw = np.minimum(R * R, ratio2)                                # the pointwise bound of Theorem 4(i), squared
                violations = int(np.sum((bar_err ** 2 > pw * (1 + 1e-9) + 1e-30) & hyp_ok))   # checked where the hypothesis holds
                ladder = []
                for u in args.tau_ladder:
                    tau = u * scale; above = t > tau
                    ladder.append(dict(tau_over_scale=u, F_t=float(np.mean(~above)),
                                       bound_middle=float(R * R * np.mean(~above) + np.mean(np.where(above, ratio2, 0.0))),
                                       bound_last=float(R * R * np.mean(~above) + epsR2 / tau ** 2),
                                       bound_old_C=float(R * R * np.mean(~above) + C_old ** 2 * eps2 / tau ** 2)))
                mse_bar = float(np.mean(bar_err ** 2))
                fam_out["rhos"][str(rho)] = dict(
                    raw=dict(nonfinite_fraction=float(np.mean(~r["raw_finite"])), nonpositive_denominator_fraction=float(np.mean(~r["raw_posden"])),
                             mse=float(np.mean(raw_err ** 2)), median_abs=float(np.median(np.abs(raw_err))),
                             p95_abs=float(np.quantile(np.abs(raw_err), 0.95)), p99_abs=float(np.quantile(np.abs(raw_err), 0.99)),
                             outside_unit_fraction=float(np.mean((r["raw"] < 0) | (r["raw"] > 1)))),
                    constrained=dict(failure_fraction=float(np.mean(r["bar_fail"])), clipped_at_zero_fraction=float(np.mean(r["clip_lo"])),
                                     clipped_at_R_fraction=float(np.mean(r["clip_hi"])), mse=mse_bar,
                                     median_abs=float(np.median(np.abs(bar_err))), p95_abs=float(np.quantile(np.abs(bar_err), 0.95)),
                                     p99_abs=float(np.quantile(np.abs(bar_err), 0.99))),
                    E_min_R2_eR2_over_t2=float(np.mean(pw)), pointwise_bound_violations=violations,
                    bound_ladder=ladder,
                    bound_middle_inf_over_mse=float(min(x["bound_middle"] for x in ladder) / max(mse_bar, 1e-300)),
                    bound_last_inf_over_mse=float(min(x["bound_last"] for x in ladder) / max(mse_bar, 1e-300)),
                    bound_old_inf_over_mse=float(min(x["bound_old_C"] for x in ladder) / max(mse_bar, 1e-300)),
                    paper_conditioned=[cr.evaluate(truth, pred, flux_scale=scale, threshold=th_, rho=rho) for th_ in args.thresholds])
            out["families"][fam] = fam_out
    out["float32_opt_in"] = bool(args.allow_float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def _finite(o):
        # a non-finite float (the bound at tau -> 0 is +inf) is written as a string, so one entry cannot kill the record
        if isinstance(o, float):
            return o if o == o and abs(o) != float("inf") else ("inf" if o > 0 else "-inf" if o < 0 else "nan")
        if isinstance(o, dict):
            return {k: _finite(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_finite(v) for v in o]
        return o
    args.output.write_text(json.dumps(_finite(out), indent=1, allow_nan=False) + "\n", encoding="utf-8")
    print("wrote", args.output, "families", list(out["families"]))


if __name__ == "__main__":
    main()
