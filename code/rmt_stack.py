"""rmt_stack - a cross-validation-free stacking estimator for an ensemble scored in a per-sample relative metric.

Per output coordinate d, given member predictions F (M x n) and targets y (n) on calibration cases:
  1. case weights s_i = 1 / ||y_i||^2 (the reported metric's weighting; pass `case_weights` to change the metric),
  2. the error covariance in that metric, Sigma = sum_i s_i (e_i - mu)(e_i - mu)' / n, e_i = F[:, i] - y_i, mu the
     weighted mean error,
  3. a cleaning of that covariance: clean="rie" (default) the Ledoit-Peche nonlinear shrinkage in the Bun-Bouchaud-Potters
     form with a regulariser eta = mean(lambda)/sqrt(n), floored at `floor` x the bulk level (the bulk level read from the
     median eigenvalue against the Marchenko-Pastur median); clean="none" the sample covariance itself; clean="lw2020"
     Ledoit-Wolf 2020 analytical nonlinear shrinkage; clean="nercome" Lam 2016 split-sample eigenvalues (both need lw2020.py).
     Measured 2026-09-05 (cleaner ladder, 40 EMIT + 30 OCO-2 cells): inside the long-only form the raw covariance, LW2020
     and NERCOME tie at full calibration (median 0.00) and beat the eta-RIE by 5-6 percent at n = 60 on OCO-2 (30/30 cells);
     floors and linear shrinkage lose. Default clean="none" (raw); "lw2020" is a wash with it. The eta-RIE ("rie") is kept
     for the sum-to-one form (long_only=False), where a cleaning is what makes the inverse usable at all.
  4. the long-only minimum-variance weights argmin w' Xi w  s.t.  w >= 0, sum w = 1  (or the sum-to-one weights
     Xi^-1 1 / 1' Xi^-1 1 with long_only=False), intercept b = -w' mu.
Numpy + scipy only. Measured 2026-09-05 on the EMIT/OCO-2 pools (see RMT_ENSEMBLE_THEORY.md): on the components where
regularisation matters it beat per-output least squares by 7-13 percent and the pool's cross-validated best by 4-5.

    from rmt_stack import rmt_stack_weights, apply_weights
    W = rmt_stack_weights(P_cal, Y_cal)          # P_cal: (M, n, D), Y_cal: (n, D)  ->  W: (D, M + 1)
    Y_hat = apply_weights(P_eval, W)             # (n_eval, D)
"""
import numpy as np
from scipy.optimize import minimize

__all__ = ["rmt_stack_weights", "apply_weights", "mp_median", "ledoit_peche_clean"]

_MP_MED = {}


def mp_median(beta, grid=200001):
    """Median of the Marchenko-Pastur law with ratio beta in (0, 1], unit variance."""
    key = round(float(beta), 6)
    if key in _MP_MED:
        return _MP_MED[key]
    a, b = (1 - np.sqrt(beta)) ** 2, (1 + np.sqrt(beta)) ** 2
    x = np.linspace(a, b, grid)[1:-1]
    f = np.sqrt(np.maximum((b - x) * (x - a), 0)) / (2 * np.pi * beta * x)
    c = np.cumsum(f); c /= c[-1]
    _MP_MED[key] = float(x[np.searchsorted(c, 0.5)])
    return _MP_MED[key]


def ledoit_peche_clean(S, n, floor=0.05, eta_scale=1.0):
    """Nonlinear shrinkage of the eigenvalues of the M x M sample covariance S built from n cases.
    Returns (Xi, info): Xi the cleaned matrix, info the noise level and the cleaned eigenvalues."""
    M = S.shape[0]; q = M / n
    lam, V = np.linalg.eigh(S)
    sigma2 = float(np.median(lam)) / mp_median(q)
    eta = eta_scale * lam.mean() / np.sqrt(n)
    z = lam - 1j * eta
    g = (1.0 / (z[:, None] - lam[None, :])).mean(1)
    xi = lam / np.abs(1 - q + q * z * g) ** 2
    xi = np.maximum(xi, floor * sigma2)
    return (V * xi) @ V.T, dict(sigma2=sigma2, xi=xi, lam=lam, q=q)


def _clean(S, X, n, clean, floor, eta_scale):
    if clean == "none":
        return S
    if clean == "rie":
        return ledoit_peche_clean(S, n, floor=floor, eta_scale=eta_scale)[0]
    if clean == "lw2020":
        from lw2020 import lw2020_clean
        return lw2020_clean(S, n)[0]
    if clean == "nercome":
        from lw2020 import nercome_clean
        return nercome_clean(X)
    raise ValueError("clean must be one of rie, none, lw2020, nercome")


def _weights_one_output(F, y, s, floor, eta_scale, long_only, clean="rie"):
    """F: (M, n) member predictions for one output; y: (n,); s: (n,) case weights."""
    M, n = F.shape
    E = F - y[None, :]
    mu = (E * s[None, :]).sum(1) / s.sum()
    Ec = (E - mu[:, None]) * np.sqrt(s)[None, :]
    S = Ec @ Ec.T / n
    Xi = _clean(S, Ec.T, n, clean, floor, eta_scale)
    if long_only:
        Xs = Xi / max(np.trace(Xi), 1e-300)
        res = minimize(lambda w: w @ Xs @ w, np.ones(M) / M, jac=lambda w: 2 * Xs @ w, bounds=[(0, 1)] * M,
                       constraints={"type": "eq", "fun": lambda w: w.sum() - 1}, method="SLSQP",
                       options=dict(maxiter=500, ftol=1e-14))
        w = np.maximum(res.x, 0); w = w / max(w.sum(), 1e-12)
    else:
        w = np.linalg.solve(Xi, np.ones(M)); w = w / w.sum()
    return w, -(w @ mu)


def rmt_stack_weights(P, Y, case_weights=None, floor=0.0, eta_scale=1.0, long_only=True, clean="none", pooled_shrink=None, boot=20, seed=0):
    """P: (M, n, D) member predictions on the calibration cases; Y: (n, D) targets.
    case_weights: (n,) or None (None = 1/||y_i||^2, the per-sample relative metric).
    pooled_shrink: None (default), "cv" or "moon" - shrink every output's weights toward the pooled long-only weights by an
      amount chosen by 5-fold cross-validation on the calibration rows ("cv": the calibration-only model-selection quantity
      that the population showed is what OCO-2-like corpora need; +4.9 / -0.5 percent vs the pool's best on EMIT / OCO-2
      with the sum-to-one form, section 3 of the report) or by a
      positive-part James-Stein amount whose noise is the m-out-of-n (m = n/2) subsampling covariance of the per-output
      weights (boot subsamples; the naive bootstrap is inconsistent for weights at zero, Andrews 2000, so "moon" is the
      only mode offered). Measured 2026-09-05 (P12'): see RMT_ENSEMBLE_THEORY.md 6m for whether it earned its keep.
    Returns W: (D, M + 1), the last column the intercept; use apply_weights."""
    P = np.asarray(P, np.float64); Y = np.asarray(Y, np.float64)
    M, n, D = P.shape
    if case_weights is None:
        s = 1.0 / np.maximum(np.linalg.norm(Y, axis=1) ** 2, 1e-300)
    else:
        s = np.asarray(case_weights, np.float64)
    W = np.zeros((D, M + 1))
    for d in range(D):
        w, b = _weights_one_output(P[:, :, d], Y[:, d], s, floor, eta_scale, long_only, clean)
        W[d, :M] = w; W[d, M] = b
    if pooled_shrink is None:
        return W
    if pooled_shrink not in ("moon", "cv") or not long_only:
        raise ValueError("pooled_shrink supports 'moon' or 'cv', with long_only=True")
    # pooled long-only weights on the sum of the trace-normalised per-output covariances (every output counts once)
    MU = np.zeros((D, M)); Spool = np.zeros((M, M))
    for d in range(D):
        E = P[:, :, d] - Y[None, :, d]; mu = (E * s[None, :]).sum(1) / s.sum(); MU[d] = mu
        Ec = (E - mu[:, None]) * np.sqrt(s)[None, :]; Sd = Ec @ Ec.T / n
        Spool += Sd / max(np.trace(Sd), 1e-300)
    wg = _nn_mv(Spool)
    if pooled_shrink == "cv":
        # the pool's recipe: 5-fold cross-validation ON THE CALIBRATION ROWS of the amount toward the pooled weights, scored in
        # the per-sample relative metric; the calibration-only model-selection quantity that P12'/P16 found is not a resampling one
        rng = np.random.default_rng(seed); folds = np.array_split(rng.permutation(n), 5)
        grid = np.linspace(0, 1, 11); err = np.zeros(len(grid))
        for f_ in folds:
            tr_ = np.setdiff1d(np.arange(n), f_)
            Wt = rmt_stack_weights(P[:, tr_, :], Y[tr_], case_weights=s[tr_], floor=floor, eta_scale=eta_scale, long_only=True, clean=clean)
            MUt = np.zeros((D, M)); Sp = np.zeros((M, M))
            for d in range(D):
                E = P[:, tr_, d] - Y[None, tr_, d]; st = s[tr_]; mu = (E * st[None, :]).sum(1) / st.sum(); MUt[d] = mu
                Ec = (E - mu[:, None]) * np.sqrt(st)[None, :]; Sd = Ec @ Ec.T / len(tr_); Sp += Sd / max(np.trace(Sd), 1e-300)
            wgt = _nn_mv(Sp)
            for k, s_ in enumerate(grid):
                Wk = wgt[None, :] + (1 - s_) * (Wt[:, :M] - wgt[None, :])
                pred = np.einsum("dm,mnd->nd", Wk, P[:, f_, :]) - (Wk * MUt).sum(1)[None, :]
                err[k] += float((np.linalg.norm(pred - Y[f_], axis=1) / np.linalg.norm(Y[f_], axis=1)).mean()) / 5
        s_pool = float(grid[int(np.argmin(err))])
        Wt = wg[None, :] + (1 - s_pool) * (W[:, :M] - wg[None, :])
        out = np.zeros((D, M + 1)); out[:, :M] = Wt; out[:, M] = -(Wt * MU).sum(1)
        rmt_stack_weights.last_info = dict(s_pool=s_pool, w_pool=wg, mode="cv")
        return out
    # m-out-of-n subsampling covariance trace of the per-output weights, scaled to size n
    rng = np.random.default_rng(seed); m = n // 2
    acc = np.zeros((boot, D, M))
    for b_ in range(boot):
        idx = rng.choice(n, m, replace=False)
        for d in range(D):
            acc[b_, d], _ = _weights_one_output(P[:, idx, d], Y[idx, d], s[idx], floor, eta_scale, True, clean)
    trV = acc.var(0, ddof=1).sum(1) * (m / n)
    dist2 = ((W[:, :M] - wg[None, :]) ** 2).sum(1)
    s_pool = float(min(1.0, trV.sum() / max(dist2.sum(), 1e-300)))
    Wt = wg[None, :] + (1 - s_pool) * (W[:, :M] - wg[None, :])
    out = np.zeros((D, M + 1)); out[:, :M] = Wt; out[:, M] = -(Wt * MU).sum(1)
    out_info = dict(s_pool=s_pool, w_pool=wg)
    rmt_stack_weights.last_info = out_info
    return out


def _nn_mv(Xi, rho=1e3):
    """long-only minimum variance by NNLS with a penalised sum constraint (agrees with SLSQP to 1e-5 on real cells)"""
    from scipy.optimize import nnls
    M = Xi.shape[0]; Xs = Xi / max(np.trace(Xi), 1e-300)
    try:
        L = np.linalg.cholesky(Xs + 1e-10 * np.eye(M))
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(Xs + 1e-6 * np.eye(M))
    A = np.vstack([L.T, rho * np.ones((1, M))]); b = np.zeros(M + 1); b[-1] = rho
    w, _ = nnls(A, b); w = np.maximum(w, 0); return w / max(w.sum(), 1e-300)


def apply_weights(P, W):
    """P: (M, n, D) member predictions; W: (D, M + 1). Returns (n, D) stacked predictions."""
    P = np.asarray(P, np.float64)
    M, n, D = P.shape
    out = np.einsum("dm,mnd->nd", W[:, :M], P)
    return out + W[:, M][None, :]


def least_squares_weights(P, Y, case_weights=None, ridge=1e-6):
    """Per-output least squares with intercept on the member predictions - the unconstrained stack, exactly as the pool's
    fitter has it (fit_pixel): ordinary, UNWEIGHTED least squares, ridge `ridge` added to the n-scaled Gram X'X/n. This is the
    least-squares vertex every blend number of the program was measured against; the P44 selftest found the earlier weighted
    form (case weights 1/||y||^2, trace-scaled ridge) scored 0.0503 against the pool's 0.0460 on the reference cell.
    case_weights: optional (n,) sample weights for a weighted variant - not the measured estimator; leave None to reproduce it.
    P: (M, n, D); Y: (n, D). Returns W (D, M + 1)."""
    P = np.asarray(P, np.float64); Y = np.asarray(Y, np.float64)
    M, n, D = P.shape
    W = np.zeros((D, M + 1)); I = np.eye(M + 1)
    for d in range(D):
        X = np.concatenate([P[:, :, d].T, np.ones((n, 1))], 1)
        if case_weights is None:
            G = X.T @ X / n; b = X.T @ Y[:, d] / n
        else:
            s = np.asarray(case_weights, np.float64); G = X.T @ (s[:, None] * X) / n; b = X.T @ (s * Y[:, d]) / n
        W[d] = np.linalg.solve(G + ridge * I, b)
    return W


def variance_removed_share(P_cal, Y_cal, P_ev, Y_ev, case_weights=None, B=8, seed=0, **stack_kw):
    """Where will the long-only stack pay? (RMT program 2026-09-05, P22/P23.) The estimation variance of the per-output
    least-squares stack at the evaluation rows, the same for the long-only stack, and their difference, each as a share of
    the least-squares evaluation error in the (weighted) squared metric. Variances are pairs-bootstrap over the calibration
    rows (B refits of each estimator exactly as used), which stays defined when the member Gram is singular (duplicate
    members). On the EMIT pools the difference (the variance the constraint removes) rank-correlated +0.65 with the cell's
    realised gain across 40 cells and +0.43..+0.87 within every output group, and replicated on a fresh split (P23).
    Returns dict(share_ls, share_nn, removed, gain_pct) - gain_pct is the realised relative-score gain if Y_ev is given."""
    P_cal = np.asarray(P_cal, np.float64); Y_cal = np.asarray(Y_cal, np.float64); P_ev = np.asarray(P_ev, np.float64)
    M, n, D = P_cal.shape; ne = P_ev.shape[1]
    rng = np.random.default_rng(seed)
    W_ls = least_squares_weights(P_cal, Y_cal, case_weights); W_nn = rmt_stack_weights(P_cal, Y_cal, case_weights=case_weights, **stack_kw)
    pred_ls = np.zeros((B, ne, D)); pred_nn = np.zeros((B, ne, D))
    for b in range(B):
        idx = rng.integers(0, n, n); cw = None if case_weights is None else np.asarray(case_weights)[idx]
        pred_ls[b] = apply_weights(P_ev, least_squares_weights(P_cal[:, idx], Y_cal[idx], cw))
        pred_nn[b] = apply_weights(P_ev, rmt_stack_weights(P_cal[:, idx], Y_cal[idx], case_weights=cw, **stack_kw))
    Y_ev = np.asarray(Y_ev, np.float64)
    se = np.maximum(np.linalg.norm(Y_ev, axis=1), 1e-300) ** -2.0 if case_weights is None else np.ones(ne)
    e_ls = apply_weights(P_ev, W_ls) - Y_ev; e_nn = apply_weights(P_ev, W_nn) - Y_ev
    err = float(np.mean(se * (e_ls ** 2).sum(1)))
    v_ls = float(np.mean(se * pred_ls.var(0, ddof=1).sum(1))) / max(err, 1e-300)
    v_nn = float(np.mean(se * pred_nn.var(0, ddof=1).sum(1))) / max(err, 1e-300)
    r_ls = np.linalg.norm(e_ls, axis=1) * np.sqrt(se); r_nn = np.linalg.norm(e_nn, axis=1) * np.sqrt(se)
    return dict(share_ls=v_ls, share_nn=v_nn, removed=v_ls - v_nn, gain_pct=float(100 * (r_ls.mean() - r_nn.mean()) / r_ls.mean()))


def blend_weights(P_cal, Y_cal, P_ev, case_weights=None, B=8, folds=5, grid=None, seed=0, metric="ratio", ev_weights=None, shift="none", **stack_kw):
    """The P24 estimator (RMT program 2026-09-05, theory 6z): a blend W(t) = (1 - t) W_simplex + t W_leastsquares whose t is chosen
    with the calibration rows and the members' predictions on the rows to be scored (no evaluation targets), by the transductive
    covariance penalty R(t) = CV_cal(t) + [V_ev(t) - V_cal(t)]: the cross-validated squared (weighted) error of the blend on the
    calibration rows plus the excess pairs-bootstrap prediction variance of the blend at the evaluation rows over the calibration
    rows. Measured on the pools: keeps the simplex where it pays (EMIT +0.08 vs the simplex, beats least squares on 39 of 40 cells)
    and beats both endpoints where neither pays (OCO-2 +0.83 vs the simplex, +0.66 vs least squares, 30 of 30 cells).
    metric="ratio" writes the same criterion in the score's own metric (theory 6ao, P39): each calibration row's out-of-fold relative
    squared residual is inflated by the variance shift before the root, R_1(t) = mean_i sqrt(max(r_i + [V_ev - V_cal], 0)), which
    orders the candidates as the mean-ratio score does when the ratios have a heavy tail (the squared loss ranks by the tail).
    The default is shift="none" since P54-P58 (2026-09-06 night): the variance-shift term never earned its place on any pool or corpus measured;
    shift="uniform" is the P24 / R_1 form, kept as the registered fallback. The default is metric="ratio" since P40-P42 (2026-09-05 night): it beat the squared-loss criterion at home on both splits, repaired the
    scarce-calibration failure (P41) and replicated on a held-out split (P42); metric="squared" is the registered fallback.
    Returns (W, t_star, R) with W of shape (D, M + 1) for apply_weights; R maps t -> criterion."""
    P_cal = np.asarray(P_cal, np.float64); Y_cal = np.asarray(Y_cal, np.float64); P_ev = np.asarray(P_ev, np.float64)
    M, n, D = P_cal.shape; ne = P_ev.shape[1]
    grid = np.round(np.arange(0, 1.0001, 0.1), 1) if grid is None else np.asarray(grid, np.float64)
    cw = None if case_weights is None else np.asarray(case_weights, np.float64)
    s_cal = np.maximum(np.linalg.norm(Y_cal, axis=1), 1e-300) ** -2.0 if cw is None else cw
    fit_ls = lambda A, Bm, w: least_squares_weights(A, Bm)                 # the measured least-squares vertex is unweighted whatever the simplex's metric
    fit_nn = lambda A, Bm, w: rmt_stack_weights(A, Bm, case_weights=w, **stack_kw)
    W_ls, W_nn = fit_ls(P_cal, Y_cal, cw), fit_nn(P_cal, Y_cal, cw)
    if shift == "none":   # P54 / P56 / P57 / P58: the criterion without the variance shift - no bootstrap, no scored-row input
        B = 0
    rng = np.random.default_rng(seed)
    parts = np.array_split(rng.permutation(n), folds); cv_ls, cv_nn = np.zeros_like(Y_cal), np.zeros_like(Y_cal)
    for f_ in parts:
        tr = np.setdiff1d(np.arange(n), f_); w_tr = None if cw is None else cw[tr]
        cv_ls[f_] = apply_weights(P_cal[:, f_], fit_ls(P_cal[:, tr], Y_cal[tr], w_tr)); cv_nn[f_] = apply_weights(P_cal[:, f_], fit_nn(P_cal[:, tr], Y_cal[tr], w_tr))
    be_ls, be_nn, bc_ls, bc_nn = (np.zeros((B,) + shp) for shp in ((ne, D), (ne, D), (n, D), (n, D)))
    for b in range(B):
        idx = rng.integers(0, n, n); w_b = None if cw is None else cw[idx]
        Wl, Wn = fit_ls(P_cal[:, idx], Y_cal[idx], w_b), fit_nn(P_cal[:, idx], Y_cal[idx], w_b)
        be_ls[b], be_nn[b], bc_ls[b], bc_nn[b] = apply_weights(P_ev, Wl), apply_weights(P_ev, Wn), apply_weights(P_cal, Wl), apply_weights(P_cal, Wn)
    # the evaluation rows' weights in the metric: 1/||y||^2 is unknown there, so the relative metric uses the least-squares prediction's norm as the scale
    s_ev = (np.maximum(np.linalg.norm(apply_weights(P_ev, W_ls), axis=1), 1e-300) ** -2.0) if cw is None else np.ones(ne)
    if ev_weights is not None:   # P55 diagnostic entry only: explicit row weights at the scored rows (e.g. 1/||y||^2 from targets a study holds); never the shipped default
        s_ev = np.asarray(ev_weights, float)
    R = {}
    for t in grid:
        cvp = (1 - t) * cv_nn + t * cv_ls; L = float(np.mean(s_cal * ((cvp - Y_cal) ** 2).sum(1)))
        if B == 0:
            Ve = Vc = 0.0
        else:
            Ve = float(np.mean(s_ev * ((1 - t) * be_nn + t * be_ls).var(0, ddof=1).sum(1))); Vc = float(np.mean(s_cal * ((1 - t) * bc_nn + t * bc_ls).var(0, ddof=1).sum(1)))
        if metric == "ratio":
            r2 = s_cal * ((cvp - Y_cal) ** 2).sum(1); R[float(t)] = float(np.mean(np.sqrt(np.maximum(r2 + (Ve - Vc), 0.0))))
        else:
            R[float(t)] = L + Ve - Vc
    t_star = min(R, key=R.get)
    return (1 - t_star) * W_nn + t_star * W_ls, t_star, R


if __name__ == "__main__":
    # smoke: recover a known long-only optimum on synthetic members
    rng = np.random.default_rng(0)
    M, n, D = 6, 400, 3
    y = 10 ** rng.uniform(0, 2, (n, D))
    good = rng.standard_normal((M, n, D)) * 0.02 * y[None]
    good[:2] *= 4.0                                                       # two bad members
    P = y[None] + good
    W = rmt_stack_weights(P, y)
    for cl in ("rie", "lw2020", "nercome"):
        Wc = rmt_stack_weights(P, y, clean=cl)
        print("clean=%-7s max|W - W_none| = %.4f" % (cl, np.abs(Wc - W).max()))
    Wp = rmt_stack_weights(P, y, pooled_shrink="moon", boot=10)
    print("pooled_shrink=moon: s_pool %.3f, max|W_pooled - W| %.4f" % (rmt_stack_weights.last_info["s_pool"], np.abs(Wp - W).max()))
    Wc = rmt_stack_weights(P, y, pooled_shrink="cv")
    print("pooled_shrink=cv:   s_pool %.3f, max|W_pooled - W| %.4f" % (rmt_stack_weights.last_info["s_pool"], np.abs(Wc - W).max()))
    print("weights per output (bad members first):")
    print(np.round(W[:, :M], 3))
    err = np.linalg.norm(apply_weights(P, W) - y, axis=1) / np.linalg.norm(y, axis=1)
    err_eq = np.linalg.norm(P.mean(0) - y, axis=1) / np.linalg.norm(y, axis=1)
    print("relative error: stack %.4f  equal-weight %.4f" % (err.mean(), err_eq.mean()))
    assert W[:, :2].max() < 0.05 and err.mean() < err_eq.mean(), "smoke failed"
    print("SMOKE PASS")

    # variance-removed diagnostic: members whose errors share a common component (collinear, as in the EMIT pools - the
    # unconstrained stack cancels the common part with large opposite weights and inherits their noise) must show a larger
    # removed share than members with independent errors. Exact duplicates are NOT the collinear case: a duplicate adds a
    # null direction that changes no prediction, so it carries no variance.
    rng = np.random.default_rng(1)
    n2, ne, D2, M2 = 300, 200, 4, 8
    yc = 10 ** rng.uniform(0, 2, (n2, D2)); ye = 10 ** rng.uniform(0, 2, (ne, D2))
    com_c = rng.standard_normal((1, n2, D2)) * 0.03 * yc[None]; com_e = rng.standard_normal((1, ne, D2)) * 0.03 * ye[None]
    col_c = com_c + rng.standard_normal((M2, n2, D2)) * 0.004 * yc[None]; col_e = com_e + rng.standard_normal((M2, ne, D2)) * 0.004 * ye[None]
    ind_c = rng.standard_normal((M2, n2, D2)) * 0.03 * yc[None]; ind_e = rng.standard_normal((M2, ne, D2)) * 0.03 * ye[None]
    r_col = variance_removed_share(yc[None] + col_c, yc, ye[None] + col_e, ye, B=8)
    r_ind = variance_removed_share(yc[None] + ind_c, yc, ye[None] + ind_e, ye, B=8)
    print("variance removed: collinear %.3f (ls share %.3f, gain %+.1f pct) | independent %.3f (ls share %.3f, gain %+.1f pct)" % (
        r_col["removed"], r_col["share_ls"], r_col["gain_pct"], r_ind["removed"], r_ind["share_ls"], r_ind["gain_pct"]))
    assert r_col["share_ls"] > r_col["share_nn"] and r_col["removed"] > r_ind["removed"], "variance-removed diagnostic control failed"
    print("variance-removed diagnostic: PASS")

    # blend control: with correlated-error members the criterion must keep t near 0 (the simplex pays); with independent members
    # it must move toward least squares, and in both cases the blend must be no worse than the better endpoint by more than 0.5 pct
    def relscore(pred, Y):
        return 100 * np.mean(np.linalg.norm(pred - Y, axis=1) / np.linalg.norm(Y, axis=1))
    for label, (Pc_, Pe_) in (("collinear", (yc[None] + col_c, ye[None] + col_e)), ("independent", (yc[None] + ind_c, ye[None] + ind_e))):
        Wb, tb, Rb = blend_weights(Pc_, yc, Pe_, B=8)
        sc = {k: relscore(apply_weights(Pe_, W), ye) for k, W in (("simplex", rmt_stack_weights(Pc_, yc)), ("ls", least_squares_weights(Pc_, yc)), ("blend", Wb))}
        print("blend control %-11s t* %.1f | simplex %.4f  ls %.4f  blend %.4f" % (label, tb, sc["simplex"], sc["ls"], sc["blend"]))
        assert sc["blend"] <= min(sc["simplex"], sc["ls"]) * 1.005, "blend control failed: worse than the better endpoint"
        if label == "collinear": assert tb <= 0.5, "blend control failed: collinear members should keep the simplex"
    print("blend estimator: PASS")

    # ratio-metric control (6ao): at delta = 0 the ratio criterion is the cross-validated mean ratio; on the collinear members it
    # must keep the simplex like the squared one, and its blend must be no worse than the better endpoint by more than 0.5 pct
    for label, (Pc_, Pe_) in (("collinear", (yc[None] + col_c, ye[None] + col_e)), ("independent", (yc[None] + ind_c, ye[None] + ind_e))):
        Wr, tr_, Rr = blend_weights(Pc_, yc, Pe_, B=8, metric="ratio")
        sc = {k: relscore(apply_weights(Pe_, W), ye) for k, W in (("simplex", rmt_stack_weights(Pc_, yc)), ("ls", least_squares_weights(Pc_, yc)), ("blend", Wr))}
        print("ratio-metric blend control %-11s t* %.1f | simplex %.4f  ls %.4f  blend %.4f" % (label, tr_, sc["simplex"], sc["ls"], sc["blend"]))
        assert sc["blend"] <= min(sc["simplex"], sc["ls"]) * 1.005, "ratio-metric control failed: worse than the better endpoint"
        if label == "collinear": assert tr_ <= 0.5, "ratio-metric control failed: collinear members should keep the simplex"
    print("ratio-metric blend estimator: PASS")
