"""OCO-2 metric-aligned selection and stacks from a seeded campaign lane's saved member predictions (plan v4, E4),
in a form that fits in memory.

Same rules as oco2_select.py: (a) the per-coordinate selection on the reduced error, (b) the same selection on the
exact radiance error (greedy over coordinates, two sweeps), (c) a simplex stack per coordinate on the reduced
relative squared error, (d) a simplex stack of all coordinates and heads minimising the exact radiance relative
squared error. Every rule is frozen on validation and read once on test, in both metrics.

What changed and why. The reconstruction is linear, R = (Z * s_z) @ P, so the radiance contribution of head h at
coordinate j is the outer product (VA[h][:, j] s_z[j] / rn) x P[j, :]. The first version stacked those outer
products as columns of a design matrix with n_val x n_rad rows (2,000 x 10,592 = 21 million rows, 162 MB per column,
several hundred columns) and was killed for memory on Kaggle and here. The Gram matrix of that design factorises:
<M_(j,h), M_(j',h')> = (a_(j,h) . a_(j',h')) (P_j . P_j'), and the right-hand side is a_(j,h) . (B P_j), so the
weighted least-squares problem is a convex quadratic in the (q x H) weights that needs only q x q and (qH) x (qH)
matrices. Rule (d) is then solved as what it is, a quadratic programme over a product of simplices (one simplex per
coordinate), by projected gradient with Nesterov momentum and a KKT check, instead of the heavy-row penalty. Rule (b)
uses the affine structure too: swapping one coordinate's head changes the radiance residual by an outer product, so
one matrix-vector product per coordinate gives every head's radiance error. numpy only apart from jpl_data.
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np


def nnls(A, b, iters=3000):
    m, n = A.shape; x = np.zeros(n); P = np.zeros(n, dtype=bool)
    w = A.T @ (b - A @ x); tol = 1e-12 * max(1.0, float(np.abs(w).max())); it = 0
    while (~P).any() and (w[~P] > tol).any() and it < iters:
        it += 1; j = int(np.argmax(np.where(P, -np.inf, w))); P[j] = True
        while True:
            s = np.zeros(n); s[P] = np.linalg.lstsq(A[:, P], b, rcond=None)[0]
            if (s[P] > 0).all():
                x = s; break
            neg = P & (s <= 0); alpha = np.min(x[neg] / (x[neg] - s[neg])); x = x + alpha * (s - x); P &= x > 1e-15; x[~P] = 0.0
        w = A.T @ (b - A @ x)
    return x


def simplex(Ph, y, wts):
    sq = np.sqrt(wts)[:, None]; big = 1e3 * np.sqrt(np.mean(wts)) * np.sqrt(len(wts))
    A = np.vstack([Ph * sq, big * np.ones((1, Ph.shape[1]))]); b = np.concatenate([y * sq[:, 0], [big]])
    w = nnls(A, b); return w / w.sum() if w.sum() > 0 else np.full(Ph.shape[1], 1.0 / Ph.shape[1])


def project_simplex(v):
    """Euclidean projection of v onto the probability simplex (Duchi et al. 2008)."""
    u = np.sort(v)[::-1]; css = np.cumsum(u)
    k = np.nonzero(u * np.arange(1, len(v) + 1) > (css - 1.0))[0][-1]
    tau = (css[k] - 1.0) / (k + 1.0)
    return np.maximum(v - tau, 0.0)


def qp_product_simplex(G, g, q, H, iters=200000, tol=1e-13):
    """min 0.5 w'Gw - g'w over w in a product of q simplices (each block of H weights sums to one, w >= 0).
    Projected gradient with Nesterov momentum and restarts; returns the weights as (q, H) and the KKT gap."""
    L = float(np.linalg.eigvalsh(G).max()); step = 1.0 / L
    def proj(w):
        return np.concatenate([project_simplex(w[j * H:(j + 1) * H]) for j in range(q)])
    def f(w):
        return 0.5 * w @ G @ w - g @ w
    w = proj(np.full(q * H, 1.0 / H)); y = w.copy(); t = 1.0; fw = f(w)
    for it in range(iters):
        grad = G @ y - g
        w_new = proj(y - step * grad)
        f_new = f(w_new)
        if f_new > fw:                      # restart the momentum when the objective rises
            y = w.copy(); t = 1.0
            w_new = proj(w - step * (G @ w - g)); f_new = f(w_new)
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        y = w_new + ((t - 1.0) / t_new) * (w_new - w)
        done = abs(fw - f_new) <= tol * max(1.0, abs(fw))
        w, fw, t = w_new, f_new, t_new
        if done and it > 50:
            break
    # KKT gap: within every block, the gradient on the support must equal the block minimum of the gradient
    grad = G @ w - g; gap = 0.0
    for j in range(q):
        gj = grad[j * H:(j + 1) * H]; wj = w[j * H:(j + 1) * H]
        gap = max(gap, float(np.max(np.where(wj > 1e-12, gj, -np.inf)) - gj.min()))
    return w.reshape(q, H), gap, it + 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--member-preds", required=True); ap.add_argument("--record", required=True); ap.add_argument("--band", required=True)
    ap.add_argument("--data-dir", required=True); ap.add_argument("--nmkc", required=True); ap.add_argument("--output", required=True)
    a = ap.parse_args()
    t0 = time.time()
    sys.path.insert(0, a.nmkc)
    import jpl_data
    jpl_data.DATA = Path(a.data_dir)
    recon = jpl_data.reconstruction(a.band)
    to_rad = lambda Z: jpl_data.to_radiance(Z, recon)
    z = np.load(a.member_preds)
    Yval, Yte = z["Yval"].astype(np.float64), z["Yte"].astype(np.float64)
    heads = sorted({k[4:] for k in z.files if k.startswith("val_") and ("te_" + k[4:]) in z.files})
    heads = [h for h in heads if h not in ("combined", "combined_w", "combined_plus_ridge", "kernel_flow")]
    VA = {h: z["val_" + h].astype(np.float64) for h in heads}; TE = {h: z["te_" + h].astype(np.float64) for h in heads}
    Rva, Rte = to_rad(Yval), to_rad(Yte)
    rel = lambda P, T: float(np.mean(np.linalg.norm(P - T, axis=1) / np.linalg.norm(T, axis=1)))
    rad = lambda P, T, RT: float(np.mean(np.linalg.norm(to_rad(P) - RT, axis=1) / np.linalg.norm(RT, axis=1)))
    score = lambda Pte: dict(reduced=100 * rel(Pte, Yte), radiance=100 * rad(Pte, Yte, Rte))
    out = dict(band=a.band, heads=heads, n_val=int(len(Yval)), n_te=int(len(Yte)), singles={h: score(TE[h]) for h in heads}, rules={})
    q = Yval.shape[1]; H = len(heads)
    s_z = np.asarray(recon["s_z"], np.float64); P_mat = np.asarray(recon["P"], np.float64)
    # the reconstruction is affine, R(z) = c + (z * s_z) @ P with c = m_z @ P + m; the check below fails loudly if the
    # module's map ever changes shape (the differences used in (b) and the target used in (d) both rely on it)
    c_off = np.asarray(recon["m_z"], np.float64) @ P_mat + np.asarray(recon["m"], np.float64)
    Zt = Yval[:5]; assert np.allclose(to_rad(Zt), c_off + (Zt * s_z) @ P_mat, rtol=1e-10, atol=1e-8), "to_radiance is not c + (Z * s_z) @ P"
    rn = np.linalg.norm(Rva, axis=1)
    PP = P_mat @ P_mat.T                                   # q x q Gram of the reconstruction rows
    # (a) per-coordinate selection on the reduced error (the campaign's rule, reproduced)
    win = [int(np.argmin([np.sqrt(((VA[h][:, j] - Yval[:, j]) ** 2).mean()) for h in heads])) for j in range(q)]
    Cte = np.stack([TE[heads[win[j]]][:, j] for j in range(q)], axis=1)
    out["rules"]["select_reduced"] = dict(winners=[heads[w] for w in win], **score(Cte))
    # (b) per-coordinate selection on the exact radiance error, greedy over coordinates from (a); the residual of a
    # swap at coordinate j is E + u x P_j with u = (z_new - z_cur)_j s_z[j], so the per-row error only needs E P_j
    cur = np.stack([VA[heads[win[j]]][:, j] for j in range(q)], axis=1); win_r = list(win)
    E = to_rad(cur) - Rva                                  # n_val x n_rad, the current radiance residual (offset included)
    e_row2 = (E * E).sum(1)
    for sweep in range(2):
        for j in range(q):
            EPj = E @ P_mat[j]; pp = PP[j, j]
            cur_err = float(np.mean(np.sqrt(np.maximum(e_row2, 0.0)) / rn))
            best = (cur_err, win_r[j]); best_u = None
            for hi, h in enumerate(heads):
                u = (VA[h][:, j] - cur[:, j]) * s_z[j]
                row2 = e_row2 + 2.0 * u * EPj + u * u * pp
                err = float(np.mean(np.sqrt(np.maximum(row2, 0.0)) / rn))
                if err < best[0] - 1e-12:
                    best = (err, hi); best_u = u
            if best[1] != win_r[j] or best_u is not None:
                if best_u is None:
                    continue
                win_r[j] = best[1]; cur[:, j] = VA[heads[best[1]]][:, j]
                E += best_u[:, None] * P_mat[j][None, :]; e_row2 = (E * E).sum(1)
    # the incremental error must agree with the direct one (a failing check means the affine bookkeeping drifted)
    inc_err = float(np.mean(np.sqrt(np.maximum(e_row2, 0.0)) / rn)); dir_err = rad(cur, Yval, Rva)
    assert abs(inc_err - dir_err) <= 1e-9 * max(1.0, dir_err), f"incremental {inc_err} vs direct {dir_err}"
    Cte_r = np.stack([TE[heads[win_r[j]]][:, j] for j in range(q)], axis=1)
    out["rules"]["select_radiance"] = dict(winners=[heads[w] for w in win_r], val_radiance_check=dict(incremental=inc_err, direct=dir_err), **score(Cte_r))
    # (c) simplex stack per coordinate on the reduced relative squared error (row weights 1/||z||^2)
    wts = 1.0 / np.maximum(np.linalg.norm(Yval, axis=1) ** 2, 1e-30)
    W = np.zeros((q, H))
    for j in range(q):
        W[j] = simplex(np.stack([VA[h][:, j] for h in heads], axis=1), Yval[:, j], wts)
    Ste = sum(W[:, hi][None, :] * TE[h] for hi, h in enumerate(heads))
    out["rules"]["stack_reduced"] = dict(**score(Ste))
    # (d) simplex stack on the exact radiance relative squared error, through the Gram of the outer-product design:
    # column (j, h) is a_(j,h) x P_j with a_(j,h) = VA[h][:, j] s_z[j] / rn; target B = (Rva - c) / rn
    Acoef = np.stack([np.stack([VA[h][:, j] * s_z[j] / rn for h in heads], axis=1) for j in range(q)], axis=0)  # q x n x H
    AA = np.einsum("jnh,knm->jhkm", Acoef, Acoef)          # (a_(j,h) . a_(k,m)) for all pairs
    G = (AA * PP[:, None, :, None]).reshape(q * H, q * H)  # times (P_j . P_k)
    # target: the radiance minus the affine offset (the offset enters the combination exactly once because every
    # coordinate's weights sum to one), whitened by the row norms
    BP = ((Rva - c_off[None, :]) / rn[:, None]) @ P_mat.T  # n x q: (B P_j)
    g = np.stack([Acoef[j].T @ BP[:, j] for j in range(q)], axis=0).reshape(q * H)
    G = 0.5 * (G + G.T)
    Wr, kkt_gap, n_it = qp_product_simplex(G, g, q, H)
    Ste_r = sum(Wr[:, hi][None, :] * TE[h] for hi, h in enumerate(heads))
    out["rules"]["stack_radiance"] = dict(kkt_gap=kkt_gap, iterations=int(n_it), **score(Ste_r))
    out["record"] = json.load(open(a.record)).get("results", {})
    out["minutes"] = round((time.time() - t0) / 60, 2)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.output, "w"), indent=1)
    print(a.band, {k: (round(v["reduced"], 3), round(v["radiance"], 4)) for k, v in out["rules"].items()}, f"kkt {kkt_gap:.2e} in {n_it} it, {out['minutes']} min")


if __name__ == "__main__":
    main()
