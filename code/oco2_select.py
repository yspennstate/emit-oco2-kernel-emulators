"""OCO-2 metric-aligned selection and stacks from a seeded campaign lane's saved member predictions (plan v4, E4).

Reads member_preds.npz (Yval, Yte and val_<head>/te_<head> in reduced coordinates, float32) written by
nmkc_public/campaign/jpl_seeded.py, rebuilds the band's reconstruction through jpl_data, and fits on the
validation rows: (a) the per-coordinate selection on the reduced error (the campaign's rule), (b) the same
selection on the exact radiance error (each coordinate's winner chosen by the radiance error of the combination
with that coordinate swapped, greedy over coordinates), (c) a simplex stack of the heads minimising the reduced
relative squared error, (d) a simplex stack minimising the exact radiance relative squared error (a convex
quadratic since the reconstruction is linear). Every rule is frozen on validation and read once on test, in both
metrics. numpy only apart from jpl_data.
"""
import argparse, json, os, sys
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--member-preds", required=True); ap.add_argument("--record", required=True); ap.add_argument("--band", required=True)
    ap.add_argument("--data-dir", required=True); ap.add_argument("--nmkc", required=True); ap.add_argument("--output", required=True)
    a = ap.parse_args()
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
    q = Yval.shape[1]
    # (a) per-coordinate selection on the reduced error (the campaign's rule, reproduced)
    win = [int(np.argmin([np.sqrt(((VA[h][:, j] - Yval[:, j]) ** 2).mean()) for h in heads])) for j in range(q)]
    Cte = np.stack([TE[heads[win[j]]][:, j] for j in range(q)], axis=1)
    out["rules"]["select_reduced"] = dict(winners=[heads[w] for w in win], **score(Cte))
    # (b) per-coordinate selection on the exact radiance error, greedy over coordinates from (a)
    cur = np.stack([VA[heads[win[j]]][:, j] for j in range(q)], axis=1); win_r = list(win)
    for sweep in range(2):
        for j in range(q):
            best = (rad(cur, Yval, Rva), win_r[j])
            for hi, h in enumerate(heads):
                trial = cur.copy(); trial[:, j] = VA[h][:, j]; e = rad(trial, Yval, Rva)
                if e < best[0] - 1e-12:
                    best = (e, hi)
            win_r[j] = best[1]; cur[:, j] = VA[heads[best[1]]][:, j]
    Cte_r = np.stack([TE[heads[win_r[j]]][:, j] for j in range(q)], axis=1)
    out["rules"]["select_radiance"] = dict(winners=[heads[w] for w in win_r], **score(Cte_r))
    # (c) simplex stack per coordinate on the reduced relative squared error (row weights 1/||z||^2)
    wts = 1.0 / np.maximum(np.linalg.norm(Yval, axis=1) ** 2, 1e-30)
    W = np.zeros((q, len(heads)))
    for j in range(q):
        W[j] = simplex(np.stack([VA[h][:, j] for h in heads], axis=1), Yval[:, j], wts)
    Ste = sum(W[:, hi][None, :] * TE[h] for hi, h in enumerate(heads))
    out["rules"]["stack_reduced"] = dict(**score(Ste))
    # (d) simplex stack on the exact radiance relative squared error: the reconstruction is linear, R = (Z * s_z) @ P,
    # so the radiance residual of a combination is affine in the weights; fitted jointly over all coordinates and heads
    # by weighted least squares on the whitened residual, with the simplex constraint per coordinate as heavy rows
    s_z = np.asarray(recon["s_z"], np.float64); P_mat = np.asarray(recon["P"], np.float64)
    rn = np.linalg.norm(Rva, axis=1)
    # design: columns (j, h) -> radiance contribution of head h's coordinate j, i.e. VA[h][:, j] * s_z[j] * P[j, :]
    cols = []; index = []
    for j in range(q):
        for hi, h in enumerate(heads):
            cols.append((VA[h][:, j] * s_z[j])[:, None] * P_mat[j][None, :] / rn[:, None]); index.append((j, hi))
    A = np.stack([c.ravel() for c in cols], axis=1); b = (Rva / rn[:, None]).ravel()
    big = 1e3 * np.sqrt(A.shape[0])
    rows = []; rhs = []
    for j in range(q):
        r = np.zeros(len(index)); r[[k for k, (jj, hi) in enumerate(index) if jj == j]] = big; rows.append(r); rhs.append(big)
    A2 = np.vstack([A, np.stack(rows)]); b2 = np.concatenate([b, rhs])
    w = nnls(A2, b2)
    Wr = np.zeros((q, len(heads)))
    for k, (j, hi) in enumerate(index):
        Wr[j, hi] = w[k]
    Wr = Wr / np.maximum(Wr.sum(1, keepdims=True), 1e-300)
    Ste_r = sum(Wr[:, hi][None, :] * TE[h] for hi, h in enumerate(heads))
    out["rules"]["stack_radiance"] = dict(**score(Ste_r))
    out["record"] = json.load(open(a.record)).get("results", {})
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.output, "w"), indent=1)
    print(a.band, {k: (round(v["reduced"], 3), round(v["radiance"], 4)) for k, v in out["rules"].items()})


if __name__ == "__main__":
    main()
