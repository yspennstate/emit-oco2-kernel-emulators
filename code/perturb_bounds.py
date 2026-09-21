"""Shared arithmetic for the same-data perturbation experiments (theory v4, Proposition 7).

Reference system H = K + lambda I (positive definite), whitened residual W = H^-1/2 (Y - M), whitened query vectors
g = H^-1/2 k_x. Perturbed system H~ with cross kernel k~_x and mean labels M~. With B = H^-1/2 (H~ - H) H^-1/2,
Q = (I + B)^-1, gamma_Q = 1/(1 + lambda_min(B)), eta_Q = max |b_i|/(1 + b_i):
  (a) exact identity   p~ - p = (h~ - h) + v^T W - (Q g~)^T (B W + U)
  (b) residual-action  ||p~ - p|| <= ||h~ - h|| + ||W^T v|| + ||Q g~|| ||B W + U||_op
  (c) signed spectrum  ||p~ - p|| <= ||h~ - h|| + ||W||_F (gamma_Q ||v|| + eta_Q ||g||) + gamma_Q ||U||_F ||g~||
  (d) ridge increase   ||p~ - p|| <= ||g|| ||W||_F s/(lambda_min(H) + s)
and the v3 bound with delta = ||B||_op < 1 for comparison. numpy only.
"""
import numpy as np


def inv_sqrt_psd(H, floor=1e-12):
    w, V = np.linalg.eigh(H); w = np.maximum(w, floor)
    return (V / np.sqrt(w)) @ V.T


def perturbation_report(H, Hm12, Ht, Kq, Kqt, Z, M, Mt, p_ref, p_t, mean_move=None):
    """All the quantities of Proposition 7 for one perturbed system; Kq, Kqt are (n_query x n) cross kernels,
    p_ref/p_t the reference and perturbed predictions on the query rows (n_query x q), M/Mt the mean labels."""
    n = len(H)
    B = Hm12 @ (Ht - H) @ Hm12; B = (B + B.T) / 2
    b_eig = np.linalg.eigvalsh(B)
    lam_min_B = float(b_eig.min()); delta = float(np.abs(b_eig).max())
    pd_ok = lam_min_B > -1.0
    W = Hm12 @ (Z - M); U = Hm12 @ (Mt - M)
    G = Hm12 @ Kq.T; Gt = Hm12 @ Kqt.T; V = Gt - G                        # n x n_query
    ell = np.linalg.norm(G, axis=0); d = np.linalg.norm(V, axis=0); ellt = np.linalg.norm(Gt, axis=0)
    move = np.linalg.norm(p_t - p_ref, axis=1)
    hmove = np.zeros_like(move) if mean_move is None else mean_move
    rec = dict(delta=delta, lambda_min_B=lam_min_B, positive_definite=pd_ok, b=float(np.linalg.norm(W)), z=float(np.linalg.norm(U)),
               median_ell=float(np.median(ell)), median_d=float(np.median(d)), median_move=float(np.median(move)), max_move=float(np.max(move)))
    if not pd_ok:
        rec.update(note="perturbed system not positive definite")
        return rec
    Q = np.linalg.inv(np.eye(n) + B); Q = (Q + Q.T) / 2
    gamma_Q = 1.0 / (1.0 + lam_min_B); eta_Q = float(np.max(np.abs(b_eig) / (1.0 + b_eig)))
    # (a) the exact identity, checked
    QGt = Q @ Gt
    ident = hmove_dummy = None
    recon = (V.T @ W) - (QGt.T @ (B @ W + U))                               # n_query x q, the kernel part of p~ - p
    diff_true = (p_t - p_ref)                                               # includes the mean movement h~ - h
    ident_err = float(np.max(np.linalg.norm(diff_true - recon, axis=1) / np.maximum(np.linalg.norm(diff_true, axis=1), 1e-300))) if mean_move is None else None
    # (b) residual action; (c) signed spectrum; (v3) delta
    WtV = np.linalg.norm(W.T @ V, axis=0)
    BWU_op = float(np.linalg.norm(B @ W + U, ord=2))
    bound_b = hmove + WtV + np.linalg.norm(QGt, axis=0) * BWU_op
    bound_c = hmove + rec["b"] * (gamma_Q * d + eta_Q * ell) + gamma_Q * rec["z"] * ellt
    rec.update(gamma_Q=gamma_Q, eta_Q=eta_Q, BWU_op=BWU_op, identity_max_rel_err=ident_err,
               bound_b_median=float(np.median(bound_b)), bound_b_holds=float(np.mean(move <= bound_b * (1 + 1e-9) + 1e-12)),
               bound_b_median_ratio=float(np.median(bound_b / np.maximum(move, 1e-300))),
               bound_c_median=float(np.median(bound_c)), bound_c_holds=float(np.mean(move <= bound_c * (1 + 1e-9) + 1e-12)),
               bound_c_median_ratio=float(np.median(bound_c / np.maximum(move, 1e-300))))
    if delta < 1:
        bound_v3 = hmove + (rec["b"] * (d + delta * ell) + rec["z"] * (ell + d)) / (1 - delta)
        rec.update(bound_v3_median=float(np.median(bound_v3)), bound_v3_holds=float(np.mean(move <= bound_v3 * (1 + 1e-9) + 1e-12)),
                   bound_v3_median_ratio=float(np.median(bound_v3 / np.maximum(move, 1e-300))))
    else:
        rec.update(bound_v3_median=None, bound_v3_holds=None, bound_v3_median_ratio=None)
    return rec


def ridge_bound(H, Hm12, Kq, Z, M, s, p_ref, p_t):
    """(d): the ridge-increase bound with the kernel and means unchanged."""
    lam_min_H = float(np.linalg.eigvalsh(H).min())
    W = Hm12 @ (Z - M); G = Hm12 @ Kq.T
    ell = np.linalg.norm(G, axis=0); move = np.linalg.norm(p_t - p_ref, axis=1)
    bound = ell * np.linalg.norm(W) * s / (lam_min_H + s)
    return dict(coefficient=float(s / (lam_min_H + s)), bound_d_median=float(np.median(bound)),
                bound_d_holds=float(np.mean(move <= bound * (1 + 1e-9) + 1e-12)), bound_d_median_ratio=float(np.median(bound / np.maximum(move, 1e-300))))
