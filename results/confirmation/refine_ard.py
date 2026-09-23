"""A proper per-component search for the input length scales.

The gradient-surrogate metric of Section 6.1 gives a shape with about a five-fold spread
across the six inputs; the manuscript's fitted metric gives the relative azimuth a
sixteen- to thirty-two-fold longer scale on the transmittances. This script searches the
six log length scales per component directly by coordinate descent on validation, using a
6,000-row subsample of the training block for ranking, and then confirms the winner at the
full 18,884 rows over a ridge ladder with the Gram built once per component.

CPU only. Peak memory: one Gram (2.66 GB) plus one Cholesky copy.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import ctypes
try:
    ctypes.windll.kernel32.SetPriorityClass(
        ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)
except Exception:
    pass

import json
import time
import numpy as np
import psutil
from scipy.linalg import cho_factor, cho_solve
from sklearn.model_selection import train_test_split

D = r"C:/Users/owner/jpl_kernel_dnn_2026_07_16/data/jpl_reg_data"
OUT = r"C:/Users/owner/paper2_bands_20260921/preds"
os.makedirs(OUT, exist_ok=True)
COMPONENTS = ["Y1", "Y2", "Y3", "Y4"]
RHO, RANK = 0.7, 64
SEED = int(os.environ.get("SEED", "101"))
NSUB = int(os.environ.get("NSUB", "6000"))
PASSES = 2
LADDER = [0.5, 0.71, 1.0, 1.41, 2.0]
LAMS = [1e-10, 1e-8, 1e-6, 1e-4]


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def gate(need=6.0):
    a = psutil.virtual_memory().available / 2 ** 30
    if a < need:
        raise SystemExit(f"REFUSE: {a:.1f} GB free, need {need}")


X = np.load(D + "/X.npy")
Ys = {c: np.load(D + f"/{c}.npy") for c in COMPONENTS}
idx = np.arange(len(X))
tr_full, te = train_test_split(idx, test_size=2331, random_state=SEED)
rng = np.random.RandomState(SEED)
perm = rng.permutation(len(tr_full))
nv = int(round(0.1 * len(tr_full)))
va, tr = tr_full[perm[:nv]], tr_full[perm[nv:]]
xm, xs = X[tr].mean(0), X[tr].std(0)
xs[xs == 0] = 1.0
Z = (X - xm) / xs
log(f"seed {SEED}: train {len(tr)} val {len(va)} test {len(te)}")


class Head:
    def __init__(self, A_tr):
        self.m, self.s = A_tr.mean(0), A_tr.std(0)
        self.s[self.s == 0] = 1.0
        T = (A_tr - self.m) / self.s
        self.c = T.mean(0)
        self.Vt = np.linalg.svd(T - self.c, full_matrices=False)[2][:RANK]

    def to_z(self, A):
        return (((A - self.m) / self.s) - self.c) @ self.Vt.T

    def from_z(self, Zc):
        return ((Zc @ self.Vt) + self.c) * self.s + self.m


heads = {c: Head(Ys[c][tr]) for c in COMPONENTS}
Ztr = {c: heads[c].to_z(Ys[c][tr]) for c in COMPONENTS}


def rel_l2(A, B):
    return float(np.mean(np.linalg.norm(A - B, axis=1) / np.linalg.norm(A, axis=1)))


def gram(A, Bm, ls, chunk=4096):
    Aw, Bw = A / ls, Bm / ls
    bn = (Bw * Bw).sum(1)
    K = np.empty((len(Aw), len(Bw)))
    s5 = np.sqrt(5.0)
    for i in range(0, len(Aw), chunk):
        a = Aw[i:i + chunk]
        d2 = (a * a).sum(1)[:, None] + bn[None, :] - 2.0 * (a @ Bw.T)
        np.maximum(d2, 0, out=d2)
        r = np.sqrt(d2, out=d2)
        blk = 1.0 + s5 * r + (5.0 / 3.0) * r * r
        np.multiply(r, -s5, out=r)
        np.exp(r, out=r)
        np.multiply(blk, r, out=blk)
        K[i:i + chunk] = blk
    return K


sub = np.random.RandomState(1).choice(len(tr), NSUB, replace=False)
Zs = Z[tr][sub]


def sub_score(c, ls, lam=1e-8):
    K = gram(Zs, Zs, ls)
    K[np.diag_indices_from(K)] += lam
    try:
        cf = cho_factor(K, lower=True, overwrite_a=True, check_finite=False)
        al = cho_solve(cf, Ztr[c][sub], check_finite=False)
    except np.linalg.LinAlgError:
        return np.inf
    finally:
        del K
    pv = gram(Z[va], Zs, ls) @ al
    return rel_l2(Ys[c][va], heads[c].from_z(pv))


chosen = {}
for c in COMPONENTS:
    ls = np.full(6, 4.0)
    best = sub_score(c, ls)
    log(f"{c}: start isotropic ls=4 -> val {100*best:.4f}%")
    for p in range(PASSES):
        for j in range(6):
            cur = ls[j]
            trial_best, trial_ls = best, cur
            for f in LADDER:
                if f == 1.0:
                    continue
                ls[j] = cur * f
                e = sub_score(c, ls)
                if e < trial_best:
                    trial_best, trial_ls = e, ls[j]
            ls[j] = trial_ls
            best = trial_best
        log(f"  {c} pass {p+1}: val {100*best:.4f}%  ls " +
            " ".join(f"{v:.2f}" for v in ls))
    chosen[c] = ls.copy()

log("")
log("full-n confirmation, one Gram per component, ridge ladder on validation")
final = {}
P_te, P_va = {}, {}
for c in COMPONENTS:
    gate(8.0)
    ls = chosen[c]
    t0 = time.time()
    K0 = gram(Z[tr], Z[tr], ls)
    Kva = gram(Z[va], Z[tr], ls)
    best, bl, bal = np.inf, None, None
    for lam in LAMS:
        K = K0.copy()
        K[np.diag_indices_from(K)] += lam
        try:
            cf = cho_factor(K, lower=True, overwrite_a=True, check_finite=False)
            al = cho_solve(cf, Ztr[c], check_finite=False)
        except np.linalg.LinAlgError:
            log(f"  {c} lam={lam}: not positive definite")
            del K
            continue
        del K, cf
        e = rel_l2(Ys[c][va], heads[c].from_z(Kva @ al))
        log(f"  {c} lam={lam}: val {100*e:.4f}%")
        if e < best:
            best, bl, bal = e, lam, al
    del Kva
    Kte = gram(Z[te], Z[tr], ls)
    P_te[c] = heads[c].from_z(Kte @ bal)
    del K0, Kte
    final[c] = {"length_scales": [float(v) for v in ls], "lambda": bl,
                "val_rel_l2_pct": 100 * best}
    log(f"  {c}: chose lam={bl}, val {100*best:.4f}%, {time.time()-t0:.0f}s")

for c in COMPONENTS:
    np.save(OUT + f"/krr_ard_matern_{c}_te_seed{SEED}.npy", P_te[c])
Yt = {c: Ys[c][te] for c in COMPONENTS}
comp = {c: 100 * rel_l2(Yt[c], P_te[c]) for c in COMPONENTS}
rad_t = Yt["Y1"] + RHO * (Yt["Y2"] + Yt["Y3"]) / (1 - RHO * Yt["Y4"])
rad_p = P_te["Y1"] + RHO * (P_te["Y2"] + P_te["Y3"]) / (1 - RHO * P_te["Y4"])
log("")
log(f"refined input-scaled kernel: radiance {100*rel_l2(rad_t, rad_p):.4f}%  "
    f"mean component {np.mean(list(comp.values())):.4f}%")
log("  per component: " + "  ".join(f"{c} {v:.4f}%" for c, v in comp.items()))
log("  published targets: radiance 0.095 +/- 0.009%")

with open("refined_ard.json", "w", encoding="utf-8") as f:
    json.dump({"seed": SEED, "per_component": final,
               "components_pct": comp,
               "radiance_pct": 100 * rel_l2(rad_t, rad_p)}, f, indent=2)
log("wrote refined_ard.json; overwrote krr_ard_matern test predictions")
