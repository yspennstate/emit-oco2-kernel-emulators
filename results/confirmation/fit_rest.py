"""Stage 'dnn': train the networks and dump predictions and features.
Stage 'kernels': the residual kernel, the feature kernel and the convex stack,
using the length scales chosen by refine_ard.py.

    python fit_rest.py dnn
    python fit_rest.py kernels

CPU only.
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

import sys
import json
import time
import numpy as np
import psutil
from itertools import combinations_with_replacement
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import nnls
from sklearn.model_selection import train_test_split

STAGE = sys.argv[1] if len(sys.argv) > 1 else "dnn"
D = r"C:/Users/owner/jpl_kernel_dnn_2026_07_16/data/jpl_reg_data"
OUT = r"C:/Users/owner/paper2_bands_20260921/preds"
CACHE = r"C:/Users/owner/paper2_bands_20260921/cache"
os.makedirs(OUT, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)
COMPONENTS = ["Y1", "Y2", "Y3", "Y4"]
RHO, RANK = 0.7, 64
SEED = int(os.environ.get("SEED", "101"))
WIDTH = int(os.environ.get("WIDTH", "512"))
EPOCHS, PATIENCE, BATCH, LR = 400, 60, 1024, 1e-3


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}] {STAGE}:", *a, flush=True)


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
np.save(OUT + f"/idx_test_seed{SEED}.npy", te)
xm, xs = X[tr].mean(0), X[tr].std(0)
xs[xs == 0] = 1.0
Z = (X - xm) / xs


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


def radiance(Y):
    return Y["Y1"] + RHO * (Y["Y2"] + Y["Y3"]) / (1.0 - RHO * Y["Y4"])


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


def krr(F_tr, Yz, F_evals, ls, lam):
    gate(8.0)
    K = gram(F_tr, F_tr, ls)
    K[np.diag_indices_from(K)] += lam
    cf = cho_factor(K, lower=True, overwrite_a=True, check_finite=False)
    al = cho_solve(cf, Yz, check_finite=False)
    del K, cf
    return [gram(Fe, F_tr, ls) @ al for Fe in F_evals]


results = {}


def score(name, P_te):
    Yt = {c: Ys[c][te] for c in COMPONENTS}
    comp = {c: 100 * rel_l2(Yt[c], P_te[c]) for c in COMPONENTS}
    rad = 100 * rel_l2(radiance(Yt), radiance(P_te))
    results[name] = {"components_pct": comp,
                     "mean_component_pct": float(np.mean(list(comp.values()))),
                     "radiance_pct": rad}
    log(f"  {name}: radiance {rad:.4f}%  mean component "
        f"{np.mean(list(comp.values())):.4f}%")


# ------------------------------------------------------------------ stage: dnn
if STAGE == "dnn":
    import torch
    import torch.nn as nn
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    log(f"training {WIDTH}-unit networks")
    for c in COMPONENTS:
        A = Ys[c]
        m, s = A[tr].mean(0), A[tr].std(0)
        s[s == 0] = 1.0
        Xt = torch.tensor(Z[tr], dtype=torch.float32)
        Yt_ = torch.tensor((A[tr] - m) / s, dtype=torch.float32)
        Xv = torch.tensor(Z[va], dtype=torch.float32)
        Yv = torch.tensor((A[va] - m) / s, dtype=torch.float32)
        body = nn.Sequential(nn.Linear(6, WIDTH), nn.GELU(),
                             nn.Linear(WIDTH, WIDTH), nn.GELU(),
                             nn.Linear(WIDTH, WIDTH), nn.GELU())
        net = nn.Sequential(body, nn.Linear(WIDTH, 285))
        opt = torch.optim.Adam(net.parameters(), lr=LR)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
        best, state, bad = np.inf, None, 0
        t0 = time.time()
        for ep in range(EPOCHS):
            net.train()
            pm = torch.randperm(len(Xt))
            for i in range(0, len(Xt), BATCH):
                b = pm[i:i + BATCH]
                opt.zero_grad()
                ((net(Xt[b]) - Yt_[b]) ** 2).mean().backward()
                opt.step()
            sch.step()
            net.eval()
            with torch.no_grad():
                vl = ((net(Xv) - Yv) ** 2).mean().item()
            if vl < best - 1e-7:
                best, state, bad = vl, {k: v.clone() for k, v in net.state_dict().items()}, 0
            else:
                bad += 1
                if bad > PATIENCE:
                    break
        net.load_state_dict(state)
        net.eval()
        with torch.no_grad():
            for k, v_ in [("tr", tr), ("va", va), ("te", te)]:
                np.save(CACHE + f"/mlp{WIDTH}_{c}_{k}.npy",
                        net(torch.tensor(Z[v_], dtype=torch.float32)).numpy() * s + m)
                np.save(CACHE + f"/feat{WIDTH}_{c}_{k}.npy",
                        body(torch.tensor(Z[v_], dtype=torch.float32)).numpy())
        log(f"  {c}: {ep+1} epochs in {time.time()-t0:.0f}s, val mse {best:.3e}")
    P = {c: np.load(CACHE + f"/mlp{WIDTH}_{c}_te.npy") for c in COMPONENTS}
    for c in COMPONENTS:
        np.save(OUT + f"/fc_dnn_{WIDTH}_{c}_te_seed{SEED}.npy", P[c])
    score(f"fc_dnn_{WIDTH}", P)
    with open(f"results_dnn_{WIDTH}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log("done")
    raise SystemExit(0)

# ------------------------------------------------------------------ stage: kernels
AR = json.load(open("refined_ard.json", encoding="utf-8"))
mlp = {c: {k: np.load(CACHE + f"/mlp{WIDTH}_{c}_{k}.npy") for k in ["tr", "va", "te"]}
       for c in COMPONENTS}
feats = {c: {k: np.load(CACHE + f"/feat{WIDTH}_{c}_{k}.npy") for k in ["tr", "va", "te"]}
         for c in COMPONENTS}
preds, preds_va = {}, {}
for c in COMPONENTS:
    pass
preds["fc_dnn_512"] = {c: mlp[c]["te"] for c in COMPONENTS}
preds_va["fc_dnn_512"] = {c: mlp[c]["va"] for c in COMPONENTS}
preds["krr_ard_matern"] = {c: np.load(OUT + f"/krr_ard_matern_{c}_te_seed{SEED}.npy")
                           for c in COMPONENTS}

log("cubic ridge")
def cubic_feats(Zm):
    cols = [np.ones(len(Zm))]
    for d in range(1, 4):
        for comb in combinations_with_replacement(range(6), d):
            v = np.ones(len(Zm))
            for k in comb:
                v = v * Zm[:, k]
            cols.append(v)
    return np.stack(cols, 1)

Fc = cubic_feats(Z)
AtA = Fc[tr].T @ Fc[tr]
P, Pv, CUBIC_LAM = {}, {}, {}
for c in COMPONENTS:
    AtY = Fc[tr].T @ Ztr[c]
    best, Wb, bl = np.inf, None, None
    for lam in [1e-8, 1e-6, 1e-4, 1e-2, 1.0]:
        W = np.linalg.solve(AtA + lam * np.eye(AtA.shape[0]), AtY)
        e = rel_l2(Ys[c][va], heads[c].from_z(Fc[va] @ W))
        if e < best:
            best, Wb, bl = e, W, lam
    CUBIC_LAM[c] = bl
    P[c] = heads[c].from_z(Fc[te] @ Wb)
    Pv[c] = heads[c].from_z(Fc[va] @ Wb)
preds["cubic_ridge"], preds_va["cubic_ridge"] = P, Pv
for c in COMPONENTS:
    np.save(OUT + f"/cubic_ridge_{c}_te_seed{SEED}.npy", P[c])
score("cubic_ridge", P)

log("validation predictions for the refined kernel")
Pv = {}
for c in COMPONENTS:
    ls = np.array(AR["per_component"][c]["length_scales"])
    lam = AR["per_component"][c]["lambda"]
    (pv,) = krr(Z[tr], Ztr[c], [Z[va]], ls, lam)
    Pv[c] = heads[c].from_z(pv)
preds_va["krr_ard_matern"] = Pv
score("krr_ard_matern", preds["krr_ard_matern"])

log("network + residual kernel")
P, Pv, RES = {}, {}, {}
for c in COMPONENTS:
    ls = np.array(AR["per_component"][c]["length_scales"])
    lam = AR["per_component"][c]["lambda"]
    R = Ys[c][tr] - mlp[c]["tr"]
    hr = Head(R)
    pt, pv = krr(Z[tr], hr.to_z(R), [Z[te], Z[va]], ls, lam)
    P[c] = mlp[c]["te"] + hr.from_z(pt)
    Pv[c] = mlp[c]["va"] + hr.from_z(pv)
    RES[c] = {"length_scales": [float(v) for v in ls], "lambda": lam}
    np.save(OUT + f"/dnn_plus_residual_krr_{c}_te_seed{SEED}.npy", P[c])
preds["dnn_plus_residual_krr"], preds_va["dnn_plus_residual_krr"] = P, Pv
score("dnn_plus_residual_krr", P)

log("kernel on the learned features")
P, Pv, DKR = {}, {}, {}
sub = np.random.RandomState(1).choice(len(tr), 6000, replace=False)
for c in COMPONENTS:
    Ftr = feats[c]["tr"].astype(np.float64)
    fm, fs = Ftr.mean(0), Ftr.std(0)
    fs[fs == 0] = 1.0
    Ftr = (Ftr - fm) / fs
    Fva = (feats[c]["va"].astype(np.float64) - fm) / fs
    Fte = (feats[c]["te"].astype(np.float64) - fm) / fs
    d = Ftr.shape[1]
    best, bm = np.inf, None
    for mm in [0.5, 1.0, 2.0, 4.0]:
        (pv,) = krr(Ftr[sub], Ztr[c][sub], [Fva], np.full(d, np.sqrt(d)) * mm, 1e-8)
        e = rel_l2(Ys[c][va], heads[c].from_z(pv))
        if e < best:
            best, bm = e, mm
    pt, pv = krr(Ftr, Ztr[c], [Fte, Fva], np.full(d, np.sqrt(d)) * bm, 1e-8)
    P[c] = heads[c].from_z(pt)
    Pv[c] = heads[c].from_z(pv)
    DKR[c] = bm
    np.save(OUT + f"/dkr_feature_kernel_{c}_te_seed{SEED}.npy", P[c])
    log(f"  {c}: multiplier {bm}")
preds["dkr_feature_kernel"], preds_va["dkr_feature_kernel"] = P, Pv
score("dkr_feature_kernel", P)

log("per-component convex stack, weights on validation")
members = ["krr_ard_matern", "dkr_feature_kernel", "dnn_plus_residual_krr", "cubic_ridge"]
P, W = {}, {}
for c in COMPONENTS:
    Av = np.stack([preds_va[m][c].ravel() for m in members], 1)
    w, _ = nnls(Av, Ys[c][va].ravel())
    if w.sum() <= 0:
        w = np.ones(len(members))
    w = w / w.sum()
    W[c] = {m: float(x) for m, x in zip(members, w)}
    P[c] = sum(x * preds[m][c] for x, m in zip(w, members))
    np.save(OUT + f"/convex_stack_{c}_te_seed{SEED}.npy", P[c])
    log(f"  {c}: " + " ".join(f"{m.split('_')[0]}={x:.3f}" for m, x in zip(members, w)))
preds["convex_stack"] = P
score("convex_stack", P)

with open("dev_hyperparams.json", "w", encoding="utf-8") as f:
    json.dump({"cubic_lambda": CUBIC_LAM, "residual_kernel": RES,
               "dkr_multiplier": DKR,
               "stack_members": members, "stack_weights": W}, f, indent=2)
with open("results_top_models.json", "w", encoding="utf-8") as f:
    json.dump({"seed": SEED, "results": results, "stack_members": members,
               "stack_weights": W, "refined_ard": AR["per_component"]}, f, indent=2)
log("wrote results_top_models.json and dev_hyperparams.json")
