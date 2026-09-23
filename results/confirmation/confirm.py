"""The single confirmatory evaluation described in PREREGISTRATION.md.

Refuses to run unless freeze.sha256 matches the current freeze.json. Fits every frozen
family on the confirmation training pool with hyperparameters copied from development,
and reads the confirmation block exactly once.

CPU only. One Gram per length-scale setting, shared across the four components.
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
import hashlib
import numpy as np
import h5py
import psutil
from itertools import combinations_with_replacement
from scipy.linalg import cho_factor, cho_solve

D = r"C:/Users/owner/jpl_kernel_dnn_2026_07_16/data/jpl_reg_data"
OUT = r"C:/Users/owner/paper2_bands_20260921/preds_confirm"
os.makedirs(OUT, exist_ok=True)
COMPONENTS = ["Y1", "Y2", "Y3", "Y4"]
RHO, RANK = 0.7, 64
TAU, KAPPA = 1e-12, 0.3
CONF_SEED = 20260921
MEM_FLOOR_GB = 6.0


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def gate():
    a = psutil.virtual_memory().available / 2 ** 30
    if a < MEM_FLOOR_GB:
        raise SystemExit(f"REFUSE: {a:.1f} GB free, floor {MEM_FLOOR_GB}")


raw = open("freeze.json", "rb").read()
h = hashlib.sha256(raw).hexdigest()
want = open("freeze.sha256", encoding="utf-8").read().split()[0].strip()
if h != want:
    raise SystemExit(f"REFUSE: freeze.json hash {h} != registered {want}")
FR = json.loads(raw.decode("utf-8"))
log("freeze.json verified", h[:16])

X = np.load(D + "/X.npy")
Ys = {c: np.load(D + f"/{c}.npy") for c in COMPONENTS}
with h5py.File(D + "/data_EMIT_24k.jld2", "r") as f:
    wls = np.array(f["wls"])
n = len(X)

perm = np.random.RandomState(CONF_SEED).permutation(n)
n_conf = int(round(0.15 * n))
conf = np.sort(perm[:n_conf])
pool = np.sort(perm[n_conf:])
p2 = np.random.RandomState(CONF_SEED).permutation(len(pool))
n_es = int(round(0.10 * len(pool)))
es, tr = pool[p2[:n_es]], pool[p2[n_es:]]
log(f"confirmation block {len(conf)}, training {len(tr)}, early-stop {len(es)}")
np.save(OUT + "/idx_confirmation.npy", conf)

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


def matern52(A, Bm, ls, chunk=4096):
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


def krr_shared(F_tr, targets, F_eval, ls, lam):
    """One Gram and one Cholesky for every target that shares these length scales."""
    gate()
    K = matern52(F_tr, F_tr, ls)
    K[np.diag_indices_from(K)] += lam
    cf = cho_factor(K, lower=True, overwrite_a=True, check_finite=False)
    keys = list(targets)
    Yc = np.concatenate([targets[k] for k in keys], axis=1)
    alpha = cho_solve(cf, Yc, check_finite=False)
    del K, cf
    Ke = matern52(F_eval, F_tr, ls)
    Pz = Ke @ alpha
    out, j = {}, 0
    for k in keys:
        w = targets[k].shape[1]
        out[k] = Pz[:, j:j + w]
        j += w
    return out


def cubic_feats(Zm):
    cols = [np.ones(len(Zm))]
    for d in range(1, 4):
        for comb in combinations_with_replacement(range(6), d):
            v = np.ones(len(Zm))
            for k in comb:
                v = v * Zm[:, k]
            cols.append(v)
    return np.stack(cols, 1)


preds = {}

log("cubic_ridge")
Fc = cubic_feats(Z)
AtA = Fc[tr].T @ Fc[tr]
P = {}
for c in COMPONENTS:
    lam = FR["cubic_ridge"]["lambda"][c]
    W = np.linalg.solve(AtA + lam * np.eye(AtA.shape[0]), Fc[tr].T @ Ztr[c])
    P[c] = heads[c].from_z(Fc[conf] @ W)
preds["cubic_ridge"] = P

log("krr_ard_matern (one Gram per component: the scales differ by component)")
P = {}
for c in COMPONENTS:
    f = FR["krr_ard_matern"][c]
    out = krr_shared(Z[tr], {c: Ztr[c]}, Z[conf],
                     np.array(f["length_scales"]), f["lambda"])
    P[c] = heads[c].from_z(out[c])
    log(f"  {c} done")
preds["krr_ard_matern"] = P

log("fc_dnn_512 (CPU)")
import torch
import torch.nn as nn
torch.set_num_threads(4)
torch.manual_seed(CONF_SEED)
width = FR["fc_dnn_512"]["width"]
epochs = FR["fc_dnn_512"]["max_epochs"]
patience = FR["fc_dnn_512"]["patience"]
lr = FR["fc_dnn_512"]["lr"]
bs = FR["fc_dnn_512"]["batch_size"]
mlp, feats = {}, {}
for c in COMPONENTS:
    A = Ys[c]
    m, s = A[tr].mean(0), A[tr].std(0)
    s[s == 0] = 1.0
    Xt = torch.tensor(Z[tr], dtype=torch.float32)
    Yt = torch.tensor((A[tr] - m) / s, dtype=torch.float32)
    Xv = torch.tensor(Z[es], dtype=torch.float32)
    Yv = torch.tensor((A[es] - m) / s, dtype=torch.float32)
    body = nn.Sequential(nn.Linear(6, width), nn.GELU(),
                         nn.Linear(width, width), nn.GELU(),
                         nn.Linear(width, width), nn.GELU())
    net = nn.Sequential(body, nn.Linear(width, 285))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best, state, bad = np.inf, None, 0
    for ep in range(epochs):
        net.train()
        pm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), bs):
            b = pm[i:i + bs]
            opt.zero_grad()
            ((net(Xt[b]) - Yt[b]) ** 2).mean().backward()
            opt.step()
        sch.step()
        net.eval()
        with torch.no_grad():
            vl = ((net(Xv) - Yv) ** 2).mean().item()
        if vl < best - 1e-7:
            best, state, bad = vl, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad > patience:
                break
    net.load_state_dict(state)
    net.eval()
    with torch.no_grad():
        mlp[c] = {k: (net(torch.tensor(Z[v_], dtype=torch.float32)).numpy() * s + m)
                  for k, v_ in [("tr", tr), ("cf", conf)]}
        feats[c] = {k: body(torch.tensor(Z[v_], dtype=torch.float32)).numpy()
                    for k, v_ in [("tr", tr), ("cf", conf)]}
    log(f"  {c}: early-stop mse {best:.3e}")
preds["fc_dnn_512"] = {c: mlp[c]["cf"] for c in COMPONENTS}

log("dnn_plus_residual_krr")
P = {}
for c in COMPONENTS:
    f = FR["dnn_plus_residual_krr"][c]
    R = Ys[c][tr] - mlp[c]["tr"]
    hr = Head(R)
    out = krr_shared(Z[tr], {c: hr.to_z(R)}, Z[conf],
                     np.array(f["length_scales"]), f["lambda"])
    P[c] = mlp[c]["cf"] + hr.from_z(out[c])
    log(f"  {c} done")
preds["dnn_plus_residual_krr"] = P

log("dkr_feature_kernel")
P = {}
for c in COMPONENTS:
    Ftr = feats[c]["tr"].astype(np.float64)
    fm, fs = Ftr.mean(0), Ftr.std(0)
    fs[fs == 0] = 1.0
    Ftr = (Ftr - fm) / fs
    Fcf = (feats[c]["cf"].astype(np.float64) - fm) / fs
    lsf = np.full(Ftr.shape[1], np.sqrt(Ftr.shape[1])) * FR["dkr_feature_kernel"]["multiplier"][c]
    out = krr_shared(Ftr, {c: Ztr[c]}, Fcf, lsf, FR["dkr_feature_kernel"]["lambda"])
    P[c] = heads[c].from_z(out[c])
preds["dkr_feature_kernel"] = P

log("convex_stack")
mem = FR["convex_stack"]["members"]
P = {}
for c in COMPONENTS:
    w = FR["convex_stack"]["weights"][c]
    P[c] = sum(w[m] * preds[m][c] for m in mem)
preds["convex_stack"] = P

# ------------------------------------------------------------------ one read
Yt = {c: Ys[c][conf] for c in COMPONENTS}
t_true = Yt["Y2"] + Yt["Y3"]
q_true = 1.0 - RHO * Yt["Y4"]
ADM = (t_true >= TAU) & (q_true >= KAPPA)
L_true = Yt["Y1"] + RHO * t_true / q_true
log(f"admissible coverage on the confirmation block {100*ADM.mean():.4f}% "
    f"({int((~ADM).sum()):,} entries excluded of {ADM.size:,})")

report = {"confirmation_seed": CONF_SEED, "n_confirmation": int(len(conf)),
          "n_train": int(len(tr)), "tau": TAU, "kappa": KAPPA,
          "admissible_coverage_pct": float(100 * ADM.mean()),
          "excluded_entries": int((~ADM).sum()), "families": {}}

for name, P in preds.items():
    for c in COMPONENTS:
        np.save(OUT + f"/{name}_{c}_conf.npy", P[c])
    comp = {c: 100 * rel_l2(Yt[c], P[c]) for c in COMPONENTS}
    rad_true = L_true
    rad_pred = P["Y1"] + RHO * (P["Y2"] + P["Y3"]) / (1.0 - RHO * P["Y4"])
    u = L_true - P["Y1"]
    den = P["Y2"] + P["Y3"] + P["Y4"] * u
    with np.errstate(divide="ignore", invalid="ignore"):
        rh = u / den
    nf = ~np.isfinite(rh)
    rh = np.where(nf, 0.0, rh)
    e = rh - RHO

    def stats(v):
        av = np.abs(v)
        return {"rmse": float(np.sqrt((v ** 2).mean())),
                "median_pp": float(100 * np.median(av)),
                "p95_pp": float(100 * np.quantile(av, 0.95)),
                "p99_pp": float(100 * np.quantile(av, 0.99))}

    report["families"][name] = {
        "components_pct": comp,
        "mean_component_pct": float(np.mean(list(comp.values()))),
        "radiance_pct": 100 * rel_l2(rad_true, rad_pred),
        "nonfinite_inversions": int(nf.sum()),
        "all_bands": stats(e.ravel()),
        "admissible": stats(e[ADM]),
    }
    r = report["families"][name]
    log(f"  {name:24s} radiance {r['radiance_pct']:.4f}%  "
        f"all-band p95 {r['all_bands']['p95_pp']:8.3f} pp  "
        f"admissible p95 {r['admissible']['p95_pp']:8.4f} pp")

fams = report["families"]
best_rad = min(fams, key=lambda k: fams[k]["radiance_pct"])
best_p95 = min(fams, key=lambda k: fams[k]["all_bands"]["p95_pp"])
h1a = best_rad != best_p95
h1b = fams[best_rad]["all_bands"]["p95_pp"] > fams["dnn_plus_residual_krr"]["all_bands"]["p95_pp"]
report["H1"] = {"best_radiance_family": best_rad, "best_all_band_p95_family": best_p95,
                "clause_a_reversal": bool(h1a), "clause_b_worse_tail": bool(h1b),
                "H1_supported": bool(h1a and h1b)}
log(f"H1: lowest radiance = {best_rad}; lowest all-band p95 = {best_p95}; "
    f"supported = {h1a and h1b}")

with open("confirmation_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)
log("wrote confirmation_report.json")
