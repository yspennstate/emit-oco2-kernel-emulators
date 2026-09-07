"""EMIT lookup-table emulation: the kernel-and-network family set at one fresh split seed.

Promotion path for the EMIT rows of the radiative-transfer paper. A fresh split (numpy
permutation, deliberately not the notebook's sklearn split), every hyperparameter selected
on the split's own validation carve, and the test block read once per family at the end.

Families (all on PCA-64 targets unless noted, physical-unit evaluation):
  ridge3     cubic-feature ridge
  krr4k      exact Matern-5/2 KRR on a 4000-point fit subset (the round-1 construction)
  krr        exact Matern KRR on the FULL training set, nu in {3/2, 5/2}, scale and nugget
             tuned on validation over a 6000-point subsample, winner refit on all rows
  ard        the same with per-input length scales (coordinate search on validation)
  dnn        3x512 ELU network on standardized 285-band outputs, early-stopped on validation
  dnn_ens    mean of M independently seeded networks (M = --members)
  dnn_corr   dnn + exact full-n Matern KRR on its residuals, selected on corrected validation error
  ens_corr   the same on the ensemble mean
  dkr        exact Matern KRR on the network's last-hidden features (full n)
  select     per-PCA-coefficient validation selection among {krr, dnn, dnn_ens, dnn_corr, ens_corr, dkr}
  stack      per-component convex combination of the same heads, weights fitted on validation

Options: --ntrain N (learning-curve rung: the first N rows of the seeded training block; the
validation and test blocks are unchanged), --pca_rank, --widths, --epochs, --members, --smoke.
Environment: EMIT_DATA (directory with X.npy, Y1..Y4.npy), P2_OUT (results root), NMKC_THREADS.
"""
import argparse, hashlib, json, os, pathlib, time

import os as _os
_T = _os.environ.get("NMKC_THREADS", "4")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    _os.environ.setdefault(_v, _T)
import numpy as np
from scipy.linalg import cho_factor, cho_solve

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, required=True)
p.add_argument("--ntrain", type=int, default=0, help="0 = the whole training block")
p.add_argument("--pca_rank", type=int, default=64)
p.add_argument("--widths", default="512,512,512")
p.add_argument("--epochs", type=int, default=150)
p.add_argument("--members", type=int, default=5)
p.add_argument("--families", default="all", help="comma list or all")
p.add_argument("--tag", default="")
p.add_argument("--smoke", action="store_true")
args = p.parse_args()

N_THREADS = int(os.environ.get("NMKC_THREADS", "4"))
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, str(N_THREADS))
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
import torch.nn as nn
torch.set_num_threads(N_THREADS)

DATA_DIR = pathlib.Path(os.environ.get("EMIT_DATA", "data/emit"))
OUT_ROOT = pathlib.Path(os.environ.get("P2_OUT", "results"))
COMPONENTS = ["Y1", "Y2", "Y3", "Y4"]
TEST_REFL = 0.7
PCA_RANK = args.pca_rank
WIDTHS = tuple(int(w) for w in args.widths.split(","))
EPOCHS = 2 if args.smoke else args.epochs
MEMBERS = 2 if args.smoke else args.members
FIT_CAP = 1000 if args.smoke else 4000
TUNE_SUB = 1500 if args.smoke else 6000
FAMS = None if args.families == "all" else set(args.families.split(","))


def want(f):
    return FAMS is None or f in FAMS


# ---- evaluation, jpl_pipeline.py verbatim ----

def fwdfun(r, tdir, tdif, s, testrefl=TEST_REFL):
    return r + testrefl * (tdir + tdif) / (1 - s * testrefl)


def refl_inv(y, r, tdir, tdif, s):
    return (y - r) / (tdir + tdif + s * (y - r))


def radiance_from(Ys_):
    return fwdfun(Ys_["Y1"], Ys_["Y2"], Ys_["Y3"], Ys_["Y4"])


def reflectance_from(rads_true, Ys_pred):
    return refl_inv(rads_true, Ys_pred["Y1"], Ys_pred["Y2"], Ys_pred["Y3"], Ys_pred["Y4"])


def rel_l2(Y_true, Y_pred):
    num = np.linalg.norm(Y_true - Y_pred, axis=1)
    den = np.linalg.norm(Y_true, axis=1)
    return float(np.mean(num / den))


def eval_predictions(Ys_true, Ys_pred):
    out = {}
    for c in COMPONENTS:
        out[f"rel_l2_{c}"] = rel_l2(Ys_true[c], Ys_pred[c])
    rads_true = radiance_from(Ys_true)
    rads_pred = radiance_from(Ys_pred)
    out["rel_l2_radiance"] = rel_l2(rads_true, rads_pred)
    refls = reflectance_from(rads_true, Ys_pred)
    refls_nonan = np.where(np.isfinite(refls), refls, 0.0)
    err = refls_nonan - TEST_REFL
    out["refl_rmse"] = float(np.sqrt(np.mean(err ** 2)))
    out["refl_mae_median"] = float(np.median(np.abs(err)))
    out["refl_p95_abs"] = float(np.quantile(np.abs(err), 0.95))
    out["refl_nan_frac"] = float(np.mean(~np.isfinite(refls)))
    refl_check = reflectance_from(rads_true, Ys_true)
    mask = np.isfinite(refl_check) & (np.abs(refl_check - TEST_REFL) < 1e-6)
    err_m = np.where(np.isfinite(refls), refls, 0.0)[mask] - TEST_REFL
    out["wellposed_frac"] = float(np.mean(mask))
    out["refl_rmse_wellposed"] = float(np.sqrt(np.mean(err_m ** 2)))
    out["refl_p95_abs_wellposed"] = float(np.quantile(np.abs(err_m), 0.95))
    out["mean_rel_l2_components"] = float(np.mean([out[f"rel_l2_{c}"] for c in COMPONENTS]))
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


# ---- data and the fresh split (emit_clean_split.py, unchanged) ----
X = np.load(DATA_DIR / "X.npy")
Ys = {c: np.load(DATA_DIR / (c + ".npy")) for c in COMPONENTS}
n_all = X.shape[0]
rng_split = np.random.RandomState(args.seed)
perm = rng_split.permutation(n_all)
n_te = int(round(0.1 * n_all))
idx_te = perm[:n_te]
tr_full = perm[n_te:]
rng_val = np.random.RandomState(args.seed + 10000)
vperm = rng_val.permutation(len(tr_full))
n_val = int(round(0.1 * len(tr_full)))
idx_val = tr_full[vperm[:n_val]]
idx_tr = tr_full[vperm[n_val:]]
if args.ntrain and args.ntrain < len(idx_tr):
    idx_tr = idx_tr[:args.ntrain]           # a learning-curve rung: the first rows of the same block
if args.smoke:
    idx_tr = idx_tr[:3000]
n = len(idx_tr)
print(f"seed {args.seed}: train={n} val={len(idx_val)} test={len(idx_te)} rank={PCA_RANK} widths={WIDTHS}", flush=True)


class Standardizer:
    def __init__(self, A):
        self.mean = A.mean(axis=0)
        self.std = np.sqrt(A.var(axis=0))
        self.std[self.std == 0] = 1.0

    def fwd(self, A):
        return (A - self.mean) / self.std

    def inv(self, A):
        return A * self.std + self.mean


class PCAReducer:
    def __init__(self, Y_tr_std, rank):
        U, S, Vt = np.linalg.svd(Y_tr_std - Y_tr_std.mean(axis=0), full_matrices=False)
        self.center = Y_tr_std.mean(axis=0)
        self.Vt = Vt[:rank]
        self.evr = float((S[:rank] ** 2).sum() / (S ** 2).sum())

    def fwd(self, Y_std):
        return (Y_std - self.center) @ self.Vt.T

    def inv(self, Z):
        return Z @ self.Vt + self.center


xstd = Standardizer(X[idx_tr])
Xs = xstd.fwd(X)
Xtr, Xva, Xte = Xs[idx_tr], Xs[idx_val], Xs[idx_te]
ystd, pca, Ztr, Zva = {}, {}, {}, {}
for c in COMPONENTS:
    ystd[c] = Standardizer(Ys[c][idx_tr])
    pca[c] = PCAReducer(ystd[c].fwd(Ys[c][idx_tr]), PCA_RANK)
    Ztr[c] = pca[c].fwd(ystd[c].fwd(Ys[c][idx_tr]))
    Zva[c] = pca[c].fwd(ystd[c].fwd(Ys[c][idx_val]))
Y_true_va = {c: Ys[c][idx_val] for c in COMPONENTS}
Y_true_te = {c: Ys[c][idx_te] for c in COMPONENTS}


def phys(c, Z):
    return ystd[c].inv(pca[c].inv(Z))


def val_err(c, Z):
    return rel_l2(Y_true_va[c], phys(c, Z))


# ---- kernels ----

def sqd(A, B):
    D2 = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * (A @ B.T)
    np.maximum(D2, 0.0, out=D2)
    return D2


def matern(D2, ls, nu):
    r = np.sqrt(D2) / ls
    if nu == 1.5:
        a = np.sqrt(3.0) * r
        return (1.0 + a) * np.exp(-a)
    a = np.sqrt(5.0) * r
    return (1.0 + a + a * a / 3.0) * np.exp(-a)


def solve_krr(K, Y, nug):
    Kr = K.copy(); Kr.flat[::len(K) + 1] += nug * len(K)
    c = cho_factor(Kr, lower=True, check_finite=False, overwrite_a=True)
    return cho_solve(c, Y, check_finite=False)


class KRR:
    """Exact Matern KRR on inputs F (rows), targets in PCA space, tuned on validation by err_fn
    over a subsample, refit on every row. `w` is a diagonal metric applied after standardization."""

    def __init__(self, Ftr, Fva, Fte, seed):
        mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-9
        self.Ftr, self.Fva, self.Fte = (Ftr - mu) / sd, (Fva - mu) / sd, (Fte - mu) / sd
        rng = np.random.RandomState(seed + 777)
        self.sub = rng.permutation(len(self.Ftr))[:min(TUNE_SUB, len(self.Ftr))]

    def tune(self, Ytr_sub_fn, err_fn, w=None, nus=(1.5, 2.5), scales=(0.5, 1.0, 2.0, 4.0),
             nugs=(1e-8, 1e-6, 1e-4, 1e-2)):
        Ftr, Fva = self.Ftr, self.Fva
        if w is not None:
            Ftr, Fva = Ftr * w, Fva * w
        S_ = Ftr[self.sub]
        D2s, D2vs = sqd(S_, S_), sqd(Fva, S_)
        med = float(np.sqrt(np.median(D2s[np.triu_indices(len(S_), 1)])))
        Ysub = Ytr_sub_fn(self.sub)
        best = (np.inf, None)
        for nu in nus:
            for sc in scales:
                Ks, Kvs = matern(D2s, sc * med, nu), matern(D2vs, sc * med, nu)
                for nug in nugs:
                    try:
                        alpha = solve_krr(Ks, Ysub, nug)
                    except np.linalg.LinAlgError:
                        continue
                    e = err_fn(Kvs @ alpha)
                    if e < best[0]:
                        best = (e, dict(nu=nu, scale=sc, nugget=nug, med=med, val_sub=e))
        return best[1]

    def fit_predict(self, Ytr, hp, w=None):
        Ftr, Fva, Fte = self.Ftr, self.Fva, self.Fte
        if w is not None:
            Ftr, Fva, Fte = Ftr * w, Fva * w, Fte * w
        ls = hp["scale"] * hp["med"]
        K = matern(sqd(Ftr, Ftr), ls, hp["nu"])
        alpha = solve_krr(K, Ytr, hp["nugget"])
        del K
        outs = []
        for F_ in (Ftr, Fva, Fte):
            pred = np.empty((len(F_), Ytr.shape[1]))
            for k in range(0, len(F_), 4000):
                pred[k:k + 4000] = matern(sqd(F_[k:k + 4000], Ftr), ls, hp["nu"]) @ alpha
            outs.append(pred)
        return outs


def ard_search(krr, Ytr_sub_fn, err_fn, hp0, d):
    """Coordinate search over per-input multipliers, two sweeps, on the tuning subsample."""
    w = np.ones(d)
    base = hp0["val_sub"]
    hp = dict(hp0)
    for sweep in range(2):
        for j in range(d):
            best_m, best_e = 1.0, base
            for m in (0.25, 0.5, 2.0, 4.0):
                wt = w.copy(); wt[j] *= m
                cand = krr.tune(Ytr_sub_fn, err_fn, w=wt, nus=(hp["nu"],), scales=(hp["scale"],))
                if cand is not None and cand["val_sub"] < best_e - 1e-7:
                    best_e, best_m = cand["val_sub"], m
            w[j] *= best_m
            base = best_e
    hp = krr.tune(Ytr_sub_fn, err_fn, w=w)   # re-tune scale, nugget and nu at the found metric
    return w, hp


# ---- the network ----

def build(d_in, d_out):
    layers, prev = [], d_in
    for wdt in WIDTHS:
        layers += [nn.Linear(prev, wdt), nn.ELU()]
        prev = wdt
    layers.append(nn.Linear(prev, d_out))
    return nn.Sequential(*layers)


def train_mlp(c, seed, max_epochs=EPOCHS, patience=25):
    torch.manual_seed(seed)
    Yt = ystd[c].fwd(Ys[c])
    xtr = torch.tensor(Xtr, dtype=torch.float32)
    ytr = torch.tensor(Yt[idx_tr], dtype=torch.float32)
    xva = torch.tensor(Xva, dtype=torch.float32)
    yva = torch.tensor(Yt[idx_val], dtype=torch.float32)
    xte = torch.tensor(Xte, dtype=torch.float32)
    model = build(xtr.shape[1], ytr.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=1e-5)
    lossf = nn.MSELoss()
    bs = 1024
    best_state, best_val, bad, ep_stop = None, np.inf, 0, 0
    for ep in range(max_epochs):
        model.train()
        bperm = torch.randperm(len(xtr))
        for i in range(0, len(xtr), bs):
            b = bperm[i:i + bs]
            opt.zero_grad()
            lossf(model(xtr[b]), ytr[b]).backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vl = lossf(model(xva), yva).item()
        ep_stop = ep
        if vl < best_val - 1e-7:
            best_val, bad = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    def feats(x):
        h = x
        for layer in list(model)[:-1]:
            h = layer(h)
        return h

    with torch.no_grad():
        preds = [model(x).numpy().astype(np.float64) for x in (xtr, xva, xte)]
        F = [feats(x).numpy().astype(np.float64) for x in (xtr, xva, xte)]
    return preds, F, ep_stop + 1


# ---- run ----
t0 = time.time()
fam_metrics, fam_te, hyper, notes = {}, {}, {}, {}
heads_va, heads_te = {}, {}   # PCA-space predictions per head per component, for select/stack


def record(name, pred_te_phys):
    fam_metrics[name] = eval_predictions(Y_true_te, pred_te_phys)
    fam_te[name] = pred_te_phys
    m = fam_metrics[name]
    print(f"== {name}: comps {100*m['mean_rel_l2_components']:.3f}%  rad {100*m['rel_l2_radiance']:.4f}%  "
          f"refl_med {100*m['refl_mae_median']:.3f}%  [{(time.time()-t0)/60:.1f} min]", flush=True)


def poly3(A):
    cols = [np.ones(len(A))]
    d = A.shape[1]
    for i in range(d):
        cols.append(A[:, i])
    for i in range(d):
        for j in range(i, d):
            cols.append(A[:, i] * A[:, j])
    for i in range(d):
        for j in range(i, d):
            for k in range(j, d):
                cols.append(A[:, i] * A[:, j] * A[:, k])
    return np.stack(cols, 1)


if want("ridge3"):
    Ftr, Fva, Fte = poly3(Xtr), poly3(Xva), poly3(Xte)
    G = Ftr.T @ Ftr
    pred = {}
    for c in COMPONENTS:
        b = Ftr.T @ Ztr[c]
        best = None
        for lam in (1e-10, 1e-8, 1e-6, 1e-4, 1e-2):
            Wc = np.linalg.solve(G + lam * len(Ftr) * np.eye(len(G)), b)
            e = val_err(c, Fva @ Wc)
            if best is None or e < best[0]:
                best = (e, Wc, lam)
        pred[c] = phys(c, Fte @ best[1])
        hyper.setdefault("ridge3", {})[c] = dict(lam=best[2], val=best[0])
    record("ridge3", pred)

if want("krr4k"):
    rng = np.random.RandomState(7)
    sel = rng.permutation(n)[:FIT_CAP]
    D2ff = sqd(Xtr[sel], Xtr[sel]); D2vf = sqd(Xva, Xtr[sel]); D2tf = sqd(Xte, Xtr[sel])
    med = float(np.sqrt(np.median(D2ff[np.triu_indices(len(sel), 1)])))
    eigs = {mm: np.linalg.eigh(matern(D2ff, mm * med, 2.5)) for mm in (1.0, 2.0)}
    pred = {}
    for c in COMPONENTS:
        best = None
        for mm in (1.0, 2.0):
            w_, V = eigs[mm]
            Kva = matern(D2vf, mm * med, 2.5)
            VtZ = V.T @ Ztr[c][sel]
            for lam in (1e-8, 1e-6, 1e-4, 1e-2):
                alpha = V @ (VtZ / (w_ + lam * len(w_))[:, None])
                e = val_err(c, Kva @ alpha)
                if best is None or e < best[0]:
                    best = (e, mm, lam, alpha)
        e, mm, lam, alpha = best
        pred[c] = phys(c, matern(D2tf, mm * med, 2.5) @ alpha)
        hyper.setdefault("krr4k", {})[c] = dict(scale=mm, nugget=lam, val=e, fit_cap=len(sel))
    record("krr4k", pred)

krr_x = KRR(Xtr, Xva, Xte, args.seed)
if want("krr"):
    pred = {}
    for c in COMPONENTS:
        hp = krr_x.tune(lambda s: Ztr[c][s], lambda Zv: val_err(c, Zv))
        ptr, pva, pte = krr_x.fit_predict(Ztr[c], hp)
        hp["val"] = val_err(c, pva)
        hyper.setdefault("krr", {})[c] = hp
        heads_va.setdefault("krr", {})[c] = pva; heads_te.setdefault("krr", {})[c] = pte
        pred[c] = phys(c, pte)
    record("krr", pred)

if want("ard"):
    pred = {}
    for c in COMPONENTS:
        hp0 = krr_x.tune(lambda s: Ztr[c][s], lambda Zv: val_err(c, Zv))
        w_ard, hp = ard_search(krr_x, lambda s: Ztr[c][s], lambda Zv: val_err(c, Zv), hp0, Xtr.shape[1])
        ptr, pva, pte = krr_x.fit_predict(Ztr[c], hp, w=w_ard)
        hp = dict(hp); hp["w"] = [round(float(x), 4) for x in w_ard]; hp["val"] = val_err(c, pva)
        hyper.setdefault("ard", {})[c] = hp
        heads_va.setdefault("ard", {})[c] = pva; heads_te.setdefault("ard", {})[c] = pte
        pred[c] = phys(c, pte)
    record("ard", pred)

need_dnn = any(want(f) for f in ("dnn", "dnn_ens", "dnn_corr", "ens_corr", "dkr", "select", "stack"))
if need_dnn:
    pred_dnn, pred_ens = {}, {}
    ens_va, ens_tr, ens_te = {}, {}, {}
    feats = {}
    for c in COMPONENTS:
        acc = None
        for m in range(MEMBERS):
            P_, F_, ep = train_mlp(c, seed=args.seed * 100 + m)
            hyper.setdefault("dnn_epochs", {}).setdefault(c, []).append(ep)
            if m == 0:
                pred_dnn[c] = ystd[c].inv(P_[2])
                feats[c] = F_
                heads_va.setdefault("dnn", {})[c] = pca[c].fwd(P_[1]); heads_te.setdefault("dnn", {})[c] = pca[c].fwd(P_[2])
                dnn_tr = pca[c].fwd(P_[0])
                m0_va, m0_te = heads_va["dnn"][c], heads_te["dnn"][c]
            acc = [a + b for a, b in zip(acc, P_)] if acc is not None else list(P_)
        ens = [a / MEMBERS for a in acc]
        pred_ens[c] = ystd[c].inv(ens[2])
        ens_tr[c], ens_va[c], ens_te[c] = pca[c].fwd(ens[0]), pca[c].fwd(ens[1]), pca[c].fwd(ens[2])
        heads_va.setdefault("dnn_ens", {})[c] = ens_va[c]; heads_te.setdefault("dnn_ens", {})[c] = ens_te[c]
        # residual corrections: dnn (member 0) and the ensemble mean, in PCA space, selected on corrected val error
        for base_name, Btr, Bva, Bte in (("dnn_corr", dnn_tr, m0_va, m0_te), ("ens_corr", ens_tr[c], ens_va[c], ens_te[c])):
            if want(base_name) or want("select") or want("stack"):
                R = Ztr[c] - Btr
                hp = krr_x.tune(lambda s, R=R: R[s], lambda Zv, Bva=Bva: val_err(c, Bva + Zv))
                rtr, rva, rte = krr_x.fit_predict(R, hp)
                hp["val"] = val_err(c, Bva + rva)
                hyper.setdefault(base_name, {})[c] = hp
                heads_va.setdefault(base_name, {})[c] = Bva + rva; heads_te.setdefault(base_name, {})[c] = Bte + rte
        if want("dkr") or want("select") or want("stack"):
            kf = KRR(feats[c][0], feats[c][1], feats[c][2], args.seed)
            hp = kf.tune(lambda s: Ztr[c][s], lambda Zv: val_err(c, Zv))
            ptr, pva, pte = kf.fit_predict(Ztr[c], hp)
            hp["val"] = val_err(c, pva)
            hyper.setdefault("dkr", {})[c] = hp
            heads_va.setdefault("dkr", {})[c] = pva; heads_te.setdefault("dkr", {})[c] = pte
        print(f"  [{c}] members done [{(time.time()-t0)/60:.1f} min]", flush=True)
    if want("dnn"):
        record("dnn", pred_dnn)
    if want("dnn_ens"):
        record("dnn_ens", pred_ens)
    for nm in ("dnn_corr", "ens_corr", "dkr"):
        if want(nm) and nm in heads_te:
            record(nm, {c: phys(c, heads_te[nm][c]) for c in COMPONENTS})

    cand = [h for h in ("krr", "dnn", "dnn_ens", "dnn_corr", "ens_corr", "dkr") if h in heads_va]
    if want("select") and len(cand) > 1:
        pred, picks = {}, {}
        for c in COMPONENTS:
            Zsel = np.empty_like(heads_te[cand[0]][c])
            pk = []
            for j in range(Zsel.shape[1]):
                errs = [np.sqrt(((heads_va[h][c][:, j] - Zva[c][:, j]) ** 2).mean()) for h in cand]
                k = int(np.argmin(errs)); pk.append(cand[k])
                Zsel[:, j] = heads_te[cand[k]][c][:, j]
            picks[c] = {h: pk.count(h) for h in cand}
            pred[c] = phys(c, Zsel)
        hyper["select"] = dict(candidates=cand, picks=picks)
        record("select", pred)
    if want("stack") and len(cand) > 1:
        from scipy.optimize import minimize
        pred, wts = {}, {}
        for c in COMPONENTS:
            # convex weights per component, fitted on validation in the physical relative metric
            Pv = [heads_va[h][c] for h in cand]

            def obj(w):
                return val_err(c, sum(wi * Pi for wi, Pi in zip(w, Pv)))
            M_ = len(cand)
            res = minimize(obj, np.ones(M_) / M_, bounds=[(0, 1)] * M_,
                           constraints={"type": "eq", "fun": lambda w: w.sum() - 1}, method="SLSQP",
                           options=dict(maxiter=300, ftol=1e-12))
            w = np.maximum(res.x, 0); w /= w.sum()
            wts[c] = {h: round(float(x), 4) for h, x in zip(cand, w)}
            pred[c] = phys(c, sum(wi * heads_te[h][c] for wi, h in zip(w, cand)))
        hyper["stack"] = dict(candidates=cand, weights=wts)
        record("stack", pred)

tag = args.tag or ("emit_s%d" % args.seed + (f"_n{args.ntrain}" if args.ntrain else "") +
                   (f"_r{PCA_RANK}" if PCA_RANK != 64 else "") + ("_big" if WIDTHS != (512, 512, 512) else ""))
out = dict(tag=tag, kind="emit_campaign", seed=args.seed, ntrain=n, n_val=len(idx_val), n_test=len(idx_te),
           pca_rank=PCA_RANK, pca_evr={c: pca[c].evr for c in COMPONENTS}, widths=list(WIDTHS), epochs=EPOCHS,
           members=MEMBERS, smoke=bool(args.smoke), families=fam_metrics, hyper=hyper,
           data_sha=dict(X=sha256(DATA_DIR / "X.npy"), **{c: sha256(DATA_DIR / (c + ".npy")) for c in COMPONENTS}),
           split="numpy RandomState(seed) permutation, 10 percent test; val carve RandomState(seed+10000), "
                 "10 percent of train; --ntrain takes the first rows of the training block",
           threads=N_THREADS, minutes=round((time.time() - t0) / 60, 1))
OUT_ROOT.mkdir(parents=True, exist_ok=True)
tmp = OUT_ROOT / (tag + ".tmp")
json.dump(out, open(tmp, "w"), indent=1)
os.replace(tmp, OUT_ROOT / (tag + ".json"))
pdir = OUT_ROOT / "preds"; pdir.mkdir(exist_ok=True)
np.savez_compressed(pdir / (tag + ".npz"), idx_te=idx_te,
                    **{f"{f}_{c}": fam_te[f][c].astype(np.float32) for f in fam_te for c in COMPONENTS})
print(f"DONE {tag} in {out['minutes']} min", flush=True)
