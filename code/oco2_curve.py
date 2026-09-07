"""OCO-2 reduced emulation: kernel and network heads at one band, seed and training size.

The protocol of campaign/jpl_seeded.py (neural-means-kernel-corrections), with one addition,
--ntrain: the network and every kernel are fitted on the first N rows of the seeded training
block (the 2000-row validation carve and the stored 2000-row test block are unchanged), so ten
seeds at nested N give the learning curve of every head on identical test cases.

Heads: kernel_flow (the published emulator's stored test predictions, a constant of the band),
kernel_raw (isotropic Matern-5/2 on standardized inputs), kernel_ard (relevance-scaled metric),
mean_flat (the residual MLP trained in the reduced relative metric), dkr_flat (Matern-5/2 on
its learned features), mean_ens (mean of M seeded networks, M = --members, 1 = off),
dkr_ens (Matern on the concatenated features of the members, when M > 1), combined
(per-coordinate validation selection over the fitted heads). Two error metrics: reduced
(relative L2 on the 40 standardized coefficients) and radiance (relative L2 on the reconstructed
monochromatic spectrum). Per-sample test errors and test predictions are saved.

Environment: NMKC_JPL_DATA (directory with the three .jld files), P2_OUT, NMKC_THREADS.
    python oco2_curve.py --band o2 --seed 3 --ntrain 4000
"""
import argparse, json, os, pathlib, sys, time
import os as _os
_T = _os.environ.get("NMKC_THREADS", "4")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    _os.environ.setdefault(_v, _T)
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--band", default="o2")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--ntrain", type=int, default=0, help="0 = all 18000")
p.add_argument("--epochs", type=int, default=250)
p.add_argument("--width", type=int, default=384)
p.add_argument("--members", type=int, default=1)
p.add_argument("--tag", default="")
p.add_argument("--smoke", action="store_true")
args = p.parse_args()

THREADS = int(os.environ.get("NMKC_THREADS", "4"))
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, str(THREADS))
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import cho_factor, cho_solve
torch.set_num_threads(THREADS)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import jpl_data
from jpl_data import load_band, reconstruction, radiance_error, kernel_flow_predictions, to_radiance

if os.environ.get("NMKC_JPL_DATA"):
    jpl_data.DATA = pathlib.Path(os.environ["NMKC_JPL_DATA"])
OUT_ROOT = pathlib.Path(os.environ.get("P2_OUT", "results"))
EPOCHS = 4 if args.smoke else args.epochs

sp = load_band(args.band, seed=args.seed)
Xtr, Ytr, Xval, Yval, Xte, Yte = (sp[k] for k in ("Xtr", "Ytr", "Xval", "Yval", "Xte", "Yte"))
if args.ntrain and args.ntrain < len(Xtr):
    Xtr, Ytr = Xtr[:args.ntrain], Ytr[:args.ntrain]
if args.smoke:
    Xtr, Ytr = Xtr[:2500], Ytr[:2500]
recon = reconstruction(args.band)
print(f"{args.band} seed {args.seed}: train={len(Xtr)} val={len(Xval)} test={len(Xte)}", flush=True)

rel = lambda Pp, T: float(np.mean(np.linalg.norm(Pp - T, axis=1) / np.linalg.norm(T, axis=1)))
rad = lambda Pp, T: radiance_error(Pp, T, recon)


class ResidualMLP(nn.Module):
    def __init__(self, d_in, d_out, width, depth=4):
        super().__init__()
        self.inp = nn.Linear(d_in, width)
        self.hidden = nn.ModuleList([nn.Linear(width, width) for _ in range(depth - 1)])
        self.out = nn.Linear(width, d_out)

    def forward(self, x, return_features=False):
        h = F.silu(self.inp(x))
        for layer in self.hidden:
            h = h + F.silu(layer(h))
        return (self.out(h), h) if return_features else self.out(h)


def train(seed):
    torch.manual_seed(seed)
    f32 = lambda a: torch.tensor(np.asarray(a, np.float32))
    xt, yt, xv = f32(Xtr), f32(Ytr), f32(Xval)
    model = ResidualMLP(Xtr.shape[1], Ytr.shape[1], args.width)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)
    n = len(xt)
    best, best_state = np.inf, None
    for ep in range(EPOCHS):
        perm = torch.randperm(n)
        for k in range(0, n, 512):
            i = perm[k:k + 512]
            pred, target = model(xt[i]), yt[i]
            loss = (torch.linalg.vector_norm(pred - target, dim=1) / torch.linalg.vector_norm(target, dim=1)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if (ep + 1) % 25 == 0 or ep == EPOCHS - 1:
            model.eval()
            with torch.no_grad():
                pv = model(xv).numpy().astype(np.float64)
            model.train()
            e = rel(pv, Yval)
            if e < best:
                best = e
                best_state = {k2: v.clone() for k2, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        preds = [model(f32(Z)).numpy().astype(np.float64) for Z in (Xtr, Xval, Xte)]
        feats = [model(f32(Z), return_features=True)[1].numpy().astype(np.float64) for Z in (Xtr, Xval, Xte)]
    return preds, feats, best


def sqd(A, B):
    return np.maximum((A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2 * A @ B.T, 0.0)


def m52(D2, ls):
    a = np.sqrt(5.0) * np.sqrt(D2) / ls
    return (1 + a + (5.0 / 3.0) * (D2 / ls ** 2)) * np.exp(-a)


def matern_head(Ztr, Zva, Zte, err_fn, label, w=None):
    """jpl_seeded.matern_head verbatim (scale grid {0.5,1,2,4} x median, nugget {1e-8,1e-6,1e-4},
    tuned on validation over a 6000-row subsample, refit on every training row)."""
    mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-9
    Ftr, Fval, Fte = (Ztr - mu) / sd, (Zva - mu) / sd, (Zte - mu) / sd
    if w is not None:
        Ftr, Fval, Fte = Ftr * w, Fval * w, Fte * w
    rng = np.random.default_rng(args.seed)
    sub = rng.choice(len(Ftr), min(6000, len(Ftr)), replace=False)
    med = np.sqrt(np.median(sqd(Ftr[sub], Ftr[sub])[np.triu_indices(len(sub), 1)]))
    best = (np.inf, None)
    D2s, D2vs = sqd(Ftr[sub], Ftr[sub]), sqd(Fval, Ftr[sub])
    for scale in (0.5, 1.0, 2.0, 4.0):
        Ks, Kvs = m52(D2s, scale * med), m52(D2vs, scale * med)
        for nug in (1e-8, 1e-6, 1e-4):
            Kr = Ks.copy(); Kr.flat[::len(sub) + 1] += nug * len(sub)
            try:
                c = cho_factor(Kr, lower=True, check_finite=False, overwrite_a=True)
            except np.linalg.LinAlgError:
                continue
            e = err_fn(Kvs @ cho_solve(c, Ytr[sub], check_finite=False))
            if e < best[0]:
                best = (e, (scale, nug))
    scale, nug = best[1]
    n = len(Ftr)
    K = m52(sqd(Ftr, Ftr), scale * med); K.flat[::n + 1] += nug * n
    c = cho_factor(K, lower=True, check_finite=False, overwrite_a=True)
    alpha = cho_solve(c, Ytr, check_finite=False)
    del K
    out = []
    for F_ in (Ftr, Fval, Fte):
        pred = np.empty((len(F_), Ytr.shape[1]))
        for k in range(0, len(F_), 4000):
            pred[k:k + 4000] = m52(sqd(F_[k:k + 4000], Ftr), scale * med) @ alpha
        out.append(pred)
    print(f"  {label}: scale {scale} nugget {nug:g} (med {med:.3f})", flush=True)
    return out, dict(scale=scale, nugget=nug, med=float(med), val=float(best[0]))


t_all = time.time()
results, val_preds, te_preds, hyper = {}, {}, {}, {}

kf = kernel_flow_predictions(args.band)
results["kernel_flow"] = dict(reduced=rel(kf, Yte), radiance=rad(kf, Yte))
te_preds["kernel_flow"] = kf

raw, h = matern_head(Xtr, Xval, Xte, lambda Pv: rel(Pv, Yval), "kernel_raw")
results["kernel_raw"] = dict(reduced=rel(raw[2], Yte), radiance=rad(raw[2], Yte)); hyper["kernel_raw"] = h
val_preds["kernel_raw"], te_preds["kernel_raw"] = raw[1], raw[2]
Xs = (Xtr - Xtr.mean(0)) / (Xtr.std(0) + 1e-9)
A_ls, *_ = np.linalg.lstsq(Xs, Ytr, rcond=None)
relevance = np.linalg.norm(A_ls, axis=1)
w_ard = relevance / relevance.mean()
ard, h = matern_head(Xtr, Xval, Xte, lambda Pv: rel(Pv, Yval), "kernel_ard", w=w_ard)
h["w"] = [round(float(x), 5) for x in w_ard]
results["kernel_ard"] = dict(reduced=rel(ard[2], Yte), radiance=rad(ard[2], Yte)); hyper["kernel_ard"] = h
val_preds["kernel_ard"], te_preds["kernel_ard"] = ard[1], ard[2]
print(f"kernels done [{(time.time()-t_all)/60:.1f} min]", flush=True)

members_P, members_F = [], []
for m in range(max(1, args.members)):
    t0 = time.time()
    Pm, Fm, bv = train(args.seed * 100 + m)
    members_P.append(Pm); members_F.append(Fm)
    if m == 0:
        results["mean_flat"] = dict(reduced=rel(Pm[2], Yte), radiance=rad(Pm[2], Yte), val=bv,
                                    minutes=round((time.time() - t0) / 60, 1))
        val_preds["mean_flat"], te_preds["mean_flat"] = Pm[1], Pm[2]
        D, h = matern_head(Fm[0], Fm[1], Fm[2], lambda Pv: rel(Pv, Yval), "dkr_flat")
        results["dkr_flat"] = dict(reduced=rel(D[2], Yte), radiance=rad(D[2], Yte)); hyper["dkr_flat"] = h
        val_preds["dkr_flat"], te_preds["dkr_flat"] = D[1], D[2]
    print(f"member {m} done [{(time.time()-t0)/60:.1f} min]", flush=True)
if args.members > 1:
    ens = [np.mean([P[k] for P in members_P], axis=0) for k in range(3)]
    results["mean_ens"] = dict(reduced=rel(ens[2], Yte), radiance=rad(ens[2], Yte))
    val_preds["mean_ens"], te_preds["mean_ens"] = ens[1], ens[2]
    cat = [np.concatenate([Fm[k] for Fm in members_F], axis=1) for k in range(3)]
    D, h = matern_head(cat[0], cat[1], cat[2], lambda Pv: rel(Pv, Yval), "dkr_ens")
    results["dkr_ens"] = dict(reduced=rel(D[2], Yte), radiance=rad(D[2], Yte)); hyper["dkr_ens"] = h
    val_preds["dkr_ens"], te_preds["dkr_ens"] = D[1], D[2]
    # members' own DKR heads averaged (the radiance lever: seed-averaged feature heads)
    heads = [D[2]]
    for m in range(1, args.members):
        Fm = members_F[m]
        Dm, _ = matern_head(Fm[0], Fm[1], Fm[2], lambda Pv: rel(Pv, Yval), f"dkr_m{m}")
        heads.append(Dm[2])
    D0 = te_preds["dkr_flat"]
    avg = np.mean([D0] + heads[1:], axis=0)
    results["dkr_avg"] = dict(reduced=rel(avg, Yte), radiance=rad(avg, Yte))
    te_preds["dkr_avg"] = avg

# per-coordinate selection on validation over the fitted heads
cand = [k for k in ("kernel_raw", "kernel_ard", "mean_flat", "dkr_flat", "mean_ens", "dkr_ens") if k in val_preds]
C_te = np.empty_like(Yte); winners = []
for j in range(Yte.shape[1]):
    errs = [np.sqrt(((val_preds[k][:, j] - Yval[:, j]) ** 2).mean()) for k in cand]
    w = int(np.argmin(errs)); winners.append(cand[w])
    C_te[:, j] = te_preds[cand[w]][:, j]
results["combined"] = dict(reduced=rel(C_te, Yte), radiance=rad(C_te, Yte))
te_preds["combined"] = C_te

den_red = np.linalg.norm(Yte, axis=1)
R_te = to_radiance(Yte, recon)
den_rad = np.linalg.norm(R_te, axis=1)
per = {}
for name, Pt in te_preds.items():
    per[name + "_red"] = np.linalg.norm(Pt - Yte, axis=1) / den_red
    per[name + "_rad"] = np.linalg.norm(to_radiance(Pt, recon) - R_te, axis=1) / den_rad

for name, r in results.items():
    print(f"{name:12s} reduced {100*r['reduced']:7.3f}%   radiance {100*r['radiance']:.4f}%", flush=True)

tag = args.tag or f"oco_{args.band}_s{args.seed}_n{len(Xtr)}" + (f"_m{args.members}" if args.members > 1 else "")
out = dict(tag=tag, kind="oco2_curve", band=args.band, seed=args.seed, ntrain=len(Xtr), n_val=len(Xval),
           n_test=len(Xte), epochs=EPOCHS, width=args.width, members=args.members, smoke=bool(args.smoke),
           results={k: {m: (round(100 * v, 5) if m in ("reduced", "radiance", "val") else v) for m, v in r.items()}
                    for k, r in results.items()},
           hyper=hyper, winners={k: winners.count(k) for k in cand}, threads=THREADS,
           minutes=round((time.time() - t_all) / 60, 1))
OUT_ROOT.mkdir(parents=True, exist_ok=True)
tmp = OUT_ROOT / (tag + ".tmp")
json.dump(out, open(tmp, "w"), indent=1)
os.replace(tmp, OUT_ROOT / (tag + ".json"))
pdir = OUT_ROOT / "preds"; pdir.mkdir(exist_ok=True)
np.savez_compressed(pdir / (tag + ".npz"), Yte=Yte.astype(np.float32),
                    **{f"te_{k}": v.astype(np.float32) for k, v in te_preds.items()},
                    **{f"val_{k}": v.astype(np.float32) for k, v in val_preds.items()},
                    **{f"per_{k}": v.astype(np.float32) for k, v in per.items()})
print(f"DONE {tag} in {out['minutes']} min", flush=True)
