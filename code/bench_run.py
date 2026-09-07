"""The kernel-and-network family set on a new corpus (papers 3-5), one seed, one JSON.

Families (all selected on the corpus's validation split, test read once per family):
  ridge        linear ridge on the standardized inputs (the sanity floor)
  krr          exact Matern-5/2 kernel ridge, isotropic, scale x nugget on validation; Nystrom with --centers
               landmarks when the training set is larger than --exact_max rows
  krr_kf       the same kernel with per-input length scales learned by the l2 kernel flow on the physical
               targets (kf_kernels.py's kf_ard, physical objective), size and nugget on validation
  mlp          a residual MLP on standardized targets (width/depth per corpus), early-stopped on validation
  mlp_ens      mean of --members networks
  mlp_resid    mlp + exact/Nystrom Matern correction of its residuals, selected on the corrected validation error
  dkr          Matern kernel on the network's last hidden layer
  lowfi_resid  (multi-fidelity corpora) the low-fidelity prediction as the mean, kernel on the gap
  select       per-output-coordinate validation selection over the heads
  stack        per-corpus convex combination of the heads, weights on validation
Writes results/<tag>.json with the corpus metric and any extra metrics the source paper reports.
    python bench_run.py --corpus climsim --seed 0 --ntrain 100000
    python bench_run.py --corpus pkanrtm --seed 0 --lowfi 1
"""
import argparse, json, os, pathlib, time
import os as _os
_T = _os.environ.get("NMKC_THREADS", "4")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    _os.environ.setdefault(_v, _T)
import numpy as np
from scipy.linalg import cho_factor, cho_solve
import torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(int(_T))
import bench_data

p = argparse.ArgumentParser()
p.add_argument("--corpus", required=True, choices=["climsim", "pkanrtm", "trl2d", "advection", "darcy", "burgers", "well", "rrtmgp", "qm9"])
p.add_argument("--well_name", default="active_matter", help="The Well dataset for --corpus well (well_data.FIELDS)")
p.add_argument("--nu", type=float, default=0.1, help="Burgers viscosity (PDEBench file)")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--ntrain", type=int, default=0)
p.add_argument("--lowfi", type=int, default=0)
p.add_argument("--beta", type=float, default=1.0)
p.add_argument("--families", default="all")
p.add_argument("--members", type=int, default=3)
p.add_argument("--epochs", type=int, default=0)
p.add_argument("--width", type=int, default=0)
p.add_argument("--exact_max", type=int, default=20000)
p.add_argument("--centers", type=int, default=6000)
p.add_argument("--kf_steps", type=int, default=300)
p.add_argument("--rank", type=int, default=0, help="TRL2D: PCA rank of the input and target representations (default 256)")
p.add_argument("--tag", default="")
p.add_argument("--hpo", type=int, default=0, help="mlp_hpo: random-search trials on validation (0 = 6)")
p.add_argument("--rrtmgp_test_files", default="", help="rrtmgp: comma list of file stems whose columns form the test set (distribution shift)")
p.add_argument("--ncal", type=int, default=1000, help="test cases carved for the split-conformal summary of the stack")
p.add_argument("--smoke", action="store_true")
args = p.parse_args()
OUT = pathlib.Path(os.environ.get("P2_OUT", "results"))
t0 = time.time()
if args.corpus == "climsim":
    D = bench_data.climsim(args.seed, args.ntrain or 100000, nval=20000, ntest=20000); EP, W, DEPTH, BS = 60, 512, 4, 1024
elif args.corpus == "pkanrtm":
    D = bench_data.pkanrtm(args.seed, args.ntrain, args.lowfi); EP, W, DEPTH, BS = 120, 384, 4, 1024
elif args.corpus == "trl2d":
    D = bench_data.trl2d(args.seed, args.ntrain, rank_in=args.rank or 256, rank_out=args.rank or 256); EP, W, DEPTH, BS = 200, 512, 4, 256
elif args.corpus == "well":
    import well_data
    D = well_data.well2d(args.well_name, args.seed, args.ntrain, rank_in=args.rank or 256, rank_out=args.rank or 256); EP, W, DEPTH, BS = 200, 512, 4, 256
elif args.corpus == "rrtmgp":
    import rrtmgp_data
    D = rrtmgp_data.rrtmgp(args.seed, args.ntrain or 100000, test_files=tuple(args.rrtmgp_test_files.split(",")) if args.rrtmgp_test_files else None)
    EP, W, DEPTH, BS = 60, 512, 4, 1024
elif args.corpus == "qm9":
    import qm9_data
    D = qm9_data.qm9(args.seed, args.ntrain or 100000); EP, W, DEPTH, BS = 200, 512, 4, 512
elif args.corpus == "burgers":
    D = bench_data.burgers(args.nu, args.seed, args.ntrain or 8000); EP, W, DEPTH, BS = 200, 512, 4, 256
elif args.corpus == "advection":
    D = bench_data.advection(args.seed, args.ntrain or 20000); EP, W, DEPTH, BS = 200, 512, 4, 512
else:
    D = bench_data.darcy(args.beta, args.seed, args.ntrain or 8000); EP, W, DEPTH, BS = 200, 512, 4, 256
if args.epochs: EP = args.epochs
if args.width: W = args.width
if args.smoke:
    EP = 3
    for k in ("Xtr", "Ztr"):
        D[k] = D[k][:3000]
    D["Yobj"], D["ynorm2"] = D["Yobj"][:3000], D["ynorm2"][:3000]
    if "lowfi_pred" in D:
        D["lowfi_pred"]["tr"] = D["lowfi_pred"]["tr"][:3000]
Xtr, Xva, Xte, Ztr, Zva, Zte = (D[k] for k in ("Xtr", "Xva", "Xte", "Ztr", "Zva", "Zte"))
n, d = Xtr.shape; q = Ztr.shape[1]
err = D["err"]
FAMS = None if args.families == "all" else set(args.families.split(","))
want = lambda f: FAMS is None or f in FAMS or "all" in FAMS          # the campaign families; "all,<extra>" adds opt-in members
want_wc = lambda f: FAMS is not None and f in FAMS                   # the world-class members run only when named
exact = n <= args.exact_max
print(f"{D['tag']}: n={n} d={d} q={q} exact={exact} epochs={EP}", flush=True)
sub_tune = np.random.default_rng(args.seed + 7).permutation(n)[:min(6000, n)]
centers = np.random.default_rng(args.seed + 9).permutation(n)[:min(args.centers, n)]


# ---------------- kernels ----------------
def sqd(A, B):
    return np.maximum((A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2 * A @ B.T, 0.0)


def m52(D2, ls):
    a = np.sqrt(5.0) * np.sqrt(D2) / ls
    return (1 + a + (5.0 / 3.0) * (D2 / ls ** 2)) * np.exp(-a)


def solve(K, Y, nug):
    Kr = K.copy(); Kr.flat[::len(K) + 1] += nug * len(K)
    return cho_solve(cho_factor(Kr, lower=True, check_finite=False, overwrite_a=True), Y, check_finite=False)


def krr_fit_predict(Ftr, Fva, Fte, Ytr_, val_fn, w=None, label="krr"):
    """Scale and nugget on validation over a subsample; then exact (n <= exact_max) or Nystrom solve on all rows.
    val_fn(pred_va) -> validation error in the corpus metric. Returns (pred_va, pred_te, hyper)."""
    if w is not None:
        Ftr, Fva, Fte = Ftr * w, Fva * w, Fte * w
    Fs = Ftr[sub_tune]; D2s, D2vs = sqd(Fs, Fs), sqd(Fva, Fs)
    med = float(np.sqrt(np.median(D2s[np.triu_indices(len(Fs), 1)]))) + 1e-12
    best = (np.inf, None)
    for sc in (0.5, 1.0, 2.0, 4.0):
        Ks, Kvs = m52(D2s, sc * med), m52(D2vs, sc * med)
        for nug in (1e-8, 1e-6, 1e-4, 1e-2):
            try:
                e = val_fn(Kvs @ solve(Ks, Ytr_[sub_tune], nug))
            except np.linalg.LinAlgError:
                continue
            if e < best[0]:
                best = (e, (sc * med, nug))
    ls, nug = best[1]
    if exact:
        K = m52(sqd(Ftr, Ftr), ls); alpha = solve(K, Ytr_, nug); del K
        outs = []
        for F_ in (Fva, Fte):
            pr = np.empty((len(F_), Ytr_.shape[1]))
            for k in range(0, len(F_), 4000):
                pr[k:k + 4000] = m52(sqd(F_[k:k + 4000], Ftr), ls) @ alpha
            outs.append(pr)
    else:
        # Nystrom ridge: alpha_m = (Knm' Knm + nug n Kmm)^-1 Knm' y
        Fm = Ftr[centers]; Kmm = m52(sqd(Fm, Fm), ls)
        KtK = np.zeros((len(Fm), len(Fm))); Kty = np.zeros((len(Fm), Ytr_.shape[1]))
        for k in range(0, n, 8000):
            Knm = m52(sqd(Ftr[k:k + 8000], Fm), ls); KtK += Knm.T @ Knm; Kty += Knm.T @ Ytr_[k:k + 8000]
        A = KtK + nug * n * Kmm + 1e-8 * np.trace(KtK) / len(Fm) * np.eye(len(Fm))
        alpha = cho_solve(cho_factor(A, lower=True, check_finite=False), Kty, check_finite=False)
        outs = [m52(sqd(F_, Fm), ls) @ alpha for F_ in (Fva, Fte)]
    print(f"  {label}: scale {ls/med:.1f} x med, nugget {nug:g}, val {100*best[0]:.4f}% ({'exact' if exact else 'nystrom %d' % len(centers)})", flush=True)
    return outs[0], outs[1], dict(scale=float(ls), nugget=nug, med=med, val_sub=float(best[0]), exact=exact)


def kf_shape(Ftr, Yobj, yn2, steps, batch=600, lr=0.05, seed=0):
    """l2 kernel flow on the physical targets for per-input log length scales (shape only)."""
    torch.set_default_dtype(torch.float64)
    Xt, Yt, Nt = torch.as_tensor(Ftr), torch.as_tensor(Yobj), torch.as_tensor(yn2)
    med0 = np.sqrt(np.median(sqd(Ftr[sub_tune[:2000]], Ftr[sub_tune[:2000]])[np.triu_indices(min(2000, len(sub_tune)), 1)])) + 1e-12
    log_ell = torch.tensor(np.full(Ftr.shape[1], np.log(med0)), requires_grad=True)
    opt = torch.optim.Adam([log_ell], lr=lr); rng = np.random.default_rng(seed)
    for it in range(steps):
        idx = torch.as_tensor(rng.choice(len(Ftr), min(batch, len(Ftr)), replace=False)); half = len(idx) // 2
        Fb = Xt[idx] / torch.exp(log_ell)[None, :]
        D2 = torch.clamp((Fb * Fb).sum(1)[:, None] + (Fb * Fb).sum(1)[None, :] - 2 * Fb @ Fb.T, min=0.0)
        a = np.sqrt(5.0) * torch.sqrt(D2 + 1e-12); K = (1 + a + a * a / 3) * torch.exp(-a)
        Kc = K[:half, :half] + 1e-4 * half * torch.eye(half)
        pred = K[half:, :half] @ torch.cholesky_solve(Yt[idx][:half], torch.linalg.cholesky(Kc))
        loss = (((Yt[idx][half:] - pred) ** 2).sum(1) / Nt[idx][half:]).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    ell = np.exp(log_ell.detach().numpy()); torch.set_default_dtype(torch.float32)
    return ell / np.exp(np.log(ell).mean())


# ---------------- network ----------------
zm, zs = Ztr.mean(0), Ztr.std(0) + 1e-9
f32 = lambda a: torch.as_tensor(np.asarray(a, np.float32))
Xt, Xv, Xe = f32(Xtr), f32(Xva), f32(Xte)
Yt, Yv = f32((Ztr - zm) / zs), f32((Zva - zm) / zs)


class Net(nn.Module):
    def __init__(self, d_in, d_out, width, depth):
        super().__init__()
        self.inp = nn.Linear(d_in, width); self.hid = nn.ModuleList([nn.Linear(width, width) for _ in range(depth - 1)]); self.out = nn.Linear(width, d_out)

    def forward(self, x, features=False):
        h = F.silu(self.inp(x))
        for l in self.hid:
            h = h + F.silu(l(h))
        return (self.out(h), h) if features else self.out(h)


def train_net(seed):
    torch.manual_seed(seed)
    net = Net(d, q, W, DEPTH)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EP, eta_min=1e-5)
    best, best_state, bad = np.inf, None, 0
    for ep in range(EP):
        net.train(); perm = torch.randperm(n)
        for k in range(0, n, BS):
            i = perm[k:k + BS]
            if len(i) < 8: continue
            loss = F.mse_loss(net(Xt[i]), Yt[i]); opt.zero_grad(); loss.backward(); opt.step()
        sched.step(); net.eval()
        with torch.no_grad():
            vl = F.mse_loss(net(Xv), Yv).item()
        if vl < best - 1e-7:
            best, bad = vl, 0; best_state = {k2: v.clone() for k2, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 25: break
    if best_state is not None: net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        P = [np.concatenate([net(X[k:k + 20000]).numpy() for k in range(0, len(X), 20000)]).astype(np.float64) * zs + zm for X in (Xt, Xv, Xe)]
        Hs = [np.concatenate([net(X[k:k + 20000], features=True)[1].numpy() for k in range(0, len(X), 20000)]).astype(np.float64) for X in (Xt, Xv, Xe)]
    return P, Hs, ep + 1


results, hyper, heads_va, heads_te = {}, {}, {}, {}


def rec(name, pv, pt, hp=None, head=True):
    results[name] = dict(val=err(pv, "va"), test=err(pt, "te"))
    for mname, fn in D.get("extra_metrics", {}).items():
        results[name]["test_" + mname] = fn(pt, "te")
    hyper[name] = hp or {}
    if head:
        heads_va[name], heads_te[name] = pv, pt
    print(f"== {name}: val {100*results[name]['val']:.4f}% test {100*results[name]['test']:.4f}% " +
          " ".join(f"{k}={v:.4g}" for k, v in results[name].items() if k.startswith("test_")) + f" [{(time.time()-t0)/60:.1f} min]", flush=True)


vfn = lambda pv: err(pv, "va")
if want("ridge"):
    Xb = np.concatenate([Xtr, np.ones((n, 1))], 1); G = Xb.T @ Xb; b = Xb.T @ Ztr
    best = (np.inf, None)
    for lam in (1e-8, 1e-6, 1e-4, 1e-2, 1.0):
        Wc = np.linalg.solve(G + lam * n * np.eye(d + 1), b); e = err(np.concatenate([Xva, np.ones((len(Xva), 1))], 1) @ Wc, "va")
        if e < best[0]: best = (e, Wc)
    rec("ridge", np.concatenate([Xva, np.ones((len(Xva), 1))], 1) @ best[1], np.concatenate([Xte, np.ones((len(Xte), 1))], 1) @ best[1])
if want("krr"):
    pv, pt, hp = krr_fit_predict(Xtr, Xva, Xte, Ztr, vfn, label="krr"); rec("krr", pv, pt, hp)
if want("krr_kf"):
    w = 1.0 / kf_shape(Xtr, D["Yobj"], D["ynorm2"], 30 if args.smoke else args.kf_steps, seed=args.seed)
    pv, pt, hp = krr_fit_predict(Xtr, Xva, Xte, Ztr, vfn, w=w, label="krr_kf"); hp["ell_rel"] = {nm: round(float(1 / x), 4) for nm, x in zip(D["names"][:len(w)], w)}
    rec("krr_kf", pv, pt, hp)
if "lowfi_pred" in D and want("lowfi_resid"):
    L = D["lowfi_pred"]; to_z = lambda Y: (Y - D["Yph"]["tr"].mean(0)) / (D["Yph"]["tr"].std(0) + 1e-12)
    lz = {k: to_z(L[k]) for k in L}
    rec("lowfi_only", lz["va"], lz["te"], head=False)
    pv, pt, hp = krr_fit_predict(Xtr, Xva, Xte, Ztr - lz["tr"], lambda pv_: err(lz["va"] + pv_, "va"), label="lowfi_resid")
    rec("lowfi_resid", lz["va"] + pv, lz["te"] + pt, hp)
need_net = any(want(f) for f in ("mlp", "mlp_ens", "mlp_resid", "dkr", "select", "stack"))
if need_net:
    Ps, Hs0 = [], None
    for m in range(max(1, args.members if want("mlp_ens") else 1)):
        P, Hs, ep = train_net(args.seed * 100 + m); Ps.append(P)
        if m == 0:
            Hs0 = Hs; rec("mlp", P[1], P[2], dict(epochs=ep))
    if len(Ps) > 1:
        ens = [np.mean([P[k] for P in Ps], 0) for k in range(3)]; rec("mlp_ens", ens[1], ens[2])
    base = Ps[0] if len(Ps) == 1 else ens
    if want("mlp_resid"):
        pv, pt, hp = krr_fit_predict(Xtr, Xva, Xte, Ztr - base[0], lambda pv_: err(base[1] + pv_, "va"), label="mlp_resid")
        rec("mlp_resid", base[1] + pv, base[2] + pt, hp)
    if want("dkr"):
        mu, sd = Hs0[0].mean(0), Hs0[0].std(0) + 1e-9
        pv, pt, hp = krr_fit_predict((Hs0[0] - mu) / sd, (Hs0[1] - mu) / sd, (Hs0[2] - mu) / sd, Ztr, vfn, label="dkr"); rec("dkr", pv, pt, hp)
# ---- world-class members (added 2026-09-03): pasted into bench_run.py by patch_bench_wc.py ----
# Network variants and training upgrades: periodic-linear input embeddings (Gorishniy et al. 2022), an
# exponential moving average of the weights (SWA/EMA), C-Mixup for regression (Yao et al. 2022), sharpness-aware
# minimization (Foret et al. 2021); a gradient-boosted-tree baseline (histogram GBDT, one model per output); and
# the infinite-width ReLU network kernel (NNGP, Lee et al. 2018) as a further exact kernel.


class PLR(nn.Module):
    """Periodic embeddings of every input coordinate (k frequencies each), then one linear map to the width."""
    def __init__(self, d_in, width, k=16, sigma=1.0):
        super().__init__()
        self.c = nn.Parameter(torch.randn(d_in, k) * sigma)
        self.lin = nn.Linear(2 * k * d_in, width)

    def forward(self, x):
        z = 2 * np.pi * x[:, :, None] * self.c[None, :, :]
        return self.lin(torch.cat([torch.cos(z), torch.sin(z)], -1).flatten(1))


class NetX(nn.Module):
    """Residual MLP with an optional PLR front end and dropout; features = the last hidden layer."""
    def __init__(self, d_in, d_out, width, depth, kind="mlp", dropout=0.0):
        super().__init__()
        self.inp = PLR(d_in, width) if kind == "plr" else nn.Linear(d_in, width)
        self.hid = nn.ModuleList([nn.Linear(width, width) for _ in range(depth - 1)])
        self.out = nn.Linear(width, d_out); self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, features=False):
        h = F.silu(self.inp(x))
        for l in self.hid:
            h = h + self.drop(F.silu(l(h)))
        return (self.out(h), h) if features else self.out(h)


def cmixup_batch(xb, yb, alpha=2.0):
    """C-Mixup: partners drawn with probability decaying in target distance; convex mix of inputs and targets."""
    with torch.no_grad():
        d2 = torch.cdist(yb, yb).pow(2); sig2 = d2.median().clamp_min(1e-6)
        P = torch.exp(-d2 / (2 * sig2)); P.fill_diagonal_(0); P = P / P.sum(1, keepdim=True).clamp_min(1e-12)
        j = torch.multinomial(P, 1).squeeze(1)
        lam = torch.distributions.Beta(alpha, alpha).sample((len(xb), 1))
    return lam * xb + (1 - lam) * xb[j], lam * yb + (1 - lam) * yb[j]


def train_netx(seed, kind="mlp", swa=False, cmixup=False, sam=False, width=None, depth=None, lr=1e-3, wd=1e-6, dropout=0.0, epochs=None):
    """train_net with the upgrades; returns (preds, features, epochs, info)."""
    torch.manual_seed(seed)
    Wd, Dp, EPx = width or W, depth or DEPTH, epochs or EP
    net = NetX(d, q, Wd, Dp, kind=kind, dropout=dropout)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPx, eta_min=lr / 100)
    ema = {k2: v.detach().clone() for k2, v in net.state_dict().items()} if swa else None
    best, best_state, bad, src = np.inf, None, 0, "raw"
    for ep in range(EPx):
        net.train(); perm = torch.randperm(n)
        for k in range(0, n, BS):
            i = perm[k:k + BS]
            if len(i) < 8: continue
            xb, yb = Xt[i], Yt[i]
            if cmixup: xb, yb = cmixup_batch(xb, yb)
            loss = F.mse_loss(net(xb), yb); opt.zero_grad(); loss.backward()
            if sam:
                with torch.no_grad():
                    gn = torch.sqrt(sum((p_.grad ** 2).sum() for p_ in net.parameters() if p_.grad is not None)) + 1e-12
                    eps = [(0.05 * p_.grad / gn) if p_.grad is not None else None for p_ in net.parameters()]
                    for p_, e_ in zip(net.parameters(), eps):
                        if e_ is not None: p_.add_(e_)
                opt.zero_grad(); F.mse_loss(net(xb), yb).backward()
                with torch.no_grad():
                    for p_, e_ in zip(net.parameters(), eps):
                        if e_ is not None: p_.sub_(e_)
            opt.step()
            if ema is not None:
                with torch.no_grad():
                    for k2, v in net.state_dict().items():
                        if v.dtype.is_floating_point: ema[k2].mul_(0.999).add_(v, alpha=0.001)
                        else: ema[k2].copy_(v)
        sched.step(); net.eval()
        with torch.no_grad():
            vl = F.mse_loss(net(Xv), Yv).item()
            if ema is not None:
                cur = {k2: v.clone() for k2, v in net.state_dict().items()}; net.load_state_dict(ema)
                vl_e = F.mse_loss(net(Xv), Yv).item(); net.load_state_dict(cur)
        cands = [(vl, "raw", None)] + ([(vl_e, "ema", ema)] if ema is not None else [])
        vbest, vsrc, vstate = min(cands, key=lambda t_: t_[0])
        if vbest < best - 1e-7:
            best, bad, src = vbest, 0, vsrc
            best_state = {k2: v.clone() for k2, v in (vstate if vstate is not None else net.state_dict()).items()}
        else:
            bad += 1
            if bad >= 25: break
    if best_state is not None: net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        P = [np.concatenate([net(X[k:k + 20000]).numpy() for k in range(0, len(X), 20000)]).astype(np.float64) * zs + zm for X in (Xt, Xv, Xe)]
        Hs = [np.concatenate([net(X[k:k + 20000], features=True)[1].numpy() for k in range(0, len(X), 20000)]).astype(np.float64) for X in (Xt, Xv, Xe)]
    return P, Hs, ep + 1, dict(kind=kind, swa=swa, cmixup=cmixup, sam=sam, width=Wd, depth=Dp, lr=lr, wd=wd, dropout=dropout, best_val_mse=float(best), source=src)


def nngp_kernel(A, B, depth, sb2, sw2=2.0):
    """Infinite-width ReLU network kernel (NNGP), He scaling, depth hidden layers; closed-form arc-cosine recursion."""
    Kab = sw2 * (A @ B.T) / A.shape[1] + sb2
    Kaa = sw2 * (A * A).sum(1) / A.shape[1] + sb2; Kbb = sw2 * (B * B).sum(1) / B.shape[1] + sb2
    for _ in range(depth - 1):
        s = np.sqrt(np.maximum(Kaa[:, None] * Kbb[None, :], 1e-30)); th = np.arccos(np.clip(Kab / s, -1.0, 1.0))
        Kab = sw2 / (2 * np.pi) * s * (np.sin(th) + (np.pi - th) * np.cos(th)) + sb2
        Kaa = sw2 / 2 * Kaa + sb2; Kbb = sw2 / 2 * Kbb + sb2
    return Kab


def nngp_fit_predict(Ftr, Fva, Fte, Ytr_, val_fn, label="krr_nngp"):
    """Depth, bias variance and nugget on validation over the tuning subsample; exact solve on all rows (n <= exact_max)."""
    Fs = Ftr[sub_tune]; best = (np.inf, None)
    for depth in (2, 3, 5):
        for sb2 in (0.0, 0.1):
            Ks, Kvs = nngp_kernel(Fs, Fs, depth, sb2), nngp_kernel(Fva, Fs, depth, sb2)
            for nug in (1e-8, 1e-6, 1e-4, 1e-2):
                try:
                    e = val_fn(Kvs @ solve(Ks, Ytr_[sub_tune], nug))
                except np.linalg.LinAlgError:
                    continue
                if e < best[0]: best = (e, (depth, sb2, nug))
    depth, sb2, nug = best[1]
    if not exact:
        print(f"  {label}: n={n} exceeds exact_max; skipped", flush=True); return None
    K = nngp_kernel(Ftr, Ftr, depth, sb2); alpha = solve(K, Ytr_, nug); del K
    outs = []
    for F_ in (Fva, Fte):
        pr = np.empty((len(F_), Ytr_.shape[1]))
        for k in range(0, len(F_), 4000):
            pr[k:k + 4000] = nngp_kernel(F_[k:k + 4000], Ftr, depth, sb2) @ alpha
        outs.append(pr)
    print(f"  {label}: depth {depth} sb2 {sb2} nugget {nug:g} val {100*best[0]:.4f}%", flush=True)
    return outs[0], outs[1], dict(depth=depth, sb2=sb2, nugget=nug, val_sub=float(best[0]))


def gbdt_fit_predict(Ftr, Fva, Fte, Ytr_, max_rows=100000):
    """Histogram gradient-boosted trees, one model per output, early stopping on a held-out tenth of the rows."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    rows = np.random.default_rng(args.seed + 11).permutation(len(Ftr))[:min(max_rows, len(Ftr))]
    pv, pt = np.empty((len(Fva), Ytr_.shape[1])), np.empty((len(Fte), Ytr_.shape[1])); iters = []
    for j in range(Ytr_.shape[1]):
        m = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1e-3,
                                          early_stopping=True, validation_fraction=0.1, n_iter_no_change=30, random_state=args.seed)
        m.fit(Ftr[rows], Ytr_[rows, j]); pv[:, j] = m.predict(Fva); pt[:, j] = m.predict(Fte); iters.append(int(m.n_iter_))
    return pv, pt, dict(rows=int(len(rows)), median_iters=float(np.median(iters)))


if want_wc("gbdt"):
    pv, pt, hp = gbdt_fit_predict(Xtr, Xva, Xte, Ztr); rec("gbdt", pv, pt, hp)
if want_wc("krr_nngp"):
    r_ = nngp_fit_predict(Xtr, Xva, Xte, Ztr, vfn)
    if r_ is not None: rec("krr_nngp", r_[0], r_[1], r_[2])
WC = [("mlp_plr", dict(kind="plr")), ("mlp_swa", dict(swa=True)), ("mlp_cmixup", dict(cmixup=True)), ("mlp_sam", dict(sam=True)),
      ("mlp_wc", dict(kind="plr", swa=True, cmixup=True))]
if want_wc("mlp_hpo"):
    # a small random search on validation (network only), then the winner is trained as mlp_hpo
    rng_h = np.random.default_rng(args.seed + 21); trials = []
    for t_ in range(args.hpo or 6):
        cfg = dict(width=int(rng_h.choice([256, 512, 1024])), depth=int(rng_h.choice([3, 4, 6])), lr=float(rng_h.choice([3e-4, 1e-3, 3e-3])),
                   wd=float(rng_h.choice([1e-6, 1e-4])), dropout=float(rng_h.choice([0.0, 0.1])), kind=str(rng_h.choice(["mlp", "plr"])))
        P_, H_, ep_, info_ = train_netx(args.seed * 100 + 50 + t_, epochs=max(EP // 2, 10), **cfg)
        trials.append((err(P_[1], "va"), cfg)); print(f"  hpo trial {t_}: val {100*trials[-1][0]:.4f}% {cfg}", flush=True)
    cfg = min(trials, key=lambda t_: t_[0])[1]
    P, Hs, ep, info = train_netx(args.seed * 100 + 70, swa=True, **cfg); rec("mlp_hpo", P[1], P[2], dict(info, epochs=ep, trials=[(round(100 * e_, 4), c_) for e_, c_ in trials]))
for name_, kw_ in WC:
    if want_wc(name_):
        P, Hs, ep, info = train_netx(args.seed * 100 + 40, **kw_); rec(name_, P[1], P[2], dict(info, epochs=ep))
        if name_ == "mlp_wc":
            mu, sd = Hs[0].mean(0), Hs[0].std(0) + 1e-9
            pv, pt, hp = krr_fit_predict((Hs[0] - mu) / sd, (Hs[1] - mu) / sd, (Hs[2] - mu) / sd, Ztr, vfn, label="dkr_wc"); rec("dkr_wc", pv, pt, hp)
            pv, pt, hp = krr_fit_predict(Xtr, Xva, Xte, Ztr - P[0], lambda pv_: err(P[1] + pv_, "va"), label="mlp_wc_resid"); rec("mlp_wc_resid", P[1] + pv, P[2] + pt, hp)
# ---- end of the world-class members block ----

cand = [h for h in ("krr", "krr_kf", "krr_nngp", "gbdt", "lowfi_resid", "mlp", "mlp_ens", "mlp_resid", "dkr", "mlp_plr", "mlp_swa", "mlp_cmixup", "mlp_sam", "mlp_wc", "dkr_wc", "mlp_wc_resid", "mlp_hpo") if h in heads_va]
if want("select") and len(cand) > 1:
    Zsel_va, Zsel_te = np.empty_like(Zva), np.empty_like(Zte); picks = {h: 0 for h in cand}
    for j in range(q):
        errs = [np.sqrt(((heads_va[h][:, j] - Zva[:, j]) ** 2).mean()) for h in cand]; k = int(np.argmin(errs)); picks[cand[k]] += 1
        Zsel_va[:, j] = heads_va[cand[k]][:, j]; Zsel_te[:, j] = heads_te[cand[k]][:, j]
    rec("select", Zsel_va, Zsel_te, dict(picks=picks), head=False)
if want("stack") and len(cand) > 1:
    from scipy.optimize import minimize
    M_ = len(cand)
    # The reported error is a mean of per-sample relative norms and every physical map is affine, so for convex weights
    # the objective is mean_i sqrt(w' G_i w) / den_i with G_i the per-sample residual Gram of the heads: built once, in
    # row chunks (the full-field corpora reconstruct gigabytes per head), instead of one reconstruction per SLSQP call.
    n_va = heads_va[cand[0]].shape[0]; Gm = np.zeros((n_va, M_, M_)); den_va = np.empty(n_va)
    for i0 in range(0, n_va, 200):
        rows = np.arange(i0, min(i0 + 200, n_va))
        T_ = np.asarray(D["Yph"]["va"][rows], np.float64)
        R_ = np.stack([np.asarray(D["phys_pred"](heads_va[h][rows], "va", rows), np.float64) - T_ for h in cand])
        Gm[rows] = np.einsum("mcd,kcd->cmk", R_, R_); den_va[rows] = D["den"]("va", rows)
    obj = lambda w: float(np.mean(np.sqrt(np.maximum(np.einsum("h,nhk,k->n", w, Gm, w), 0.0)) / den_va))
    res = minimize(obj, np.ones(M_) / M_, bounds=[(0, 1)] * M_, constraints={"type": "eq", "fun": lambda w: w.sum() - 1}, method="SLSQP", options=dict(maxiter=300, ftol=1e-12))
    w = np.maximum(res.x, 0); w /= w.sum()
    rec("stack", sum(wi * heads_va[h] for wi, h in zip(w, cand)), sum(wi * heads_te[h] for wi, h in zip(w, cand)), dict(weights={h: round(float(x), 4) for h, x in zip(cand, w)}), head=False)

# ---- split-conformal summary of the final head (experiment 16): calibration cases carved from the test block,
# which no selection touched; raw per-case relative error as the score; coverage on the disjoint remainder ----
conformal = None
final_head = next((h for h in ("stack", "select") if h in results), None)
ncal = min(args.ncal, len(Zte) // 4)
if final_head is not None and ncal >= 100:
    Pf_te = heads_te[final_head] if final_head in heads_te else None
    if final_head == "stack" and Pf_te is None:
        Pf_te = sum(wi * heads_te[h] for wi, h in zip(w, cand))
    if final_head == "select" and Pf_te is None:
        Pf_te = Zsel_te
    if Pf_te is not None:
        e_case = np.empty(len(Zte))
        for i0 in range(0, len(Zte), 500):
            rows = np.arange(i0, min(i0 + 500, len(Zte)))
            T_ = np.asarray(D["Yph"]["te"][rows], np.float64); R_ = np.asarray(D["phys_pred"](Pf_te[rows], "te", rows), np.float64) - T_
            e_case[rows] = np.linalg.norm(R_, axis=1) / D["den"]("te", rows)
        perm_c = np.random.default_rng(args.seed + 31).permutation(len(Zte)); cal, ev_ = perm_c[:ncal], perm_c[ncal:]
        conformal = dict(head=final_head, n_cal=int(len(cal)), n_eval=int(len(ev_)), score="relative L2 per case")
        for alpha in (0.1, 0.05):
            k_ = int(np.ceil((1 - alpha) * (len(cal) + 1))); qv = np.sort(e_case[cal])[min(k_, len(cal)) - 1]
            conformal[f"a{alpha:g}"] = dict(target=1 - alpha, q=float(qv), coverage=float(np.mean(e_case[ev_] <= qv)))
        print(f"== conformal ({final_head}): 90% band q {conformal['a0.1']['q']:.4f} cover {conformal['a0.1']['coverage']:.4f}; 95% q {conformal['a0.05']['q']:.4f} cover {conformal['a0.05']['coverage']:.4f}", flush=True)

tag = args.tag or (D["tag"] + "_bench")
out = dict(tag=tag, kind="bench", corpus=args.corpus, seed=args.seed, n=n, d=d, q=q, exact=exact, epochs=EP, width=W, smoke=bool(args.smoke), conformal=conformal,
           results={k: {m: (100 * v if m in ("val", "test") else v) for m, v in r.items()} for k, r in results.items()}, hyper=hyper,
           extra={k: v for k, v in D.items() if k in ("pca_evr_in", "pca_evr_out", "pca_fit_rows", "persistence_err", "pca_recon_err", "extra_metrics_lowfi", "err_floor", "frac_below_floor_te", "target", "persistence_vrmse", "rank_in", "rank_out")},
           minutes=round((time.time() - t0) / 60, 1))
OUT.mkdir(parents=True, exist_ok=True)
tmp = OUT / (tag + ".tmp"); json.dump(out, open(tmp, "w"), indent=1); os.replace(tmp, OUT / (tag + ".json"))
np.savez_compressed(OUT / "preds" / (tag + ".npz"), **{k: v.astype(np.float32) for k, v in heads_te.items()}) if (OUT / "preds").exists() else None
print(f"DONE {tag} in {out['minutes']} min", flush=True)
