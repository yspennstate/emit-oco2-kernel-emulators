"""Equal-budget regularization suite for the mean network, each arm followed by the kernel correction (GPU lane).

One lane = one corpus, one seed, every regularizer in the suite, one JSON. Each regularizer trains the same
residual MLP under the same budget (the same number of training runs of the same length, every hyperparameter
chosen on validation), and each selected mean is then corrected twice by the same Matern-5/2 kernel machinery
the rest of the programme uses:

    <arm>          the regularized mean network on its own
    <arm>_resid    exact (or Nystrom) Matern-5/2 kernel ridge on the network's residual, added back to the mean
    <arm>_dkr      the same kernel on the network's last hidden layer

so the table answers the question the suite exists for: which regularizer leaves the best residual for the
kernel, which is not the same question as which regularizer has the best network. `krr` (the kernel alone on
the standardized inputs) is the reference row.

Arms (method names say what they are; --methods selects a subset):
    mlp_base       AdamW, weight decay 1e-6, cosine schedule, best-epoch checkpoint on validation (the control)
    mlp_wd         weight-decay grid                      mlp_ema        exponential moving average of the weights
    mlp_swa        equal weight average over the tail      mlp_sam        sharpness-aware minimization, rho grid
    mlp_asam       adaptive SAM (elementwise |w| metric)   mlp_cmixup     C-Mixup (Yao et al. 2022), alpha grid
    mlp_dropout    dropout on the residual branches        mlp_specnorm   hard spectral normalization of the hidden stack
    mlp_specpen    soft spectral-norm penalty              mlp_noise      Gaussian input noise
    mlp_jac        Jacobian Frobenius penalty (Hutchinson) mlp_earlystop  patience grid, no other regularizer
    mlp_realmlp    the RealMLP-TD training bundle: periodic-linear input embeddings, smooth input clipping,
                   decay on weights only, warmup + cosine, dropout (Holzmueller et al. 2024)

EQUAL BUDGET. Every arm gets exactly --budget training runs of the same epoch count, and the winner is the run
with the lowest CORPUS validation error; an arm whose grid is shorter than the budget spends the rest of it on
extra seeds. --budget_mode passes additionally shortens the two-pass arms (SAM, ASAM, Jacobian) so that the
number of forward/backward passes matches the one-pass arms instead of the number of epochs. The realized
budget of every arm (runs, epochs, steps, passes, seconds) is written into the result JSON.

Corpora, through the loaders and splits already in ~/p23/code, so the numbers are comparable with the lanes
already run: emit (emit_campaign.py's split and physical metric, PCA-64 per component, the four components
regressed jointly), oco2 (jpl_data / oco2_curve.py's reduced and radiance metrics), climsim, pkanrtm
(bench_data.py), rrtmgp (rrtmgp_data.py). The test block is read once per head, at the very end.

MEMORY. n = training rows, d = inputs, q = regression targets, w = width, L = depth, b = batch,
m = Nystrom centers, c = prediction chunk (4,000), H = number of stored heads.

  VRAM (the float64 kernel step dominates; the network is float32 and negligible). The Gram is built one
  row block at a time into a preallocated matrix and the Matern is applied in place, so only one block-sized
  temporary is ever alive beside it; the Cholesky factor is the second full copy.
      exact    16 n^2 + 8 n q + 16 B n + 16 c n      bytes   (B = 2,048 build block, c = 2,000 predict chunk)
               n = 18,883 -> 6.9 GB   n = 24,000 -> 10.8 GB   n = 30,000 -> 16.4 GB   n = 36,000 -> 23.2 GB
      nystrom  8 (5 m^2 + 2 K m + m q)               bytes   (K = 8,000 pass block)
               m = 6,000 -> 1.6 GB    m = 20,000 (CAP_CENTERS) -> 18.6 GB
      network  4 b (d + q + 2 w L) * 3 + 4 (n + n_va + n_te)(d + q)  -> under 1 GB at b = 1024, w = 1024
      The exact branch REFUSES (raises) above CAP_EXACT_GPU = 36,000 rows, above --exact_max, or when the
      formula above exceeds the card fraction the runner granted - on a 32 GB V100 at 0.65 that bites at
      about n = 31,000, so a bigger corpus must use Nystrom (the driver never silently downgrades).
  RAM (host):
      8 (n + n_va + n_te)(d + q + w) for the data, the features and the standardized targets
    + 8 (n_va + n_te) q H            for the stored heads (H = 2 + 3 x arms)
    + the loader's own peak (EMIT holds 4 x 23,313 x 285 float64 = 0.21 GB; ClimSim copies ntrain rows)

Environment: P2_OUT, NMKC_THREADS, EMIT_DATA, NMKC_JPL_DATA, DATA_NEW; GPU lanes CUDA_VISIBLE_DEVICES=<UUID>
(one card, addressed as cuda:0) and P23_GPU_MEM_FRAC. Nothing is downloaded and nothing is written outside P2_OUT.

    python reg_suite_gpu.py --corpus emit --seed 0 --tag emit_s0_regsuite
    python reg_suite_gpu.py --corpus climsim --seed 0 --ntrain 100000 --tag climsim_s0_regsuite
"""
import argparse, copy, hashlib, json, os, pathlib, sys, time

try:
    import resource                                   # Linux (the box); absent on the laptop's CPU-only test
except ImportError:
    resource = None

_T = os.environ.get("NMKC_THREADS", "4")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, _T)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(int(_T))

# ---- caps and refusal constants (DRIVER_CONTRACT: never build a Gram you have not sized) ----
CAP_EXACT_GPU = 36000        # exact fp64 Gram on one V100 at the 0.65 fraction (contract)
CAP_EXACT_CPU = 20000        # the existing exact_max of bench_run.py
CAP_CENTERS = 20000          # Nystrom landmarks
PRED_CHUNK = 2000            # rows of one prediction block
BUILD_BLOCK = 2048           # rows of one Gram build block
PASS_BLOCK = 8000            # rows of one Nystrom pass block
TUNE_SUB = 6000              # rows of the kernel hyperparameter search (krr_fit_predict's subsample)
SCALES = (0.5, 1.0, 2.0, 4.0)
NUGGETS = (1e-8, 1e-6, 1e-4, 1e-2)

# ---- the arms: name -> hyperparameter grid (a run is one dict) ----
GRIDS = {
    "mlp_base": [dict()],
    "mlp_wd": [dict(wd=1e-5), dict(wd=1e-4), dict(wd=1e-3), dict(wd=1e-2)],
    "mlp_ema": [dict(ema=0.999), dict(ema=0.99), dict(ema=0.9999)],
    "mlp_swa": [dict(swa=0.5), dict(swa=0.25)],
    "mlp_sam": [dict(sam=0.05), dict(sam=0.01), dict(sam=0.1), dict(sam=0.2)],
    "mlp_asam": [dict(asam=1.0), dict(asam=0.5), dict(asam=2.0)],
    "mlp_cmixup": [dict(cmixup=2.0), dict(cmixup=0.5)],
    "mlp_dropout": [dict(dropout=0.1), dict(dropout=0.05), dict(dropout=0.2)],
    "mlp_specnorm": [dict(specnorm=True)],
    "mlp_specpen": [dict(specpen=1e-3), dict(specpen=1e-4), dict(specpen=1e-2)],
    "mlp_noise": [dict(noise=0.05), dict(noise=0.01), dict(noise=0.1)],
    "mlp_jac": [dict(jac=1e-3), dict(jac=1e-4), dict(jac=1e-2)],
    "mlp_earlystop": [dict(patience=5), dict(patience=3), dict(patience=10)],
    "mlp_realmlp": [dict(kind="plr", dropout=0.15, wd=2e-2, warmup=0.05, smooth_clip=True, decay_weights_only=True)],
}
TWO_PASS = ("sam", "asam", "jac")     # cost two forward/backward passes per step

CORPUS_DEFAULTS = {                    # epochs, width, depth, batch, training loss, validation stride
    "emit": dict(epochs=150, width=512, depth=3, batch=1024, loss="mse", val_every=1),
    "oco2": dict(epochs=250, width=384, depth=4, batch=512, loss="rel", val_every=5),
    "climsim": dict(epochs=60, width=512, depth=4, batch=1024, loss="mse", val_every=1),
    "rrtmgp": dict(epochs=60, width=512, depth=4, batch=1024, loss="mse", val_every=1),
    "pkanrtm": dict(epochs=120, width=384, depth=4, batch=1024, loss="mse", val_every=1),
}


# ============================== device ==============================

def setup_device(mem_frac=None):
    """The one card the runner assigned (CUDA_VISIBLE_DEVICES=<UUID>, addressed as cuda:0), with the memory
    fraction set BEFORE the first allocation. Never touches nvidia-smi / NVML."""
    on_gpu = torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES", "") != ""
    if not on_gpu:
        return torch.device("cpu"), 0.0
    frac = float(mem_frac if mem_frac is not None else os.environ.get("P23_GPU_MEM_FRAC", "0.65"))
    torch.cuda.set_per_process_memory_fraction(frac, 0)
    torch.backends.cudnn.benchmark = True
    total = torch.cuda.get_device_properties(0).total_memory
    return torch.device("cuda:0"), frac * total


# ============================== kernel ==============================

def exact_bytes(n, q, block=BUILD_BLOCK, chunk=PRED_CHUNK):
    """Sized peak of the exact float64 step: the Gram, its Cholesky factor, the right-hand side, one build
    block's temporary and one prediction block (with its temporary)."""
    return 16.0 * n * n + 8.0 * n * q + 16.0 * block * n + 16.0 * chunk * n


def nystrom_bytes(n, q, m, block=PASS_BLOCK, chunk=PRED_CHUNK):
    return 8.0 * (5.0 * m * m + 2.0 * max(block, chunk) * m + m * q)


def check_exact_cap(n, device, exact_max, q=1, budget_bytes=0.0):
    """Refuse an oversized exact Gram. Raises RuntimeError; returns the sized peak in bytes when it is safe."""
    hard = CAP_EXACT_GPU if getattr(device, "type", str(device)) == "cuda" else CAP_EXACT_CPU
    if exact_max > hard:
        raise RuntimeError(f"exact_max {exact_max} exceeds the cap {hard} for {device}: refusing")
    if n > exact_max:
        raise RuntimeError(f"exact Matern Gram on n={n} rows exceeds exact_max={exact_max}: use Nystrom")
    need = exact_bytes(n, q)
    if budget_bytes and need > budget_bytes:
        raise RuntimeError(f"exact Gram on n={n}, q={q} needs {need/2**30:.1f} GB, "
                           f"budget is {budget_bytes/2**30:.1f} GB: refusing (lower --exact_max)")
    return need


def _sqdist_into(out, A, B, sa, sb):
    """out <- squared distances between the rows of A and B (sa, sb are the squared row norms), in place."""
    torch.mm(A, B.T, out=out)
    return out.mul_(-2.0).add_(sa[:, None]).add_(sb[None, :]).clamp_min_(0.0)


def _m52_inplace(out, ls):
    """out (squared distances) <- Matern-5/2 at length scale ls. One temporary of the same shape."""
    out.div_(ls * ls)                        # r^2
    a = out.mul(5.0).sqrt_()                 # a = sqrt(5) r
    out.mul_(5.0 / 3.0).add_(a).add_(1.0)    # 1 + a + a^2 / 3
    a.neg_().exp_()
    out.mul_(a)
    del a
    return out


def _kern_into(out, A, B, sa, sb, ls):
    return _m52_inplace(_sqdist_into(out, A, B, sa, sb), ls)


def _gram(A, B, sa, sb, ls, out=None, block=BUILD_BLOCK, kernel=True):
    if out is None:
        out = torch.empty((A.shape[0], B.shape[0]), dtype=A.dtype, device=A.device)
    for k in range(0, A.shape[0], block):
        v = out[k:k + block]
        _sqdist_into(v, A[k:k + block], B, sa[k:k + block], sb)
        if kernel:
            _m52_inplace(v, ls)
    return out


def _chol_solve(K, Y, nug):
    """Nugget added and removed in place around the factorization (no copy of K). Returns None when the
    Cholesky fails, so the nugget grid can skip that point."""
    n = K.shape[0]
    K.diagonal().add_(nug * n)
    L, info = torch.linalg.cholesky_ex(K)
    K.diagonal().sub_(nug * n)
    if int(info) != 0:
        return None
    return torch.cholesky_solve(Y, L)


def kernel_head(Ftr, Fva, Fte, Ytr, val_fn, device, exact_max, centers, seed=0, budget_bytes=0.0, label="krr"):
    """Matern-5/2 kernel ridge on the rows of F: scale x nugget chosen on validation over a subsample (exactly
    bench_run.krr_fit_predict's grid), then an exact solve on every row when n <= exact_max, else a Nystrom
    ridge with `centers` landmarks. Everything in float64 on `device`. Returns (pred_va, pred_te, hyper)."""
    t0 = time.time()
    n = Ftr.shape[0]
    q = Ytr.shape[1]
    F64 = lambda A: torch.as_tensor(np.ascontiguousarray(A, dtype=np.float64), device=device)
    Xt, Xv, Xe = F64(Ftr), F64(Fva), F64(Fte)
    Yt = F64(Ytr)
    st, sv, se = ((A * A).sum(1) for A in (Xt, Xv, Xe))
    rng = np.random.default_rng(seed + 7)
    sub = torch.as_tensor(np.sort(rng.permutation(n)[:min(TUNE_SUB, n)]).copy(), device=device)
    Fs, ss, Ysub = Xt[sub].contiguous(), st[sub], Yt[sub].contiguous()
    ns = len(Fs)
    # the median pairwise distance of the tuning subsample, in the buffer the Gram will reuse. The median is
    # taken over the whole symmetric matrix rather than its upper triangle: duplicating every pair does not
    # move a quantile, and the ns zeros on the diagonal shift it by ns / ns^2 of the population.
    Ks = _gram(Fs, Fs, ss, ss, 1.0, kernel=False)
    med = float(torch.sqrt(torch.median(Ks))) + 1e-12
    Kvs = torch.empty((len(Xv), ns), dtype=torch.float64, device=device)
    best = (np.inf, None)
    for sc in SCALES:
        _gram(Fs, Fs, ss, ss, sc * med, out=Ks)
        _gram(Xv, Fs, sv, ss, sc * med, out=Kvs)
        for nug in NUGGETS:
            alpha = _chol_solve(Ks, Ysub, nug)
            if alpha is None:
                continue
            e = val_fn((Kvs @ alpha).cpu().numpy())
            if e < best[0]:
                best = (float(e), (sc * med, nug, sc))
    del Ks, Kvs, Fs, Ysub, ss
    if best[1] is None:
        raise RuntimeError(f"{label}: every nugget on the tuning subsample failed to factorize")
    ls, nug, sc = best[1]
    if device.type == "cuda":
        torch.cuda.empty_cache()
    exact = n <= exact_max
    if exact:
        peak = check_exact_cap(n, device, exact_max, q=q, budget_bytes=budget_bytes)
        K = _gram(Xt, Xt, st, st, ls)
        K.diagonal().add_(nug * n)
        L, info = torch.linalg.cholesky_ex(K)
        del K
        if int(info) != 0:
            raise RuntimeError(f"{label}: exact Gram is not positive definite at nugget {nug:g}")
        alpha = torch.cholesky_solve(Yt, L)
        del L
        outs = []
        Kx = torch.empty((min(PRED_CHUNK, max(len(Xv), len(Xe))), n), dtype=torch.float64, device=device)
        for F_, s_ in ((Xv, sv), (Xe, se)):
            pr = torch.empty((len(F_), q), dtype=torch.float64, device=device)
            for k in range(0, len(F_), PRED_CHUNK):
                b = min(PRED_CHUNK, len(F_) - k)
                _kern_into(Kx[:b], F_[k:k + b], Xt, s_[k:k + b], st, ls)
                pr[k:k + b] = Kx[:b] @ alpha
            outs.append(pr.cpu().numpy())
        del Kx
        mode = "exact"
    else:
        m = int(min(centers, CAP_CENTERS, n))
        peak = nystrom_bytes(n, q, m)
        if budget_bytes and peak > budget_bytes:
            raise RuntimeError(f"{label}: Nystrom with m={m} needs {peak/2**30:.1f} GB > "
                               f"{budget_bytes/2**30:.1f} GB: refusing (lower --centers)")
        cidx = torch.as_tensor(np.sort(np.random.default_rng(seed + 9).permutation(n)[:m]).copy(), device=device)
        Fm, sm = Xt[cidx].contiguous(), st[cidx]
        Kmm = _gram(Fm, Fm, sm, sm, ls)
        KtK = torch.zeros((m, m), dtype=torch.float64, device=device)
        Kty = torch.zeros((m, q), dtype=torch.float64, device=device)
        blk = max(1, min(PASS_BLOCK, n))
        Knm = torch.empty((blk, m), dtype=torch.float64, device=device)
        for k in range(0, n, blk):
            b = min(blk, n - k)
            _kern_into(Knm[:b], Xt[k:k + b], Fm, st[k:k + b], sm, ls)
            KtK += Knm[:b].T @ Knm[:b]
            Kty += Knm[:b].T @ Yt[k:k + b]
        del Knm
        A = KtK + (nug * n) * Kmm
        A.diagonal().add_(1e-8 * float(torch.diagonal(KtK).sum()) / m)
        La, info = torch.linalg.cholesky_ex(A)
        if int(info) != 0:
            raise RuntimeError(f"{label}: Nystrom normal equations are not positive definite")
        alpha = torch.cholesky_solve(Kty, La)
        del A, La, KtK, Kmm
        outs = []
        Kx = torch.empty((min(PRED_CHUNK, max(len(Xv), len(Xe))), m), dtype=torch.float64, device=device)
        for F_, s_ in ((Xv, sv), (Xe, se)):
            pr = torch.empty((len(F_), q), dtype=torch.float64, device=device)
            for k in range(0, len(F_), PRED_CHUNK):
                b = min(PRED_CHUNK, len(F_) - k)
                _kern_into(Kx[:b], F_[k:k + b], Fm, s_[k:k + b], sm, ls)
                pr[k:k + b] = Kx[:b] @ alpha
            outs.append(pr.cpu().numpy())
        del Kx
        mode = f"nystrom{m}"
    hp = dict(scale=float(sc), length_scale=float(ls), median=float(med), nugget=float(nug),
              val_sub=float(best[0]), mode=mode, n=int(n), sized_bytes_gb=round(peak / 2 ** 30, 3),
              seconds=round(time.time() - t0, 1))
    print(f"  {label}: scale {sc} x med, nugget {nug:g}, {mode}, sub-val {100*best[0]:.4f}% "
          f"[{hp['seconds']:.0f} s]", flush=True)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return outs[0], outs[1], hp


# ============================== network ==============================

class PLR(nn.Module):
    """Periodic embedding of every input coordinate (k frequencies), then one linear map (bench_run.PLR)."""

    def __init__(self, d_in, width, k=16, sigma=1.0):
        super().__init__()
        self.c = nn.Parameter(torch.randn(d_in, k) * sigma)
        self.lin = nn.Linear(2 * k * d_in, width)

    def forward(self, x):
        z = 2 * np.pi * x[:, :, None] * self.c[None, :, :]
        return self.lin(torch.cat([torch.cos(z), torch.sin(z)], -1).flatten(1))


class Net(nn.Module):
    """The residual MLP of bench_run.NetX, with the options the suite's arms need."""

    def __init__(self, d_in, d_out, width, depth, kind="mlp", dropout=0.0, specnorm=False, smooth_clip=False):
        super().__init__()
        self.inp = PLR(d_in, width) if kind == "plr" else nn.Linear(d_in, width)
        hid = [nn.Linear(width, width) for _ in range(depth - 1)]
        if specnorm:
            from torch.nn.utils.parametrizations import spectral_norm
            hid = [spectral_norm(l, n_power_iterations=1) for l in hid]
        self.hid = nn.ModuleList(hid)
        self.out = nn.Linear(width, d_out)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.smooth_clip = bool(smooth_clip)

    def forward(self, x, features=False):
        if self.smooth_clip:                                  # RealMLP's smooth clipping of standardized inputs
            x = x / torch.sqrt(1.0 + (x / 3.0) ** 2)
        h = F.silu(self.inp(x))
        for l in self.hid:
            h = h + self.drop(F.silu(l(h)))
        return (self.out(h), h) if features else self.out(h)


def hidden_weights(net):
    """The square hidden weights (the only layers whose ESD alphas are comparable with each other)."""
    out = []
    for i, l in enumerate(net.hid):
        w = l.weight if hasattr(l, "weight") else None
        if w is not None and w.ndim == 2:
            out.append((f"hid{i}", w))
    return out


def hill_alpha(w, xmin_pos=2, eigs_thresh=50):
    """TempBalance's median Hill exponent of one weight matrix's empirical spectral density (its own formula:
    eigs = svdvals(W)^2 ascending, i = N // xmin_pos, alpha = 1 + n / (sum log eigs[i:] - n log eigs[i])).
    Only comparable BETWEEN LAYERS OF THE SAME SHAPE: on pure Gaussian noise it reads 2.1 at 512x512 and 3.5 at
    512x128, an aspect-ratio bias larger than any training effect."""
    a = np.asarray(w.detach().cpu().numpy(), dtype=np.float64)
    e = np.sort(np.linalg.svd(a, compute_uv=False) ** 2)
    e = e[e > 0]
    N = len(e)
    if N < max(eigs_thresh, 2 * xmin_pos):
        return float("nan")
    i = int(N / xmin_pos)
    n = float(N - i)
    le = np.log(e)
    return float(1.0 + n / (le[i:].sum() - n * le[i]))


def top_hessian_eig(net, xb, yb, loss_fn, iters=15, seed=0):
    """Largest eigenvalue of the loss Hessian at the current weights, by power iteration on Hessian-vector
    products over one fixed subsample (the sharpness that SAM claims to reduce). Diagnostic only: it selects
    nothing. Cost: `iters` extra backward passes on `xb`."""
    ps = [p for p in net.parameters() if p.requires_grad]
    g = torch.autograd.grad(loss_fn(net(xb), yb), ps, create_graph=True)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    v = [torch.randn(p.shape, generator=gen).to(p.device) for p in ps]
    nrm = torch.sqrt(sum((x * x).sum() for x in v))
    v = [x / nrm for x in v]
    lam = float("nan")
    for _ in range(iters):
        hv = torch.autograd.grad(g, ps, grad_outputs=v, retain_graph=True)
        lam = float(sum((a * b).sum() for a, b in zip(hv, v)))
        nrm = torch.sqrt(sum((a * a).sum() for a in hv))
        if not torch.isfinite(nrm) or float(nrm) == 0.0:
            break
        v = [(a / nrm).detach() for a in hv]
    return lam


def cmixup_batch(xb, yb, alpha=2.0):
    """C-Mixup (bench_run.cmixup_batch verbatim, including its degenerate-row fallback)."""
    with torch.no_grad():
        d2 = torch.cdist(yb, yb).pow(2)
        sig2 = d2.median().clamp_min(1e-6)
        P = torch.exp(-d2 / (2 * sig2))
        P.fill_diagonal_(0)
        rs = P.sum(1, keepdim=True)
        degenerate = (rs <= 0).squeeze(1)
        if bool(degenerate.any()):
            U = torch.ones_like(P)
            U.fill_diagonal_(0)
            P[degenerate] = U[degenerate]
            rs = P.sum(1, keepdim=True)
        P = P / rs.clamp_min(1e-12)
        j = torch.multinomial(P, 1).squeeze(1)
        lam = torch.distributions.Beta(alpha, alpha).sample((len(xb), 1)).to(xb.device)
    return lam * xb + (1 - lam) * xb[j], lam * yb + (1 - lam) * yb[j]


def _param_groups(net, wd, decay_weights_only):
    if not decay_weights_only:
        return [dict(params=list(net.parameters()), weight_decay=wd)]
    dec, nod = [], []
    for nm, p in net.named_parameters():
        (nod if (p.ndim < 2 or nm.endswith("bias")) else dec).append(p)
    return [dict(params=dec, weight_decay=wd), dict(params=nod, weight_decay=0.0)]


def train_mean(T, cfg, seed, epochs, arch, device, loss_kind, batch, val_every, patience_default=25):
    """One training run of one arm. Selection and early stopping both read the CORPUS validation error, never
    the test block. Returns (val_err, state_dict on cpu, info)."""
    torch.manual_seed(seed)
    kind = cfg.get("kind", "mlp")
    net = Net(arch["d"], arch["q"], arch["width"], arch["depth"], kind=kind,
              dropout=float(cfg.get("dropout", 0.0)), specnorm=bool(cfg.get("specnorm", False)),
              smooth_clip=bool(cfg.get("smooth_clip", False))).to(device)
    lr = float(cfg.get("lr", 1e-3))
    wd = float(cfg.get("wd", 1e-6))
    opt = torch.optim.AdamW(_param_groups(net, wd, bool(cfg.get("decay_weights_only", False))), lr=lr)
    warm = int(round(float(cfg.get("warmup", 0.0)) * epochs))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs - warm, 1), eta_min=lr / 100)
    patience = int(cfg.get("patience", patience_default))
    rho_sam, rho_asam = float(cfg.get("sam", 0.0)), float(cfg.get("asam", 0.0))
    c_jac, c_spec = float(cfg.get("jac", 0.0)), float(cfg.get("specpen", 0.0))
    sig_noise, a_mix = float(cfg.get("noise", 0.0)), float(cfg.get("cmixup", 0.0))
    ema_decay, swa_frac = float(cfg.get("ema", 0.0)), float(cfg.get("swa", 0.0))
    avg = {k: v.detach().clone().float() for k, v in net.state_dict().items()} if (ema_decay > 0 or swa_frac > 0) else None
    swa_start = epochs - max(1, int(round(swa_frac * epochs))) if swa_frac > 0 else None
    swa_count = 0
    n = T["Xt"].shape[0]

    def raw_loss(pred, target):
        if loss_kind == "rel":
            return (torch.linalg.vector_norm(pred - target, dim=1) /
                    torch.linalg.vector_norm(target, dim=1).clamp_min(1e-30)).mean()
        return F.mse_loss(pred, target)

    def penalties():
        p = 0.0
        if c_spec > 0:
            p = p + c_spec * sum(torch.linalg.matrix_norm(l.weight, 2) ** 2 for l in net.hid)
        return p

    best, best_state, best_src, bad, steps, passes = np.inf, None, "raw", 0, 0, 0
    ep = 0
    for ep in range(epochs):
        net.train()
        if warm and ep < warm:
            for g in opt.param_groups:
                g["lr"] = lr * (ep + 1) / warm
        perm = torch.randperm(n, device=device)
        for k in range(0, n, batch):
            i = perm[k:k + batch]
            if len(i) < 8:
                continue
            xb, yb = T["Xt"][i], T["Yt"][i]
            if a_mix > 0:
                xb, yb = cmixup_batch(xb, yb, a_mix)
            if sig_noise > 0:
                xb = xb + sig_noise * torch.randn_like(xb)
            opt.zero_grad(set_to_none=True)
            if c_jac > 0:
                xb = xb.detach().requires_grad_(True)
                pred = net(xb)
                v = torch.randn_like(pred)
                gx = torch.autograd.grad((pred * v).sum(), xb, create_graph=True)[0]
                loss = raw_loss(pred, yb) + c_jac * gx.pow(2).sum(1).mean() + penalties()
                passes += 2
            else:
                loss = raw_loss(net(xb), yb) + penalties()
                passes += 1
            loss.backward()
            if rho_sam > 0 or rho_asam > 0:
                with torch.no_grad():
                    if rho_sam > 0:
                        gn = torch.sqrt(sum((p_.grad ** 2).sum() for p_ in net.parameters() if p_.grad is not None)) + 1e-12
                        eps = [(rho_sam * p_.grad / gn) if p_.grad is not None else None for p_ in net.parameters()]
                    else:
                        tw = [(p_.abs() * p_.grad) if p_.grad is not None else None for p_ in net.parameters()]
                        gn = torch.sqrt(sum((t * t).sum() for t in tw if t is not None)) + 1e-12
                        eps = [(rho_asam * p_.abs() * t / gn) if t is not None else None
                               for p_, t in zip(net.parameters(), tw)]
                    for p_, e_ in zip(net.parameters(), eps):
                        if e_ is not None:
                            p_.add_(e_)
                opt.zero_grad(set_to_none=True)
                (raw_loss(net(xb), yb) + penalties()).backward()
                passes += 1
                with torch.no_grad():
                    for p_, e_ in zip(net.parameters(), eps):
                        if e_ is not None:
                            p_.sub_(e_)
            opt.step()
            steps += 1
            if ema_decay > 0:
                with torch.no_grad():
                    for k2, v2 in net.state_dict().items():
                        if v2.dtype.is_floating_point:
                            avg[k2].mul_(ema_decay).add_(v2.float(), alpha=1.0 - ema_decay)
                        else:
                            avg[k2] = v2.clone()
        if not (warm and ep < warm):
            sched.step()
        if swa_frac > 0 and ep >= swa_start:
            with torch.no_grad():
                swa_count += 1
                for k2, v2 in net.state_dict().items():
                    if v2.dtype.is_floating_point:
                        avg[k2].add_((v2.float() - avg[k2]) / swa_count) if swa_count > 1 else avg[k2].copy_(v2.float())
                    else:
                        avg[k2] = v2.clone()
        if (ep + 1) % val_every and ep != epochs - 1:
            continue
        net.eval()
        cands = [("raw", None, T["val_fn"](predict_np(net, T["Xv"])))]
        if avg is not None and (ema_decay > 0 or swa_count > 0):
            cur = copy.deepcopy(net.state_dict())
            net.load_state_dict({k2: v2.to(cur[k2].dtype) for k2, v2 in avg.items()})
            cands.append(("avg", copy.deepcopy(net.state_dict()), T["val_fn"](predict_np(net, T["Xv"]))))
            net.load_state_dict(cur)
        vsrc, vstate, vbest = min(cands, key=lambda t_: t_[2])
        if vbest < best - 1e-12:
            best, bad, best_src = float(vbest), 0, vsrc
            st = vstate if vstate is not None else net.state_dict()
            best_state = {k2: v2.detach().cpu().clone() for k2, v2 in st.items()}
        else:
            bad += 1
            if bad >= patience:
                break
    info = dict(cfg={k2: (float(v2) if isinstance(v2, (int, float)) and not isinstance(v2, bool) else v2)
                     for k2, v2 in cfg.items()},
                seed=int(seed), epochs_run=ep + 1, epochs_budget=int(epochs), steps=int(steps),
                fwd_bwd_passes=int(passes), checkpoint=best_src, val=float(best))
    return float(best), best_state, info


@torch.no_grad()
def predict_np(net, X, chunk=20000, features=False):
    net.eval()
    outs = []
    for k in range(0, len(X), chunk):
        o = net(X[k:k + chunk], features=True)
        outs.append((o[1] if features else o[0]).detach().cpu().numpy())
    return np.concatenate(outs).astype(np.float64)


# ============================== corpora ==============================

class _Std:
    def __init__(self, A):
        self.m = A.mean(0)
        self.s = A.std(0)
        self.s[self.s == 0] = 1.0

    def fwd(self, A):
        return (A - self.m) / self.s

    def inv(self, A):
        return A * self.s + self.m


def _rel_l2(T, P):
    return float(np.mean(np.linalg.norm(T - P, axis=1) / np.maximum(np.linalg.norm(T, axis=1), 1e-30)))


def load_emit(seed, ntrain=0, pca_rank=64, smoke=False):
    """emit_campaign.py's split (RandomState(seed) permutation, 10 % test; RandomState(seed+10000) 10 % val
    carve) and its physical metric, verbatim. The four components are regressed jointly as 4 x pca_rank
    coefficients; the reported error is the mean relative L2 over the components, with the radiance relative
    L2 and the reflectance RMSE of jpl_pipeline.py as extra metrics."""
    DATA = pathlib.Path(os.environ.get("EMIT_DATA", "data/emit"))
    COMP = ["Y1", "Y2", "Y3", "Y4"]
    TEST_REFL = 0.7
    X = np.load(DATA / "X.npy")
    Ys = {c: np.load(DATA / (c + ".npy")) for c in COMP}
    perm = np.random.RandomState(seed).permutation(len(X))
    n_te = int(round(0.1 * len(X)))
    idx_te, tr_full = perm[:n_te], perm[n_te:]
    vperm = np.random.RandomState(seed + 10000).permutation(len(tr_full))
    n_val = int(round(0.1 * len(tr_full)))
    idx_val, idx_tr = tr_full[vperm[:n_val]], tr_full[vperm[n_val:]]
    if ntrain and ntrain < len(idx_tr):
        idx_tr = idx_tr[:ntrain]
    if smoke:
        idx_tr, idx_val, idx_te = idx_tr[:2000], idx_val[:500], idx_te[:500]
    xs = _Std(X[idx_tr].astype(np.float64))
    Xs = {k: xs.fwd(X[i].astype(np.float64)) for k, i in (("tr", idx_tr), ("va", idx_val), ("te", idx_te))}
    ystd, Vt, cen, Z = {}, {}, {}, {"tr": [], "va": [], "te": []}
    for c in COMP:
        ystd[c] = _Std(Ys[c][idx_tr].astype(np.float64))
        A = ystd[c].fwd(Ys[c][idx_tr].astype(np.float64))
        cen[c] = A.mean(0)
        _, S, Vt_ = np.linalg.svd(A - cen[c], full_matrices=False)
        Vt[c] = Vt_[:pca_rank]
        for k, i in (("tr", idx_tr), ("va", idx_val), ("te", idx_te)):
            Z[k].append((ystd[c].fwd(Ys[c][i].astype(np.float64)) - cen[c]) @ Vt[c].T)
    Zs = {k: np.concatenate(v, 1) for k, v in Z.items()}
    Ytrue = {k: {c: Ys[c][i].astype(np.float64) for c in COMP} for k, i in
             (("va", idx_val), ("te", idx_te))}
    sl = {c: slice(j * pca_rank, (j + 1) * pca_rank) for j, c in enumerate(COMP)}

    def phys(Zp):
        return {c: ystd[c].inv(Zp[:, sl[c]] @ Vt[c] + cen[c]) for c in COMP}

    def rad(P):
        return P["Y1"] + TEST_REFL * (P["Y2"] + P["Y3"]) / (1 - P["Y4"] * TEST_REFL)

    def err(Zp, split):
        P = phys(Zp)
        return float(np.mean([_rel_l2(Ytrue[split][c], P[c]) for c in COMP]))

    def err_rad(Zp, split):
        return _rel_l2(rad(Ytrue[split]), rad(phys(Zp)))

    def refl_rmse(Zp, split):
        R = rad(Ytrue[split])
        P = phys(Zp)
        r = (R - P["Y1"]) / (P["Y2"] + P["Y3"] + P["Y4"] * (R - P["Y1"]))
        r = np.where(np.isfinite(r), r, 0.0) - TEST_REFL
        return float(np.sqrt(np.mean(r ** 2)))

    return dict(name="emit", tag=f"emit_s{seed}", Xtr=Xs["tr"], Xva=Xs["va"], Xte=Xs["te"],
                Ztr=Zs["tr"], Zva=Zs["va"], Zte=Zs["te"], err=err,
                extra_metrics={"rel_l2_radiance": err_rad, "refl_rmse": refl_rmse},
                note=f"PCA rank {pca_rank} per component, four components regressed jointly")


def load_oco2(seed, band="o2", ntrain=0, smoke=False):
    """jpl_data's band split (oco2_curve.py's protocol): the reported error is the reduced relative L2 on the
    40 standardized coefficients, the radiance relative L2 is the extra metric."""
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import jpl_data
    if os.environ.get("NMKC_JPL_DATA"):
        jpl_data.DATA = pathlib.Path(os.environ["NMKC_JPL_DATA"])
    sp = jpl_data.load_band(band, seed=seed)
    recon = jpl_data.reconstruction(band)
    Xtr, Ytr, Xva, Yva, Xte, Yte = (sp[k] for k in ("Xtr", "Ytr", "Xval", "Yval", "Xte", "Yte"))
    if ntrain and ntrain < len(Xtr):
        Xtr, Ytr = Xtr[:ntrain], Ytr[:ntrain]
    if smoke:
        Xtr, Ytr, Xva, Yva, Xte, Yte = Xtr[:2000], Ytr[:2000], Xva[:400], Yva[:400], Xte[:400], Yte[:400]
    truth = {"va": Yva, "te": Yte}
    return dict(name="oco2", tag=f"oco_{band}_s{seed}", Xtr=Xtr, Xva=Xva, Xte=Xte,
                Ztr=Ytr, Zva=Yva, Zte=Yte,
                err=lambda Zp, split: _rel_l2(truth[split], Zp),
                extra_metrics={"radiance": lambda Zp, split: jpl_data.radiance_error(Zp, truth[split], recon)},
                note=f"band {band}, reduced relative L2 on 40 coefficients")


def load_bench(name, seed, ntrain, args, smoke=False):
    """climsim / pkanrtm (bench_data.py) and rrtmgp (rrtmgp_data.py): the loaders' own splits, targets and
    error function, unchanged, so the rows line up with the bench_run lanes."""
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import bench_data
    if name == "climsim":
        D = bench_data.climsim(seed, ntrain or 100000, nval=20000, ntest=20000)
    elif name == "pkanrtm":
        D = bench_data.pkanrtm(seed, ntrain or 100000, args.lowfi)
    elif name == "rrtmgp":
        import rrtmgp_data
        D = rrtmgp_data.rrtmgp(seed, ntrain or 100000,
                               test_files=tuple(args.rrtmgp_test_files.split(",")) if args.rrtmgp_test_files else None)
    else:
        raise ValueError(name)
    keys = ("Xtr", "Xva", "Xte", "Ztr", "Zva", "Zte")
    if smoke:
        for k in ("Xtr", "Ztr"):
            D[k] = D[k][:2000]
    C = dict(name=name, tag=D["tag"], err=D["err"], extra_metrics=dict(D.get("extra_metrics", {})),
             note=D.get("problem", name))
    C.update({k: np.asarray(D[k], np.float64) for k in keys})
    return C


def load_corpus(args):
    if args.corpus == "emit":
        return load_emit(args.seed, args.ntrain, args.pca_rank, args.smoke)
    if args.corpus == "oco2":
        return load_oco2(args.seed, args.band, args.ntrain, args.smoke)
    return load_bench(args.corpus, args.seed, args.ntrain, args, args.smoke)


# ============================== the suite ==============================

def build_configs(method, budget, base_seed, mi):
    """The arm's runs: the grid first, then extra seeds of the grid's first entries until the budget is spent."""
    grid = GRIDS[method]
    runs = []
    for r in range(budget):
        runs.append((dict(grid[r % len(grid)]), base_seed * 1000 + mi * 37 + r))
    return runs


def run_suite(C, args, device, vram_budget=0.0):
    t0 = time.time()
    Xtr, Xva, Xte = (np.ascontiguousarray(C[k], np.float64) for k in ("Xtr", "Xva", "Xte"))
    Ztr, Zva, Zte = (np.ascontiguousarray(C[k], np.float64) for k in ("Ztr", "Zva", "Zte"))
    n, d = Xtr.shape
    q = Ztr.shape[1]
    dflt = dict(CORPUS_DEFAULTS.get(C["name"], CORPUS_DEFAULTS["climsim"]))
    epochs = args.epochs or (3 if args.smoke else dflt["epochs"])
    width = args.width or dflt["width"]
    depth = args.depth or dflt["depth"]
    batch = args.batch or dflt["batch"]
    val_every = 1 if args.smoke else (args.val_every or dflt["val_every"])
    loss_kind = args.loss or dflt["loss"]
    budget = 1 if args.smoke else args.budget
    exact_max = min(args.exact_max, CAP_EXACT_GPU if device.type == "cuda" else CAP_EXACT_CPU)
    centers = min(args.centers, CAP_CENTERS)
    err = C["err"]
    zm, zs = Ztr.mean(0), Ztr.std(0) + 1e-9
    f32 = lambda A: torch.as_tensor(np.asarray(A, np.float32), device=device)
    T = dict(Xt=f32(Xtr), Yt=f32((Ztr - zm) / zs), Xv=f32(Xva), Xe=f32(Xte))
    T["val_fn"] = lambda P: err(P * zs + zm, "va")
    arch = dict(d=d, q=q, width=width, depth=depth)
    print(f"{C['tag']}: n={n} d={d} q={q} epochs={epochs} width={width} depth={depth} batch={batch} "
          f"budget={budget} exact_max={exact_max} device={device}", flush=True)

    methods = [m for m in (list(GRIDS) if args.methods in ("", "all") else args.methods.split(",")) if m in GRIDS]
    if args.methods not in ("", "all"):
        unknown = [m for m in args.methods.split(",") if m not in GRIDS]
        if unknown:
            raise ValueError(f"unknown methods: {unknown}")
    results, hyper, budgets = {}, {}, {}
    heads_va, heads_te = {}, {}

    def rec(name, pv, pt, hp=None):
        r = dict(val=float(err(pv, "va")), test=float(err(pt, "te")))
        for mn, fn in C.get("extra_metrics", {}).items():
            r["test_" + mn] = float(fn(pt, "te"))
            r["val_" + mn] = float(fn(pv, "va"))
        results[name] = r
        hyper[name] = hp or {}
        heads_va[name], heads_te[name] = pv, pt
        print(f"== {name}: val {100*r['val']:.4f}%  test {100*r['test']:.4f}%  [{(time.time()-t0)/60:.1f} min]",
              flush=True)
        return r

    # the reference row: the kernel alone on the standardized inputs
    if args.krr_ref:
        pv, pt, hp = kernel_head(Xtr, Xva, Xte, Ztr, lambda P: err(P, "va"), device, exact_max, centers,
                                 seed=args.seed, budget_bytes=vram_budget, label="krr")
        rec("krr", pv, pt, hp)

    # a fixed subsample for the sharpness diagnostic, drawn once so every arm is measured on the same rows
    sh_idx = torch.as_tensor(np.random.default_rng(args.seed + 5).permutation(n)[:min(2048, n)], device=device)

    for mi, method in enumerate(methods):
        tm = time.time()
        runs = build_configs(method, budget, args.seed, mi)
        ep_m = epochs
        if args.budget_mode == "passes" and any(k in GRIDS[method][0] for k in TWO_PASS):
            ep_m = max(2, epochs // 2)
        trials, best = [], (np.inf, None, None)
        for cfg, seed_r in runs:
            v, state, info = train_mean(T, cfg, seed_r, ep_m, arch, device, loss_kind, batch, val_every)
            trials.append(info)
            print(f"  {method} run seed {seed_r} {cfg}: val {100*v:.4f}% "
                  f"({info['epochs_run']} ep, {info['fwd_bwd_passes']} passes)", flush=True)
            if v < best[0]:
                best = (v, state, info)
        v_best, state, info = best
        net = _rebuild(state, arch, info, device)
        Ptr, Pva, Pte = (predict_np(net, T[k]) * zs + zm for k in ("Xt", "Xv", "Xe"))
        Hs = [predict_np(net, T[k], features=True) for k in ("Xt", "Xv", "Xe")] \
            if "dkr" in args.kernel_heads else None
        diag = {}
        if args.diag:
            al = [hill_alpha(w) for _, w in hidden_weights(net)]
            al = [a for a in al if np.isfinite(a)]
            diag["esd_alpha_hidden_mean"] = float(np.mean(al)) if al else None
            lf = (lambda p, y: F.mse_loss(p, y)) if loss_kind == "mse" else \
                 (lambda p, y: (torch.linalg.vector_norm(p - y, dim=1) /
                                torch.linalg.vector_norm(y, dim=1).clamp_min(1e-30)).mean())
            diag["top_hessian_eig"] = top_hessian_eig(net, T["Xt"][sh_idx], T["Yt"][sh_idx], lf,
                                                      iters=3 if args.smoke else 15, seed=args.seed)
        bud = dict(runs=len(runs), epochs_budget=int(ep_m), epochs_run=int(sum(t["epochs_run"] for t in trials)),
                   steps=int(sum(t["steps"] for t in trials)),
                   fwd_bwd_passes=int(sum(t["fwd_bwd_passes"] for t in trials)),
                   seconds=round(time.time() - tm, 1))
        budgets[method] = bud
        rec(method, Pva, Pte, dict(info, trials=[dict(cfg=t["cfg"], seed=t["seed"], val=round(100 * t["val"], 5))
                                                for t in trials], budget=bud, **diag))
        del net
        if device.type == "cuda":
            torch.cuda.empty_cache()
        # ---- the kernel corrections of this arm's mean ----
        if "resid" in args.kernel_heads:
            pv, pt, hp = kernel_head(Xtr, Xva, Xte, Ztr - Ptr, lambda P: err(Pva + P, "va"), device,
                                     exact_max, centers, seed=args.seed, budget_bytes=vram_budget,
                                     label=method + "_resid")
            r = rec(method + "_resid", Pva + pv, Pte + pt, hp)
            r["gain_over_mean_val"] = round(1.0 - r["val"] / max(results[method]["val"], 1e-30), 5)
            r["gain_over_mean_test"] = round(1.0 - r["test"] / max(results[method]["test"], 1e-30), 5)
        if Hs is not None:
            Htr, Hva, Hte = Hs
            mu, sd = Htr.mean(0), Htr.std(0) + 1e-9
            pv, pt, hp = kernel_head((Htr - mu) / sd, (Hva - mu) / sd, (Hte - mu) / sd, Ztr,
                                     lambda P: err(P, "va"), device, exact_max, centers, seed=args.seed,
                                     budget_bytes=vram_budget, label=method + "_dkr")
            r = rec(method + "_dkr", pv, pt, hp)
            r["gain_over_mean_val"] = round(1.0 - r["val"] / max(results[method]["val"], 1e-30), 5)
            r["gain_over_mean_test"] = round(1.0 - r["test"] / max(results[method]["test"], 1e-30), 5)
            del Htr, Hva, Hte
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out = dict(tag=args.tag, kind="reg_suite", corpus=C["name"], corpus_tag=C["tag"], note=C.get("note", ""),
               seed=args.seed, n_train=int(n), n_val=int(len(Xva)), n_test=int(len(Xte)), d=int(d), q=int(q),
               device=str(device), gpu_uuid=(os.environ.get("P23_GPU_UUID") or os.environ.get("CUDA_VISIBLE_DEVICES"))
               if device.type == "cuda" else None,
               methods=results, hyper=hyper, budget=dict(per_arm=budgets, mode=args.budget_mode,
                                                         runs_per_arm=budget, epochs=epochs, width=width,
                                                         depth=depth, batch=batch, loss=loss_kind,
                                                         val_every=val_every,
                                                         selection="lowest corpus validation error"),
               exact_max=int(exact_max), centers=int(centers), smoke=bool(args.smoke),
               minutes=round((time.time() - t0) / 60, 2),
               peak_ram_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, 3),
               peak_vram_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 3) if device.type == "cuda" else 0.0)
    return out, heads_te


def _rebuild(state, arch, info, device):
    net = Net(arch["d"], arch["q"], arch["width"], arch["depth"], kind=info["cfg"].get("kind", "mlp"),
              dropout=float(info["cfg"].get("dropout", 0.0)), specnorm=bool(info["cfg"].get("specnorm", False)),
              smooth_clip=bool(info["cfg"].get("smooth_clip", False))).to(device)
    net.load_state_dict({k: v.to(device) for k, v in state.items()})
    return net


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--corpus", required=True, choices=["emit", "oco2", "climsim", "rrtmgp", "pkanrtm"])
    p.add_argument("--tag", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ntrain", type=int, default=0)
    p.add_argument("--band", default="o2", help="oco2 spectral band")
    p.add_argument("--pca_rank", type=int, default=64, help="emit: PCA rank per component")
    p.add_argument("--lowfi", type=int, default=0, help="pkanrtm: add the 6S coefficients as inputs")
    p.add_argument("--rrtmgp_test_files", default="")
    p.add_argument("--methods", default="all", help="comma list of arm names, or all")
    p.add_argument("--budget", type=int, default=3, help="training runs per arm (the equal budget)")
    p.add_argument("--budget_mode", default="runs", choices=["runs", "passes"],
                   help="runs: same epochs for every arm; passes: halve the epochs of the two-pass arms")
    p.add_argument("--kernel_heads", default="resid,dkr")
    p.add_argument("--krr_ref", type=int, default=1, help="also fit the kernel alone on the inputs")
    p.add_argument("--epochs", type=int, default=0)
    p.add_argument("--width", type=int, default=0)
    p.add_argument("--depth", type=int, default=0)
    p.add_argument("--batch", type=int, default=0)
    p.add_argument("--val_every", type=int, default=0)
    p.add_argument("--loss", default="", choices=["", "mse", "rel"])
    p.add_argument("--exact_max", type=int, default=CAP_EXACT_GPU)
    p.add_argument("--centers", type=int, default=6000)
    p.add_argument("--diag", type=int, default=1, help="record the ESD alpha and the top Hessian eigenvalue")
    p.add_argument("--save_preds", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.exact_max > CAP_EXACT_GPU:
        raise RuntimeError(f"--exact_max {args.exact_max} exceeds CAP_EXACT_GPU={CAP_EXACT_GPU}: refusing")
    device, vram = setup_device()
    OUT = pathlib.Path(os.environ.get("P2_OUT", "results"))
    C = load_corpus(args)
    out, heads_te = run_suite(C, args, device, vram)
    out["code_sha256"] = sha256_file(pathlib.Path(__file__).resolve())
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / (args.tag + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=float)
    os.replace(tmp, OUT / (args.tag + ".json"))
    if args.save_preds:
        pdir = OUT / "preds"
        pdir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(pdir / (args.tag + ".npz"), **{k: v.astype(np.float32) for k, v in heads_te.items()})
    print(f"DONE {args.tag} in {out['minutes']} min "
          f"(peak_ram {out['peak_ram_gb']} GB, peak_vram {out['peak_vram_gb']} GB)", flush=True)
    return out


if __name__ == "__main__":
    main()
