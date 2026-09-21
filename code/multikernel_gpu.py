"""Many kernels on one GPU for the emulator corpora: EMIT, OCO-2, pKANrtm, RRTMGP, ClimSim, QM9.

One lane = one corpus, one seed, the whole kernel family set, one result JSON. Everything is exactly the
protocol the CPU drivers already use (bench_run.py, emit_campaign.py, oco2_curve.py): the loaders and their
splits, the corpus's own error function, every hyperparameter picked on validation over a tuning subsample
and the winner refitted on all rows, the test block read once per method at the end.

Kernels on the inputs (families, `--kernels`):
  krr_matern12/32/52   isotropic Matern, nu = 1/2, 3/2, 5/2, length scale = scale x median distance
  krr_rbf              squared exponential
  krr_sm<Q>            spectral mixture of Q components (Wilson and Adams 2013), the mixture fitted by
                       exact-GP empirical Bayes on a subsample; separable, so each component costs one
                       weighted distance matrix and one rank-one phase
  krr_nngp             infinite-width ReLU network kernel (arc-cosine recursion, depth and bias variance
                       on validation) - the same closed form bench_run.py uses on CPU
  krr_ntk              the neural tangent kernel of the same network (Theta^{l+1} = Sigma^{l+1} + Theta^l Sigmadot^{l+1})
  krr_add              first-order additive Matern-5/2 (mean of d one-dimensional kernels), d <= ADD_MAX_D
  krr_ard_kf           Matern-5/2 with per-input length scales learned by the l2 kernel flow (Owhadi-Yoo),
                       minibatch, in the corpus's own target geometry
  krr_ard_eb           the same with the length scales, amplitude and noise learned by empirical Bayes
                       (exact GP log marginal likelihood on a subsample)
Kernels on the network's features (`--fkernels`, prefix dkr_): the network is the residual MLP of
bench_run.py trained on the GPU; its last hidden layer feeds the same kernels.
Combiners: `mkl_sum` (one kernel K(theta) = sum_k theta_k K_k, theta on the simplex chosen on validation from
alignment, uniform, one-hot and Dirichlet candidates, then refitted on all rows), `mkl_stack` (convex
combination of the heads' predictions, weights on validation), `mkl_ridge` (GCV ridge of the target on the
stack of head predictions, per output coordinate), `select` (per-output-coordinate validation selection).
The combiners' weights are fitted on validation, so their reported `val` is the fit's own error and carries
`val_is_in_sample` in the hyper; only their `test` is out of sample. Every head's `val` is honest.

Solvers. Exact fp64 Cholesky while n <= --exact_max, Falkon (Nystrom + Cholesky-preconditioned CG, Rudi
et al. 2017) above it; --also_falkon adds a Falkon twin of the Matern-5/2 head as a receipt on the
approximation. A request for an exact Gram that has not been sized against the card RAISES (GramTooLarge);
it is never built silently.

Memory - the peak this lane must declare
  VRAM, exact solve      8 * (2 n^2 + c n + n q) bytes
                         (the Gram and its Cholesky factor coexist for one call; c = --pred_chunk = 4096)
                         n = 18,883, q = 64, c = 4096:  2*18883^2 + 4096*18883 + 18883*64 = 7.9e8 -> 5.9 GB
                         n = 33,000, q = 64:  2.32e9 -> 17.3 GB   (the practical ceiling at the 0.65 fraction)
  VRAM, Falkon           8 * (3 M^2 + c M + M q + [n M when it is cached]) bytes; K_nM is cached when
                         8 n M <= min(--cache_gb, 0.45 x the fraction), and recomputed in row chunks otherwise
                         n = 150,000, M = 8,000, q = 3, c = 4096:  cached 1.4e9 -> 11.1 GB
                         n = 390,000, M = 8,000, no cache: 8 * (1.92e8 + 3.3e7) -> 1.8 GB, 2 chunked
                         passes over the rows per CG iteration instead
  VRAM, tuning           8 * (2 m^2 + n_val m + m q) bytes, m = --tune_sub 6000, n_val 20,000 -> 1.5 GB
                         (mkl_sum holds every base Gram at once, so it uses --mkl_sub 3000 rows and
                         --mkl_val 4000 validation rows: 8 K (m^2 + m n_val) = 1.5 GB at K = 10 kernels)
  host RAM               the loader's own peak plus 8 * (n + n_val + n_test) * (d + q) bytes for the
                         float64 splits and 8 H (n_val + n_test) q bytes for the H heads' predictions
The exact caps are named constants: EXACT_MAX_GPU = 36,000 rows (one fp64 Gram = 10.4 GB, two = 20.7 GB),
EXACT_MAX_CPU = 20,000 (the existing exact_max), FALKON_MAX_CENTERS = 20,000. Beyond the sized budget the
driver refuses; it never falls back quietly.

Environment: CUDA_VISIBLE_DEVICES (one GPU UUID), P23_GPU_MEM_FRAC, P23_GPU_UUID, NMKC_THREADS, P2_OUT,
EMIT_DATA, NMKC_JPL_DATA, DATA_NEW.
    python multikernel_gpu.py --corpus emit --comp Y2 --seed 101 --tag mk_emit_Y2_s101
    python multikernel_gpu.py --corpus climsim --seed 0 --ntrain 100000 --tag mk_climsim_s0
"""
import argparse, hashlib, json, math, os, pathlib, sys, time

_T = os.environ.get("NMKC_THREADS", "4")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, _T)

import numpy as np
import torch

try:                                   # POSIX only; the local CPU test runs on Windows
    import resource
except ImportError:                    # pragma: no cover
    resource = None

torch.set_num_threads(int(_T))

# ---- caps that must never be crossed silently (contract, and the 2026-09-06 outage) ----
EXACT_MAX_GPU = 36000          # exact fp64 Gram on one V100 at the 0.65 fraction (10.4 GB per copy)
EXACT_MAX_CPU = 20000          # the existing exact_max of bench_run.py
FALKON_MAX_CENTERS = 20000
VRAM_SAFETY = 0.90             # of the declared memory fraction; the rest is cuBLAS workspace and fragments
ADD_MAX_D = 32                 # the additive kernel builds d one-dimensional Grams
NUGGETS = (1e-8, 1e-6, 1e-4, 1e-2)
SCALES = (0.5, 1.0, 2.0, 4.0)
DT = torch.float64
SQ3, SQ5 = math.sqrt(3.0), math.sqrt(5.0)


class GramTooLarge(RuntimeError):
    """Raised instead of building a Gram that has not been sized against the device."""


# ------------------------------------------------------------------ device


def setup_device(prefer_gpu=True):
    """(device, byte budget, info). Sets the process memory fraction BEFORE the first CUDA allocation."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if prefer_gpu and cvd.strip() and torch.cuda.is_available():
        frac = float(os.environ.get("P23_GPU_MEM_FRAC", "0.65"))
        torch.cuda.set_per_process_memory_fraction(frac, 0)
        props = torch.cuda.get_device_properties(0)
        info = dict(device="cuda:0", device_name=props.name, mem_frac=frac,
                    gpu_uuid=os.environ.get("P23_GPU_UUID", cvd.strip()),
                    visible_device_count=torch.cuda.device_count(),
                    total_vram_gb=round(props.total_memory / 2 ** 30, 2))
        return torch.device("cuda:0"), frac * props.total_memory, info
    budget = float(os.environ.get("P23_CPU_GRAM_GB", "8")) * 2 ** 30
    return torch.device("cpu"), budget, dict(device="cpu", mem_frac=None, gpu_uuid=None)


def exact_bytes(n, q, chunk):
    """Peak bytes of the exact path: the Gram, its Cholesky factor, one prediction block, the coefficients."""
    return 8.0 * (2.0 * n * n + float(chunk) * n + float(n) * q)


def assert_exact_fits(n, q, chunk, budget_bytes, device, exact_max):
    """Refuse an exact Gram that the cap or the sized budget does not allow. Never silently proceeds."""
    hard = EXACT_MAX_GPU if device.type == "cuda" else EXACT_MAX_CPU
    if exact_max > hard:
        raise GramTooLarge(f"--exact_max {exact_max} exceeds the {device.type} cap {hard}")
    if n > exact_max:
        raise GramTooLarge(f"exact solve asked for n={n} rows, cap is {exact_max}")
    need = exact_bytes(n, q, chunk)
    if need > VRAM_SAFETY * budget_bytes:
        raise GramTooLarge(f"exact Gram for n={n}, q={q} needs {need / 2**30:.1f} GB, budget is "
                           f"{VRAM_SAFETY * budget_bytes / 2**30:.1f} GB; use Falkon")
    return need


# ------------------------------------------------------------------ kernels (torch, float64)


def sqdist(A, B):
    d2 = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * (A @ B.T)
    return d2.clamp_(min=0.0)


def k_matern(A, B, ls, nu):
    r = torch.sqrt(sqdist(A, B) + 1e-30) / ls
    if nu == 0.5:
        return torch.exp(-r)
    if nu == 1.5:
        a = SQ3 * r
        return (1.0 + a) * torch.exp(-a)
    if nu == 2.5:
        a = SQ5 * r
        return (1.0 + a + a * a / 3.0) * torch.exp(-a)
    raise ValueError(f"matern nu {nu}")


def k_rbf(A, B, ls):
    return torch.exp(-0.5 * sqdist(A, B) / (ls * ls))


def k_sm(A, B, w, mu, v):
    """Spectral mixture, Wilson and Adams (2013), separable over dimensions:
    k(x, x') = sum_q w_q exp(-2 pi^2 sum_d v_qd (x_d - x'_d)^2) cos(2 pi sum_d mu_qd (x_d - x'_d)).
    The Gaussian factor is a weighted squared distance and the phase is rank one, so no (n, m, d) tensor."""
    K = None
    two_pi = 2.0 * math.pi
    for q in range(w.shape[0]):
        s = torch.sqrt(v[q])
        E = torch.exp(-2.0 * math.pi ** 2 * sqdist(A * s, B * s))
        ph = (A @ mu[q])[:, None] - (B @ mu[q])[None, :]
        term = w[q] * E * torch.cos(two_pi * ph)
        K = term if K is None else K + term
    return K


def k_nn(A, B, depth, sb2, sw2=2.0, ntk=False):
    """Infinite-width ReLU network: NNGP (arc-cosine recursion) and, with ntk=True, the neural tangent kernel."""
    dA, dB = A.shape[1], B.shape[1]
    Kab = sw2 * (A @ B.T) / dA + sb2
    Kaa = sw2 * (A * A).sum(1) / dA + sb2
    Kbb = sw2 * (B * B).sum(1) / dB + sb2
    Th = Kab.clone() if ntk else None
    for _ in range(depth - 1):
        s = torch.sqrt((Kaa[:, None] * Kbb[None, :]).clamp_(min=1e-30))
        th = torch.arccos((Kab / s).clamp_(-1.0, 1.0))
        if ntk:
            dot = sw2 / (2.0 * math.pi) * (math.pi - th)
        Kab = sw2 / (2.0 * math.pi) * s * (torch.sin(th) + (math.pi - th) * torch.cos(th)) + sb2
        Kaa = sw2 / 2.0 * Kaa + sb2
        Kbb = sw2 / 2.0 * Kbb + sb2
        if ntk:
            Th = Kab + Th * dot
    return Th if ntk else Kab


def k_add(A, B, ls_vec, nu=2.5):
    """First-order additive Matern: the mean of d one-dimensional kernels (one per input coordinate)."""
    d = A.shape[1]
    K = None
    for j in range(d):
        r = torch.abs(A[:, j][:, None] - B[:, j][None, :]) / ls_vec[j]
        a = SQ5 * r if nu == 2.5 else SQ3 * r
        term = (1.0 + a + a * a / 3.0) * torch.exp(-a) if nu == 2.5 else (1.0 + a) * torch.exp(-a)
        K = term if K is None else K + term
    return K / d


def median_dist(F, cap=2000, seed=0):
    """Median pairwise distance over at most `cap` rows (the house length-scale unit)."""
    m = min(cap, len(F))
    idx = torch.as_tensor(np.random.default_rng(seed).permutation(len(F))[:m], device=F.device)
    D2 = sqdist(F[idx], F[idx])
    iu = torch.triu_indices(m, m, offset=1, device=F.device)
    return float(torch.sqrt(D2[iu[0], iu[1]]).median()) + 1e-12


# ------------------------------------------------------------------ solvers


def chol_solve(K, Y, nug, n_scale=None):
    """(K + nug * n I)^{-1} Y with a Cholesky that reports failure instead of raising. Returns None on failure.
    K is consumed: its diagonal is shifted in place."""
    n = K.shape[0]
    K.diagonal().add_(nug * float(n_scale if n_scale is not None else n))
    L, info = torch.linalg.cholesky_ex(K)
    if int(info) != 0:
        return None
    return torch.cholesky_solve(Y, L)


def exact_fit_predict(kfn, Ftr, Ytr, preds_on, nug, chunk):
    """Exact KRR: build the Gram, factor it, free it, predict in row chunks. Peak = two n x n copies."""
    n = Ftr.shape[0]
    K = kfn(Ftr, Ftr)
    K.diagonal().add_(nug * n)
    L, info = torch.linalg.cholesky_ex(K)
    del K
    if int(info) != 0:
        return None
    alpha = torch.cholesky_solve(Ytr, L)
    del L
    out = []
    for F_ in preds_on:
        P = torch.empty((F_.shape[0], Ytr.shape[1]), dtype=DT, device=Ftr.device)
        for i in range(0, F_.shape[0], chunk):
            P[i:i + chunk] = kfn(F_[i:i + chunk], Ftr) @ alpha
        out.append(P)
    return out


class Falkon:
    """Nystrom KRR with the Cholesky preconditioner and conjugate gradients (Rudi, Carratino, Rosasco 2017).

    K_MM = T^T T,  A^T A = T T^T / M + lam I,  preconditioner P = T^{-1} A^{-1} / sqrt(n).
    The system solved is  P^T H P beta = P^T (1/n) K_nM^T y  with  H = (1/n) K_nM^T K_nM + lam K_MM,
    and alpha = P beta * sqrt(n) = T^{-1} A^{-1} beta. K_nM is cached when it fits the byte budget, and
    recomputed in row chunks when it does not."""

    def __init__(self, kfn, Ftr, centers, lam, chunk, cache_bytes):
        self.kfn, self.Ftr, self.lam, self.chunk = kfn, Ftr, float(lam), chunk
        self.Fm = Ftr[centers].contiguous()
        self.n, self.M = Ftr.shape[0], self.Fm.shape[0]
        if self.M > FALKON_MAX_CENTERS:
            raise GramTooLarge(f"{self.M} Nystrom centers exceeds FALKON_MAX_CENTERS {FALKON_MAX_CENTERS}")
        Kmm = kfn(self.Fm, self.Fm)
        # the smallest jitter that factors: it enters the penalty term, so a jitter of j perturbs the
        # solution by about j / lam (a 1e-10 jitter at lam = 1e-4 already moves the fifth digit)
        # It is not enough that the Cholesky returns info = 0: at a smooth kernel with many centers K_MM can
        # factor into a T so ill-conditioned that the triangular solves destroy the CG operator (measured
        # 2026-09-12: Matern-5/2 at 2 x median with 600 of 700 centers broke CG down at iteration one and
        # returned a near-zero solution). T must also be usable, so the jitter climbs until it is.
        scale = float(torch.diagonal(Kmm).mean()) + 1e-30
        self.jitter, T, self.ok = 0.0, None, False
        for e in range(-12, -2):
            j = scale * self.M * (10.0 ** e)
            Kj = Kmm.clone()
            Kj.diagonal().add_(j)
            Tc, info = torch.linalg.cholesky_ex(Kj, upper=True)
            del Kj
            if int(info) == 0:
                dg = torch.diagonal(Tc).abs()
                if float(dg.min()) > 1e-7 * float(dg.max()):
                    self.jitter, T, self.ok = j, Tc, True
                    break
            del Tc
        del Kmm
        if not self.ok:
            return
        AA = (T @ T.T) / self.M
        AA.diagonal().add_(self.lam)
        A, info2 = torch.linalg.cholesky_ex(AA, upper=True)
        del AA
        self.ok = int(info2) == 0
        if not self.ok:
            return
        self.T, self.A = T, A
        self.TTt = None
        self.Knm = None
        if cache_bytes > 0 and 8.0 * self.n * self.M <= cache_bytes:
            self.Knm = torch.empty((self.n, self.M), dtype=DT, device=Ftr.device)
            for i in range(0, self.n, chunk):
                self.Knm[i:i + chunk] = kfn(Ftr[i:i + chunk], self.Fm)

    def set_lam(self, lam):
        """Change the ridge without touching K_MM's Cholesky or the K_nM cache: only A depends on lambda.
        This is what lets the ridge be re-selected against the Falkon fit itself rather than inherited from
        an exact solve on a subsample - a Nystrom estimator wants a larger ridge than an exact one."""
        self.lam = float(lam)
        if self.TTt is None:
            self.TTt = (self.T @ self.T.T) / self.M
        AA = self.TTt.clone()
        AA.diagonal().add_(self.lam)
        A, info = torch.linalg.cholesky_ex(AA, upper=True)
        del AA
        if int(info) != 0:
            return False
        self.A = A
        return True

    # triangular solves
    def _iT(self, v):
        return torch.linalg.solve_triangular(self.T, v, upper=True)

    def _iTt(self, v):
        return torch.linalg.solve_triangular(self.T.mT, v, upper=False)

    def _iA(self, v):
        return torch.linalg.solve_triangular(self.A, v, upper=True)

    def _iAt(self, v):
        return torch.linalg.solve_triangular(self.A.mT, v, upper=False)

    def _KtK(self, V):
        """(1/n) K_nM^T (K_nM V)."""
        if self.Knm is not None:
            return self.Knm.T @ (self.Knm @ V) / self.n
        acc = torch.zeros((self.M, V.shape[1]), dtype=DT, device=V.device)
        for i in range(0, self.n, self.chunk):
            Kc = self.kfn(self.Ftr[i:i + self.chunk], self.Fm)
            acc += Kc.T @ (Kc @ V)
            del Kc
        return acc / self.n

    def _Kty(self, Y):
        if self.Knm is not None:
            return self.Knm.T @ Y / self.n
        acc = torch.zeros((self.M, Y.shape[1]), dtype=DT, device=Y.device)
        for i in range(0, self.n, self.chunk):
            Kc = self.kfn(self.Ftr[i:i + self.chunk], self.Fm)
            acc += Kc.T @ Y[i:i + self.chunk]
            del Kc
        return acc / self.n

    def solve(self, Y, iters=25, tol=1e-7):
        rhs = self._iAt(self._iTt(self._Kty(Y)))

        def H(B):
            v = self._iA(B)
            return self._iAt(self._iTt(self._KtK(self._iT(v))) + self.lam * v)

        beta = torch.zeros_like(rhs)
        r = rhs.clone()
        p = r.clone()
        rs = (r * r).sum(0)
        rs0 = rs.clone()
        floor = rs0.clamp_min(1e-300)
        used = 0
        self.cg_breakdown = False
        for it in range(iters):
            Hp = H(p)
            denom = (p * Hp).sum(0)
            if bool((denom <= 0).any()):
                self.cg_breakdown = True          # H is not positive on p: the preconditioner is unusable
            a = torch.where(denom > 0, rs / denom.clamp_min(1e-300), torch.zeros_like(denom))
            beta += a[None, :] * p
            r -= a[None, :] * Hp
            rs_new = (r * r).sum(0)
            used = it + 1
            # converged, or stalled at the rounding floor: either way the reported residual is this one
            stalled = bool((rs_new >= rs).all())
            if stalled and it == 0:
                self.cg_breakdown = True          # no progress at all on the first step
            done = bool((rs_new <= (tol ** 2) * floor).all()) or stalled
            if not done:
                p = r + (rs_new / rs.clamp_min(1e-300))[None, :] * p
            rs = rs_new
            if done:
                break
        self.cg_iters = used
        self.cg_resid = float(torch.sqrt((rs / floor).max()))
        return self._iT(self._iA(beta))

    def predict(self, F_, alpha):
        P = torch.empty((F_.shape[0], alpha.shape[1]), dtype=DT, device=F_.device)
        for i in range(0, F_.shape[0], self.chunk):
            P[i:i + self.chunk] = self.kfn(F_[i:i + self.chunk], self.Fm) @ alpha
        return P

    def free(self):
        self.Knm = None
        self.T = self.A = self.TTt = None


# ------------------------------------------------------------------ learned metrics


def kf_ard(Ftr, Yobj, ynorm2, steps=300, batch=600, lr=0.05, seed=0, nug=1e-4):
    """Per-input length scales from the l2 kernel flow (Owhadi and Yoo): a random half of each minibatch
    predicts the other half, the loss is the corpus's own relative-square objective. Returns ell (d,),
    normalized to geometric mean one, as a numpy array."""
    dev = Ftr.device
    n, d = Ftr.shape
    Y = torch.as_tensor(Yobj, dtype=DT, device=dev)
    N2 = torch.as_tensor(ynorm2, dtype=DT, device=dev)
    med0 = median_dist(Ftr, seed=seed)
    log_ell = torch.full((d,), math.log(med0), dtype=DT, device=dev, requires_grad=True)
    opt = torch.optim.Adam([log_ell], lr=lr)
    rng = np.random.default_rng(seed)
    b = min(batch, n)
    for _ in range(steps):
        idx = torch.as_tensor(rng.choice(n, b, replace=False), device=dev)
        half = b // 2
        Fb = Ftr[idx] / torch.exp(log_ell)[None, :]
        D2 = torch.clamp((Fb * Fb).sum(1)[:, None] + (Fb * Fb).sum(1)[None, :] - 2.0 * Fb @ Fb.T, min=0.0)
        a = SQ5 * torch.sqrt(D2 + 1e-12)
        K = (1.0 + a + a * a / 3.0) * torch.exp(-a)
        Kc = K[:half, :half] + nug * half * torch.eye(half, dtype=DT, device=dev)
        L, info = torch.linalg.cholesky_ex(Kc)
        if int(info) != 0:
            continue
        pred = K[half:, :half] @ torch.cholesky_solve(Y[idx][:half], L)
        loss = (((Y[idx][half:] - pred) ** 2).sum(1) / N2[idx][half:]).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    ell = torch.exp(log_ell).detach().cpu().numpy()
    return ell / np.exp(np.log(ell).mean())


def eb_ard(Ftr, Ytr, rows, steps=150, lr=0.05, seed=0, n_out=8):
    """Per-input length scales, amplitude and noise by exact-GP empirical Bayes on `rows` training rows and
    the leading `n_out` output coordinates (PCA coefficients are ordered by variance). Returns (ell, info)."""
    dev = Ftr.device
    m = len(rows)
    F = Ftr[rows]
    Y = Ytr[rows][:, :min(n_out, Ytr.shape[1])].contiguous()
    Y = Y - Y.mean(0)
    ys = Y.std(0).clamp_min(1e-12)
    Y = Y / ys
    d = F.shape[1]
    med0 = median_dist(F, seed=seed)
    log_ell = torch.full((d,), math.log(med0), dtype=DT, device=dev, requires_grad=True)
    log_amp = torch.zeros((), dtype=DT, device=dev, requires_grad=True)
    log_noise = torch.full((), math.log(1e-2), dtype=DT, device=dev, requires_grad=True)
    opt = torch.optim.Adam([log_ell, log_amp, log_noise], lr=lr)
    eye = torch.eye(m, dtype=DT, device=dev)
    qe = Y.shape[1]
    last = float("nan")
    for _ in range(steps):
        Fs = F / torch.exp(log_ell)[None, :]
        D2 = torch.clamp((Fs * Fs).sum(1)[:, None] + (Fs * Fs).sum(1)[None, :] - 2.0 * Fs @ Fs.T, min=0.0)
        a = SQ5 * torch.sqrt(D2 + 1e-12)
        K = torch.exp(log_amp) * (1.0 + a + a * a / 3.0) * torch.exp(-a) + (torch.exp(log_noise) + 1e-8) * eye
        L, info = torch.linalg.cholesky_ex(K)
        if int(info) != 0:
            break
        quad = (Y * torch.cholesky_solve(Y, L)).sum()
        logdet = 2.0 * torch.log(torch.diagonal(L)).sum()
        nll = (0.5 * quad + 0.5 * qe * logdet) / (m * qe)
        opt.zero_grad()
        nll.backward()
        opt.step()
        last = float(nll.detach())
    ell = torch.exp(log_ell).detach().cpu().numpy()
    g = np.exp(np.log(ell).mean())
    return ell / g, dict(nll_per_point=last, amp=float(torch.exp(log_amp).detach()),
                         noise=float(torch.exp(log_noise).detach()),
                         rows=int(m), outputs=int(qe), ell_geomean=float(g))


def eb_sm(Ftr, Ytr, rows, Q=4, steps=200, lr=0.05, seed=0, n_out=6):
    """Spectral-mixture parameters (weights, frequencies, bandwidths) by exact-GP empirical Bayes on a
    subsample. Returns (w, mu, v) tensors on the device and an info dict."""
    dev = Ftr.device
    m = len(rows)
    F = Ftr[rows]
    Y = Ytr[rows][:, :min(n_out, Ytr.shape[1])].contiguous()
    Y = Y - Y.mean(0)
    Y = Y / Y.std(0).clamp_min(1e-12)
    d = F.shape[1]
    med0 = median_dist(F, seed=seed)
    g = torch.Generator(device="cpu").manual_seed(seed + 5)
    log_w = torch.zeros(Q, dtype=DT, device=dev, requires_grad=True)
    mu0 = torch.abs(torch.randn(Q, d, generator=g, dtype=DT)) / (2.0 * med0)
    mu = mu0.to(dev).requires_grad_(True)
    log_v = torch.full((Q, d), math.log(1.0 / (med0 * med0)), dtype=DT, device=dev).requires_grad_(True)
    log_noise = torch.full((), math.log(1e-2), dtype=DT, device=dev, requires_grad=True)
    opt = torch.optim.Adam([log_w, mu, log_v, log_noise], lr=lr)
    eye = torch.eye(m, dtype=DT, device=dev)
    qe = Y.shape[1]
    last = float("nan")
    for _ in range(steps):
        w = torch.softmax(log_w, 0)
        K = k_sm(F, F, w, mu, torch.exp(log_v)) + (torch.exp(log_noise) + 1e-8) * eye
        L, info = torch.linalg.cholesky_ex(K)
        if int(info) != 0:
            break
        quad = (Y * torch.cholesky_solve(Y, L)).sum()
        logdet = 2.0 * torch.log(torch.diagonal(L)).sum()
        nll = (0.5 * quad + 0.5 * qe * logdet) / (m * qe)
        opt.zero_grad()
        nll.backward()
        opt.step()
        last = float(nll.detach())
    with torch.no_grad():
        w = torch.softmax(log_w, 0).detach()
        v = torch.exp(log_v).detach()
        mu_d = mu.detach()
    return w, mu_d, v, dict(nll_per_point=last, Q=int(Q), rows=int(m), outputs=int(qe),
                            noise=float(torch.exp(log_noise).detach()),
                            weights=[round(float(x), 4) for x in w])


# ------------------------------------------------------------------ kernel candidate grids


def kernel_grid(name, Fs, learned, args):
    """[(params, kfn)] for one kernel family, evaluated on the tuning subsample Fs (standardized)."""
    out = []
    if name.startswith("krr_matern") or name.startswith("dkr_matern"):
        nu = {"12": 0.5, "32": 1.5, "52": 2.5}[name[-2:]]
        med = median_dist(Fs, seed=args.seed)
        for sc in SCALES:
            out.append((dict(nu=nu, scale=sc, med=med), lambda A, B, ls=sc * med, nu=nu: k_matern(A, B, ls, nu)))
    elif name.endswith("_rbf"):
        med = median_dist(Fs, seed=args.seed)
        for sc in SCALES:
            out.append((dict(scale=sc, med=med), lambda A, B, ls=sc * med: k_rbf(A, B, ls)))
    elif "_sm" in name:
        w, mu, v, info = learned["sm"]
        for sc in (0.5, 1.0, 2.0):
            out.append((dict(scale=sc, sm=info), lambda A, B, s=sc, w=w, mu=mu, v=v: k_sm(A, B, w, mu / s, v / (s * s))))
    elif name.endswith("_nngp") or name.endswith("_ntk"):
        ntk = name.endswith("_ntk")
        for depth in (2, 3, 5):
            for sb2 in (0.0, 0.1):
                out.append((dict(depth=depth, sb2=sb2, ntk=ntk),
                            lambda A, B, dp=depth, sb=sb2, nk=ntk: k_nn(A, B, dp, sb, ntk=nk)))
    elif name.endswith("_add"):
        d = Fs.shape[1]
        if d > ADD_MAX_D:
            return []
        ls1 = torch.as_tensor(np.full(d, 1.0), dtype=DT, device=Fs.device)
        for sc in SCALES:
            out.append((dict(scale=sc, per_dim_ls=1.0), lambda A, B, s=sc, l=ls1: k_add(A, B, l * s)))
    elif name.endswith("_ard_kf") or name.endswith("_ard_eb"):
        key = "ard_kf" if name.endswith("_ard_kf") else "ard_eb"
        ell = learned[key]
        w = torch.as_tensor(1.0 / ell, dtype=DT, device=Fs.device)
        med = median_dist(Fs * w[None, :], seed=args.seed)
        for sc in SCALES:
            out.append((dict(scale=sc, med=med, ell_rel=[round(float(x), 4) for x in ell]),
                        lambda A, B, ls=sc * med, w=w: k_matern(A * w[None, :], B * w[None, :], ls, 2.5)))
    else:
        raise ValueError(f"unknown kernel {name!r}")
    return out


# ------------------------------------------------------------------ one head


def fit_head(name, cands, Ftr, Fva, Fte, Ytr, val_fn, sub, args, budget, device, force_falkon=False):
    """Tune (scale-like parameters x nugget) on the subsample, then refit on every row: exact while
    n <= --exact_max, Falkon above it. Returns (pred_va, pred_te, hyper) as numpy float64, or None."""
    if not cands:
        return None
    n = Ftr.shape[0]
    Fs = Ftr[sub]
    Ysub = Ytr[sub]
    best = (float("inf"), None, None)
    for params, kfn in cands:
        Ks = kfn(Fs, Fs)
        Kvs = kfn(Fva, Fs)
        for nug in NUGGETS:
            al = chol_solve(Ks.clone(), Ysub, nug)
            if al is None:
                continue
            e = val_fn((Kvs @ al).detach().cpu().numpy())
            if e < best[0]:
                best = (e, dict(params, nugget=nug), kfn)
        del Ks, Kvs
    if best[2] is None:
        return None
    val_sub, hp, kfn = best
    hp = dict(hp)
    hp["val_sub"] = float(val_sub)
    hp["tune_rows"] = int(len(sub))
    hp["tune_budget"] = int(len(cands) * len(NUGGETS))
    use_exact = (n <= args.exact_max) and not force_falkon
    if use_exact:
        assert_exact_fits(n, Ytr.shape[1], args.pred_chunk, budget, device, args.exact_max)
        got = exact_fit_predict(kfn, Ftr, Ytr, (Fva, Fte), hp["nugget"], args.pred_chunk)
        if got is None:
            return None
        pv, pt = got
        hp["solver"] = "exact"
    else:
        centers = torch.as_tensor(np.random.default_rng(args.seed + 9).permutation(n)[:min(args.centers, n)],
                                  device=Ftr.device)
        cache = min(args.cache_gb * 2 ** 30, 0.45 * budget)     # the cache never takes half the card
        fk = Falkon(kfn, Ftr, centers, hp["nugget"], args.pred_chunk, cache)
        if not fk.ok:
            return None
        # the ridge is re-selected on validation against the Nystrom fit (K_MM's factor and the K_nM cache
        # are reused, so this costs one M x M Cholesky and one CG per candidate)
        best_f = (float("inf"), None, None)
        broke = 0
        for nug in NUGGETS:
            if not fk.set_lam(nug):
                continue
            a_ = fk.solve(Ytr, iters=args.cg_iters)
            if fk.cg_breakdown:
                # the CG made no progress: the solution is not a fit, and reporting it would be a lie
                broke += 1
                print(f"  {name}: CG breakdown at nugget {nug:g}, candidate dropped", flush=True)
                continue
            p_ = fk.predict(Fva, a_)
            e = val_fn(p_.detach().cpu().numpy())
            if e < best_f[0]:
                best_f = (e, nug, a_, fk.cg_iters, fk.cg_resid)
            del p_
        if best_f[1] is None:
            return None
        alpha = best_f[2]
        pv, pt = fk.predict(Fva, alpha), fk.predict(Fte, alpha)
        hp["solver"] = "falkon"
        hp["centers"] = int(fk.M)
        hp["nugget_subsample"] = hp["nugget"]
        hp["nugget"] = best_f[1]
        hp["val_falkon"] = float(best_f[0])
        hp["tune_budget"] = int(hp["tune_budget"] + len(NUGGETS))
        hp["cg_iters"] = int(best_f[3])
        hp["cg_resid"] = float(best_f[4])
        hp["knm_cached"] = fk.Knm is not None
        hp["falkon_jitter"] = fk.jitter
        hp["cg_breakdowns"] = int(broke)
        fk.free()
    pvn, ptn = pv.detach().cpu().numpy(), pt.detach().cpu().numpy()
    del pv, pt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return pvn, ptn, hp


# ------------------------------------------------------------------ the network (bench_run.py's residual MLP)


class Net(torch.nn.Module):
    def __init__(self, d_in, d_out, width, depth):
        super().__init__()
        self.inp = torch.nn.Linear(d_in, width)
        self.hid = torch.nn.ModuleList([torch.nn.Linear(width, width) for _ in range(depth - 1)])
        self.out = torch.nn.Linear(width, d_out)

    def forward(self, x, features=False):
        h = torch.nn.functional.silu(self.inp(x))
        for l in self.hid:
            h = h + torch.nn.functional.silu(l(h))
        return (self.out(h), h) if features else self.out(h)


def train_net(Xtr, Xva, Xte, Ztr, Zva, hp, seed, device, patience=25):
    """bench_run.py's network, trained on the device in float32. Returns (preds tr/va/te, features tr/va/te, epochs)."""
    torch.manual_seed(seed)
    zm, zs = Ztr.mean(0), Ztr.std(0) + 1e-9
    f32 = lambda a: torch.as_tensor(np.asarray(a, np.float32), device=device)
    Xt, Xv, Xe = f32(Xtr), f32(Xva), f32(Xte)
    Yt, Yv = f32((Ztr - zm) / zs), f32((Zva - zm) / zs)
    n, d, q = Xt.shape[0], Xt.shape[1], Yt.shape[1]
    net = Net(d, q, hp["WIDTH"], hp["DEPTH"]).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=hp["EPOCHS"], eta_min=1e-5)
    best, best_state, bad, ep = float("inf"), None, 0, 0
    for ep in range(hp["EPOCHS"]):
        net.train()
        perm = torch.randperm(n, device=device)
        for k in range(0, n, hp["BS"]):
            i = perm[k:k + hp["BS"]]
            if len(i) < 8:
                continue
            loss = torch.nn.functional.mse_loss(net(Xt[i]), Yt[i])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            vl = float(torch.nn.functional.mse_loss(net(Xv), Yv))
        if vl < best - 1e-7:
            best, bad = vl, 0
            best_state = {k2: v.detach().clone() for k2, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    P, H = [], []
    with torch.no_grad():
        for X in (Xt, Xv, Xe):
            ps, hs = [], []
            for k in range(0, X.shape[0], 20000):
                y, h = net(X[k:k + 20000], features=True)
                ps.append(y.double().cpu().numpy())
                hs.append(h.double().cpu().numpy())
            P.append(np.concatenate(ps) * zs + zm)
            H.append(np.concatenate(hs))
    del net, Xt, Xv, Xe, Yt, Yv
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return P, H, ep + 1, float(best)


# ------------------------------------------------------------------ corpora


def rel_l2(T, P):
    return float(np.mean(np.linalg.norm(T - P, axis=1) / np.maximum(np.linalg.norm(T, axis=1), 1e-30)))


class Std:
    def __init__(self, A):
        self.m = A.mean(0)
        self.s = A.std(0)
        self.s[self.s == 0] = 1.0

    def fwd(self, A):
        return (A - self.m) / self.s

    def inv(self, A):
        return A * self.s + self.m


def load_emit(args):
    """emit_campaign.py's split, verbatim: RandomState(seed) permutation, 10 % test, a 10 % validation carve
    with RandomState(seed + 10000); PCA-<rank> targets of one component; the error is its rel_l2 in physical units."""
    root = pathlib.Path(os.environ.get("EMIT_DATA", "data/emit"))
    X = np.load(root / "X.npy")
    Y = np.load(root / (args.comp + ".npy"))
    n_all = len(X)
    perm = np.random.RandomState(args.seed).permutation(n_all)
    n_te = int(round(0.1 * n_all))
    idx_te, tr_full = perm[:n_te], perm[n_te:]
    vperm = np.random.RandomState(args.seed + 10000).permutation(len(tr_full))
    n_val = int(round(0.1 * len(tr_full)))
    idx_va, idx_tr = tr_full[vperm[:n_val]], tr_full[vperm[n_val:]]
    if args.ntrain and args.ntrain < len(idx_tr):
        idx_tr = idx_tr[:args.ntrain]
    xs = Std(X[idx_tr].astype(np.float64))
    ys = Std(Y[idx_tr].astype(np.float64))
    Ys_tr = ys.fwd(Y[idx_tr].astype(np.float64))
    c = Ys_tr.mean(0)
    _, S, Vt = np.linalg.svd(Ys_tr - c, full_matrices=False)
    Vt = Vt[:args.pca_rank]
    evr = float((S[:args.pca_rank] ** 2).sum() / (S ** 2).sum())
    fwd = lambda A: (ys.fwd(A) - c) @ Vt.T
    inv = lambda Z: ys.inv(Z @ Vt + c)
    Yph = {"tr": Y[idx_tr].astype(np.float64), "va": Y[idx_va].astype(np.float64), "te": Y[idx_te].astype(np.float64)}
    Z = {k: fwd(v) for k, v in Yph.items()}
    D = dict(problem="emit", tag=f"emit_{args.comp}_s{args.seed}",
             Xtr=xs.fwd(X[idx_tr].astype(np.float64)), Xva=xs.fwd(X[idx_va].astype(np.float64)),
             Xte=xs.fwd(X[idx_te].astype(np.float64)),
             Ztr=Z["tr"], Zva=Z["va"], Zte=Z["te"], names=[f"in{j}" for j in range(X.shape[1])],
             err=lambda Zp, split: rel_l2(Yph[split], inv(Zp)),
             err_rows=lambda Zp, split, rows: rel_l2(Yph[split][rows], inv(Zp)),
             Yobj=Z["tr"] - Z["tr"].mean(0), ynorm2=np.maximum((Yph["tr"] ** 2).sum(1), 1e-30),
             extra=dict(pca_rank=args.pca_rank, pca_evr=evr, component=args.comp))
    return D, dict(EPOCHS=150, WIDTH=512, DEPTH=3, BS=1024)


def load_oco2(args):
    """oco2_curve.py's protocol: jpl_data's seeded 2000-row validation carve and the stored test block;
    the target is the 40 reduced coefficients, the error the reduced relative L2, radiance as an extra metric."""
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import jpl_data
    if os.environ.get("NMKC_JPL_DATA"):
        jpl_data.DATA = pathlib.Path(os.environ["NMKC_JPL_DATA"])
    sp = jpl_data.load_band(args.band, seed=args.seed)
    Xtr, Ytr, Xva, Yva, Xte, Yte = (sp[k] for k in ("Xtr", "Ytr", "Xval", "Yval", "Xte", "Yte"))
    if args.ntrain and args.ntrain < len(Xtr):
        Xtr, Ytr = Xtr[:args.ntrain], Ytr[:args.ntrain]
    recon = jpl_data.reconstruction(args.band)
    xs = Std(Xtr)
    Yt = {"tr": Ytr, "va": Yva, "te": Yte}
    D = dict(problem="oco2", tag=f"oco2_{args.band}_s{args.seed}",
             Xtr=xs.fwd(Xtr), Xva=xs.fwd(Xva), Xte=xs.fwd(Xte),
             Ztr=Ytr, Zva=Yva, Zte=Yte, names=[f"x{j}" for j in range(Xtr.shape[1])],
             err=lambda Zp, split: rel_l2(Yt[split], Zp),
             err_rows=lambda Zp, split, rows: rel_l2(Yt[split][rows], Zp),
             Yobj=Ytr - Ytr.mean(0), ynorm2=np.maximum((Ytr ** 2).sum(1), 1e-30),
             extra_metrics={"radiance": lambda Zp, split: jpl_data.radiance_error(Zp, Yt[split], recon)},
             extra=dict(band=args.band))
    return D, dict(EPOCHS=250, WIDTH=384, DEPTH=4, BS=512)


def load_synth(args):
    """The corpus the local CPU test runs on: a smooth vector-valued function of d inputs, no files."""
    rng = np.random.default_rng(args.seed)
    n, nva, nte, d, q = 700, 200, 200, 5, 3
    X = rng.standard_normal((n + nva + nte, d))
    W = rng.standard_normal((d, q))
    Y = np.sin(X @ W) + 0.3 * X[:, :1] ** 2 + 0.05 * rng.standard_normal((len(X), q))
    xs, ys = Std(X[:n]), Std(Y[:n])
    Xs, Zs = xs.fwd(X), ys.fwd(Y)
    sl = {"tr": slice(0, n), "va": slice(n, n + nva), "te": slice(n + nva, None)}
    Yph = {k: Y[v] for k, v in sl.items()}
    D = dict(problem="synth", tag=f"synth_s{args.seed}",
             Xtr=Xs[sl["tr"]], Xva=Xs[sl["va"]], Xte=Xs[sl["te"]],
             Ztr=Zs[sl["tr"]], Zva=Zs[sl["va"]], Zte=Zs[sl["te"]], names=[f"x{j}" for j in range(d)],
             err=lambda Zp, split: rel_l2(Yph[split], ys.inv(Zp)),
             err_rows=lambda Zp, split, rows: rel_l2(Yph[split][rows], ys.inv(Zp)),
             Yobj=Zs[sl["tr"]] - Zs[sl["tr"]].mean(0), ynorm2=np.maximum((Yph["tr"] ** 2).sum(1), 1e-30),
             extra=dict(note="synthetic corpus for the CPU test"))
    return D, dict(EPOCHS=3, WIDTH=32, DEPTH=2, BS=128)


def load_corpus(args):
    if args.corpus == "emit":
        return load_emit(args)
    if args.corpus == "oco2":
        return load_oco2(args)
    if args.corpus == "synth":
        return load_synth(args)
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import bench_data
    if args.corpus == "climsim":
        D = bench_data.climsim(args.seed, args.ntrain or 100000, nval=20000, ntest=20000)
        hp = dict(EPOCHS=60, WIDTH=512, DEPTH=4, BS=1024)
    elif args.corpus == "pkanrtm":
        D = bench_data.pkanrtm(args.seed, args.ntrain, args.lowfi)
        hp = dict(EPOCHS=120, WIDTH=384, DEPTH=4, BS=1024)
    elif args.corpus == "rrtmgp":
        import rrtmgp_data
        D = rrtmgp_data.rrtmgp(args.seed, args.ntrain or 100000,
                               test_files=tuple(args.rrtmgp_test_files.split(",")) if args.rrtmgp_test_files else None)
        hp = dict(EPOCHS=60, WIDTH=512, DEPTH=4, BS=1024)
    elif args.corpus == "qm9":
        import qm9_data
        D = qm9_data.qm9(args.seed, args.ntrain or 100000)
        hp = dict(EPOCHS=200, WIDTH=512, DEPTH=4, BS=512)
    else:
        raise SystemExit(f"unknown corpus {args.corpus!r}")
    return D, hp


# ------------------------------------------------------------------ combiners


def gcv_ridge_stack(Pva, Zva, Pte):
    """Ridge of the target on the stack of head predictions, one fit per output coordinate, the ridge
    parameter by generalized cross-validation on validation (no test, no second selection pass).
    Pva/Pte: (H, n, q). Returns (pred_va, pred_te, info)."""
    H, nva, q = Pva.shape
    pv = np.empty((nva, q))
    pt = np.empty((Pte.shape[1], q))
    lams, used = np.logspace(-8, 2, 21), []
    for j in range(q):
        A = np.concatenate([Pva[:, :, j].T, np.ones((nva, 1))], 1)
        y = Zva[:, j]
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        Uty = U.T @ y
        best = (np.inf, None)
        for lam in lams:
            f = s ** 2 / (s ** 2 + lam * nva)
            resid = y - U @ (f * Uty)
            dof = f.sum()
            g = (resid ** 2).mean() / max((1.0 - dof / nva) ** 2, 1e-12)
            if g < best[0]:
                best = (g, lam)
        lam = best[1]
        used.append(lam)
        w = Vt.T @ ((s / (s ** 2 + lam * nva)) * Uty)
        pv[:, j] = A @ w
        pt[:, j] = np.concatenate([Pte[:, :, j].T, np.ones((Pte.shape[1], 1))], 1) @ w
    return pv, pt, dict(lam_median=float(np.median(used)), lam_min=float(np.min(used)), lam_max=float(np.max(used)))


def make_err_rows(D):
    """err(Zp, split, rows): the corpus metric on a subset of a split's rows, so a search that has to hold
    several Grams at once can run on fewer validation rows. bench_data's _pack already exposes the two
    pieces (phys_pred and den, both row-sliceable); the loaders in this file supply their own."""
    if "err_rows" in D:
        return D["err_rows"]
    if "Yph" in D and "phys_pred" in D and "den" in D:
        def f(Zp, split, rows):
            T = np.asarray(D["Yph"][split][rows], np.float64)
            P = np.asarray(D["phys_pred"](Zp, split, rows), np.float64)
            return float(np.mean(np.linalg.norm(P - T, axis=1) / D["den"](split, rows)))
        return f
    return None


def simplex_candidates(K, seed, n_draw=48):
    rng = np.random.default_rng(seed + 33)
    cands = [np.ones(K) / K]
    for k in range(K):
        e = np.zeros(K)
        e[k] = 1.0
        cands.append(e)
    for _ in range(n_draw):
        cands.append(rng.dirichlet(np.ones(K) * 0.7))
    return cands


# ------------------------------------------------------------------ main


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--corpus", required=True,
                   choices=["emit", "oco2", "pkanrtm", "rrtmgp", "climsim", "qm9", "synth"])
    p.add_argument("--comp", default="Y2", help="EMIT component Y1..Y4")
    p.add_argument("--band", default="o2", help="OCO-2 band o2 / wco2 / sco2")
    p.add_argument("--lowfi", type=int, default=0, help="pKANrtm: 1 = the 6S coefficients as extra inputs")
    p.add_argument("--rrtmgp_test_files", default="")
    p.add_argument("--pca_rank", type=int, default=64, help="EMIT target rank")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ntrain", type=int, default=0, help="0 = the corpus default")
    p.add_argument("--kernels", default="krr_matern12,krr_matern32,krr_matern52,krr_rbf,krr_sm4,krr_nngp,"
                                        "krr_ntk,krr_add,krr_ard_kf,krr_ard_eb")
    p.add_argument("--fkernels", default="dkr_matern52,dkr_rbf,dkr_ntk,dkr_ard_kf",
                   help="kernels on the network features; empty string = no network")
    p.add_argument("--combiners", default="mkl_sum,mkl_stack,mkl_ridge,select")
    p.add_argument("--tune_sub", type=int, default=6000)
    p.add_argument("--mkl_sub", type=int, default=3000, help="mkl_sum weight search: training rows")
    p.add_argument("--mkl_val", type=int, default=4000, help="mkl_sum weight search: validation rows")
    p.add_argument("--exact_max", type=int, default=0, help="0 = the device cap (GPU 36000, CPU 20000)")
    p.add_argument("--centers", type=int, default=8000, help="Falkon Nystrom centers")
    p.add_argument("--cg_iters", type=int, default=25)
    p.add_argument("--pred_chunk", type=int, default=4096)
    p.add_argument("--cache_gb", type=float, default=12.0,
                   help="cache K_nM when it fits in this many GB and in 45 percent of the memory fraction")
    p.add_argument("--kf_steps", type=int, default=300)
    p.add_argument("--eb_steps", type=int, default=150)
    p.add_argument("--eb_rows", type=int, default=2000)
    p.add_argument("--eb_outputs", type=int, default=8)
    p.add_argument("--sm_q", type=int, default=4, help="spectral-mixture components (the krr_sm<Q> name is a label)")
    p.add_argument("--sm_rows", type=int, default=1500, help="rows of the spectral-mixture empirical-Bayes fit")
    p.add_argument("--also_falkon", action="store_true",
                   help="add a Falkon twin of the Matern-5/2 head as a receipt on the approximation")
    p.add_argument("--cpu", action="store_true", help="force the CPU path (tests only)")
    p.add_argument("--epochs", type=int, default=0)
    p.add_argument("--width", type=int, default=0)
    p.add_argument("--tag", required=True)
    p.add_argument("--smoke", action="store_true")
    return p


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def peak_ram_gb():
    if resource is None:
        return None
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, 3)


def main(argv=None):
    t0 = time.time()
    parser = build_parser()
    args = parser.parse_args(argv)
    at_default = lambda k: getattr(args, k) == parser.get_default(k)
    device, budget, dinfo = setup_device(prefer_gpu=not args.cpu)
    hard_cap = EXACT_MAX_GPU if device.type == "cuda" else EXACT_MAX_CPU
    if args.exact_max <= 0:
        args.exact_max = hard_cap
    if args.exact_max > hard_cap:
        raise GramTooLarge(f"--exact_max {args.exact_max} exceeds the {device.type} cap {hard_cap}")
    if args.centers > FALKON_MAX_CENTERS:
        raise GramTooLarge(f"--centers {args.centers} exceeds FALKON_MAX_CENTERS {FALKON_MAX_CENTERS}")

    D, nethp = load_corpus(args)
    if args.epochs:
        nethp["EPOCHS"] = args.epochs
    if args.width:
        nethp["WIDTH"] = args.width
    if args.smoke:
        # The whole code path on a tiny training slice; the loaders already ran in full, so the smoke's
        # peak_ram_gb is the real loading peak. A smoke shrinks only the settings left at their default:
        # an option given on the command line is the operator's, not the smoke's, and is never overridden.
        for k, v in (("tune_sub", 400), ("mkl_sub", 300), ("mkl_val", 400), ("centers", 250), ("cg_iters", 5),
                     ("kf_steps", 15), ("eb_steps", 15), ("eb_rows", 300), ("sm_rows", 300), ("pred_chunk", 512),
                     ("cache_gb", 0.5)):
            if at_default(k):
                setattr(args, k, v)
        if at_default("epochs"):
            nethp["EPOCHS"] = 2
        for k in ("Xtr", "Ztr", "Yobj"):
            D[k] = D[k][:1200]
        D["ynorm2"] = D["ynorm2"][:1200]
        args.also_falkon = True          # a smoke always exercises the Nystrom path as well as the exact one
    Xtr, Xva, Xte = (np.ascontiguousarray(D[k], np.float64) for k in ("Xtr", "Xva", "Xte"))
    Ztr, Zva, Zte = (np.ascontiguousarray(D[k], np.float64) for k in ("Ztr", "Zva", "Zte"))
    n, d = Xtr.shape
    q = Ztr.shape[1]
    err = D["err"]
    err_rows = make_err_rows(D)
    print(f"{D['tag']}: n={n} d={d} q={q} device={dinfo['device']} exact_max={args.exact_max} "
          f"budget={budget / 2**30:.1f} GB", flush=True)
    if dinfo.get("visible_device_count", 1) > 1:
        print(f"WARNING: {dinfo['visible_device_count']} CUDA devices are visible; a lane must hold exactly "
              f"one card (CUDA_VISIBLE_DEVICES=<one GPU UUID>). Using cuda:0 only.", flush=True)

    to_t = lambda A: torch.as_tensor(A, dtype=DT, device=device)
    Ftr, Fva, Fte, Yt = to_t(Xtr), to_t(Xva), to_t(Xte), to_t(Ztr)
    sub = torch.as_tensor(np.random.default_rng(args.seed + 7).permutation(n)[:min(args.tune_sub, n)], device=device)
    eb_rows = torch.as_tensor(np.random.default_rng(args.seed + 13).permutation(n)[:min(args.eb_rows, n)], device=device)

    methods, hyper, heads_va, heads_te = {}, {}, {}, {}

    def record(name, pv, pt, hp=None, head=True):
        r = dict(val=100.0 * err(pv, "va"), test=100.0 * err(pt, "te"))
        for mn, fn in D.get("extra_metrics", {}).items():
            r["test_" + mn] = fn(pt, "te")
            r["val_" + mn] = fn(pv, "va")
        methods[name] = r
        hyper[name] = hp or {}
        if head:
            heads_va[name], heads_te[name] = pv, pt
        print(f"== {name}: val {r['val']:.4f}% test {r['test']:.4f}% [{(time.time() - t0) / 60:.1f} min]", flush=True)

    want_k = [k.strip() for k in args.kernels.split(",") if k.strip()]
    want_f = [k.strip() for k in args.fkernels.split(",") if k.strip()]
    want_c = [k.strip() for k in args.combiners.split(",") if k.strip()]

    # ---- learned metrics (on the inputs), used by the ARD and spectral-mixture families
    learned = {}
    if any(k.endswith("_ard_kf") for k in want_k + want_f):
        learned["ard_kf"] = kf_ard(Ftr, D["Yobj"], D["ynorm2"], steps=args.kf_steps, seed=args.seed)
        hyper["_ard_kf_input"] = dict(ell_rel=[round(float(x), 4) for x in learned["ard_kf"]], steps=args.kf_steps)
        print(f"  kernel-flow ARD done [{(time.time() - t0) / 60:.1f} min]", flush=True)
    if any(k.endswith("_ard_eb") for k in want_k + want_f):
        ell, info = eb_ard(Ftr, Yt, eb_rows, steps=args.eb_steps, seed=args.seed, n_out=args.eb_outputs)
        learned["ard_eb"] = ell
        hyper["_ard_eb_input"] = dict(info, ell_rel=[round(float(x), 4) for x in ell])
        print(f"  empirical-Bayes ARD done [{(time.time() - t0) / 60:.1f} min]", flush=True)
    if any("_sm" in k for k in want_k + want_f):
        w_, mu_, v_, info = eb_sm(Ftr, Yt, eb_rows[:min(args.sm_rows, len(eb_rows))], Q=args.sm_q,
                                  steps=args.eb_steps, seed=args.seed)
        learned["sm"] = (w_, mu_, v_, info)
        hyper["_sm_input"] = info
        print(f"  spectral mixture (Q={args.sm_q}) fitted [{(time.time() - t0) / 60:.1f} min]", flush=True)

    vfn = lambda pv: err(pv, "va")
    for name in want_k:
        cands = kernel_grid(name, Ftr[sub], learned, args)
        if not cands:
            print(f"  {name}: skipped (d={d} above ADD_MAX_D {ADD_MAX_D})", flush=True)
            continue
        got = fit_head(name, cands, Ftr, Fva, Fte, Yt, vfn, sub, args, budget, device)
        if got is None:
            print(f"  {name}: no positive-definite candidate, skipped", flush=True)
            continue
        record(name, got[0], got[1], got[2])
    if args.also_falkon and "krr_matern52" in want_k:
        cands = kernel_grid("krr_matern52", Ftr[sub], learned, args)
        got = fit_head("krr_matern52_falkon", cands, Ftr, Fva, Fte, Yt, vfn, sub, args, budget, device,
                       force_falkon=True)
        if got is not None:
            record("krr_matern52_falkon", got[0], got[1], got[2], head=False)

    # ---- one kernel from a convex combination of the base kernels (multiple kernel learning)
    if "mkl_sum" in want_c:
        base = [(nm, kernel_grid(nm, Ftr[sub], learned, args)) for nm in want_k]
        chosen = []
        for nm, cds in base:
            if not cds or nm not in hyper:
                continue
            hp = hyper[nm]
            pick = None
            for params, kfn in cds:
                same = all(abs(params.get(kk, 0) - hp.get(kk, 0)) < 1e-12 for kk in ("scale", "depth", "sb2")
                           if kk in params and isinstance(params.get(kk), (int, float)))
                if same:
                    pick = kfn
                    break
            chosen.append((nm, pick if pick is not None else cds[0][1], hp.get("nugget", 1e-6)))
        if len(chosen) >= 2:
            # every base Gram is held at once here, so the weight search runs on --mkl_sub training rows and
            # --mkl_val validation rows (the corpus metric restricted to those rows); the winner is then
            # refitted on every row through the ordinary head path below.
            msub = sub[:min(args.mkl_sub, len(sub))]
            nva = Xva.shape[0]
            vrows = np.sort(np.random.default_rng(args.seed + 17).permutation(nva)[:min(args.mkl_val, nva)])
            if err_rows is None:
                vrows = np.arange(nva)
                sub_val = lambda P: vfn(P)
            else:
                sub_val = lambda P: err_rows(P, "va", vrows)
            Fv = Fva[torch.as_tensor(vrows, device=device)]
            Fs = Ftr[msub]
            Ysub = Yt[msub]
            Ks = [kfn(Fs, Fs) for _, kfn, _ in chosen]
            Kvs = [kfn(Fv, Fs) for _, kfn, _ in chosen]
            # centered kernel alignment of each base kernel with the target Gram, as one candidate weighting
            Yc = Ysub - Ysub.mean(0)
            Ky = Yc @ Yc.T
            al = np.array([float((Kb * Ky).sum() / (torch.linalg.norm(Kb) * torch.linalg.norm(Ky) + 1e-30))
                           for Kb in Ks])
            al = np.maximum(al, 0)
            al = al / al.sum() if al.sum() > 0 else np.ones(len(Ks)) / len(Ks)
            cands = simplex_candidates(len(Ks), args.seed, n_draw=8 if args.smoke else 48) + [al]
            best = (float("inf"), None, None)
            for th in cands:
                Kc = None
                for wgt, Kb in zip(th, Ks):
                    if wgt <= 0:
                        continue
                    Kc = Kb * wgt if Kc is None else Kc + wgt * Kb
                if Kc is None:
                    continue
                Kvc = None
                for wgt, Kb in zip(th, Kvs):
                    if wgt <= 0:
                        continue
                    Kvc = Kb * wgt if Kvc is None else Kvc + wgt * Kb
                for nug in NUGGETS:
                    a_ = chol_solve(Kc.clone(), Ysub, nug)
                    if a_ is None:
                        continue
                    e = sub_val((Kvc @ a_).detach().cpu().numpy())
                    if e < best[0]:
                        best = (e, np.asarray(th, float), nug)
                del Kc, Kvc
            del Ks, Kvs, Ky, Yc, Fv
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if best[1] is not None:
                theta, nug = best[1], best[2]
                fns = [kfn for _, kfn, _ in chosen]

                def mix(A, B, th=theta, fns=fns):
                    K = None
                    for wgt, f in zip(th, fns):
                        if wgt <= 0:
                            continue
                        Kb = f(A, B)
                        if K is None:
                            K = Kb.mul_(wgt)
                        else:
                            K.add_(Kb, alpha=float(wgt))
                            del Kb
                    return K
                got = fit_head("mkl_sum", [(dict(theta=[round(float(x), 4) for x in theta]), mix)],
                               Ftr, Fva, Fte, Yt, vfn, sub, args, budget, device)
                if got is not None:
                    hp = dict(got[2])
                    hp["weights"] = {nm: round(float(x), 4) for (nm, _, _), x in zip(chosen, theta)}
                    hp["alignment"] = {nm: round(float(x), 4) for (nm, _, _), x in zip(chosen, al)}
                    hp["weight_search"] = dict(rows=int(len(msub)), val_rows=int(len(vrows)),
                                               candidates=int(len(cands)), val_sub_restricted=float(best[0]))
                    record("mkl_sum", got[0], got[1], hp)

    # ---- the network and the kernels on its features
    if want_f:
        P, Hs, ep, best_mse = train_net(Xtr, Xva, Xte, Ztr, Zva, nethp, args.seed * 100, device)
        record("mlp", P[1], P[2], dict(epochs=ep, width=nethp["WIDTH"], depth=nethp["DEPTH"],
                                       best_val_mse=best_mse, tune_budget="fixed schedule, one run"))
        mu, sd = Hs[0].mean(0), Hs[0].std(0) + 1e-9
        Gtr, Gva, Gte = (to_t((H - mu) / sd) for H in Hs)
        flearn = {}
        if any(k.endswith("_ard_kf") for k in want_f):
            flearn["ard_kf"] = kf_ard(Gtr, D["Yobj"], D["ynorm2"], steps=args.kf_steps, seed=args.seed + 1)
        if any(k.endswith("_ard_eb") for k in want_f):
            flearn["ard_eb"] = eb_ard(Gtr, Yt, eb_rows, steps=args.eb_steps, seed=args.seed + 1,
                                      n_out=args.eb_outputs)[0]
        if any("_sm" in k for k in want_f):
            flearn["sm"] = eb_sm(Gtr, Yt, eb_rows[:min(args.sm_rows, len(eb_rows))], Q=args.sm_q,
                                 steps=args.eb_steps, seed=args.seed + 1)
        for name in want_f:
            cands = kernel_grid(name, Gtr[sub], flearn, args)
            if not cands:
                print(f"  {name}: skipped (feature dimension {Gtr.shape[1]} above ADD_MAX_D)", flush=True)
                continue
            got = fit_head(name, cands, Gtr, Gva, Gte, Yt, vfn, sub, args, budget, device)
            if got is None:
                continue
            record(name, got[0], got[1], got[2])
        # the network as the mean, an exact kernel on its residual (the paper's correction head)
        R = to_t(Ztr - P[0])
        cands = kernel_grid("krr_matern52", Ftr[sub], learned, args)
        got = fit_head("mlp_resid_matern52", cands, Ftr, Fva, Fte, R, lambda pv: err(P[1] + pv, "va"),
                       sub, args, budget, device)
        if got is not None:
            record("mlp_resid_matern52", P[1] + got[0], P[2] + got[1], got[2])
        del Gtr, Gva, Gte, R
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- combiners over the heads
    cand = list(heads_va)
    if len(cand) > 1:
        if "select" in want_c:
            Sv, St = np.empty_like(Zva), np.empty_like(Zte)
            picks = {h: 0 for h in cand}
            for j in range(q):
                e = [np.sqrt(((heads_va[h][:, j] - Zva[:, j]) ** 2).mean()) for h in cand]
                k = int(np.argmin(e))
                picks[cand[k]] += 1
                Sv[:, j], St[:, j] = heads_va[cand[k]][:, j], heads_te[cand[k]][:, j]
            record("select", Sv, St, dict(candidates=cand, picks=picks, val_is_in_sample=True), head=False)
        if "mkl_stack" in want_c:
            from scipy.optimize import minimize
            Pv = [heads_va[h] for h in cand]
            Pt = [heads_te[h] for h in cand]
            M_ = len(cand)
            obj = lambda w: err(sum(wi * Pi for wi, Pi in zip(w, Pv)), "va")
            res = minimize(obj, np.ones(M_) / M_, bounds=[(0, 1)] * M_,
                           constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
                           method="SLSQP", options=dict(maxiter=300, ftol=1e-12))
            w = np.maximum(res.x, 0)
            w = w / w.sum() if w.sum() > 0 else np.ones(M_) / M_
            record("mkl_stack", sum(wi * Pi for wi, Pi in zip(w, Pv)), sum(wi * Pi for wi, Pi in zip(w, Pt)),
                   dict(weights={h: round(float(x), 4) for h, x in zip(cand, w)}, val_is_in_sample=True), head=False)
        if "mkl_ridge" in want_c:
            Av = np.stack([heads_va[h] for h in cand])
            At = np.stack([heads_te[h] for h in cand])
            pv, pt, info = gcv_ridge_stack(Av, Zva, At)
            record("mkl_ridge", pv, pt, dict(info, candidates=cand, val_is_in_sample=True), head=False)

    out = dict(tag=args.tag, kind="multikernel", corpus=args.corpus, seed=args.seed,
               n_train=int(n), n_val=int(len(Xva)), n_test=int(len(Xte)), d=int(d), q=int(q),
               device=dinfo["device"], device_info=dinfo, methods=methods, hyper=hyper,
               exact_max=int(args.exact_max), centers=int(args.centers), tune_sub=int(args.tune_sub),
               kernels=want_k, fkernels=want_f, combiners=want_c,
               corpus_extra=dict({k: v for k, v in D.items()
                                  if k in ("pca_evr_in", "pca_evr_out", "pca_fit_rows", "persistence_err",
                                           "pca_recon_err", "extra_metrics_lowfi", "err_floor",
                                           "frac_below_floor_te", "target", "rank_in", "rank_out")},
                                 **D.get("extra", {})), threads=int(_T),
               minutes=round((time.time() - t0) / 60, 2),
               peak_ram_gb=peak_ram_gb(),
               peak_vram_gb=(round(torch.cuda.max_memory_allocated() / 2 ** 30, 3) if device.type == "cuda" else None),
               code_sha256=sha256_of(__file__), smoke=bool(args.smoke))
    root = pathlib.Path(os.environ.get("P2_OUT", "results"))
    root.mkdir(parents=True, exist_ok=True)
    tmp = root / (args.tag + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=float)
    os.replace(tmp, root / (args.tag + ".json"))
    pdir = root / "preds"
    pdir.mkdir(exist_ok=True)
    keep = 200 if args.smoke else None          # a smoke writes the file but not 200 MB of it
    np.savez_compressed(pdir / (args.tag + ".npz"),
                        **{h: heads_te[h][:keep].astype(np.float32) for h in heads_te})
    print(f"DONE {args.tag} in {out['minutes']} min "
          f"(peak_ram {out['peak_ram_gb']} GB, peak_vram {out['peak_vram_gb']} GB)", flush=True)
    return out


if __name__ == "__main__":
    main()
