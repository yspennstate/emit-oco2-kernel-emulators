"""A streaming loader for The Well's 2D datasets (active_matter, helmholtz_staircase, and TRL2D again), one-step maps
state_t -> state_{t+1} on the full multi-channel field, built the way bench_data.trl2d builds its representation but
without ever holding the whole training set in memory: two passes over the HDF5 files, a reservoir subsample for the
PCA fits, chunked projections, standardized validation/test fields kept as float32 memmaps for the error.

Fields are taken from the file's t0_fields (scalars), t1_fields (vectors, last axis 2) and t2_fields (tensors, last two
axes 2 x 2) and concatenated per grid point into C channels. The regression target is the increment (default) or the
next state; the reported error is the relative L2 error of the predicted standardized state against the full field,
with the persistence baseline and the PCA reconstruction floor recorded beside it, and The Well's per-field VRMSE as
extra metrics (per channel group).

Dataset -> fields:
  active_matter        concentration | velocity | D, E                (256 x 256, 11 channels, 81 steps, 3 traj/file)
  helmholtz_staircase  pressure_re, pressure_im | mask                (1024 x 256; check the file)
  turbulent_radiative_layer_2D  density, pressure | velocity         (128 x 384, 4 channels)
Cache under <root>/cache/<key>_*.npy|npz; a lock serializes concurrent builders. Environment: DATA_NEW.
"""
import glob, os, pathlib
import numpy as np

DN = pathlib.Path(os.environ.get("DATA_NEW", os.path.expanduser("~/data_new")))
CHUNK = int(os.environ.get("WELL_CHUNK", "128"))      # rows held at once by err/den/the metrics; 128 x 256 x 256 x 11
                                                      # float64 is 0.7 GB, so a three-array step stays near 2 GB
FIELDS = {
    "active_matter": [("t0_fields", "concentration"), ("t1_fields", "velocity"), ("t2_fields", "D"), ("t2_fields", "E")],
    "helmholtz_staircase": [("t0_fields", "pressure_re"), ("t0_fields", "pressure_im")],
    "turbulent_radiative_layer_2D": [("t0_fields", "density"), ("t0_fields", "pressure"), ("t1_fields", "velocity")],
}


def rel_l2(Yt, Yp):
    return float(np.mean(np.linalg.norm(Yt - Yp, axis=1) / np.maximum(np.linalg.norm(Yt, axis=1), 1e-30)))


def vrmse_paper(pred_rows, T, C, chunk=128, eps=1e-7, per_row=False):
    """The Well's VRMSE as its paper (Ohana et al. 2024, App. E.3) and its code (the_well/benchmark/metrics/spatial.py,
    NMSE with norm_mode="std") define it: for every row and every channel, sqrt(mean_cells (P - T)^2 / (var_cells T + eps))
    with the variance taken over the cells of THAT row (torch.std, N - 1), and then the MEAN over rows. Table 2 of the
    paper is this quantity averaged over channels. It is invariant to the per-channel standardization the loaders apply.
    `pred_rows(i, j)` returns the predicted standardized state of rows i..j as (j - i, F * C); T is the (n, F * C) truth
    (a memmap is fine); the rows are walked in chunks so the whole prediction is never held twice. Returns (C,), or the
    per-row (n, C) values with per_row=True (what traj_median needs)."""
    n = T.shape[0]; F = T.shape[1] // C; rows = np.empty((n, C))
    for i in range(0, n, chunk):
        j = min(i + chunk, n)
        p = np.asarray(pred_rows(i, j), np.float64).reshape(j - i, F, C); t = np.asarray(T[i:j], np.float64).reshape(j - i, F, C)
        rows[i:j] = np.sqrt(((p - t) ** 2).mean(1) / (t.var(1, ddof=1) + eps))
    return rows if per_row else rows.mean(0)


def traj_median(rows, tid):
    """Walrus's convention (McCabe et al. 2025, Tables 1/13): the MEDIAN over trajectories of the trajectory-averaged
    one-step VRMSE. `rows` is vrmse_paper(..., per_row=True), `tid` the trajectory id of every row. Returns (C,)."""
    tid = np.asarray(tid); ids = np.unique(tid)
    return np.median(np.stack([rows[tid == u].mean(0) for u in ids]), axis=0)


def vrmse_pooled(pred_rows, T, C, chunk=128):
    """The convention the first lanes reported under `vrmse_*`: RMSE over (rows, cells) / std over (rows, cells), per
    channel. It differs from the paper's in two ways - the pooled std counts the variation BETWEEN rows (time,
    trajectory, physical parameter), and it is a ratio of aggregates rather than a mean of per-row ratios - and the two
    move in opposite directions, so it can read either lower or higher than the paper's (TRL2D persistence, test block:
    pressure 0.62 pooled vs 1.13 paper, vx 0.42 pooled vs 0.30 paper). Not comparable with the Well's tables; kept so
    the old result files stay readable."""
    n = T.shape[0]; F = T.shape[1] // C; se = np.zeros(C); s1 = np.zeros(C); s2 = np.zeros(C); cnt = 0
    for i in range(0, n, chunk):
        j = min(i + chunk, n)
        p = np.asarray(pred_rows(i, j), np.float64).reshape(-1, C); t = np.asarray(T[i:j], np.float64).reshape(-1, C)
        se += ((p - t) ** 2).sum(0); s1 += t.sum(0); s2 += (t ** 2).sum(0); cnt += len(t)
    var = np.maximum(s2 / cnt - (s1 / cnt) ** 2, 0.0)
    return np.sqrt(se / cnt) / np.sqrt(var)


class Std:
    def __init__(self, A):
        self.m = A.mean(0); self.s = A.std(0); self.s[self.s == 0] = 1.0

    def fwd(self, A):
        return (A - self.m) / self.s


def _files(root, split):
    fs = sorted(glob.glob(str(root / "data" / split / "*.hdf5"))) or sorted(glob.glob(str(root / f"{split}_*.hdf5")))
    return fs


def _split_files(root):
    """The three file lists as _build_cache sees them: the release's train/valid/test, with a validation split carved
    from the last tenth of the training files when a release ships none."""
    splits = {s: _files(root, s) for s in ("train", "valid", "test")}
    if splits["train"] and not splits["valid"]:
        k = max(1, len(splits["train"]) // 10); splits["valid"] = splits["train"][-k:]; splits["train"] = splits["train"][:-k]
    return splits


def _row_index(files, fields, stride):
    """Trajectory id and step of every cached row of one split, in cache order (file, trajectory, t), read from the
    HDF5 shapes only - no field data is loaded."""
    import h5py
    grp, nm = fields[0]; tid, step = [], []; base = 0
    for f in files:
        with h5py.File(f, "r") as h:
            traj, T = h[grp][nm].shape[:2]
        ts = np.arange(0, T - 1, stride)
        for tr_ in range(traj):
            tid.append(np.full(len(ts), base + tr_)); step.append(ts)
        base += traj
    return (np.concatenate(tid), np.concatenate(step)) if tid else (np.zeros(0, int), np.zeros(0, int))


def _read_states(path, fields):
    """(traj, T, H, W, C) float32 for one file, channels concatenated in the order of `fields`."""
    import h5py
    parts, names = [], []
    with h5py.File(path, "r") as h:
        for grp, nm in fields:
            a = h[grp][nm][:]                                  # scalar (traj,T,H,W) / vector (...,2) / tensor (...,2,2)
            a = a.reshape(a.shape[:4] + (-1,)).astype(np.float32)
            parts.append(a); names += [f"{nm}{j}" if a.shape[-1] > 1 else nm for j in range(a.shape[-1])]
    return np.concatenate(parts, -1), names


def _build_cache(root, cdir, key, name, fields, rank_in, rank_out, stride, target, nfit=3000, seed=0):
    rng = np.random.default_rng(seed)
    splits = _split_files(root)                                # a release without a validation split has one carved from train
    if not splits["train"] or not splits["test"]:
        raise SystemExit(f"{name}: train/test files missing under {root}")
    # pass 1: channel statistics over the training rows, row counts, and a reservoir subsample of (x_t, x_{t+1}) pairs
    csum = None; csq = None; ncell = 0; n_rows = {}; reservoir_x, reservoir_y, seen = [], [], 0
    shape_hw = None
    for split, fs in splits.items():
        n_rows[split] = 0
        for f in fs:
            st, names = _read_states(f, fields); traj, T, H, W, C = st.shape; shape_hw = (H, W, C)
            ts = np.arange(0, T - 1, stride); n_rows[split] += traj * len(ts)
            if split == "train":
                flat = st.reshape(-1, C).astype(np.float64)
                csum = flat.sum(0) if csum is None else csum + flat.sum(0); csq = (flat ** 2).sum(0) if csq is None else csq + (flat ** 2).sum(0); ncell += len(flat)
                for tr_ in range(traj):
                    for t in ts:                               # reservoir sampling of nfit pairs
                        seen += 1
                        if len(reservoir_x) < nfit:
                            reservoir_x.append(st[tr_, t].reshape(-1).copy()); reservoir_y.append(st[tr_, t + 1].reshape(-1).copy())
                        else:
                            j = rng.integers(0, seen)
                            if j < nfit:
                                reservoir_x[j] = st[tr_, t].reshape(-1).copy(); reservoir_y[j] = st[tr_, t + 1].reshape(-1).copy()
            del st
    H, W, C = shape_hw; F = H * W
    mu = csum / ncell; sd = np.sqrt(np.maximum(csq / ncell - mu ** 2, 0)) + 1e-8
    mu32, sd32 = mu.astype(np.float32), sd.astype(np.float32)
    # standardize the reservoir IN PLACE in float32 (the first version promoted to float64 and peaked near 90 GB)
    Xf = np.stack(reservoir_x); del reservoir_x
    Xv = Xf.reshape(-1, F, C); Xv -= mu32; Xv /= sd32
    Yf = np.stack(reservoir_y); del reservoir_y
    Yv = Yf.reshape(-1, F, C); Yv -= mu32; Yv /= sd32
    if target == "increment":
        Yf -= Xf                                                # Yf now holds the standardized increment
    Tf = Yf

    def fit(A, rank):
        """PCA of the rows of A (float32) through the small Gram matrix: no n x D SVD workspace."""
        c = A.mean(0, dtype=np.float64).astype(np.float32)
        A -= c                                                  # in place; float32 throughout, no n x D copies
        G = (A @ A.T).astype(np.float64)                        # (n, n) Gram
        w, U = np.linalg.eigh(G); order = np.argsort(w)[::-1]; w, U = w[order], U[:, order]
        keep = w[:rank] > 1e-12 * w[0]; U = U[:, :rank][:, keep]; s = np.sqrt(w[:rank][keep])
        Vt = ((U.T.astype(np.float32) @ A) / s[:, None].astype(np.float32)).astype(np.float64)   # (rank, D)
        A += c
        return c.astype(np.float64), Vt, float((w[:rank][keep]).sum() / w[w > 0].sum())
    c_in, Vt_in, evr_in = fit(Xf, rank_in); del Xf
    c_out, Vt_out, evr_out = fit(Tf, rank_out); del Tf, Yf
    # pass 2: scores for every row; standardized full fields of the validation/test rows to memmaps
    out = {k: [] for k in ("Xtr", "Xva", "Xte", "Ztr", "Zva", "Zte")}; tnorm2 = []
    mm = {}
    for split in ("valid", "test"):
        tag = "va" if split == "valid" else "te"
        mm["X" + tag] = np.lib.format.open_memmap(cdir / f"{key}_X{tag}_s.npy", mode="w+", dtype=np.float32, shape=(n_rows[split], F * C))
        mm["Y" + tag] = np.lib.format.open_memmap(cdir / f"{key}_Y{tag}_s.npy", mode="w+", dtype=np.float32, shape=(n_rows[split], F * C))
    pos = {"valid": 0, "test": 0}
    pers = {"va": [], "te": []}; recon = {"va": [], "te": []}
    for split, fs in splits.items():
        tag = {"train": "tr", "valid": "va", "test": "te"}[split]
        for f in fs:
            st, _ = _read_states(f, fields); traj, T = st.shape[:2]; ts = np.arange(0, T - 1, stride)
            sts = ((st.reshape(traj, T, F, C) - mu) / sd).reshape(traj, T, F * C).astype(np.float32); del st
            X = sts[:, ts].reshape(-1, F * C); Y = sts[:, ts + 1].reshape(-1, F * C); del sts
            Tg = Y - X if target == "increment" else Y
            zx = (X.astype(np.float64) - c_in) @ Vt_in.T; zt = (Tg.astype(np.float64) - c_out) @ Vt_out.T
            out["X" + tag].append(zx); out["Z" + tag].append(zt)
            if split == "train":
                tnorm2.append((Tg.astype(np.float64) ** 2).sum(1))
            else:
                n_ = len(X); mm["X" + tag][pos[split]:pos[split] + n_] = X; mm["Y" + tag][pos[split]:pos[split] + n_] = Y; pos[split] += n_
                pred = zt @ Vt_out + c_out + (X.astype(np.float64) if target == "increment" else 0.0)
                pers[tag].append(np.linalg.norm(Y - X, axis=1) / np.linalg.norm(Y, axis=1)); recon[tag].append(np.linalg.norm(Y - pred, axis=1) / np.linalg.norm(Y, axis=1))
            del X, Y, Tg
    for k, v in out.items():
        np.save(cdir / f"{key}_{k}.npy", np.concatenate(v))
    for m_ in mm.values():
        m_.flush()
    np.savez(cdir / f"{key}_meta.npz", mu=mu, sd=sd, c_in=c_in, Vt_in=Vt_in, c_out=c_out, Vt_out=Vt_out, evr_in=evr_in, evr_out=evr_out,
             tnorm2_tr=np.concatenate(tnorm2), pers_va=float(np.concatenate(pers["va"]).mean()), pers_te=float(np.concatenate(pers["te"]).mean()),
             recon_va=float(np.concatenate(recon["va"]).mean()), recon_te=float(np.concatenate(recon["te"]).mean()), nfit=nfit, target=target,
             H=H, W=W, C=C, n_tr=n_rows["train"], n_va=n_rows["valid"], n_te=n_rows["test"], channel_names=np.array(names))


def well2d(name="active_matter", seed=0, ntrain=0, rank_in=256, rank_out=256, stride=1, target="increment"):
    fields = FIELDS[name]
    root = DN / "the_well" / name; cdir = root / "cache"; cdir.mkdir(exist_ok=True)
    key = f"r{rank_in}x{rank_out}_st{stride}_{target}"
    if not (cdir / f"{key}_meta.npz").exists():
        import fcntl                                                # POSIX only; the metric helpers above import cleanly anywhere
        with open(cdir / f"{key}.lock", "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            if not (cdir / f"{key}_meta.npz").exists():
                _build_cache(root, cdir, key, name, fields, rank_in, rank_out, stride, target)
    m = np.load(cdir / f"{key}_meta.npz", allow_pickle=True)
    Vt_out, c_out = m["Vt_out"], m["c_out"]; H, W, C = int(m["H"]), int(m["W"]), int(m["C"])
    Xs = {k: np.load(cdir / f"{key}_X{k}.npy") for k in ("tr", "va", "te")}
    Zs = {k: np.load(cdir / f"{key}_Z{k}.npy") for k in ("tr", "va", "te")}
    Xfull = {k: np.load(cdir / f"{key}_X{k}_s.npy", mmap_mode="r") for k in ("va", "te")}
    Yph = {k: np.load(cdir / f"{key}_Y{k}_s.npy", mmap_mode="r") for k in ("va", "te")}
    tnorm2 = m["tnorm2_tr"]
    if ntrain and ntrain < len(Xs["tr"]):
        keep = np.sort(np.random.default_rng(seed).choice(len(Xs["tr"]), ntrain, replace=False))
        Xs["tr"], Zs["tr"], tnorm2 = Xs["tr"][keep], Zs["tr"][keep], tnorm2[keep]
    incr = str(m["target"]) == "increment"
    inv = lambda Z: Z @ Vt_out + c_out

    def state_of(Zp, split, rows=None):
        base = Xfull[split] if rows is None else Xfull[split][rows]
        return inv(Zp) + (np.asarray(base, np.float64) if incr else 0.0)
    xs = Std(Xs["tr"])
    D = dict(problem=name, tag=f"{name}_s{seed}" + (f"_n{ntrain}" if ntrain else ""), Xtr=xs.fwd(Xs["tr"]), Xva=xs.fwd(Xs["va"]), Xte=xs.fwd(Xs["te"]),
             Ztr=Zs["tr"], Zva=Zs["va"], Zte=Zs["te"], names=[f"pc{j}" for j in range(rank_in)], to_phys=inv)
    # err and den walk the rows in chunks. A whole-split float64 state is 5.8 MB PER ROW on active_matter
    # (256 x 256 x 11), so the old one-shot `rel_l2(asarray(Yph), state_of(Zp))` held the truth, the prediction and
    # their difference at 12 GB each - about 36 GB per CALL, and bench_run calls err once per head per split. That,
    # not the fit, is what OVERCAP-killed both Well smokes at 64 GB against a 40 GB declaration on 2026-09-12.
    def err_chunked(Zp, split, chunk=CHUNK):
        n = Yph[split].shape[0]; acc = 0.0
        for i in range(0, n, chunk):
            j = min(i + chunk, n)
            t = np.asarray(Yph[split][i:j], np.float64); p = state_of(Zp[i:j], split, rows=slice(i, j))
            acc += float((np.linalg.norm(t - p, axis=1) / np.maximum(np.linalg.norm(t, axis=1), 1e-30)).sum())
        return acc / max(n, 1)

    def den_chunked(split, rows=None, chunk=CHUNK):
        Y = Yph[split] if rows is None else Yph[split][rows]
        out = np.empty(len(Y))
        for i in range(0, len(Y), chunk):
            j = min(i + chunk, len(Y))
            out[i:j] = np.linalg.norm(np.asarray(Y[i:j], np.float64), axis=1)
        return np.maximum(out, 1e-30)
    D["err"] = err_chunked
    D["phys_pred"] = lambda Zp, split, rows=None: state_of(Zp, split, rows)   # the caller asks for the rows it can hold
    D["den"] = den_chunked
    D["Yobj"] = Zs["tr"] - Zs["tr"].mean(0); D["ynorm2"] = np.maximum(tnorm2, 1e-30)
    D["Yph"] = Yph
    chan = [str(x) for x in m["channel_names"]]

    # Two VRMSE conventions, both per channel. `vrmse_paper_*` is The Well's own (per-row spatial variance, mean over
    # rows) and is the only one that compares with the paper's Table 2 / the Well leaderboard; `vrmse_*` is the pooled
    # convention the first lanes reported (see vrmse_pooled). Both are computed in chunks: a whole-set (n, F*C) float64
    # prediction plus its truth plus the squared difference is what overran the 60 GB declaration on active_matter.
    def _rows(Zp, split):
        return lambda i, j: state_of(Zp[i:j], split, rows=slice(i, j))
    _cache = {}

    # trajectory ids of the validation/test rows (HDF5 shapes only), for Walrus's median-over-trajectories convention
    sf = _split_files(root); tids = {}
    for s, split in (("va", "valid"), ("te", "test")):
        t_ = _row_index(sf[split], fields, stride)[0]
        tids[s] = t_ if len(t_) == Yph[s].shape[0] else None      # a cache built from other files: no median, said below

    def both(Zp, split):
        import hashlib                                                        # one pass per (prediction, split): the
        k = (split, Zp.shape, hashlib.md5(np.ascontiguousarray(Zp).tobytes()).hexdigest())   # table asks per channel
        if k not in _cache:
            _cache.clear()
            rows = vrmse_paper(_rows(Zp, split), Yph[split], C, per_row=True)
            med = traj_median(rows, tids[split]) if tids[split] is not None else np.full(C, np.nan)
            _cache[k] = (rows.mean(0), vrmse_pooled(_rows(Zp, split), Yph[split], C), med)
        return _cache[k]
    D["extra_metrics"] = {"vrmse_paper_mean": lambda Zp, split: float(both(Zp, split)[0].mean()),
                          "vrmse_mean": lambda Zp, split: float(both(Zp, split)[1].mean()),
                          "vrmse_paper_medtraj_mean": lambda Zp, split: float(both(Zp, split)[2].mean())}
    for j, nm in enumerate(chan):
        D["extra_metrics"][f"vrmse_paper_{nm}"] = (lambda jj: (lambda Zp, split: float(both(Zp, split)[0][jj])))(j)
        D["extra_metrics"][f"vrmse_{nm}"] = (lambda jj: (lambda Zp, split: float(both(Zp, split)[1][jj])))(j)
        D["extra_metrics"][f"vrmse_paper_medtraj_{nm}"] = (lambda jj: (lambda Zp, split: float(both(Zp, split)[2][jj])))(j)
    pers_rows = lambda i, j: Xfull["te"][i:j]
    prow = vrmse_paper(pers_rows, Yph["te"], C, per_row=True); pvp = prow.mean(0); pv = vrmse_pooled(pers_rows, Yph["te"], C)
    pvm = traj_median(prow, tids["te"]) if tids["te"] is not None else np.full(C, np.nan)
    D.update(pca_evr_in=float(m["evr_in"]), pca_evr_out=float(m["evr_out"]), pca_fit_rows=int(m["nfit"]), target=str(m["target"]),
             persistence_err={"va": float(m["pers_va"]), "te": float(m["pers_te"])}, pca_recon_err={"va": float(m["recon_va"]), "te": float(m["recon_te"])},
             persistence_vrmse={"mean": float(pv.mean()), **{nm: float(pv[j]) for j, nm in enumerate(chan)}},
             persistence_vrmse_paper={"mean": float(pvp.mean()), **{nm: float(pvp[j]) for j, nm in enumerate(chan)}},
             persistence_vrmse_paper_medtraj={"mean": float(pvm.mean()), **{nm: float(pvm[j]) for j, nm in enumerate(chan)}},
             n_traj={"va": int(len(np.unique(tids["va"]))) if tids["va"] is not None else None,
                     "te": int(len(np.unique(tids["te"]))) if tids["te"] is not None else None},
             rank_in=rank_in, rank_out=rank_out, grid=[H, W, C],
             protocol="The Well official train/valid/test files; one-step state_t -> state_{t+1} from ONE input step (the Well "
                      "baselines see 4); vrmse_paper_* is the paper's metric (App. E.3, mean over rows), vrmse_paper_medtraj_* "
                      "Walrus's median over trajectories of the trajectory mean, vrmse_* the pooled convention")
    return D
