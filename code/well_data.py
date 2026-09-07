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
import fcntl, glob, os, pathlib
import numpy as np

DN = pathlib.Path(os.environ.get("DATA_NEW", os.path.expanduser("~/data_new")))
FIELDS = {
    "active_matter": [("t0_fields", "concentration"), ("t1_fields", "velocity"), ("t2_fields", "D"), ("t2_fields", "E")],
    "helmholtz_staircase": [("t0_fields", "pressure_re"), ("t0_fields", "pressure_im")],
    "turbulent_radiative_layer_2D": [("t0_fields", "density"), ("t0_fields", "pressure"), ("t1_fields", "velocity")],
}


def rel_l2(Yt, Yp):
    return float(np.mean(np.linalg.norm(Yt - Yp, axis=1) / np.maximum(np.linalg.norm(Yt, axis=1), 1e-30)))


class Std:
    def __init__(self, A):
        self.m = A.mean(0); self.s = A.std(0); self.s[self.s == 0] = 1.0

    def fwd(self, A):
        return (A - self.m) / self.s


def _files(root, split):
    fs = sorted(glob.glob(str(root / "data" / split / "*.hdf5"))) or sorted(glob.glob(str(root / f"{split}_*.hdf5")))
    return fs


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
    splits = {s: _files(root, s) for s in ("train", "valid", "test")}
    if not splits["train"] or not splits["test"]:
        raise SystemExit(f"{name}: train/test files missing under {root}")
    if not splits["valid"]:                                    # some releases carry no validation split: carve it from train
        k = max(1, len(splits["train"]) // 10); splits["valid"] = splits["train"][-k:]; splits["train"] = splits["train"][:-k]
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
    D["err"] = lambda Zp, split: rel_l2(np.asarray(Yph[split], np.float64), state_of(Zp, split))
    D["phys_pred"] = lambda Zp, split, rows=None: state_of(Zp, split, rows)
    D["den"] = lambda split, rows=None: np.maximum(np.linalg.norm(np.asarray(Yph[split] if rows is None else Yph[split][rows], np.float64), axis=1), 1e-30)
    D["Yobj"] = Zs["tr"] - Zs["tr"].mean(0); D["ynorm2"] = np.maximum(tnorm2, 1e-30)
    D["Yph"] = Yph
    chan = [str(x) for x in m["channel_names"]]

    def vrmse(Zp, split):
        P = state_of(Zp, split).reshape(-1, H * W, C); T = np.asarray(Yph[split], np.float64).reshape(-1, H * W, C)
        return np.sqrt(((P - T) ** 2).mean((0, 1))) / T.std((0, 1))
    D["extra_metrics"] = {"vrmse_mean": lambda Zp, split: float(vrmse(Zp, split).mean())}
    for j, nm in enumerate(chan):
        D["extra_metrics"][f"vrmse_{nm}"] = (lambda jj: (lambda Zp, split: float(vrmse(Zp, split)[jj])))(j)
    Pp = np.asarray(Xfull["te"], np.float64).reshape(-1, H * W, C); Tt = np.asarray(Yph["te"], np.float64).reshape(-1, H * W, C)
    pv = np.sqrt(((Pp - Tt) ** 2).mean((0, 1))) / Tt.std((0, 1)); del Pp, Tt
    D.update(pca_evr_in=float(m["evr_in"]), pca_evr_out=float(m["evr_out"]), pca_fit_rows=int(m["nfit"]), target=str(m["target"]),
             persistence_err={"va": float(m["pers_va"]), "te": float(m["pers_te"])}, pca_recon_err={"va": float(m["recon_va"]), "te": float(m["recon_te"])},
             persistence_vrmse={"mean": float(pv.mean()), **{nm: float(pv[j]) for j, nm in enumerate(chan)}}, rank_in=rank_in, rank_out=rank_out,
             grid=[H, W, C])
    return D
