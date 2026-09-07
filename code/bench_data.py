"""Loaders for the new corpora on the DGX (~/data_new), one interface for bench_run.py.

Each loader returns a dict: Xtr, Xva, Xte (standardized inputs, float64), Ztr, Zva, Zte (the targets the
models regress: standardized outputs, or PCA coefficients of fields), err(Zpred, split) -> the corpus's
error in its natural units (mean relative L2 per sample on the reconstructed physical targets), extra
metrics where the source paper reports others, names of the inputs, and for the multi-fidelity corpus
the low-fidelity prediction that can serve as a mean.

  climsim(seed, ntrain)          LEAP ClimSim subsampled low-res: 124 inputs -> 128 tendencies; train rows
                                 subsampled from the 10.1M, validation from val_*, test from scoring_*
  pkanrtm(seed, ntrain, lowfi)   paired 6S / libRadtran Sentinel-2 coefficients: 9 state inputs + band
                                 wavelength -> (rho_path, T_total, spher_alb) of libRadtran; lowfi=1 adds the
                                 6S coefficients of the same state as inputs (physics-mean / multi-fidelity)
  trl2d(seed, ntrain)            The Well turbulent radiative layer 2D: one-step map state_t -> state_{t+1}
                                 on four fields (128 x 384), PCA in and out (fitted on training rows)
  advection(seed, ntrain)        Caltech operator suite Advection (200-point input function -> 200-point output)
  darcy(beta, seed, ntrain)      PDEBench 2D Darcy (128 x 128 coefficient -> solution), PCA in and out
Environment: DATA_NEW (default ~/data_new).
"""
import json, os, pathlib
import numpy as np

DN = pathlib.Path(os.environ.get("DATA_NEW", os.path.expanduser("~/data_new")))


def rel_l2(Yt, Yp):
    return float(np.mean(np.linalg.norm(Yt - Yp, axis=1) / np.maximum(np.linalg.norm(Yt, axis=1), 1e-30)))


class Std:
    def __init__(self, A):
        self.m = A.mean(0); self.s = A.std(0); self.s[self.s == 0] = 1.0

    def fwd(self, A):
        return (A - self.m) / self.s

    def inv(self, A):
        return A * self.s + self.m


class PCA:
    def __init__(self, Y, rank, nfit=4000, seed=0):
        idx = np.random.default_rng(seed).permutation(len(Y))[:min(nfit, len(Y))]
        self.c = Y[idx].mean(0)
        _, S, Vt = np.linalg.svd(Y[idx] - self.c, full_matrices=False)
        self.Vt = Vt[:rank]; self.evr = float((S[:rank] ** 2).sum() / (S ** 2).sum())

    def fwd(self, Y):
        return (Y - self.c) @ self.Vt.T

    def inv(self, Z):
        return Z @ self.Vt + self.c


def _pack(problem, tag, Xs, Zs, phys_of_Z, Yph, names, extra=None):
    xs = Std(Xs["tr"])
    D = dict(problem=problem, tag=tag, Xtr=xs.fwd(Xs["tr"]), Xva=xs.fwd(Xs["va"]), Xte=xs.fwd(Xs["te"]),
             Ztr=Zs["tr"], Zva=Zs["va"], Zte=Zs["te"], names=names, to_phys=phys_of_Z)
    D["err"] = lambda Zp, split: rel_l2(Yph[split], phys_of_Z(Zp))
    # the two pieces of err, row-sliceable, for combiners that build a per-sample residual Gram once instead of
    # calling err hundreds of times: phys_pred(Zp_rows, split, rows) and den(split, rows) = the error denominators
    D["phys_pred"] = lambda Zp, split, rows=None: phys_of_Z(Zp)
    D["den"] = lambda split, rows=None: np.maximum(np.linalg.norm(Yph[split] if rows is None else Yph[split][rows], axis=1), 1e-30)
    D["Yobj"] = Yph["tr"] - Yph["tr"].mean(0); D["ynorm2"] = np.maximum((Yph["tr"] ** 2).sum(1), 1e-30)
    D["Yph"] = Yph
    if extra:
        D.update(extra)
    return D


def climsim(seed=0, ntrain=100000, nval=20000, ntest=20000):
    root = DN / "climsim"
    rng = np.random.default_rng(seed)
    Xall = np.load(root / "train_input.npy", mmap_mode="r"); Yall = np.load(root / "train_target.npy", mmap_mode="r")
    itr = np.sort(rng.choice(Xall.shape[0], ntrain, replace=False))
    Xtr, Ytr = np.asarray(Xall[itr], np.float64), np.asarray(Yall[itr], np.float64)
    Xv = np.load(root / "val_input.npy", mmap_mode="r"); Yv = np.load(root / "val_target.npy", mmap_mode="r")
    iva = np.sort(rng.choice(Xv.shape[0], nval, replace=False))
    Xva, Yva = np.asarray(Xv[iva], np.float64), np.asarray(Yv[iva], np.float64)
    Xs_ = np.load(root / "scoring_input.npy", mmap_mode="r"); Ys_ = np.load(root / "scoring_target.npy", mmap_mode="r")
    ite = np.sort(rng.choice(Xs_.shape[0], ntest, replace=False))
    Xte, Yte = np.asarray(Xs_[ite], np.float64), np.asarray(Ys_[ite], np.float64)
    ys = Std(Ytr)                       # the targets are regressed standardized per output
    Zs = {"tr": ys.fwd(Ytr), "va": ys.fwd(Yva), "te": ys.fwd(Yte)}
    Yph = {"tr": Ytr, "va": Yva, "te": Yte}
    D = _pack("climsim", f"climsim_s{seed}_n{ntrain}", {"tr": Xtr, "va": Xva, "te": Xte}, Zs, ys.inv, Yph, [f"in{j}" for j in range(124)])
    # ClimSim reports per-variable R2 and MAE; the standardized targets make the mean R2 over outputs meaningful
    def _r2(Zp, split):
        Yt = Yph[split]; Yp = ys.inv(Zp)
        ss = ((Yt - Yp) ** 2).sum(0); st = ((Yt - Yt.mean(0)) ** 2).sum(0)
        ok = st > 0
        return 1 - ss[ok] / st[ok]
    # a few of the 128 targets are nearly constant on the scoring set; their R2 is meaningless and hugely negative,
    # so the mean is reported with each target's R2 clipped at zero (the convention of the public ClimSim scoring),
    # together with the median over targets.
    D["extra_metrics"] = {"r2_mean_clip0": lambda Zp, split: float(np.mean(np.maximum(_r2(Zp, split), 0.0))),
                          "r2_median": lambda Zp, split: float(np.median(_r2(Zp, split))),
                          "mae_std": lambda Zp, split: float(np.abs(Zp - Zs[split]).mean())}
    return D


def _read_jsonl(path, keys):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            rows.append([d.get(k) for k in keys])
    return rows


def pkanrtm(seed=0, ntrain=0, lowfi=0, cache=True):
    root = DN / "pkanrtm"; npz = root / "paired_arrays.npz"
    keys_in = ["wvl_nm", "sza_deg", "vza_deg", "raa_deg", "aod550", "cwv_cm", "o3_cm", "elev_km"]
    keys_out = ["rho_path", "T_total", "spher_alb"]
    if npz.exists() and cache:
        z = np.load(npz, allow_pickle=True)
        X, Yh, Yl, sid, band = z["X"], z["Yh"], z["Yl"], z["sid"], z["band"]
    else:
        hi = _read_jsonl(root / "dataset_rows_libradtran.jsonl", ["state_id", "band"] + keys_in + keys_out)
        lo = _read_jsonl(root / "dataset_rows_6s.jsonl", ["state_id", "band"] + keys_out)
        lo_map = {(r[0], r[1]): r[2:] for r in lo}
        X, Yh, Yl, sid, band = [], [], [], [], []
        for r in hi:
            k = (r[0], r[1])
            if k not in lo_map or any(v is None for v in r[2:]) or any(v is None for v in lo_map[k]):
                continue
            X.append(r[2:2 + len(keys_in)]); Yh.append(r[2 + len(keys_in):]); Yl.append(lo_map[k]); sid.append(r[0]); band.append(r[1])
        X, Yh, Yl = np.array(X, np.float64), np.array(Yh, np.float64), np.array(Yl, np.float64)
        sid, band = np.array(sid), np.array(band)
        np.savez_compressed(npz, X=X, Yh=Yh, Yl=Yl, sid=sid, band=band)
    # split by STATE (all bands of a state travel together), 80/10/10 by seed
    states = np.unique(sid); rng = np.random.default_rng(seed); perm = rng.permutation(len(states))
    n_te = len(states) // 10; te_states = set(states[perm[:n_te]]); va_states = set(states[perm[n_te:2 * n_te]])
    is_te = np.array([s in te_states for s in sid]); is_va = np.array([s in va_states for s in sid]); is_tr = ~(is_te | is_va)
    if ntrain and ntrain < is_tr.sum():
        tr_idx = np.where(is_tr)[0]; keep = np.sort(rng.choice(tr_idx, ntrain, replace=False)); is_tr = np.zeros_like(is_tr); is_tr[keep] = True
    Xin = np.concatenate([X, Yl], 1) if lowfi else X
    names = keys_in + (["6s_" + k for k in keys_out] if lowfi else [])
    Xs = {"tr": Xin[is_tr], "va": Xin[is_va], "te": Xin[is_te]}
    Yph = {"tr": Yh[is_tr], "va": Yh[is_va], "te": Yh[is_te]}
    ys = Std(Yph["tr"])
    Zs = {k: ys.fwd(v) for k, v in Yph.items()}
    D = _pack("pkanrtm", f"pkanrtm_s{seed}" + ("_lowfi" if lowfi else ""), Xs, Zs, ys.inv, Yph, names,
              extra=dict(lowfi_pred={"tr": Yl[is_tr], "va": Yl[is_va], "te": Yl[is_te]}))
    # The water-vapour bands (B9, B10) have coefficients of order 1e-3 or below: a per-sample relative error
    # explodes there (the 6S pair itself reads 83% mean / 11% median). The reported error is therefore relative
    # with an absolute floor of 0.05 on the coefficient norm; the same floor enters the kernel-flow objective.
    FLOOR = 0.05
    D["err"] = lambda Zp, split: float((np.linalg.norm(Yph[split] - ys.inv(Zp), axis=1) / np.maximum(np.linalg.norm(Yph[split], axis=1), FLOOR)).mean())
    D["phys_pred"] = lambda Zp, split, rows=None: ys.inv(Zp)
    D["den"] = lambda split, rows=None: np.maximum(np.linalg.norm(Yph[split] if rows is None else Yph[split][rows], axis=1), FLOOR)
    D["ynorm2"] = np.maximum((Yph["tr"] ** 2).sum(1), FLOOR ** 2)
    D["err_floor"] = FLOOR; D["frac_below_floor_te"] = float((np.linalg.norm(Yph["te"], axis=1) < FLOOR).mean())
    def rmse(Zp, split):
        return float(np.sqrt(((ys.inv(Zp) - Yph[split]) ** 2).mean()))
    def smape(Zp, split):
        P, T = ys.inv(Zp), Yph[split]
        return float(100 * np.mean(2 * np.abs(P - T) / np.maximum(np.abs(P) + np.abs(T), 1e-12)))
    def nrmse_mean(Zp, split):
        P, T = ys.inv(Zp), Yph[split]
        return float(np.mean(np.sqrt(((P - T) ** 2).mean(0)) / T.std(0)))
    D["extra_metrics"] = {"rmse": rmse, "smape_pct": smape, "nrmse_mean": nrmse_mean}
    D["extra_metrics_lowfi"] = {"rmse_6s": float(np.sqrt(((Yl[is_te] - Yh[is_te]) ** 2).mean())),
                                "rel_floor_6s": float((np.linalg.norm(Yh[is_te] - Yl[is_te], axis=1) / np.maximum(np.linalg.norm(Yh[is_te], axis=1), FLOOR)).mean())}
    return D


def _trl2d_build_cache(root, cdir, key, rank_in, rank_out, stride, target, nfit=4000):
    """One pass over the 27 HDF5 files: per-field standardization (statistics from the training rows), PCA of the
    flattened 4 x 128 x 384 state fitted on nfit random training rows (inputs), PCA of the TARGET fitted the same way
    (target = the next state, or the increment x_{t+1} - x_t, both in standardized units), scores for every row, the
    standardized validation/test states x_t and x_{t+1} kept in full (float32) for the error, per-row training norms
    of the target for the kernel-flow objective."""
    import h5py
    F = 128 * 384

    def load_split(split):
        Xl, Yl = [], []
        for f in sorted(root.glob(f"{split}_tcool_*.hdf5")):
            with h5py.File(f, "r") as h:
                den = h["t0_fields/density"][:]; pre = h["t0_fields/pressure"][:]; vel = h["t1_fields/velocity"][:]
            st = np.concatenate([den[..., None], pre[..., None], vel], -1).astype(np.float32)   # (traj, T, 128, 384, 4)
            del den, pre, vel
            traj, T = st.shape[:2]; ts = np.arange(0, T - 1, stride)
            for tr_ in range(traj):
                Xl.append(st[tr_, ts].reshape(len(ts), F * 4)); Yl.append(st[tr_, ts + 1].reshape(len(ts), F * 4))
        return np.concatenate(Xl), np.concatenate(Yl)

    Xtr, Ytr = load_split("train"); Xva, Yva = load_split("valid"); Xte, Yte = load_split("test")
    mu = Xtr.reshape(len(Xtr), F, 4).mean((0, 1), dtype=np.float64); sd = Xtr.reshape(len(Xtr), F, 4).std((0, 1), dtype=np.float64) + 1e-8

    def stdz(A):                                   # in place, float32
        B = A.reshape(len(A), F, 4); B -= mu; B /= sd
        return A

    for A in (Xtr, Ytr, Xva, Yva, Xte, Yte):
        stdz(A)
    if target == "increment":                      # the regression target is the change of state; the error stays on the state
        Ttr, Tva, Tte = Ytr - Xtr, Yva - Xva, Yte - Xte
    else:
        Ttr, Tva, Tte = Ytr, Yva, Yte
    rng = np.random.default_rng(0)
    idx = np.sort(rng.permutation(len(Xtr))[:min(nfit, len(Xtr))])

    def fit(A, rank):
        c = A[idx].mean(0, dtype=np.float64)
        _, S, Vt = np.linalg.svd(A[idx].astype(np.float64) - c, full_matrices=False)
        return c, Vt[:rank].copy(), float((S[:rank] ** 2).sum() / (S ** 2).sum())

    c_in, Vt_in, evr_in = fit(Xtr, rank_in); c_out, Vt_out, evr_out = fit(Ttr, rank_out)

    def scores(A, c, Vt, chunk=400):
        return np.concatenate([(A[i:i + chunk].astype(np.float64) - c) @ Vt.T for i in range(0, len(A), chunk)])

    out = {}
    for nm, A in (("Xtr", Xtr), ("Xva", Xva), ("Xte", Xte)):
        out[nm] = scores(A, c_in, Vt_in)
    for nm, A in (("Ztr", Ttr), ("Zva", Tva), ("Zte", Tte)):
        out[nm] = scores(A, c_out, Vt_out)
    tnorm2_tr = np.concatenate([(Ttr[i:i + 400].astype(np.float64) ** 2).sum(1) for i in range(0, len(Ttr), 400)])
    inv = lambda Z: Z @ Vt_out + c_out

    def state_pred(Zs, Xs_):                       # predicted next state in standardized units
        return inv(Zs) + (Xs_.astype(np.float64) if target == "increment" else 0.0)
    pers = {k: rel_l2(Y_.astype(np.float64), X_.astype(np.float64)) for k, X_, Y_ in (("va", Xva, Yva), ("te", Xte, Yte))}
    recon = {k: rel_l2(Y_.astype(np.float64), state_pred(out["Z" + k], X_)) for k, X_, Y_ in (("va", Xva, Yva), ("te", Xte, Yte))}
    for nm, A in out.items():
        np.save(cdir / f"{key}_{nm}.npy", A)
    np.save(cdir / f"{key}_Xva_s.npy", Xva); np.save(cdir / f"{key}_Xte_s.npy", Xte)
    np.save(cdir / f"{key}_Yva_s.npy", Yva); np.save(cdir / f"{key}_Yte_s.npy", Yte)
    np.savez(cdir / f"{key}_meta.npz", mu=mu, sd=sd, c_in=c_in, Vt_in=Vt_in, c_out=c_out, Vt_out=Vt_out, evr_in=evr_in, evr_out=evr_out,
             tnorm2_tr=tnorm2_tr, pers_va=pers["va"], pers_te=pers["te"], recon_va=recon["va"], recon_te=recon["te"], nfit=len(idx),
             n_tr=len(Xtr), n_va=len(Xva), n_te=len(Xte), target=target)


def trl2d(seed=0, ntrain=0, rank_in=256, rank_out=256, stride=1, target="increment"):
    """One-step map on The Well's turbulent radiative layer: x = state at t (density, pressure, two velocity
    components on 128 x 384), y = state at t+1. Training pairs from the nine tcool training files, validation and
    test from the valid/test files. The representation (per-field standardization, PCA fitted on 4000 training
    rows) is built once and cached under the_well/trl2d/cache (a lock keeps concurrent lanes from building it
    twice). The regression target is the increment x_{t+1} - x_t (target="state" regresses the next state); the
    reported error is the relative L2 error of the predicted standardized state against the full field, with the
    persistence baseline (x_t for x_{t+1}) and the PCA reconstruction floor recorded beside it, and the per-field
    VRMSE of The Well (RMSE over the field divided by its standard deviation) as extra metrics."""
    import fcntl
    root = DN / "the_well" / "trl2d"; cdir = root / "cache"; cdir.mkdir(exist_ok=True)
    key = f"r{rank_in}x{rank_out}_st{stride}_{target}"
    if not (cdir / f"{key}_meta.npz").exists():
        with open(cdir / f"{key}.lock", "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            if not (cdir / f"{key}_meta.npz").exists():
                _trl2d_build_cache(root, cdir, key, rank_in, rank_out, stride, target)
    m = np.load(cdir / f"{key}_meta.npz")
    Vt_out, c_out = m["Vt_out"], m["c_out"]
    Xs = {k: np.load(cdir / f"{key}_X{k}.npy") for k in ("tr", "va", "te")}
    Zs = {k: np.load(cdir / f"{key}_Z{k}.npy") for k in ("tr", "va", "te")}
    Xfull = {"va": np.load(cdir / f"{key}_Xva_s.npy", mmap_mode="r"), "te": np.load(cdir / f"{key}_Xte_s.npy", mmap_mode="r")}
    Yph = {"va": np.load(cdir / f"{key}_Yva_s.npy", mmap_mode="r"), "te": np.load(cdir / f"{key}_Yte_s.npy", mmap_mode="r")}
    tnorm2 = m["tnorm2_tr"]
    if ntrain and ntrain < len(Xs["tr"]):
        keep = np.sort(np.random.default_rng(seed).choice(len(Xs["tr"]), ntrain, replace=False))
        Xs["tr"], Zs["tr"], tnorm2 = Xs["tr"][keep], Zs["tr"][keep], tnorm2[keep]
    incr = str(m["target"]) == "increment"
    inv = lambda Z: Z @ Vt_out + c_out

    def state_of(Zp, split):
        return inv(Zp) + (np.asarray(Xfull[split], np.float64) if incr else 0.0)
    xs = Std(Xs["tr"])
    D = dict(problem="trl2d", tag=f"trl2d_s{seed}" + (f"_n{ntrain}" if ntrain else ""), Xtr=xs.fwd(Xs["tr"]), Xva=xs.fwd(Xs["va"]), Xte=xs.fwd(Xs["te"]),
             Ztr=Zs["tr"], Zva=Zs["va"], Zte=Zs["te"], names=[f"pc{j}" for j in range(rank_in)], to_phys=inv)
    D["err"] = lambda Zp, split: rel_l2(np.asarray(Yph[split], np.float64), state_of(Zp, split))
    D["phys_pred"] = lambda Zp, split, rows=None: inv(Zp) + (np.asarray(Xfull[split] if rows is None else Xfull[split][rows], np.float64) if incr else 0.0)
    D["den"] = lambda split, rows=None: np.maximum(np.linalg.norm(np.asarray(Yph[split] if rows is None else Yph[split][rows], np.float64), axis=1), 1e-30)
    D["Yobj"] = Zs["tr"] - Zs["tr"].mean(0); D["ynorm2"] = np.maximum(tnorm2, 1e-30)     # objective in score space, norms of the full target
    D["Yph"] = Yph

    def vrmse_fields(Zp, split):                  # The Well's VRMSE per field: RMSE over the field / std of the field (test set)
        P = state_of(Zp, split).reshape(-1, 128 * 384, 4); T = np.asarray(Yph[split], np.float64).reshape(-1, 128 * 384, 4)
        num = np.sqrt(((P - T) ** 2).mean((0, 1))); den = T.std((0, 1))
        return num / den
    D["extra_metrics"] = {"vrmse_mean": lambda Zp, split: float(vrmse_fields(Zp, split).mean()),
                          "vrmse_density": lambda Zp, split: float(vrmse_fields(Zp, split)[0]),
                          "vrmse_pressure": lambda Zp, split: float(vrmse_fields(Zp, split)[1]),
                          "vrmse_vx": lambda Zp, split: float(vrmse_fields(Zp, split)[2]),
                          "vrmse_vy": lambda Zp, split: float(vrmse_fields(Zp, split)[3])}
    # the persistence baseline in The Well's own metric, per field, on the test block
    Pp = np.asarray(Xfull["te"], np.float64).reshape(-1, 128 * 384, 4); Tt = np.asarray(Yph["te"], np.float64).reshape(-1, 128 * 384, 4)
    pers_vrmse = np.sqrt(((Pp - Tt) ** 2).mean((0, 1))) / Tt.std((0, 1)); del Pp, Tt
    D.update(pca_evr_in=float(m["evr_in"]), pca_evr_out=float(m["evr_out"]), pca_fit_rows=int(m["nfit"]), target=str(m["target"]),
             persistence_err={"va": float(m["pers_va"]), "te": float(m["pers_te"])}, pca_recon_err={"va": float(m["recon_va"]), "te": float(m["recon_te"])},
             persistence_vrmse={"mean": float(pers_vrmse.mean()), "density": float(pers_vrmse[0]), "pressure": float(pers_vrmse[1]), "vx": float(pers_vrmse[2]), "vy": float(pers_vrmse[3])},
             rank_in=rank_in, rank_out=rank_out)
    return D


def burgers(nu=0.1, seed=0, ntrain=8000, ntest=1000, t_index=100, stride=4, rank_in=0, rank_out=128):
    """PDEBench 1D Burgers (10,000 trajectories, 201 steps on [0, 2], 1024 cells): the operator from the initial
    condition to the solution at t = 1 (step 100), the map Batlle et al. and PDEBench both benchmark. The grid is
    subsampled by `stride` (1024 -> 256) for both input and output; the output is regressed through PCA of rank
    rank_out (the reconstruction floor is recorded), the input is used raw (rank_in = 0) or through PCA."""
    import h5py
    f = DN / "pdebench" / f"1D_Burgers_Sols_Nu{nu}.hdf5"
    with h5py.File(f, "r") as h:
        U = h["tensor"]                                    # (N, T, X)
        N = U.shape[0]
        rng = np.random.default_rng(seed); perm = rng.permutation(N)
        te, tr = np.sort(perm[:ntest]), np.sort(perm[ntest:ntest + ntrain])
        X0 = U[:, 0, ::stride].astype(np.float64); Y1 = U[:, t_index, ::stride].astype(np.float64)
    nva = max(500, ntrain // 10)
    tr, va = tr[:-nva], tr[-nva:]
    Xs = {"tr": X0[tr], "va": X0[va], "te": X0[te]}
    Yph = {"tr": Y1[tr], "va": Y1[va], "te": Y1[te]}
    if rank_in:
        pin = PCA(Xs["tr"], rank_in, seed=seed); Xs = {k: pin.fwd(v) for k, v in Xs.items()}
    pout = PCA(Yph["tr"], rank_out, seed=seed)
    Zs = {k: pout.fwd(v) for k, v in Yph.items()}
    D = _pack("burgers", f"burgers_nu{nu}_s{seed}" + (f"_n{ntrain}" if ntrain != 8000 else ""), Xs, Zs, pout.inv, Yph,
              [f"x{j}" for j in range(Xs["tr"].shape[1])],
              extra=dict(pca_evr_out=pout.evr, pca_recon_err={k: rel_l2(Yph[k], pout.inv(pout.fwd(Yph[k]))) for k in ("va", "te")},
                         nu=nu, t_index=t_index, stride=stride, idx=dict(tr=tr, va=va, te=te), to_z=pout.fwd, grid=(X0.shape[1],)))
    return D


def advection(seed=0, ntrain=20000, ntest=2000, rank_in=0):
    root = DN / "caltech_suite"
    X = np.load(root / "Advection_inputs.npy").T.astype(np.float64); Y = np.load(root / "Advection_outputs.npy").T.astype(np.float64)
    rng = np.random.default_rng(seed); perm = rng.permutation(len(X))
    tr, va, te = perm[:ntrain - 1000], perm[ntrain - 1000:ntrain], perm[ntrain:ntrain + ntest]
    Xs = {"tr": X[tr], "va": X[va], "te": X[te]}
    Yph = {"tr": Y[tr], "va": Y[va], "te": Y[te]}
    Zs = dict(Yph); to_phys = lambda Z: Z
    D = _pack("advection", f"advection_s{seed}", Xs, Zs, to_phys, Yph, [f"x{j}" for j in range(X.shape[1])])
    return D


def darcy(beta=1.0, seed=0, ntrain=8000, ntest=1000, rank_in=200, rank_out=100):
    import h5py
    f = DN / "pdebench" / f"2D_DarcyFlow_beta{beta}_Train.hdf5"
    with h5py.File(f, "r") as h:
        A = h["nu"][:].astype(np.float32); U = h["tensor"][:].astype(np.float32)
    A = A.reshape(len(A), -1); U = U.reshape(len(U), -1)
    rng = np.random.default_rng(seed); perm = rng.permutation(len(A))
    tr, va, te = perm[:ntrain - 1000], perm[ntrain - 1000:ntrain], perm[ntrain:ntrain + ntest]
    xs = Std(A[tr].astype(np.float64)); pin = PCA(xs.fwd(A[tr].astype(np.float64)), rank_in, seed=seed)
    pout = PCA(U[tr].astype(np.float64), rank_out, seed=seed)
    Xs = {k: pin.fwd(xs.fwd(A[i].astype(np.float64))) for k, i in (("tr", tr), ("va", va), ("te", te))}
    Yph = {k: U[i].astype(np.float64) for k, i in (("tr", tr), ("va", va), ("te", te))}
    Zs = {k: pout.fwd(Yph[k]) for k in Yph}
    D = _pack("darcy", f"darcy_b{beta}_s{seed}", Xs, Zs, pout.inv, Yph, [f"pc{j}" for j in range(rank_in)],
              extra=dict(pca_evr_in=pin.evr, pca_evr_out=pout.evr, idx=dict(tr=tr, va=va, te=te), to_z=pout.fwd, grid=(128, 128)))
    return D
