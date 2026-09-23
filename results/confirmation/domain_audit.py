"""The theorem's admissible domain, measured on the TRUE table.

Proposition 1 / Theorem 8 need, per sample-band: t = Y2+Y3 > 0 and q = 1 - rho*Y4 > 0
(so that the forward map is defined and the inverse is stable). This script measures
how often the tabulated truth itself violates those hypotheses, and where.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import h5py
import hashlib
import json

D = r"C:/Users/owner/jpl_kernel_dnn_2026_07_16/data/jpl_reg_data"
RHO = 0.7
Ys = {c: np.load(D + f"/{c}.npy") for c in ["Y1", "Y2", "Y3", "Y4"]}
with h5py.File(D + "/data_EMIT_24k.jld2", "r") as f:
    wls = np.array(f["wls"])

t = Ys["Y2"] + Ys["Y3"]
s = Ys["Y4"]
n, B = t.shape
tot = t.size
print(f"table {n} states x {B} bands = {tot:,} sample-band entries")
print()

print("--- albedo Y4 ---")
exact2 = int((s == 2.0).sum())
print(f"  exactly 2.0            : {exact2:,}  ({100*exact2/tot:.5f}%)")
print(f"  >= 1  (q = 1-0.7s <= 0.3 and non-physical) : {int((s >= 1).sum()):,}")
print(f"  < 0                    : {int((s < 0).sum()):,}   min {s.min():.6g}")
print(f"  in [0,1)               : {int(((s >= 0) & (s < 1)).sum()):,}")
q = 1.0 - RHO * s
print(f"  q = 1 - 0.7*s <= 0     : {int((q <= 0).sum()):,}   min q {q.min():.6g}")

print()
print("--- transmitted flux t = Y2 + Y3 ---")
for thr in [0.0, 1e-20, 1e-13, 1e-12, 1e-8, 1e-4]:
    print(f"  t <= {thr:<8g} : {int((t <= thr).sum()):>9,}  ({100*(t <= thr).sum()/tot:.4f}%)")
print(f"  min t {t.min():.6g}")

print()
print("--- joint admissible domain  A = {t > 0 and 0 <= s < 1} ---")
adm = (t > 0) & (s >= 0) & (s < 1)
print(f"  admissible entries     : {int(adm.sum()):,}  ({100*adm.mean():.4f}%)")
print(f"  inadmissible entries   : {int((~adm).sum()):,}  ({100*(~adm).mean():.4f}%)")
rows_all_ok = adm.all(axis=1).sum()
print(f"  states with every band admissible : {int(rows_all_ok):,} of {n:,}")
bad_bands = (~adm).sum(axis=0)
nz = np.nonzero(bad_bands)[0]
print(f"  bands carrying any violation      : {len(nz)} of {B}")
if len(nz):
    order = np.argsort(-bad_bands)
    print("  worst bands (wavelength nm : violating states):")
    for k in order[:12]:
        if bad_bands[k] == 0:
            break
        print(f"     {wls[k]:8.1f} nm : {int(bad_bands[k]):,}")

print()
print("--- a stricter operating domain  A_kappa = {t >= tau and q >= kappa} ---")
for tau, kap in [(1e-12, 0.3), (1e-8, 0.3), (1e-6, 0.3), (1e-4, 0.3)]:
    m = (t >= tau) & (q >= kap)
    print(f"  tau={tau:<8g} kappa={kap}: coverage {100*m.mean():7.4f}%  "
          f"states fully covered {int(m.all(axis=1).sum()):,}")

print()
print("--- round trip on EXACT components (the paper's 0.0054 RMSE diagnostic) ---")
L = Ys["Y1"] + RHO * t / q
den = t + s * (L - Ys["Y1"])
with np.errstate(divide="ignore", invalid="ignore"):
    rho_hat = (L - Ys["Y1"]) / den
nonfinite = int((~np.isfinite(rho_hat)).sum())
rh = np.where(np.isfinite(rho_hat), rho_hat, 0.0)
err_all = rh - RHO
print(f"  non-finite inverses           : {nonfinite:,}")
print(f"  all-band RMSE                 : {np.sqrt((err_all**2).mean()):.6f}")
adm_mask = adm
err_adm = rh[adm_mask] - RHO
print(f"  RMSE on the admissible domain : {np.sqrt((err_adm**2).mean()):.3e}  "
      f"(coverage {100*adm_mask.mean():.4f}%)")
m2 = (t >= 1e-12) & (q >= 0.3)
err2 = rh[m2] - RHO
print(f"  RMSE on t>=1e-12, q>=0.3      : {np.sqrt((err2**2).mean()):.3e}  "
      f"(coverage {100*m2.mean():.4f}%)")

print()
h = hashlib.sha256()
with open(D + "/data_EMIT_24k.jld2", "rb") as fh:
    for chunk in iter(lambda: fh.read(1 << 22), b""):
        h.update(chunk)
print("data_EMIT_24k.jld2 sha256 =", h.hexdigest())
print("bytes =", os.path.getsize(D + "/data_EMIT_24k.jld2"))

out = {
    "entries": int(tot), "states": int(n), "bands": int(B),
    "albedo_exact_2": exact2,
    "albedo_ge_1": int((s >= 1).sum()),
    "albedo_lt_0": int((s < 0).sum()),
    "t_le_0": int((t <= 0).sum()),
    "t_lt_1e_12": int((t < 1e-12).sum()),
    "admissible_frac": float(adm.mean()),
    "states_fully_admissible": int(rows_all_ok),
    "rmse_all_band": float(np.sqrt((err_all ** 2).mean())),
    "rmse_admissible": float(np.sqrt((err_adm ** 2).mean())),
    "jld2_sha256": h.hexdigest(),
    "jld2_bytes": os.path.getsize(D + "/data_EMIT_24k.jld2"),
}
with open("domain_audit.json", "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=2)
print("wrote domain_audit.json")
