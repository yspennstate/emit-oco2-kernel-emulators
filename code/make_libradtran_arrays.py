"""Arrays for the second radiative-transfer code: the libRadtran half of the paired 6S/libRadtran Sentinel-2 corpus.

The corpus (paired_arrays.npz, the release accompanying the physics-guided emulation work cited as [pkan]) holds one row
per state and band: eight numeric inputs (band wavelength, solar and view zenith, relative azimuth, aerosol optical
depth at 550 nm, water-vapour column, ozone column, elevation) and three libRadtran coefficients, the path reflectance
rho_path, the total transmittance T_total and the spherical albedo S. Each state also has an aerosol model and an
atmosphere profile, stored as codes in paired_cats.npz, aligned row by row with paired_arrays.npz. Top-of-atmosphere
reflectance is
    L(rho) = rho_path + rho T_total / (1 - rho S),
which is equation (1) of the paper with Y1 = rho_path, t = Y2 + Y3 = T_total and Y4 = S.

States are kept only when all thirteen bands are present (9,722 of 50,000; band B10 at 1372 nm is missing in the
others). The script writes, in the layout of the EMIT export read by emit_campaign.py:
    X.npy   (states, 7)   the seven numeric state inputs, in the order above without the wavelength;
            (states, 16)  with --cats, followed by the one-hot aerosol model (4) and atmosphere profile (5)
    Y1.npy  (states, 13)  rho_path, bands ordered by wavelength
    Y2.npy  (states, 13)  T_total
    Y3.npy  (states, 13)  1e-6 T_total: the driver fits four components, and a negligible second transmission term
                          lets it run unmodified; the scored transmission is Y2 + Y3 = (1 + 1e-6) T_total
    Y4.npy  (states, 13)  S
and MANIFEST.json with the source digests, the selection, the band order and the digest of every array. The Y arrays
do not depend on --cats.

usage: python code/make_libradtran_arrays.py <paired_arrays.npz> <out_dir> [--cats <paired_cats.npz>]
"""
import hashlib
import json
import os
import sys

import numpy as np

src, out = sys.argv[1], sys.argv[2]
cats_src = sys.argv[sys.argv.index("--cats") + 1] if "--cats" in sys.argv else None
os.makedirs(out, exist_ok=True)
EPS = 1e-6
# the codes in paired_cats.npz index these levels (the loader that wrote the file uses the same tuples)
AERO_LEVELS = ("continental", "desert", "maritime", "urban")
PROF_LEVELS = ("midlatitude_summer", "midlatitude_winter", "subarctic_summer", "subarctic_winter", "tropical")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


z = np.load(src, allow_pickle=True)
X, Yh, sid, band = z["X"], z["Yh"], z["sid"], z["band"].astype(str)
if cats_src:
    c = np.load(cats_src, allow_pickle=True)
    aero, prof = c["aero"], c["prof"]
    if len(aero) != len(sid) or len(prof) != len(sid):
        raise ValueError("paired_cats.npz is not aligned with paired_arrays.npz")
bands = sorted(set(band.tolist()), key=lambda b: float(np.mean(X[band == b, 0])))
wvl = {b: float(np.mean(X[band == b, 0])) for b in bands}
states, counts = np.unique(sid, return_counts=True)
full = states[counts == len(bands)]
col = {b: j for j, b in enumerate(bands)}
row_of = {}
for i, (s, b) in enumerate(zip(sid, band)):
    row_of[(s, b)] = i
n = len(full)
ncat = len(AERO_LEVELS) + len(PROF_LEVELS) if cats_src else 0
Xs = np.empty((n, 7 + ncat))
Y = {k: np.empty((n, len(bands))) for k in ("Y1", "Y2", "Y3", "Y4")}
for k, s in enumerate(full):
    rows = [row_of[(s, b)] for b in bands]
    ins = X[rows, 1:]
    if not np.all(ins == ins[0]):
        raise ValueError(f"state {s}: inputs differ across bands")
    Xs[k, :7] = ins[0]
    if cats_src:
        a, p = aero[rows], prof[rows]
        if not (np.all(a == a[0]) and np.all(p == p[0])):
            raise ValueError(f"state {s}: aerosol model or profile differs across bands")
        Xs[k, 7:] = 0.0
        Xs[k, 7 + int(a[0])] = 1.0
        Xs[k, 7 + len(AERO_LEVELS) + int(p[0])] = 1.0
    Y["Y1"][k] = Yh[rows, 0]
    Y["Y2"][k] = Yh[rows, 1]
    Y["Y3"][k] = EPS * Yh[rows, 1]
    Y["Y4"][k] = Yh[rows, 2]
if not np.isfinite(Xs).all() or not all(np.isfinite(a).all() for a in Y.values()):
    raise ValueError("nonfinite values")
np.save(os.path.join(out, "X.npy"), Xs)
for k, a in Y.items():
    np.save(os.path.join(out, k + ".npy"), a)
inputs = ["sza_deg", "vza_deg", "raa_deg", "aod550", "cwv_cm", "o3_cm", "elev_km"]
man = {"source": os.path.basename(src), "source_sha256": sha256(src), "states_total": int(len(states)),
       "states_kept": int(n), "rule": "states with all thirteen bands", "bands": bands, "wavelength_nm": wvl}
if cats_src:
    inputs += ["aerosol_" + a for a in AERO_LEVELS] + ["profile_" + p for p in PROF_LEVELS]
    man.update(categories_source=os.path.basename(cats_src), categories_sha256=sha256(cats_src),
               categories="one-hot aerosol model and atmosphere profile, constant over the bands of a state")
man.update(inputs=inputs,
           components={"Y1": "rho_path", "Y2": "T_total", "Y3": f"{EPS:g} * T_total", "Y4": "spher_alb"},
           sha256={k: sha256(os.path.join(out, k + ".npy")) for k in ("X", "Y1", "Y2", "Y3", "Y4")})
with open(os.path.join(out, "MANIFEST.json"), "w", encoding="utf-8", newline="\n") as f:
    json.dump(man, f, indent=1)
t = Y["Y2"] + Y["Y3"]
print(json.dumps({"states": n, "inputs": len(inputs), "bands": bands, "t_min": float(t.min()),
                  "t_q01": float(np.quantile(t, 0.01)), "s_max": float(Y["Y4"].max()), "sha256": man["sha256"]}))
