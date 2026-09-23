"""Theorem (error transfer), item (i), applied band by band on one split of the main pipeline.

At every entry of the physical domain the constrained retrieval satisfies |rho_bar - rho| <= min{R, e_R/t}. A
pointwise bound passes to every quantile, so at each band the 95th percentile of the constrained retrieval error is at
most the 95th percentile of min{R, e_R/t} over the same states. This script computes both per band and family and
compares the wall sets, the bands where the 95th percentile exceeds two reflectance points.

Conventions are those of transmission_conditioned.py: a = Y1, t = Y2 + Y3, s = Y4 in physical units; R is the largest
prescribed reflectance, 0.9; S is the largest training albedo not above one; the projections t_hat -> max(t_hat, 0) and
s_hat -> [0, S] are applied before e_R = |e_a| + R|e_t| + R^2 t |e_s|; a nonpositive denominator returns 0 and counts as
a failure. The domain is t > 0 and 0 <= s <= S; the retrieval is evaluated at rho = 0.7.

usage: EMIT_DATA=<dir> [TQ_RES=<dir with preds/>] [TQ_OUT=<dir>] python code/band_transfer_check.py [tag]
       (default tag tq_s101_raw_w512; TQ_RES and TQ_OUT default to results/target_quality)
"""
import json
import os
import sys

import numpy as np

W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAG = sys.argv[1] if len(sys.argv) > 1 else "tq_s101_raw_w512"
RES = os.environ.get("TQ_RES", os.path.join(W, "results", "target_quality"))
OUTDIR = os.environ.get("TQ_OUT", RES)
DATA = os.environ["EMIT_DATA"]
C = ("Y1", "Y2", "Y3", "Y4")
RHO, R, WALL = 0.7, 0.9, 0.02
FAMILIES = ("ridge3", "dnn", "dnn_corr", "ard", "dkr", "stack")

pred = np.load(os.path.join(RES, "preds", TAG + ".npz"), allow_pickle=False)
te, tr = pred["idx_te"], (pred["idx_tr"] if "idx_tr" in pred.files else None)
full = {c: np.load(os.path.join(DATA, c + ".npy"), allow_pickle=False).astype(float) for c in C}
Y = {c: full[c][te] for c in C}
a, t, s = Y["Y1"], Y["Y2"] + Y["Y3"], Y["Y4"]
if tr is None:
    raise SystemExit("predictions carry no idx_tr; S must come from the training rows")
s_tr = full["Y4"][tr]
S = float(s_tr[np.isfinite(s_tr) & (s_tr <= 1.0)].max())
assert R * S < 1, (R, S)
q = 1.0 - RHO * s
L = a + RHO * t / q
DOM = (t > 0) & (s >= 0) & (s <= S)


def p95_by_band(x, mask):
    A = np.where(mask, x, np.nan)
    return np.nanquantile(A, 0.95, axis=0)


out = {"tag": TAG, "rho": RHO, "R": R, "S": S, "domain_fraction": float(DOM.mean()), "families": {}}
walls_actual, walls_bound = {}, {}
for k in FAMILIES:
    if f"{k}_Y1" not in pred.files:
        continue
    P = {c: pred[f"{k}_{c}"].astype(float) for c in C}
    ah, th, sh = P["Y1"], np.maximum(P["Y2"] + P["Y3"], 0.0), np.clip(P["Y4"], 0.0, S)
    u = L - ah
    den = th + sh * u
    with np.errstate(divide="ignore", invalid="ignore"):
        rho_bar = np.where(den > 0, np.clip(u / den, 0.0, R), 0.0)
        eR = np.abs(ah - a) + R * np.abs(th - t) + R * R * t * np.abs(sh - s)
        bound = np.minimum(R, eR / t)
    err = np.abs(rho_bar - RHO)
    viol = int(np.sum((err > bound * (1 + 1e-9) + 1e-12) & DOM))
    pa, pb = p95_by_band(err, DOM), p95_by_band(bound, DOM)
    wa, wb = set(np.nonzero(pa > WALL)[0].tolist()), set(np.nonzero(pb > WALL)[0].tolist())
    walls_actual[k], walls_bound[k] = wa, wb
    ratio = pb / np.maximum(pa, 1e-15)
    out["families"][k] = dict(pointwise_violations_on_domain=viol, walls_actual=len(wa), walls_bound=len(wb),
                              bound_walls_not_actual=len(wb - wa), actual_walls_not_bound=len(wa - wb),
                              p95_ratio_bound_over_actual_median=float(np.median(ratio)),
                              p95_ratio_on_actual_walls_median=float(np.median(ratio[sorted(wa)])) if wa else None,
                              p95_ratio_on_actual_walls_max=float(np.max(ratio[sorted(wa)])) if wa else None,
                              failures_on_domain=int(np.sum((den <= 0) & DOM)))
core_a = set.intersection(*walls_actual.values())
core_b = set.intersection(*walls_bound.values())
out.update(core_actual=len(core_a), core_bound=len(core_b), core_bound_not_actual=len(core_b - core_a),
           core_actual_not_bound=len(core_a - core_b))
with open(os.path.join(OUTDIR, f"band_transfer_{TAG}.json"), "w", encoding="utf-8", newline="\n") as f:
    json.dump(out, f, indent=1)
print(json.dumps(out, indent=1))
