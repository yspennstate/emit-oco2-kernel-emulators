"""Transmission statistics of the libRadtran arrays, band by band.

Reads Y2.npy and Y3.npy of the arrays written by make_libradtran_arrays.py (transmission t = Y2 + Y3) and writes
results/libradtran/transmission_by_band.json: for each band the minimum, first percentile, median and the fraction of
states below 1e-6, and over the twelve bands other than B10 the smallest transmission and first percentile.

usage: EMIT_DATA=<array dir> python code/lrt_transmission_stats.py
"""
import json
import os

import numpy as np

W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.environ["EMIT_DATA"]
man = json.load(open(os.path.join(D, "MANIFEST.json"), encoding="utf-8"))
bands = man["bands"]
t = np.load(os.path.join(D, "Y2.npy")) + np.load(os.path.join(D, "Y3.npy"))
per = {}
for j, b in enumerate(bands):
    x = t[:, j]
    per[b] = {"min": float(x.min()), "q01": float(np.quantile(x, 0.01)), "median": float(np.median(x)),
              "frac_below_1e-6": float(np.mean(x < 1e-6))}
rest = [j for j, b in enumerate(bands) if b != "B10"]
out = {"arrays_sha256": man["sha256"], "states": int(t.shape[0]), "bands": per,
       "without_B10": {"min": float(t[:, rest].min()), "min_q01": float(min(per[bands[j]]["q01"] for j in rest)),
                       "max": float(t[:, rest].max())}}
with open(os.path.join(W, "results", "libradtran", "transmission_by_band.json"), "w", encoding="utf-8",
          newline="\n") as f:
    json.dump(out, f, indent=1)
print(json.dumps({"B10": per["B10"], "without_B10": out["without_B10"]}))
