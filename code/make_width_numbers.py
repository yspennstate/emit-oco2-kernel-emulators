"""Rebuild results/scaling/width_numbers.json from the per-split records.

Cells per width: network radiance error, network median reflectance error, feature-kernel radiance error and
feature-kernel median reflectance error, each as mean and sample standard deviation over the available splits, in
percent (radiance) and percentage points (reflectance).
"""
import glob
import json
import os
import statistics as st

W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCES = {
    "512": os.path.join(W, "results", "emit", "emit_s1[0-9][0-9].json"),
    "1000": os.path.join(W, "results", "scaling", "per_seed", "emit_s1[0-9][0-9]_w1000.json"),
    "2000": os.path.join(W, "results", "scaling", "per_seed", "emit_s1[0-9][0-9]_wide.json"),
    "4000": os.path.join(W, "results", "scaling", "per_seed", "emit_s1[0-9][0-9]_w4000.json"),
}


def cell(values):
    return f"{st.mean(values):.3f}$\\pm${st.stdev(values):.3f}"


out = {}
for width, pattern in SOURCES.items():
    recs = [json.load(open(p, encoding="utf-8")) for p in sorted(glob.glob(pattern))]
    recs = [r for r in recs if r.get("widths") == [int(width)] * 3]
    fam = lambda r, f, m: 100 * r["families"][f][m]  # noqa: E731
    out[width] = {"seeds": len(recs), "seed_list": [r["seed"] for r in recs], "cells": [
        cell([fam(r, "dnn", "rel_l2_radiance") for r in recs]),
        cell([fam(r, "dnn", "refl_mae_median") for r in recs]),
        cell([fam(r, "dkr", "rel_l2_radiance") for r in recs]),
        cell([fam(r, "dkr", "refl_mae_median") for r in recs])]}
path = os.path.join(W, "results", "scaling", "width_numbers.json")
old = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
for width in out:
    if width in old and old[width]["cells"] != out[width]["cells"]:
        print(f"width {width}: {old[width]['seeds']} -> {out[width]['seeds']} splits, "
              f"{old[width]['cells']} -> {out[width]['cells']}")
json.dump(out, open(path, "w", encoding="utf-8", newline="\n"), indent=1)
print(json.dumps({w: (v["seeds"], v["cells"]) for w, v in out.items()}))
