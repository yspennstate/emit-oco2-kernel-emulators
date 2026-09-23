"""The ten-split comparison of learned kernel metrics on OCO-2 quoted in the appendix on OCO-2.

Four ways of choosing an exact kernel's metric on the state (kf_kernels.py: the kernel-flow loss, the same with a
full Mahalanobis metric, the gradient outer product of the recursive feature machine, per-input scales by empirical
Bayes) and an additive kernel are compared with the isotropic kernel of the same run, and with the network and the
kernel on its features from the baseline runs of oco2_curve.py on the same splits. The isotropic kernel appears in
both sets. The metric runs choose its Matern smoothness on validation (3/2 or 5/2) while the baseline fixes 5/2, so
the script prints the selected smoothness, scale and nugget of both at every split.

usage: python code/oco2_metric_summary.py results/oco2_metrics results/oco2_ensembles
Writes results/oco2_metrics/summary.json.
"""
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KFK_DIR, PROT_DIR = sys.argv[1], sys.argv[2]
KFK, PROT = defaultdict(dict), defaultdict(dict)
for f in glob.glob(os.path.join(KFK_DIR, "oco_*_s*_kfk.json")):
    m = re.fullmatch(r"oco_(o2|wco2|sco2)_s(\d+)_kfk", os.path.basename(f)[:-5])
    if m:
        KFK[m.group(1)][int(m.group(2))] = json.load(open(f, encoding="utf-8"))
for f in glob.glob(os.path.join(PROT_DIR, "oco_*_s*_n18000.json")):
    m = re.fullmatch(r"oco_(o2|wco2|sco2)_s(\d+)_n18000", os.path.basename(f)[:-5])
    if m:
        PROT[m.group(1)][int(m.group(2))] = json.load(open(f, encoding="utf-8"))

BANDS = ("o2", "wco2", "sco2")
HEADS = [("val_iso", "isotropic"), ("kf_ard", "kernel-flow per-input"), ("kf_mahal", "kernel-flow Mahalanobis"),
         ("rfm", "recursive feature machine"), ("eb_ard", "empirical-Bayes per-input"), ("add_kf", "additive kernel")]
PROT_HEADS = [("kernel_raw", "isotropic (baseline runs)"), ("mean_flat", "network"),
              ("dkr_flat", "kernel on network features")]


def ms(v):
    a = np.asarray(v, float)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0


out = {"heads": {}, "per_split": {}}
print(f"{'band':6s} {'head':28s} {'coefficients [%]':>20s} {'radiance [%]':>20s}")
for band in BANDS:
    if sorted(KFK[band]) != sorted(PROT[band]):
        raise SystemExit(f"{band}: the two record sets hold different splits")
    seeds = sorted(KFK[band])
    for key, name in HEADS:
        v = [KFK[band][s]["results"][key]["test"] for s in seeds]
        r = [KFK[band][s]["results"][key]["test_radiance"] for s in seeds]
        (a, sd), (b, sdb) = ms(v), ms(r)
        out["heads"][f"{band}/{key}"] = dict(coefficients=[a, sd], radiance=[b, sdb], n=len(v))
        print(f"{band:6s} {name:28s} {a:9.2f} +- {sd:<7.2f} {b:9.4f} +- {sdb:<7.4f}")
    for key, name in PROT_HEADS:
        a, sd = ms([PROT[band][s]["results"][key]["reduced"] for s in seeds])
        out["heads"][f"{band}/{key}"] = dict(coefficients=[a, sd], n=len(seeds))
        print(f"{band:6s} {name:28s} {a:9.2f} +- {sd:<7.2f}")
    rows = []
    for s in seeds:
        hk, hp = KFK[band][s]["hyper"]["val_iso"], PROT[band][s]["hyper"]["kernel_raw"]
        rows.append(dict(seed=s, metric_run=dict(nu=hk["nu"], scale=hk["scale"], nugget=hk["nugget"],
                                                 test=KFK[band][s]["results"]["val_iso"]["test"]),
                         baseline=dict(nu=2.5, scale=hp["scale"] * hp["med"], nugget=hp["nugget"],
                                       test=PROT[band][s]["results"]["kernel_raw"]["reduced"]),
                         eb_ard=KFK[band][s]["results"]["eb_ard"]["test"]))
    out["per_split"][band] = rows
    print("  isotropic kernel per split (metric run | baseline): " + "; ".join(
        f"s{r['seed']} nu {r['metric_run']['nu']} {r['metric_run']['test']:.2f} | {r['baseline']['test']:.2f}"
        for r in rows))
    print()
with open(os.path.join(W, "results", "oco2_metrics", "summary.json"), "w", encoding="utf-8", newline="\n") as f:
    json.dump(out, f, indent=1)
print("wrote results/oco2_metrics/summary.json")
