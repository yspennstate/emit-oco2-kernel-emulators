"""The forward-inverse comparison on libRadtran: hypotheses H1-H3 of the protocol and the table, from the run records.

Inputs, under results/libradtran/: <prefix>_s<seed>_w512.json (emit_campaign.py record) and the matching
_conditioned.json (conditioned_reflectance.py). The prefix is lrtc (the default: sixteen inputs, the lanes of addendum 1)
or lrt (the seven numeric inputs, seeds 101-104). The protocol is PREREGISTRATION_LRT_20260923.md with its addendum 1:
  H1  the family with the lowest radiance error does not have the lowest all-band 95th-percentile reflectance error;
  H2  the convex stack's radiance error is at most the feature kernel's and its all-band 95th percentile is larger;
  H3  the Kendall correlation between the seven families' radiance errors and their 95th percentiles on the physical
      domain at the floor 1e-3 exceeds the same correlation over all bands;
each confirmed if it holds on at least 9 of the 10 seeds. Also reported, without a decision rule: how far apart the families
sit in radiance on each seed and how often the family with the lowest radiance error changes between seeds.
Writes paper/table_lrt.tex and results/libradtran/lrt_summary.json (prefix lrtc), or table_lrt_numeric.tex and
lrt_summary_numeric.json (prefix lrt).

usage: python code/make_lrt_tables.py [lrtc|lrt]
"""
import glob
import json
import os
import re
import statistics as st
import sys

from scipy.stats import kendalltau

W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(W, "results", "libradtran")
ORDER = ("ridge3", "krr", "ard", "dnn", "dnn_corr", "dkr", "stack")
NAMES = {"ridge3": "cubic ridge", "krr": "isotropic kernel", "ard": "input-scaled kernel", "dnn": "network",
         "dnn_corr": "network + residual kernel", "dkr": "kernel on features", "stack": "convex stack"}
FLOORS = (1e-12, 1e-3, 1e-2)
PREFIX = sys.argv[1] if len(sys.argv) > 1 else "lrtc"
SUFFIX = "" if PREFIX == "lrtc" else "_numeric"


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def at_floor(cond, fam, floor):
    rows = [r for r in cond["results"][fam] if abs(r["threshold"] - floor) <= 1e-9 * floor]
    if len(rows) != 1:
        raise ValueError(f"{fam}: floor {floor} found {len(rows)} times")
    return rows[0]


def ms(v, d):
    v = [x for x in v if x is not None]
    if not v:
        return "--"
    return f"{v[0]:.{d}f}" if len(v) == 1 else f"{st.mean(v):.{d}f}$\\pm${st.stdev(v):.{d}f}"


runs = {}
for p in glob.glob(os.path.join(RES, f"{PREFIX}_s*_w512.json")):
    m = re.match(PREFIX + r"_s(\d+)_w512\.json$", os.path.basename(p))
    c = p[:-5] + "_conditioned.json"
    if m and os.path.exists(c):
        runs[int(m.group(1))] = (load(p), load(c))
seeds = sorted(runs)
per, hyp = {}, {"H1": {}, "H2": {}, "H3": {}}
for s in seeds:
    rec, cond = runs[s]
    fams = [f for f in ORDER if f in rec["families"]]
    row = {}
    for f in fams:
        r = rec["families"][f]
        row[f] = {"radiance": 100 * r["rel_l2_radiance"], "allband_p95": 100 * r["refl_p95_abs"],
                  "components": 100 * r["mean_rel_l2_components"]}
        for fl in FLOORS:
            c = at_floor(cond, f, fl)
            row[f][f"p95@{fl:g}"] = 100 * c["p95_absolute_error"] if c["p95_absolute_error"] is not None else None
            row[f][f"failed_pct@{fl:g}"] = 100 * c["failed_inversions"] / c["retained_entries"]
            row[f][f"coverage@{fl:g}"] = 100 * c["retained_fraction_all_entries"]
    per[s] = row
    rad = {f: row[f]["radiance"] for f in fams}
    allp = {f: row[f]["allband_p95"] for f in fams}
    phys = {f: row[f]["p95@0.001"] for f in fams}
    best_rad, best_tail = min(rad, key=rad.get), min(allp, key=allp.get)
    hyp["H1"][s] = {"holds": best_rad != best_tail, "lowest_radiance": best_rad, "lowest_allband_p95": best_tail}
    hyp["H2"][s] = {"holds": rad["stack"] <= rad["dkr"] and allp["stack"] > allp["dkr"],
                    "d_radiance": rad["stack"] - rad["dkr"], "tail_ratio": allp["stack"] / allp["dkr"]}
    t_all = kendalltau([rad[f] for f in fams], [allp[f] for f in fams]).statistic
    t_phy = kendalltau([rad[f] for f in fams], [phys[f] for f in fams]).statistic
    hyp["H3"][s] = {"holds": bool(t_phy > t_all), "tau_all_band": float(t_all), "tau_floor_1e-3": float(t_phy)}
count = {h: sum(1 for v in hyp[h].values() if v["holds"]) for h in hyp}
resolution = {}
for s in seeds:
    rad = [per[s][f]["radiance"] for f in per[s]]
    resolution[s] = {"radiance_range": max(rad) - min(rad), "radiance_mean": st.mean(rad),
                     "lowest_radiance": hyp["H1"][s]["lowest_radiance"]}
lows = [resolution[s]["lowest_radiance"] for s in seeds]
fam_sd = {f: (st.stdev([per[s][f]["radiance"] for s in seeds]) if len(seeds) > 1 else None) for f in ORDER
          if seeds and f in per[seeds[0]]}
summary = {"prefix": PREFIX, "seeds": seeds, "counts": count,
           "confirmed": {h: (count[h] >= 9 and len(seeds) == 10) for h in count},
           "hypotheses": hyp, "per_seed": per,
           "forward_resolution": {"per_seed": resolution, "lowest_radiance_families": sorted(set(lows)),
                                  "lowest_radiance_counts": {f: lows.count(f) for f in sorted(set(lows))},
                                  "radiance_sd_across_seeds": fam_sd}}
rows = []
for f in ORDER:
    if not seeds or f not in per[seeds[0]]:
        continue
    col = lambda k: [per[s][f][k] for s in seeds]  # noqa: E731
    r = [NAMES[f], ms(col("radiance"), 3), ms(col("allband_p95"), 2)]
    for fl in FLOORS:
        r += [ms(col(f"failed_pct@{fl:g}"), 2), ms(col(f"p95@{fl:g}"), 2)]
    rows.append(r)
hdr = ("family & radiance [\\%] & \\thead{all bands\\\\$p_{95}$ [pp]}" + "".join(
    f" & \\thead{{$\\tau_0=10^{{{e}}}$\\\\failed [\\%]}} & $p_{{95}}$ [pp]" for e in (-12, -3, -2)))
body = "\\begin{tabular}{l" + "r" * 8 + "}\n\\toprule\n" + hdr + " \\\\\n\\midrule\n"
body += "\n".join(" & ".join(r) + " \\\\" for r in rows) + "\n\\bottomrule\n\\end{tabular}\n"
with open(os.path.join(W, "paper", f"table_lrt{SUFFIX}.tex"), "w", encoding="utf-8", newline="\n") as fh:
    fh.write(f"% generated by code/make_lrt_tables.py {PREFIX}; seeds {seeds}\n" + body)
with open(os.path.join(RES, f"lrt_summary{SUFFIX}.json"), "w", encoding="utf-8", newline="\n") as fh:
    json.dump(summary, fh, indent=1)
print(json.dumps({"prefix": PREFIX, "seeds": seeds, "counts": count, "confirmed": summary["confirmed"],
                  "lowest_radiance_counts": summary["forward_resolution"]["lowest_radiance_counts"]}))
