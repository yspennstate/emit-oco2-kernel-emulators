"""Tables for the training-target experiment (Section 5 of the manuscript), from the run records.

Inputs, under results/target_quality/:
  tq_s<seed>_<arm>_<config>.json              the campaign record (emit_campaign.py --training-policy)
  tq_s<seed>_<arm>_<config>_conditioned.json   the physical-domain scores (conditioned_reflectance.py)
with <arm> in raw, admissible, matched and <config> in w512, w2000. Only seeds with all three arms of a
configuration are used, so every paired difference is taken on one split. Writes:
  paper/table_tq_domain.tex   the raw arm: all-band and physical-domain reflectance tails, coverage, failures
  paper/table_tq_policy.tex   admissible and matched arms against raw: paired differences and win counts
  results/target_quality/tq_summary.json   every number in both tables, per seed and pooled
Sample standard deviations use ddof=1. Reflectance errors are reported in percentage points.
"""
import glob
import json
import os
import re
import statistics as st

W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(W, "results", "target_quality")
OUT = os.path.join(W, "paper")
ARMS = ("raw", "admissible", "matched")
NAMES = {"ridge3": "cubic ridge", "krr": "isotropic kernel", "ard": "input-scaled kernel", "dnn": "network",
         "dnn_ens": "five-network mean", "dnn_corr": "network + residual kernel",
         "ens_corr": "ensemble + residual kernel", "dkr": "kernel on features", "select": "per-coordinate selection",
         "stack": "convex stack"}
ORDER = ("ridge3", "krr", "ard", "dnn", "dnn_ens", "dnn_corr", "ens_corr", "dkr", "select", "stack")
THRESH_INDEX = 0          # the first prespecified threshold, 1e-12 of the training flux scale


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def ms(values, digits):
    if not values:
        return "--"
    if len(values) == 1:
        return f"{values[0]:.{digits}f}"
    return f"{st.mean(values):.{digits}f}$\\pm${st.stdev(values):.{digits}f}"


def collect():
    runs = {}
    for p in glob.glob(os.path.join(RES, "tq_s*_*_w*.json")):
        m = re.match(r"tq_s(\d+)_(raw|admissible|matched)_(w512|w2000)\.json$", os.path.basename(p))
        if not m:
            continue
        seed, arm, cfg = int(m.group(1)), m.group(2), m.group(3)
        cond = p[:-5] + "_conditioned.json"
        if not os.path.exists(cond):
            continue
        runs[(cfg, seed, arm)] = (load(p), load(cond))
    complete = {}
    for cfg in ("w512", "w2000"):
        seeds = sorted({s for (c, s, a) in runs if c == cfg})
        complete[cfg] = [s for s in seeds if all((cfg, s, a) in runs for a in ARMS)]
    return runs, complete


def metrics(record, cond, fam):
    f = record["families"][fam]
    c = cond["results"][fam][THRESH_INDEX]
    c3 = cond["results"][fam][1]                 # the second prespecified floor, 1e-3 of the training flux scale
    pct = lambda r, k: 100 * r[k] if r[k] is not None else None  # noqa: E731
    return {"radiance": 100 * f["rel_l2_radiance"], "albedo": 100 * f["rel_l2_Y4"],
            "components": 100 * f["mean_rel_l2_components"],
            "allband_median": 100 * f["refl_mae_median"], "allband_p95": 100 * f["refl_p95_abs"],
            "phys_median": pct(c, "median_absolute_error"), "phys_p95": pct(c, "p95_absolute_error"),
            "coverage": 100 * c["retained_fraction_all_entries"], "failed": c["failed_inversions"],
            "failed_pct": 100 * c["failed_inversions"] / c["retained_entries"],
            "retained": c["retained_entries"], "threshold": c["threshold"],
            "p95_1e3": pct(c3, "p95_absolute_error"), "coverage_1e3": 100 * c3["retained_fraction_all_entries"],
            "failed_pct_1e3": 100 * c3["failed_inversions"] / c3["retained_entries"]}


def main():
    runs, complete = collect()
    summary = {"seeds": complete, "per_seed": {}, "pooled": {}}
    dom_rows, pol_rows = [], []
    for cfg in ("w512", "w2000"):
        seeds = complete[cfg]
        if not seeds:
            continue
        fams = [f for f in ORDER if f in runs[(cfg, seeds[0], "raw")][0]["families"]]
        width = "3$\\times$512" if cfg == "w512" else "3$\\times$2000"
        for fam in fams:
            per = {s: {a: metrics(*runs[(cfg, s, a)], fam) for a in ARMS} for s in seeds}
            summary["per_seed"][f"{cfg}/{fam}"] = per
            raw = [per[s]["raw"] for s in seeds]
            dom_rows.append([f"{NAMES[fam]} ({width})",
                             ms([r["radiance"] for r in raw], 3),
                             ms([r["allband_p95"] for r in raw], 2),
                             ms([r["failed_pct"] for r in raw], 2),
                             ms([r["phys_p95"] for r in raw if r["phys_p95"] is not None], 2),
                             ms([r["failed_pct_1e3"] for r in raw], 3),
                             ms([r["p95_1e3"] for r in raw if r["p95_1e3"] is not None], 2)])
            for arm in ("admissible", "matched"):
                d_rad = [per[s][arm]["radiance"] - per[s]["raw"]["radiance"] for s in seeds]
                d_alb = [per[s][arm]["albedo"] - per[s]["raw"]["albedo"] for s in seeds]
                d_p95 = [per[s][arm]["phys_p95"] - per[s]["raw"]["phys_p95"] for s in seeds
                         if per[s][arm]["phys_p95"] is not None and per[s]["raw"]["phys_p95"] is not None]
                summary["pooled"][f"{cfg}/{fam}/{arm}"] = {"d_radiance": d_rad, "d_albedo": d_alb, "d_phys_p95": d_p95}
                pol_rows.append([f"{NAMES[fam]} ({width})" if arm == "admissible" else "", arm,
                                 ms(d_rad, 4), f"{sum(x < 0 for x in d_rad)}/{len(d_rad)}",
                                 ms(d_alb, 3), ms(d_p95, 2), f"{sum(x < 0 for x in d_p95)}/{len(d_p95)}"])
        # the stack-to-feature-kernel ratio of the reflectance tail, per arm
        if "stack" in fams and "dkr" in fams:
            for arm in ARMS:
                r_all = [runs[(cfg, s, arm)][0]["families"]["stack"]["refl_p95_abs"] /
                         runs[(cfg, s, arm)][0]["families"]["dkr"]["refl_p95_abs"] for s in seeds]
                r_phy = [runs[(cfg, s, arm)][1]["results"]["stack"][THRESH_INDEX]["p95_absolute_error"] /
                         runs[(cfg, s, arm)][1]["results"]["dkr"][THRESH_INDEX]["p95_absolute_error"] for s in seeds]
                summary["pooled"][f"{cfg}/stack_over_dkr/{arm}"] = {"all_band": r_all, "physical": r_phy}
    cov = {cfg: (st.mean(runs[(cfg, s, "raw")][1]["results"]["stack"][0]["retained_fraction_all_entries"] for s in complete[cfg]),
                 st.mean(runs[(cfg, s, "raw")][1]["results"]["stack"][1]["retained_fraction_all_entries"] for s in complete[cfg]))
           for cfg in complete if complete[cfg]}
    summary["coverage"] = {cfg: {"floor_1e-12": 100 * a, "floor_1e-3": 100 * b} for cfg, (a, b) in cov.items()}
    hdr = ("family & radiance [\\%] & all bands: $p_{95}$ [pp] & $\\tau_0=10^{-12}$: failed [\\%] & $p_{95}$ [pp] "
           "& $\\tau_0=10^{-3}$: failed [\\%] & $p_{95}$ [pp]")
    body = "\\begin{tabular}{lrrrrrr}\n\\toprule\n" + hdr + " \\\\\n\\midrule\n"
    body += "\n".join(" & ".join(r) + " \\\\" for r in dom_rows) + "\n\\bottomrule\n\\end{tabular}\n"
    with open(os.path.join(OUT, "table_tq_domain.tex"), "w", encoding="utf-8", newline="\n") as f:
        f.write(f"% seeds w512 {complete.get('w512')} w2000 {complete.get('w2000')}\n" + body)
    hdr = ("family & arm & $\\Delta$ radiance [pp] & lower & $\\Delta$ albedo [\\%] & $\\Delta$ physical $p_{95}$ [pp] "
           "& lower")
    body = "\\begin{tabular}{llrrrrr}\n\\toprule\n" + hdr + " \\\\\n\\midrule\n"
    body += "\n".join(" & ".join(r) + " \\\\" for r in pol_rows) + "\n\\bottomrule\n\\end{tabular}\n"
    with open(os.path.join(OUT, "table_tq_policy.tex"), "w", encoding="utf-8", newline="\n") as f:
        f.write(f"% seeds w512 {complete.get('w512')} w2000 {complete.get('w2000')}\n" + body)
    with open(os.path.join(RES, "tq_summary.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({"complete_seeds": complete, "domain_rows": len(dom_rows), "policy_rows": len(pol_rows)}))


if __name__ == "__main__":
    main()
