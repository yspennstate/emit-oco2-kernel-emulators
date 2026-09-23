"""Learning curves, their exponents and the output-rank ablation on EMIT, from the per-split records.

Reads the full-size records results/emit/emit_s<seed>.json and the records of the secondary runs in
results/scaling/per_seed/ (emit_s<seed>_n<rows>, emit_s<seed>_n<rows>_ard, emit_s<seed>_r<rank>, emit_s<seed>_big).
Writes paper/table_emit_curves.tex, paper/table_emit_slopes.tex, paper/table_emit_rank.tex and
results/scaling/scaling_numbers.json, and, when matplotlib is installed, figures/emit_curves.png.
Every cell is the mean and sample standard deviation over splits.

usage: python code/make_scaling_tables.py
"""
import glob
import json
import math
import os
import re
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(ROOT, "paper")
FIGS = os.path.join(ROOT, "figures")
RES = os.path.join(ROOT, "results")

R = {}
for f in sorted(glob.glob(os.path.join(RES, "scaling", "per_seed", "emit_s*.json"))) + \
        sorted(glob.glob(os.path.join(RES, "emit", "emit_s*.json"))):
    with open(f, encoding="utf-8") as fh:
        R[os.path.basename(f)[:-5]] = json.load(fh)
full = {t: d for t, d in R.items() if re.fullmatch(r"emit_s\d+", t)}
if len(full) != 10:
    raise SystemExit(f"expected ten full-size records, found {len(full)}")

METRICS = ["rel_l2_Y1", "rel_l2_Y2", "rel_l2_Y3", "rel_l2_Y4", "rel_l2_radiance", "refl_mae_median"]
numbers = {}


def ms(vals):
    v = np.asarray(vals, float)
    return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else 0.0), len(v)


def cell(vals, scale=100.0, dec=3):
    m, s, n = ms(vals)
    return f"{scale*m:.{dec}f}$\\pm${scale*s:.{dec}f}"


def collect(recs, fam, m):
    return [float(d["families"][fam][m]) for d in recs if fam in d["families"] and m in d["families"][fam]]


def write(name, lines):
    with open(os.path.join(PAPER, name), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")


# learning curves; the per-input kernel runs alone on its own lanes, tagged emit_s<seed>_n<n>_ard
curves, ard_curves = defaultdict(list), defaultdict(list)
for t, d in R.items():
    mm = re.fullmatch(r"emit_s(\d+)_n(\d+)", t)
    if mm:
        curves[int(mm.group(2))].append(d)
    mm = re.fullmatch(r"emit_s(\d+)_n(\d+)_ard", t)
    if mm:
        ard_curves[int(mm.group(2))].append(d)
curves[18884] = list(full.values())
ard_curves[18884] = list(full.values())

ns = sorted(curves)
fams = ["ridge3", "krr", "dnn", "dnn_corr", "dkr"]
lines = ["\\begin{tabular}{lrcccccc}", "\\toprule",
         "training rows & seeds & Cubic ridge & KRR, isotropic & KRR, per-input & FC-DNN & DNN + residual KRR & "
         "kernel on features \\\\", "\\midrule"]
numbers["curves"] = {}
numbers["ard_curve"] = {}
for n in ns:
    recs = curves[n]
    arec = ard_curves.get(n, [])
    av = collect(arec, "ard", "rel_l2_radiance")
    numbers["ard_curve"][n] = {m: ms(collect(arec, "ard", m)) for m in METRICS if collect(arec, "ard", m)}
    row = [f"{n:,}", str(len(collect(recs, 'krr', 'rel_l2_radiance')))]
    numbers["curves"][n] = {}
    for fam in fams:
        v = collect(recs, fam, "rel_l2_radiance")
        numbers["curves"][n][fam] = {m: ms(collect(recs, fam, m)) for m in METRICS}
        row.append(cell(v))
        if fam == "krr":
            row.append(cell(av) if av else "--")
    lines.append(" & ".join(row) + " \\\\")
lines += ["\\bottomrule", "\\end{tabular}"]
write("table_emit_curves.tex", lines)

# exponents: least-squares slope of log error against log n over the rungs, and the slope of the last step
numbers["slopes"] = {}
for fam in fams + ["ard"]:
    numbers["slopes"][fam] = {}
    src = numbers["ard_curve"] if fam == "ard" else None
    for m in METRICS[:5]:
        xs = [math.log(n) for n in ns]
        if src is not None:
            if not all(m in src[n] for n in ns):
                continue
            ys = [math.log(src[n][m][0]) for n in ns]
        else:
            ys = [math.log(numbers["curves"][n][fam][m][0]) for n in ns]
        s_all = float(np.polyfit(xs, ys, 1)[0])
        s_lo = float(np.polyfit(xs[:-1], ys[:-1], 1)[0])
        s_last = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
        numbers["slopes"][fam][m] = dict(all=s_all, upto8000=s_lo, last=s_last)
numbers["ratio_dnn_krr"] = {}
numbers["ratio_corr_krr"] = {}
numbers["corr_below_both"] = {}
for n in ns:
    recs = curves[n]
    r1 = [float(d["families"]["dnn"]["rel_l2_radiance"]) / float(d["families"]["krr"]["rel_l2_radiance"]) for d in recs]
    r2 = [float(d["families"]["dnn_corr"]["rel_l2_radiance"]) / float(d["families"]["krr"]["rel_l2_radiance"])
          for d in recs]
    numbers["ratio_dnn_krr"][n] = ms(r1)
    numbers["ratio_corr_krr"][n] = ms(r2)
    numbers["corr_below_both"][n] = {}
    for m in METRICS[:5]:
        below = sum(1 for d in recs if float(d["families"]["dnn_corr"][m]) <
                    min(float(d["families"]["krr"][m]), float(d["families"]["dnn"][m])))
        numbers["corr_below_both"][n][m] = [below, len(recs)]
sfams = [("ridge3", "Cubic ridge"), ("krr", "KRR, isotropic"), ("ard", "KRR, per-input"), ("dnn", "FC-DNN"),
         ("dnn_corr", "DNN + residual KRR"), ("dkr", "Kernel on features")]
smets = [("rel_l2_Y1", "$Y_1$"), ("rel_l2_Y2", "$Y_2$"), ("rel_l2_Y3", "$Y_3$"), ("rel_l2_Y4", "$Y_4$"),
         ("rel_l2_radiance", "radiance")]
lines = ["\\begin{tabular}{lccccc|ccccc}", "\\toprule",
         " & \\multicolumn{5}{c|}{slope over the six rungs} & \\multicolumn{5}{c}{slope of the last step} \\\\",
         "family & " + " & ".join(m for _, m in smets) + " & " + " & ".join(m for _, m in smets) + " \\\\",
         "\\midrule"]
for f, name in sfams:
    s = numbers["slopes"][f]
    lines.append(name + " & " + " & ".join(f"${s[m]['all']:.2f}$" for m, _ in smets) + " & "
                 + " & ".join(f"${s[m]['last']:.2f}$" for m, _ in smets) + " \\\\")
lines += ["\\bottomrule", "\\end{tabular}"]
write("table_emit_slopes.tex", lines)

# output-rank ablation; rank 64 is the main campaign's own rank at the same seeds
ranks = defaultdict(list)
for t, d in R.items():
    mm = re.fullmatch(r"emit_s(\d+)_r(\d+)", t)
    if mm:
        ranks[int(mm.group(2))].append(d)
seeds_r = sorted({d["seed"] for k in ranks for d in ranks[k]})
ranks[64] = [d for d in full.values() if d["seed"] in seeds_r]
lines = ["\\begin{tabular}{lcccccc}", "\\toprule",
         "PCA rank & variance outside the basis & KRR, isotropic & FC-DNN & DNN + residual KRR & kernel on features \\\\",
         "\\midrule"]
numbers["rank"] = {"seeds": seeds_r}


def sci(x):
    if x <= 0:
        return "$0$"
    e = int(math.floor(math.log10(x)))
    return f"${x/10**e:.1f}\\times10^{{{e}}}$"


for r in sorted(ranks):
    recs = ranks[r]
    evr = np.mean([min(float(d["pca_evr"][c]) for c in ("Y1", "Y2", "Y3", "Y4")) for d in recs])
    row = [str(r), sci(1.0 - evr)]
    numbers["rank"][r] = {"evr_min_mean": float(evr)}
    for fam in ["krr", "dnn", "dnn_corr", "dkr"]:
        row.append(cell(collect(recs, fam, "rel_l2_radiance")))
        numbers["rank"][r][fam] = {m: ms(collect(recs, fam, m)) for m in METRICS}
    lines.append(" & ".join(row) + " \\\\")
lines += ["\\bottomrule", "\\end{tabular}"]
write("table_emit_rank.tex", lines)

# the width-2000 network trained for 500 epochs beside the main records at the same seeds (quoted in the text)
big = [d for t, d in R.items() if t.endswith("_big")]
seeds_b = sorted({d["seed"] for d in big})
base = [d for d in full.values() if d["seed"] in seeds_b]
numbers["wide"] = {"seeds": seeds_b, "base": {}, "big": {}, "paired": {}}
for recs, key in ((base, "base"), (big, "big")):
    for fam in ("dnn", "dnn_corr", "dkr"):
        numbers["wide"][key][fam] = {m: ms(collect(recs, fam, m)) for m in METRICS}
bb = {d["seed"]: d for d in big}
ba = {d["seed"]: d for d in base}
for fam in ("dnn", "dnn_corr", "dkr"):
    wins = sum(1 for s in seeds_b if float(bb[s]["families"][fam]["rel_l2_radiance"]) <
               float(ba[s]["families"][fam]["rel_l2_radiance"]))
    numbers["wide"]["paired"][fam + "_big_better"] = [wins, len(seeds_b)]
numbers["wide"]["paired"]["big_dkr_below_ard"] = [sum(1 for s in seeds_b if float(bb[s]["families"]["dkr"]["rel_l2_radiance"])
                                                      < float(ba[s]["families"]["ard"]["rel_l2_radiance"])), len(seeds_b)]
numbers["wide"]["paired"]["big_dkr_below_stack"] = [sum(1 for s in seeds_b if float(bb[s]["families"]["dkr"]["rel_l2_radiance"])
                                                        < float(ba[s]["families"]["stack"]["rel_l2_radiance"])), len(seeds_b)]
numbers["wide"]["epochs"] = {s: bb[s]["hyper"]["dnn_epochs"] for s in seeds_b}
numbers["wide"]["ard_base"] = {m: ms(collect(base, "ard", m)) for m in METRICS}
numbers["wide"]["stack_base"] = {m: ms(collect(base, "stack", m)) for m in METRICS}

with open(os.path.join(RES, "scaling", "scaling_numbers.json"), "w", encoding="utf-8", newline="\n") as fh:
    json.dump(numbers, fh, indent=1, default=float)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
if plt is not None:
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.3))
    panels = [("rel_l2_radiance", "radiance rel. $L^2$ [%]"), ("rel_l2_Y1", "$Y_1$ rel. $L^2$ [%]"),
              ("rel_l2_Y2", "$Y_2$ rel. $L^2$ [%]"), ("rel_l2_Y4", "$Y_4$ rel. $L^2$ [%]")]
    style = {"ridge3": ("Cubic ridge", "0.5", "s"), "krr": ("KRR, isotropic", "C0", "o"), "dnn": ("FC-DNN", "C3", "^"),
             "dnn_corr": ("DNN + residual KRR", "C2", "v"), "dkr": ("Kernel on features", "C1", "D")}
    for ax, (m, lab) in zip(axes, panels):
        for fam in fams:
            ys = [100 * numbers["curves"][n][fam][m][0] for n in ns]
            es = [100 * numbers["curves"][n][fam][m][1] for n in ns]
            name, col, mk = style[fam]
            ax.errorbar(ns, ys, yerr=es, color=col, marker=mk, ms=4, lw=1.2, capsize=2, label=name)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("training rows")
        ax.set_ylabel(lab)
        ax.grid(True, which="both", alpha=0.25)
        ax.set_xticks(ns)
        ax.set_xticklabels([str(n) for n in ns], rotation=45, fontsize=7)
        ax.minorticks_off()
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "emit_curves.png"), dpi=170)
print("slopes (radiance):", {f: round(numbers["slopes"][f]["rel_l2_radiance"]["all"], 3) for f in fams})
print("figure:", "written" if plt is not None else "skipped (matplotlib not installed)")
