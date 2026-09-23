"""Kernel tuning on EMIT: every row in the solve, the per-input scales read from the cubic surrogate's gradients,
and the wide-network pipeline beside the main records at the same splits.

Reads results/emit/emit_s<seed>.json and, from results/scaling/per_seed/, the records emit_s<seed>_ft (tuning with
every row: krr_full, ard_full, and the gradient scales at exponent 1, ard_phys), emit_s<seed>_ph05 and _ph025 (the
gradient scales at exponents 1/2 and 1/4), emit_s<seed>_wide and emit_s<seed>_w1000. Writes
paper/table_emit_retune.tex and results/scaling/retune_numbers.json; the wide-pipeline table itself is written by
make_emit_tables.py.

usage: python code/make_retune_table.py
"""
import glob
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
M = ["rel_l2_Y1", "rel_l2_Y2", "rel_l2_Y3", "rel_l2_Y4", "rel_l2_radiance", "refl_mae_median"]


def load(pattern):
    out = {}
    for f in sorted(glob.glob(os.path.join(RES, pattern))):
        with open(f, encoding="utf-8") as fh:
            d = json.load(fh)
        out[d["seed"]] = d
    return out


base = load("emit/emit_s1??.json")
ft = load("scaling/per_seed/emit_s1??_ft.json")
ph05 = load("scaling/per_seed/emit_s1??_ph05.json")
ph025 = load("scaling/per_seed/emit_s1??_ph025.json")
wide = load("scaling/per_seed/emit_s1??_wide.json")
w1000 = load("scaling/per_seed/emit_s1??_w1000.json")
numbers = {}


def cell(vals, dec=3):
    v = 100 * np.asarray(vals, float)
    return f"{v.mean():.{dec}f}$\\pm${v.std(ddof=1) if len(v) > 1 else 0:.{dec}f}"


def row(label, recs, fam, seeds, dec=(2, 2, 2, 2, 3, 3)):
    vals = [[float(recs[s]["families"][fam][m]) for s in seeds] for m in M]
    numbers[label] = {m: (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, len(v))
                      for m, v in zip(M, vals)}
    return label + " & " + " & ".join(cell(v, d) for v, d in zip(vals, dec)) + " \\\\"


def selected_row(label, arms, seeds, fam="ard_phys", dec=(2, 2, 2, 2, 3, 3)):
    """One exponent per split, the one with the lowest mean validation error over the four components; records the
    choice at each split."""
    picks, vals = {}, [[] for _ in M]
    for s in seeds:
        best, bestv = None, None
        for name, recs in arms:
            d = recs.get(s)
            if not d or fam not in d.get("families", {}):
                best = None
                break
            h = d.get("hyper", {}).get(fam, {})
            vs = [h[c]["val"] for c in ("Y1", "Y2", "Y3", "Y4") if c in h and "val" in h[c]]
            if len(vs) != 4:
                best = None
                break
            mv = float(np.mean(vs))
            if bestv is None or mv < bestv:
                best, bestv = (name, d), mv
        if best is None:
            continue
        picks[s] = best[0]
        for j, m in enumerate(M):
            vals[j].append(float(best[1]["families"][fam][m]))
    if not picks:
        return None
    numbers[label] = {m: (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, len(v))
                      for m, v in zip(M, vals)}
    numbers[label + " :: picks"] = dict(picks)
    return label + " & " + " & ".join(cell(v, d) for v, d in zip(vals, dec)) + " \\\\"


seeds = sorted(set(base) & set(ft))
lines = ["\\begin{tabular}{lcccccc}", "\\toprule",
         " & \\multicolumn{4}{c}{component rel.\\ $L^2$ [\\%]} & radiance & reflectance \\\\", "\\cmidrule(lr){2-5}",
         "kernel & $Y_1$ & $Y_2$ & $Y_3$ & $Y_4$ & rel.\\ $L^2$ [\\%] & med.\\ $|\\hat\\rho-\\rho|$ [\\%] \\\\",
         "\\midrule",
         row("isotropic, tuned on 6{,}000 rows", base, "krr", seeds),
         row("isotropic, tuned on every row", ft, "krr_full", seeds),
         row("per-input search, tuned on 6{,}000 rows", base, "ard", seeds),
         row("per-input search, re-tuned on every row", ft, "ard_full", seeds),
         row("per-input from the cubic's gradients, $p=1$", ft, "ard_phys", seeds)]
for lab, recs in (("per-input from the cubic's gradients, $p=1/2$", ph05),
                  ("per-input from the cubic's gradients, $p=1/4$", ph025)):
    ss = sorted(set(recs) & set(seeds))
    if ss:
        lines.append(row(lab + f" ({len(ss)} seeds)" if len(ss) != len(seeds) else lab, recs, "ard_phys", ss))
# the exponent chosen on validation; the row is printed only when the choice differs between splits
arms = [("1", ft), ("1/2", ph05), ("1/4", ph025)]
ss = sorted(set(seeds).intersection(*[set(r) for _, r in arms]))
if ss:
    lab = "\\quad with $p$ chosen on validation"
    sel = selected_row(lab + (f" ({len(ss)} seeds)" if len(ss) != len(seeds) else ""), arms, ss)
    picked = set(numbers.get([k for k in numbers if k.endswith(":: picks")][-1], {}).values()) if sel else set()
    if sel and len(picked) > 1:
        lines.append(sel)
    elif sel:
        print(f"validation chose p={picked.pop()} at all {len(ss)} splits; the row would repeat that exponent's row")
lines += ["\\bottomrule", "\\end{tabular}"]
with open(os.path.join(ROOT, "paper", "table_emit_retune.tex"), "w", encoding="utf-8", newline="\n") as fh:
    fh.write("\n".join(lines) + "\n")
numbers["seeds"] = seeds
numbers["paired"] = {}
for a, fa, b, fb in (("ft", "krr_full", "base", "krr"), ("ft", "ard_full", "base", "ard"),
                     ("ft", "ard_phys", "base", "ard"), ("ft", "ard_phys", "base", "krr")):
    A, B = {"ft": ft, "base": base}[a], {"ft": ft, "base": base}[b]
    wins = sum(1 for s in seeds if float(A[s]["families"][fa]["rel_l2_radiance"]) <
               float(B[s]["families"][fb]["rel_l2_radiance"]))
    ratio = [float(A[s]["families"][fa]["rel_l2_radiance"]) / float(B[s]["families"][fb]["rel_l2_radiance"])
             for s in seeds]
    numbers["paired"][f"{fa}_vs_{fb}"] = dict(wins=wins, n=len(seeds), ratio_mean=float(np.mean(ratio)),
                                             ratio_min=float(min(ratio)), ratio_max=float(max(ratio)))
numbers["sub_vs_full_scale"] = {s: {c: (ft[s]["hyper"]["krr_full"][c].get("sub_hp", {}).get("scale"),
                                        ft[s]["hyper"]["krr_full"][c].get("scale")) for c in ("Y1", "Y2", "Y3", "Y4")}
                                for s in seeds}
numbers["phys_w_Y2"] = {s: ft[s]["hyper"]["ard_phys"]["Y2"]["w"] for s in seeds}
numbers["ard_w_Y2"] = {s: base[s]["hyper"]["ard"]["Y2"]["w"] for s in seeds}

ws = sorted(set(base) & set(wide))
if ws:
    for label, recs in (("3$\\times$512", base), ("3$\\times$2000", wide)):
        for fam in ("dnn", "dnn_corr", "dkr", "select", "stack"):
            if not all(fam in recs[s]["families"] for s in ws):
                continue
            vals = [[float(recs[s]["families"][fam][m]) for s in ws] for m in M]
            numbers[f"{label}_{fam}"] = {m: (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, len(v))
                                         for m, v in zip(M, vals)}
    numbers["wide_seeds"] = ws
    numbers["wide_paired"] = {}
    for fam in ("dkr", "stack", "select"):
        if all(fam in wide[s]["families"] for s in ws):
            numbers["wide_paired"][fam + "_wide_below_base_stack"] = [sum(
                1 for s in ws if float(wide[s]["families"][fam]["rel_l2_radiance"]) <
                float(base[s]["families"]["stack"]["rel_l2_radiance"])), len(ws)]
            numbers["wide_paired"][fam + "_wide_below_base_ard"] = [sum(
                1 for s in ws if float(wide[s]["families"][fam]["rel_l2_radiance"]) <
                float(base[s]["families"]["ard"]["rel_l2_radiance"])), len(ws)]
    numbers["wide_stack_weights"] = {s: wide[s]["hyper"].get("stack", {}).get("weights") for s in ws}
if w1000:
    numbers["w1000"] = {s: {f: float(w1000[s]["families"][f]["rel_l2_radiance"]) for f in ("dnn", "dkr", "dnn_corr")
                            if f in w1000[s]["families"]} for s in w1000}

with open(os.path.join(RES, "scaling", "retune_numbers.json"), "w", encoding="utf-8", newline="\n") as fh:
    json.dump(numbers, fh, indent=1, default=float)
print("seeds", seeds, "wide seeds", ws if ws else None)
print("paired:", json.dumps(numbers["paired"]))
