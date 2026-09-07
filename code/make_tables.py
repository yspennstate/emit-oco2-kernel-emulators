"""Rebuild the generated LaTeX tables from the per-run records in results/.

Writes paper/table_oco2_losses.tex, paper/table_oco2_seeds.tex and paper/table_corpora.tex,
and prints the paired counts, bootstrap intervals and replication figures quoted in the text,
so the prose can be checked against the records without reading the LaTeX.

    python code/make_tables.py            # from the repository root

Requires only the standard library.
"""
import glob
import json
import os
import statistics as st
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "paper")

BANDS = [("o2", "O$_2$"), ("wco2", "weak CO$_2$"), ("sco2", "strong CO$_2$")]
MEMBERS = ["mean_flat", "dkr_flat", "mean_wnum", "dkr_wnum", "mean_radx", "dkr_radx"]


def load(pattern):
    out = []
    for f in sorted(glob.glob(pattern)):
        with open(f, encoding="utf-8") as fh:
            out.append(json.load(fh))
    return out


def cell(values, nd):
    """mean +- sd, with the spread dropped when it rounds to zero at this precision."""
    m = st.mean(values)
    s = st.stdev(values) if len(values) > 1 else 0.0
    if s < 0.5 * 10 ** (-nd):
        return "%.*f" % (nd, m)
    return "%.*f$\\pm$%.*f" % (nd, m, nd, s)


def write(name, lines):
    path = os.path.join(OUT, name)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote", os.path.relpath(path, ROOT))


# --------------------------------------------------------------- OCO-2 by loss and readout
def oco2_losses():
    D = {}
    for b, _ in BANDS:
        for d in load(os.path.join(RES, "oco2_losses", "oco_%s_s*.json" % b)):
            for fam, v in d["results"].items():
                if isinstance(v, dict) and "reduced" in v:
                    D.setdefault((b, fam), {})[d["seed"]] = (v["reduced"], v["radiance"])

    rows = [
        ("kernel_flow", "reference emulator, stored predictions"),
        (None, "\\emph{trained on the reduced coefficients}"),
        ("mean_flat", "network"),
        ("ridge_flat", "\\quad ridge readout on its features"),
        ("dkr_flat", "\\quad Mat\\'ern kernel on its features"),
        (None, "\\emph{trained on the weighted coefficients}"),
        ("mean_wnum", "network"),
        ("ridge_wnum", "\\quad ridge readout on its features"),
        ("dkr_wnum", "\\quad Mat\\'ern kernel on its features"),
        (None, "\\emph{trained on the reconstructed radiance}"),
        ("mean_radx", "network"),
        ("ridge_radx", "\\quad ridge readout on its features"),
        ("dkr_radx", "\\quad Mat\\'ern kernel on its features"),
        (None, None),
        ("combined", "per-coordinate selection over the six heads"),
    ]
    L = ["\\begin{tabular}{lcccccc}", "\\toprule",
         " & \\multicolumn{3}{c}{reduced coefficients rel.\\ $L^2$ [\\%]}"
         " & \\multicolumn{3}{c}{radiance rel.\\ $L^2$ [\\%]} \\\\",
         "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
         "model & " + " & ".join(n for _, n in BANDS) + " & " + " & ".join(n for _, n in BANDS) + " \\\\",
         "\\midrule"]
    for fam, label in rows:
        if fam is None:
            L.append("\\midrule" if label is None else "\\multicolumn{7}{l}{%s} \\\\" % label)
            continue
        cells = [cell([v[0] for v in D[(b, fam)].values()], 2) for b, _ in BANDS]
        cells += [cell([v[1] for v in D[(b, fam)].values()], 4) for b, _ in BANDS]
        L.append(label + " & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    write("table_oco2_losses.tex", L)

    print("\n-- OCO-2 loss campaign, statistics quoted in the text")
    for b, _ in BANDS:
        ref = D[(b, "kernel_flow")]
        sel = D[(b, "combined")]
        seeds = sorted(sel)
        print("  %-5s selection beats the reference: reduced %d/%d, radiance %d/%d" % (
            b,
            sum(1 for s in seeds if sel[s][0] < ref[s][0]), len(seeds),
            sum(1 for s in seeds if sel[s][1] < ref[s][1]), len(seeds)))
        for loss in ("flat", "wnum", "radx"):
            h, g, n = D[(b, "dkr_" + loss)], D[(b, "ridge_" + loss)], D[(b, "mean_" + loss)]
            diff = [g[s][0] - h[s][0] for s in seeds]
            print("        %-4s kernel beats ridge readout %d/%d (by %.3f +- %.3f points); "
                  "ridge readout beats its network %d/%d" % (
                      loss,
                      sum(1 for s in seeds if h[s][0] < g[s][0]), len(seeds),
                      st.mean(diff), st.stdev(diff),
                      sum(1 for s in seeds if g[s][0] < n[s][0]), len(seeds)))
        picks = Counter()
        for d in load(os.path.join(RES, "oco2_losses", "oco_%s_s*.json" % b)):
            for j, w in enumerate(d["winners"]):
                picks[(j, MEMBERS[w])] += 1
        lead = {k[1]: v for k, v in picks.items() if k[0] == 0}
        other = Counter()
        for k, v in picks.items():
            if k[0] != 0:
                other[k[1]] += v
        print("        leading coefficient -> %s ; remaining coefficients -> %s" % (lead, dict(other)))
    return D


# --------------------------------------------------------------- OCO-2 ensembles
def oco2_ensembles(losses):
    rows = [
        ("kernel_flow", "reference emulator, stored predictions"),
        ("kernel_raw", "Mat\\'ern KRR on the state, isotropic"),
        ("kernel_ard", "Mat\\'ern KRR on the state, scaled metric"),
        ("mean_flat", "network"),
        ("dkr_flat", "DKR: Mat\\'ern KRR on network features"),
        ("mean_ens", "mean of three networks"),
        ("dkr_ens", "DKR on concatenated features (3)"),
        ("dkr_avg", "mean of three DKR heads"),
        ("combined", "per-coordinate selection"),
    ]
    D = {}
    for b, _ in BANDS:
        for d in load(os.path.join(RES, "oco2_ensembles", "oco_%s_s*_n18000_m3.json" % b)):
            for fam, v in d["results"].items():
                if isinstance(v, dict) and "reduced" in v:
                    D.setdefault((b, fam), {})[d["seed"]] = (v["reduced"], v["radiance"])
    L = ["\\begin{tabular}{lcccccc}", "\\toprule",
         " & \\multicolumn{3}{c}{reduced coefficients rel.\\ $L^2$ [\\%]}"
         " & \\multicolumn{3}{c}{radiance rel.\\ $L^2$ [\\%]} \\\\",
         "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
         "model & " + " & ".join(n for _, n in BANDS) + " & " + " & ".join(n for _, n in BANDS) + " \\\\",
         "\\midrule"]
    for fam, label in rows:
        cells = [cell([v[0] for v in D[(b, fam)].values()], 2) for b, _ in BANDS]
        cells += [cell([v[1] for v in D[(b, fam)].values()], 4) for b, _ in BANDS]
        L.append(label + " & " + " & ".join(cells) + " \\\\")
        if fam == "kernel_flow":
            L.append("\\midrule")
    L += ["\\bottomrule", "\\end{tabular}"]
    write("table_oco2_seeds.tex", L)

    print("\n-- the two campaigns share their splits; the rows they have in common are a replication")
    for b, _ in BANDS:
        for fam in ("mean_flat", "dkr_flat"):
            A, C = losses[(b, fam)], D[(b, fam)]
            seeds = sorted(set(A) & set(C))
            print("  %-5s %-9s loss campaign %.3f, ensemble campaign %.3f, largest per-split gap %.3f" % (
                b, fam,
                st.mean([A[s][0] for s in seeds]), st.mean([C[s][0] for s in seeds]),
                max(abs(A[s][0] - C[s][0]) for s in seeds)))


# --------------------------------------------------------------- the further corpora
def corpora():
    """One row per configuration. The globs are exact: the rank-1024 rerun of the turbulent
    radiative layer and the multi-fidelity configuration of the correction-coefficient corpus are
    separate problems from the rows they would otherwise be pooled into, and are excluded or
    given rows of their own."""
    order = [
        ("Advection", "advection_s*_bench.json", None),
        ("Burgers $\\nu=0.1$", "burgers_nu0.1_s*_bench.json", None),
        ("Burgers $\\nu=0.01$", "burgers_nu0.01_s*_bench.json", None),
        ("Burgers $\\nu=0.001$", "burgers_nu0.001_s*_bench.json", None),
        ("Turbulent radiative layer", "trl2d_s*_bench.json", "_r1024_"),
        ("Active matter", "well_active_matter_s*_bench.json", None),
        ("Helmholtz staircase", "well_helmholtz_staircase_s*_bench.json", None),
        ("ClimSim", "climsim_s*_bench.json", None),
        ("RT corrections, state inputs", "pkanrtm_s*_bench.json", "_lowfi_"),
        ("RT corrections, multi-fidelity inputs", "pkanrtm_s*_lowfi_bench.json", None),
    ]
    fams = [("ridge", "ridge (linear)"), ("krr", "KRR"), ("krr_kf", "KRR, learned metric"),
            ("mlp", "network"), ("mlp_ens", "network ensemble"),
            ("mlp_resid", "network + residual KRR"), ("dkr", "kernel on features"),
            ("stack", "convex stack")]
    L = ["\\begin{tabular}{lrrrr" + "c" * len(fams) + "}", "\\toprule",
         "corpus & $n$ & $d$ & $q$ & runs & " + " & ".join(n for _, n in fams) + " \\\\", "\\midrule"]
    print("\n-- the further corpora")
    for name, pattern, drop in order:
        files = sorted(glob.glob(os.path.join(RES, "corpora", pattern)))
        if drop:
            files = [f for f in files if drop not in os.path.basename(f)]
        recs = []
        for f in files:
            with open(f, encoding="utf-8") as fh:
                recs.append(json.load(fh))
        if not recs:
            print("  MISSING", name)
            continue
        m = recs[0]
        ns = sorted({r["n"] for r in recs})
        ncol = format(ns[0], ",") if len(ns) == 1 else "%s--%s" % (format(ns[0], ","), format(ns[-1], ","))
        cells, means = [], {}
        for fam, _ in fams:
            vals = [r["results"][fam]["test"] for r in recs if fam in r["results"]]
            cells.append(cell(vals, 2) if vals else "--")
            if vals:
                means[fam] = st.mean(vals)
        L.append("%s & %s & %d & %d & %d & %s \\\\" % (
            name, ncol, m["d"], m["q"], len(recs), " & ".join(cells)))
        best = sorted(means, key=means.get)
        print("  %-38s %2d runs, exact kernel solve: %s, lowest %s %.3f then %s %.3f" % (
            name, len(recs), m.get("exact"), best[0], means[best[0]], best[1], means[best[1]]))
    L += ["\\bottomrule", "\\end{tabular}"]
    write("table_corpora.tex", L)


if __name__ == "__main__":
    losses = oco2_losses()
    oco2_ensembles(losses)
    corpora()
