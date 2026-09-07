"""LaTeX table fragments for the radiative-transfer paper from the p2_analyze summary.
  python p2_tables.py <summary.json> <out_dir>
Writes table_emit_seeds.tex (ten-seed EMIT family table), table_emit_curve.tex (learning curve),
table_emit_rank.tex (output-rank ablation, if present), table_oco2_seeds.tex (OCO-2 at the full
block) and table_oco2_curve.tex (OCO-2 learning curve). Every cell is mean +- sd over seeds.
"""
import json, os, sys

summ = json.load(open(sys.argv[1], encoding="utf-8"))
out = sys.argv[2]
os.makedirs(out, exist_ok=True)
EM_NAMES = [("ridge3", "Cubic ridge"), ("krr4k", "Mat\\'ern KRR, 4000-point fit"), ("krr", "Mat\\'ern KRR, exact on all rows"),
            ("ard", "Mat\\'ern KRR, per-input scales"), ("dnn", "FC-DNN (3$\\times$512)"), ("dnn_ens", "Five-network mean"),
            ("dnn_corr", "DNN + residual KRR"), ("ens_corr", "Ensemble + residual KRR"), ("dkr", "Kernel on network features"),
            ("select", "Per-coordinate selection"), ("stack", "Convex stack")]


def cell(g, k, dp=2):
    if k not in g:
        return "--"
    return ("%%.%df$\\pm$%%.%df" % (dp, dp)) % (g[k]["mean"], g[k]["sd"])


def emit_table(g, path, caption_note):
    lines = ["\\begin{tabular}{lcccccc}", "\\toprule",
             " & \\multicolumn{4}{c}{component rel.\\ $L^2$ [\\%]} & radiance & reflectance \\\\",
             "\\cmidrule(lr){2-5}",
             "model & $Y_1$ & $Y_2$ & $Y_3$ & $Y_4$ & rel.\\ $L^2$ [\\%] & med.\\ $|\\hat\\rho-\\rho|$ [\\%] \\\\", "\\midrule"]
    for key, name in EM_NAMES:
        if key not in g:
            continue
        r = g[key]
        lines.append("%s & %s & %s & %s & %s & %s & %s \\\\" % (
            name, cell(g, key, 2) if False else "%.2f$\\pm$%.2f" % (r["rel_l2_Y1"]["mean"], r["rel_l2_Y1"]["sd"]),
            "%.2f$\\pm$%.2f" % (r["rel_l2_Y2"]["mean"], r["rel_l2_Y2"]["sd"]),
            "%.2f$\\pm$%.2f" % (r["rel_l2_Y3"]["mean"], r["rel_l2_Y3"]["sd"]),
            "%.2f$\\pm$%.2f" % (r["rel_l2_Y4"]["mean"], r["rel_l2_Y4"]["sd"]),
            "%.3f$\\pm$%.3f" % (r["rel_l2_radiance"]["mean"], r["rel_l2_radiance"]["sd"]),
            "%.3f$\\pm$%.3f" % (r["refl_mae_median"]["mean"], r["refl_mae_median"]["sd"])))
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("wrote", path, caption_note)


emit = summ.get("emit", {})
full_key = [k for k in emit if k.startswith("full block")]
if full_key:
    emit_table(emit[full_key[0]], os.path.join(out, "table_emit_seeds.tex"), "(ten fresh seeds)")

curve_keys = sorted([k for k in emit if k.startswith("n=")], key=lambda k: int(k[2:]))
if curve_keys:
    fams = [("ridge3", "Cubic ridge"), ("krr", "Exact KRR"), ("dnn", "FC-DNN"), ("dnn_corr", "DNN + residual KRR"), ("dkr", "Kernel on features")]
    lines = ["\\begin{tabular}{l" + "c" * (len(curve_keys) + 1) + "}", "\\toprule",
             "model & " + " & ".join(["$N=%d$" % int(k[2:]) for k in curve_keys] + (["full block"] if full_key else [])) + " \\\\", "\\midrule"]
    for key, name in fams:
        cells = []
        for k in curve_keys + (full_key[:1] if full_key else []):
            g = emit[k].get(key)
            cells.append("%.2f$\\pm$%.2f" % (g["mean_rel_l2_components"]["mean"], g["mean_rel_l2_components"]["sd"]) if g else "--")
        lines.append("%s & %s \\\\" % (name, " & ".join(cells)))
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(out, "table_emit_curve.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("wrote table_emit_curve.tex")

rank_keys = sorted([k for k in emit if k.startswith("rank ")], key=lambda k: int(k.split()[1]))
if rank_keys:
    fams = [("krr", "Exact KRR"), ("dnn", "FC-DNN"), ("dnn_corr", "DNN + residual KRR"), ("dkr", "Kernel on features")]
    cols = rank_keys[:]
    lines = ["\\begin{tabular}{l" + "c" * (len(cols) + 1) + "}", "\\toprule",
             "model & " + " & ".join(["rank %s" % k.split()[1] for k in cols] + ["rank 64"]) + " \\\\", "\\midrule"]
    for key, name in fams:
        cells = []
        for k in cols + (full_key[:1] if full_key else []):
            g = emit[k].get(key)
            cells.append("%.2f$\\pm$%.2f" % (g["mean_rel_l2_components"]["mean"], g["mean_rel_l2_components"]["sd"]) if g else "--")
        lines.append("%s & %s \\\\" % (name, " & ".join(cells)))
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(out, "table_emit_rank.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("wrote table_emit_rank.tex")

oco = summ.get("oco2", {})
if oco:
    heads = [("kernel_flow", "Kernel-flow emulator (published)"), ("kernel_raw", "Mat\\'ern KRR, state"), ("kernel_ard", "Mat\\'ern KRR, relevance-scaled state"),
             ("mean_flat", "Residual MLP"), ("dkr_flat", "Kernel on MLP features"), ("mean_ens", "Three-network mean"), ("dkr_ens", "Kernel on ensemble features"),
             ("dkr_avg", "Mean of feature-kernel heads"), ("combined", "Per-coordinate selection")]
    bands = [("o2", "O$_2$"), ("wco2", "weak CO$_2$"), ("sco2", "strong CO$_2$")]
    for M in sorted({int(k.split("|")[2]) for k in oco}):
        keys = {b: "%s|18000|%d" % (b, M) for b, _ in bands}
        if not all(k in oco for k in keys.values()):
            continue
        lines = ["\\begin{tabular}{lcccccc}", "\\toprule",
                 " & \\multicolumn{2}{c}{O$_2$} & \\multicolumn{2}{c}{weak CO$_2$} & \\multicolumn{2}{c}{strong CO$_2$} \\\\",
                 "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
                 "head & reduced & radiance & reduced & radiance & reduced & radiance \\\\", "\\midrule"]
        for h, name in heads:
            if not all(h in oco[keys[b]] for b, _ in bands):
                continue
            cells = []
            for b, _ in bands:
                g = oco[keys[b]][h]
                cells.append("%.2f$\\pm$%.2f & %.4f$\\pm$%.4f" % (g["reduced"], g["reduced_sd"], g["radiance"], g["radiance_sd"]))
            lines.append("%s & %s \\\\" % (name, " & ".join(cells)))
        lines += ["\\bottomrule", "\\end{tabular}"]
        fn = "table_oco2_seeds.tex" if M == 1 else "table_oco2_ens.tex"
        open(os.path.join(out, fn), "w", encoding="utf-8").write("\n".join(lines) + "\n")
        print("wrote", fn)
    ns = sorted({int(k.split("|")[1]) for k in oco if k.endswith("|1")})
    if len(ns) > 1:
        lines = ["\\begin{tabular}{ll" + "c" * len(ns) + "}", "\\toprule",
                 "band & head & " + " & ".join("$N=%d$" % n for n in ns) + " \\\\", "\\midrule"]
        for b, bname in bands:
            for h, name in (("kernel_raw", "KRR, state"), ("kernel_ard", "KRR, scaled state"), ("mean_flat", "Residual MLP"), ("dkr_flat", "Kernel on features")):
                cells = []
                for n in ns:
                    g = oco.get("%s|%d|1" % (b, n), {}).get(h)
                    cells.append("%.2f$\\pm$%.2f" % (g["reduced"], g["reduced_sd"]) if g else "--")
                lines.append("%s & %s & %s \\\\" % (bname if h == "kernel_raw" else "", name, " & ".join(cells)))
        lines += ["\\bottomrule", "\\end{tabular}"]
        open(os.path.join(out, "table_oco2_curve.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
        print("wrote table_oco2_curve.tex")
