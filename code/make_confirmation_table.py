"""The fresh-partition evaluation table, from its report.

Reads results/confirmation/confirmation_report.json (written by results/confirmation/confirm.py) and writes
paper/table_confirmation.tex. The screened columns use the admissible domain of the protocol, t >= 1e-12 and
q = 1 - rho*Y4 >= 0.3 at rho = 0.7.

usage: python code/make_confirmation_table.py
"""
import json
import os

W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROWS = (("cubic_ridge", "cubic ridge"),
        ("fc_dnn_512", "network $3\\times512$"),
        ("dnn_plus_residual_krr", "network + residual kernel"),
        ("krr_ard_matern", "input-scaled Mat\\'ern kernel"),
        ("dkr_feature_kernel", "kernel on learned features"),
        ("convex_stack", "convex stack"))

with open(os.path.join(W, "results", "confirmation", "confirmation_report.json"), encoding="utf-8") as f:
    rep = json.load(f)
fam = rep["families"]
if sorted(fam) != sorted(k for k, _ in ROWS):
    raise SystemExit(f"report families {sorted(fam)} differ from the table rows")
lines = ["\\begin{tabular}{lrrrrrr}", "\\toprule",
         "family & radiance rel.\\ $L_2$ [\\%] & \\multicolumn{3}{c}{all bands [pp]} & "
         "\\multicolumn{2}{c}{screened [pp]} \\\\",
         "\\cmidrule(lr){3-5}\\cmidrule(lr){6-7}",
         " & & median & $p_{95}$ & $p_{99}$ & median & $p_{95}$ \\\\",
         "\\midrule"]
for key, name in ROWS:
    r = fam[key]
    a, s = r["all_bands"], r["admissible"]
    lines.append(f"{name} & {r['radiance_pct']:.4f} & {a['median_pp']:.3f} & {a['p95_pp']:.2f} & {a['p99_pp']:.1f} & "
                 f"{s['median_pp']:.3f} & {s['p95_pp']:.2f} \\\\")
lines += ["\\bottomrule", "\\end{tabular}"]
with open(os.path.join(W, "paper", "table_confirmation.tex"), "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(lines) + "\n")
h = rep["H1"]
print(json.dumps({"n_confirmation": rep["n_confirmation"], "n_train": rep["n_train"],
                  "admissible_coverage_pct": rep["admissible_coverage_pct"], "H1": h}))
