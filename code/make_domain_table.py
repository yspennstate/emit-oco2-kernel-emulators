"""The table of entries outside the physical domain, from the audit of the tabulated truth.

Reads results/confirmation/domain_audit.json (written by results/confirmation/domain_audit.py, whose printed output
is domain_audit.log beside it) and writes paper/table_domain_audit.tex.

usage: python code/make_domain_table.py
"""
import json
import os

W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(W, "results", "confirmation", "domain_audit.json"), encoding="utf-8") as f:
    a = json.load(f)
n = a["entries"]
if n != a["states"] * a["bands"]:
    raise SystemExit("entries != states x bands")
adm = round(a["admissible_frac"] * n)
rows = (("transmitted flux $t=Y_2+Y_3 \\le 0$", a["t_le_0"]),
        ("$t < 10^{-12}$", a["t_lt_1e_12"]),
        ("albedo $Y_4 < 0$", a["albedo_lt_0"]),
        ("albedo $Y_4 \\ge 1$", a["albedo_ge_1"]),
        ("albedo exactly $2$ (reported fill marker)", a["albedo_exact_2"]))
lines = ["\\begin{tabular}{lrr}", "\\toprule", "condition on the tabulated truth & entries & share \\\\", "\\midrule"]
lines += [f"{name} & {k:,} & ${100 * k / n:.4f}\\%$ \\\\" for name, k in rows]
lines += ["\\midrule", f"admissible $\\mathcal{{A}}=\\{{t>0,\\ 0\\le Y_4<1\\}}$ & {adm:,} & ${100 * adm / n:.4f}\\%$ \\\\",
          "\\bottomrule", "\\end{tabular}"]
with open(os.path.join(W, "paper", "table_domain_audit.tex"), "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(lines) + "\n")
print(json.dumps({"entries": n, "admissible": adm, "outside": n - adm, "jld2_sha256": a["jld2_sha256"]}))
