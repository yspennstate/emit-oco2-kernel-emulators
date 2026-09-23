"""Build the main EMIT tables from the per-split records.

Writes the main and wide-pipeline tables, the paired contrasts and the all-band retrieval tails, and a JSON
summary with the source digests. Sample SD uses ddof=1.
Run from any directory: python code/make_revision_tables.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEEDS = tuple(range(101, 111))
BASE_FILES = [ROOT / f"results/emit/emit_s{s}.json" for s in SEEDS]
WIDE_FILES = [ROOT / f"results/scaling/per_seed/emit_s{s}_wide.json" for s in SEEDS]
CAT_FILES = [ROOT / f"results/scaling/per_seed/emit_s{s}_cat.json" for s in SEEDS]
METRICS = ("rel_l2_Y1", "rel_l2_Y2", "rel_l2_Y3", "rel_l2_Y4", "rel_l2_radiance", "refl_mae_median")
NAMES = [("ridge3", "Cubic ridge"), ("krr4k", "Mat\\'ern KRR, 4000-point fit"),
         ("krr", "Mat\\'ern KRR, exact on all rows"), ("ard", "Mat\\'ern KRR, per-input scales"),
         ("dnn", "FC-DNN (3$\\times$512)"), ("dnn_ens", "Five-network mean"),
         ("dnn_corr", "DNN + residual KRR"), ("ens_corr", "Ensemble + residual KRR"),
         ("dkr", "Kernel on network features"), ("select", "Per-coordinate selection"), ("stack", "Convex stack")]


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values) -> dict[str, float | int]:
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Expected a nonempty finite vector")
    return {"n": int(len(x)), "mean": float(x.mean()), "sd": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
            "min": float(x.min()), "max": float(x.max())}


def values(records, family, metric):
    return np.array([records[s]["families"][family][metric] for s in SEEDS]) * 100


def cell(x, digits=3):
    d = stats(x)
    return f"{d['mean']:.{digits}f}$\\pm${d['sd']:.{digits}f}"


def write_table(filename, lines):
    (ROOT / "paper" / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    base = {s: read(p) for s, p in zip(SEEDS, BASE_FILES)}
    wide = {s: read(p) for s, p in zip(SEEDS, WIDE_FILES)}
    cat = {s: read(p) for s, p in zip(SEEDS, CAT_FILES)}
    reference_hashes = base[101]["data_sha"]
    for records, width in ((base, 512), (wide, 2000)):
        for s, record in records.items():
            if record["seed"] != s or record["widths"] != [width] * 3:
                raise ValueError(f"Unexpected configuration at seed {s}")
            if record["ntrain"] != 18884 or record["n_test"] != 2331 or record["n_val"] != 2098:
                raise ValueError("Unexpected split sizes")
            if record["data_sha"] != reference_hashes or record.get("smoke"):
                raise ValueError("Mismatching data provenance or a smoke run")
    # The shared deterministic rows are an additional pairing check.
    for s in SEEDS:
        for family in ("ridge3", "krr", "ard"):
            for metric in METRICS:
                if base[s]["families"][family][metric] != wide[s]["families"][family][metric]:
                    raise ValueError("Deterministic paired rows disagree")

    header = [r"\begin{tabular}{lcccccc}", r"\toprule",
              r" & \multicolumn{4}{c}{component rel.\ $L^2$ [\%]} & radiance & reflectance \\",
              r"\cmidrule(lr){2-5}",
              r"model & $Y_1$ & $Y_2$ & $Y_3$ & $Y_4$ & rel.\ $L^2$ [\%] & med.\ $|\hat\rho-\rho|$ [pp] \\", r"\midrule"]
    for fam, label in NAMES:
        header.append(label + " & " + " & ".join(cell(values(base, fam, m), 2 if j < 4 else 3)
                                                   for j, m in enumerate(METRICS)) + r" \\")
    write_table("table_emit_seeds.tex", header + [r"\bottomrule", r"\end{tabular}"])

    rows = [("dnn", "network"), ("dnn_corr", "network + residual KRR"),
            ("dkr", "kernel on its features"), ("select", "per-coordinate selection"), ("stack", "convex stack")]
    lines = [r"\begin{tabular}{llcccccc}", r"\toprule",
             r" & & \multicolumn{4}{c}{component rel.\ $L^2$ [\%]} & radiance & reflectance \\",
             r"\cmidrule(lr){3-6}",
             r"network & head & $Y_1$ & $Y_2$ & $Y_3$ & $Y_4$ & rel.\ $L^2$ [\%] & med.\ $|\hat\rho-\rho|$ [pp] \\", r"\midrule"]
    for records, width in ((base, 512), (wide, 2000)):
        for j, (fam, label) in enumerate(rows):
            name = f"3$\\times${width}" if j == 0 else ""
            lines.append(name + " & " + label + " & " + " & ".join(
                cell(values(records, fam, m), 2 if i < 4 else 3) for i, m in enumerate(METRICS)) + r" \\")
        if width == 512:
            lines.append(r"\midrule")
    write_table("table_emit_widepipe.tex", lines + [r"\bottomrule", r"\end{tabular}"])

    contrasts = [("Input-scaled / isotropic KRR", base, "ard", base, "krr"),
                 ("Residual correction / network", base, "dnn_corr", base, "dnn"),
                 ("Narrow stack / input-scaled KRR", base, "stack", base, "ard"),
                 ("Wide feature kernel / input-scaled KRR", wide, "dkr", base, "ard"),
                 ("Wide / narrow feature kernel", wide, "dkr", base, "dkr"),
                 ("Wide stack / wide feature kernel", wide, "stack", wide, "dkr")]
    paired = []
    lines = [r"\begin{tabular}{lrrr}", r"\toprule",
             r"candidate / reference & mean gain $\pm$ SD [pp] & gain range [pp] & wins/10 \\", r"\midrule"]
    for label, a, af, b, bf in contrasts:
        va, vb = values(a, af, "rel_l2_radiance"), values(b, bf, "rel_l2_radiance")
        delta = vb - va
        d = stats(delta)
        wins = int(np.sum(delta > 0))
        paired.append({"comparison": label, "gain_pp": d, "wins": wins,
                       "seed_gains_pp": delta.tolist(), "mean_relative_gain_pct": float(np.mean(100 * (1 - va / vb)))})
        lines.append(f"{label} & {cell(delta, 4)} & [{d['min']:.4f}, {d['max']:.4f}] & {wins} "+r"\\")
    write_table("table_emit_paired.tex", lines + [r"\bottomrule", r"\end{tabular}"])

    tail_rows = [("Cubic ridge", base, "ridge3"), ("Isotropic KRR", base, "krr"),
                 ("Input-scaled KRR", base, "ard"), ("Narrow feature kernel", base, "dkr"),
                 ("Narrow convex stack", base, "stack"),
                 ("Concatenated features, 5$\\times$512", cat, "dkr_cat"), ("Wide feature kernel", wide, "dkr"),
                 ("Wide convex stack", wide, "stack")]
    tail_metrics = ("rel_l2_radiance", "refl_mae_median", "refl_p95_abs", "refl_rmse")
    tails = {}
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"model & radiance [\%] & median [pp] & 95th percentile [pp] & RMSE [pp] \\", r"\midrule"]
    for label, records, fam in tail_rows:
        d = {m: stats(values(records, fam, m)) for m in tail_metrics}
        tails[label] = d
        rm = d["refl_rmse"]
        rmcell = f"{rm['mean']:.2e}"
        lines.append(label + " & " + " & ".join(cell(values(records, fam, m), 3 if j < 2 else 2)
                                                 for j, m in enumerate(tail_metrics[:3])) + " & " + rmcell + r" \\")
    write_table("table_emit_tails.tex", lines + [r"\bottomrule", r"\end{tabular}"])
    tail_contrasts = {}
    for label, records in (("narrow", base), ("wide", wide)):
        x, y = values(records, "stack", "refl_p95_abs"), values(records, "dkr", "refl_p95_abs")
        tail_contrasts[label] = {"stack_worse_splits": int(np.sum(x > y)),
                                "ratio_stack_to_feature": stats(x / y), "seed_ratios": (x / y).tolist()}

    # Report absent per-seed records instead of fabricating them from aggregate means.
    groups = {}
    for n in (500, 1000, 2000, 4000, 8000):
        for suffix in ("", "_ard"):
            group = f"n{n}{suffix}"
            expected = [f"results/scaling/per_seed/emit_s{s}_n{n}{suffix}.json" for s in SEEDS]
            groups[group] = {"expected": 10, "available": sum((ROOT / p).exists() for p in expected),
                             "missing": [p for p in expected if not (ROOT / p).exists()]}
    for suffix in ("w1000", "ft"):
        expected = [f"results/scaling/per_seed/emit_s{s}_{suffix}.json" for s in SEEDS]
        groups[suffix] = {"expected": 10, "available": sum((ROOT / p).exists() for p in expected),
                          "missing": [p for p in expected if not (ROOT / p).exists()]}
    sources = BASE_FILES + WIDE_FILES + CAT_FILES + [ROOT / f"results/scaling/{name}.json" for name in
                                       ("scaling_numbers", "retune_numbers", "width_numbers")]
    summary = {"analysis": "reanalysis of stored metrics, not new training or a raw-array rerun",
               "spread": "sample standard deviation (ddof=1); descriptive, not confidence intervals",
               "quantiles": "mean of within-split all-band quantiles, not a pooled quantile",
               "seeds": list(SEEDS), "data_sha": reference_hashes,
               "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sources},
               "paired_radiance": paired, "all_band_tails": tails,
               "tail_tradeoff": tail_contrasts, "individual_record_inventory": groups,
               "any_nonfinite_inverse_reported_in_core_records": bool(any(
                   rec["families"][fam]["refl_nan_frac"] > 0
                   for rec in list(base.values()) + list(wide.values()) for fam in rec["families"]))}
    output = ROOT / "results/revision_20260918"
    output.mkdir(exist_ok=True)
    (output / "reanalysis.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"paired_radiance": paired, "tail_tradeoff": tail_contrasts,
                      "individual_record_inventory": groups}, indent=2))


if __name__ == "__main__":
    main()
