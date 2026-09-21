"""Check arithmetic in the retained one-split table, without inventing run records.

This is a table audit, not model training, raw-array verification or rescoring.
"""
from pathlib import Path
import hashlib
import json
import re
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def retained_rows(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    body = text.split(r"\midrule", 1)[1].split(r"\bottomrule", 1)[0]
    rows = []
    for line in body.splitlines():
        if " & " not in line:
            continue
        cells = [c.strip() for c in line.strip().removesuffix(r"\\").split(" & ")]
        if len(cells) != 6:
            raise ValueError("Expected one name and five numeric cells in the retained refit")
        values = [float(v) for v in cells[1:]]
        if not all(np.isfinite(values)):
            raise ValueError("The retained table contains a nonfinite numerical cell")
        rows.append(dict(zip(("model", "radiance_percent", "all_median_pp", "all_p95_pp",
                              "screened_median_pp", "screened_p95_pp"), [cells[0], *values])))
    if len(rows) != 6 or len({r["model"] for r in rows}) != 6:
        raise ValueError("Expected the six distinct retained refit rows")
    return rows


def audit() -> dict:
    path = ROOT / "paper/table_constrained_retrieval.tex"
    rows = retained_rows(path)
    feature = next(r for r in rows if r["model"] == "kernel on learned features")
    stack = next(r for r in rows if r["model"] == "convex stack")
    for row in rows:
        for key in ("radiance_percent", "all_p95_pp", "screened_p95_pp"):
            row[key + "_rank"] = 1 + sum(other[key] < row[key] for other in rows)
        row["p95_reduction_percent"] = 100 * (1 - row["screened_p95_pp"] / row["all_p95_pp"])
    network_errors = [100 * json.loads(p.read_text())["families"]["dnn"]["rel_l2_radiance"]
                      for p in sorted((ROOT / "results/emit").glob("emit_s*.json"))
                      if re.fullmatch(r"emit_s10[1-9]\.json|emit_s110\.json", p.name)]
    if len(network_errors) != 10:
        raise ValueError("Expected the complete main ten-split network records")
    return {
        "scope": "Arithmetic on retained rounded values; no fitting or sample-level rescoring",
        "source": str(path.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rows": rows,
        "stack_over_feature_p95_ratio_all": stack["all_p95_pp"] / feature["all_p95_pp"],
        "stack_over_feature_p95_ratio_screened": stack["screened_p95_pp"] / feature["screened_p95_pp"],
        "unchanged_displayed_medians": sum(r["all_median_pp"] == r["screened_median_pp"] for r in rows),
        "main_network_radiance_mean_percent": float(np.mean(network_errors)),
        "main_network_radiance_sample_sd_percent": float(np.std(network_errors, ddof=1)),
        "retained_refit_network_radiance_percent": next(r["radiance_percent"] for r in rows if r["model"].startswith("network $3")),
        "refit_seed": None, "refit_test_indices": None, "refit_hyperparameters": None,
        "refit_test_mask_coverage": None, "refit_failure_counts": None,
        "missing_metadata_note": "Not recoverable from these aggregate cells; whole-table coverage is not test-split coverage",
        "clean_target_training_completed": False,
    }


def main() -> None:
    out = ROOT / "results/review_20260921/retained_refit_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(audit(), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Wrote", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
