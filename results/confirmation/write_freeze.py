"""Freeze the development-selected hyperparameters, then hash the freeze.

Run after the development chain and before confirm.py. Writes freeze.json and
freeze.sha256; confirm.py refuses to run unless they agree.
"""
import json
import hashlib

R = json.load(open("results_top_models.json", encoding="utf-8"))
X = json.load(open("dev_hyperparams.json", encoding="utf-8"))
AR = json.load(open("refined_ard.json", encoding="utf-8"))
COMPONENTS = ["Y1", "Y2", "Y3", "Y4"]

freeze = {
    "source": "development run, seed %d" % R["seed"],
    "cubic_ridge": {"lambda": X["cubic_lambda"]},
    "krr_ard_matern": {
        c: {"length_scales": AR["per_component"][c]["length_scales"],
            "lambda": AR["per_component"][c]["lambda"]}
        for c in COMPONENTS
    },
    "fc_dnn_512": {"width": 512, "max_epochs": 400, "patience": 60,
                   "lr": 1e-3, "batch_size": 1024},
    "dnn_plus_residual_krr": {
        c: {"length_scales": X["residual_kernel"][c]["length_scales"],
            "lambda": X["residual_kernel"][c]["lambda"]}
        for c in COMPONENTS
    },
    "dkr_feature_kernel": {"multiplier": X["dkr_multiplier"], "lambda": 1e-8},
    "convex_stack": {"members": X["stack_members"], "weights": X["stack_weights"]},
}
raw = json.dumps(freeze, indent=2, sort_keys=True).encode("utf-8")
open("freeze.json", "wb").write(raw)
h = hashlib.sha256(raw).hexdigest()
open("freeze.sha256", "w", encoding="utf-8").write(h + "  freeze.json\n")
print("freeze.json sha256", h)
for k in freeze:
    print(" ", k)
