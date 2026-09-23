"""Regenerate supported supplementary tables from public JSON records.

Default output is paper/. A table whose input records are missing is left as it is and reported,
never rebuilt from fewer runs. The transmission comparison uses the float64 records of seeds 104--107.
Use code/format_publication_tables.py after this command for readable table layouts.
"""
import glob, io, json, os, re, sys
import numpy as np

W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(W, "results"); OUT = os.path.join(W, "paper")


def load(p):
    return json.load(io.open(p, encoding="utf-8")) if os.path.exists(p) else None


def write(name, body, pending=False):
    p = os.path.join(OUT, name)
    if pending:
        if not os.path.exists(p):
            raise FileNotFoundError(f"No public inputs or retained table for {name}")
        print("retained historical aggregate (inputs absent): " + name)
        return
    io.open(p, "w", encoding="utf-8", newline="\n").write(body)
    print("wrote   " + name)


def tabular(cols, header, rows, mid=None):
    s = "\\begin{tabular}{" + cols + "}\n\\toprule\n" + header + " \\\\\n\\midrule\n"
    for i, r in enumerate(rows):
        if mid and i in mid:
            s += "\\midrule\n"
        s += " & ".join(r) + " \\\\\n"
    return s + "\\bottomrule\n\\end{tabular}\n"


def msd(vals, digits=3, scale=1.0):
    v = np.array(vals, float) * scale
    if len(v) == 0:
        return "--"
    if len(v) == 1:
        return f"{v[0]:.{digits}f}"
    return f"{v.mean():.{digits}f}$\\pm${v.std(ddof=1):.{digits}f}"


FAM_NAMES = {"krr": "isotropic kernel", "ard": "input-scaled kernel", "dnn": "network", "dnn_ens": "network ensemble", "dnn_corr": "network + residual kernel",
             "ens_corr": "ensemble + residual kernel", "dkr": "kernel on features", "dkr_cat": "kernel on concatenated features",
             "select": "coordinatewise selection", "stack": "convex stack"}

# ---- 1. transmission-conditioned retrieval (E1) ----
tc_files = sorted(glob.glob(os.path.join(RES, "tc", "*_tc.json")))
if tc_files:
    per_fam = {}
    seeds = []
    for f in tc_files:
        d = load(f)
        if d["seed"] not in (104, 105, 106, 107):
            print("excluded from common float64 cohort: " + os.path.basename(f))
            continue
        if d.get("float32_opt_in") or any(dt != ["float64"] for dt in d["prediction_dtypes"].values()):
            raise ValueError(f"Precision mismatch in common cohort: {f}")
        seeds.append(d["seed"])
        for fam, v in d["families"].items():
            x = v["rhos"].get("0.7")
            if not x:
                continue
            per_fam.setdefault(fam, []).append(dict(epsR=v["epsR"], nu2=v["profile_exponent_2nu_observed_range"], raw95=x["raw"]["p95_abs"], raw99=x["raw"]["p99_abs"],
                                                  con95=x["constrained"]["p95_abs"], con99=x["constrained"]["p99_abs"], fail=x["constrained"]["failure_fraction"],
                                                  clipR=x["constrained"]["clipped_at_R_fraction"], emin=x["E_min_R2_eR2_over_t2"], viol=x["pointwise_bound_violations"],
                                                  ratio=x["bound_middle_inf_over_mse"]))
    rows = []
    for fam in ("krr", "dnn", "dnn_ens", "dnn_corr", "ens_corr", "dkr", "dkr_cat", "select", "stack"):
        L = per_fam.get(fam)
        if not L:
            continue
        g = lambda k, dg=3, sc=1.0: msd([r[k] for r in L if r[k] is not None], dg, sc)
        rows.append([FAM_NAMES[fam], g("epsR", 4), g("nu2", 2), g("raw95", 3) + " / " + g("raw99", 2), g("con95", 3) + " / " + g("con99", 2),
                     g("fail", 2, 100) + " / " + g("clipR", 2, 100), g("emin", 3), str(sum(r["viol"] for r in L)), g("ratio", 1)])
    hdr = ("family & $\\varepsilon_R$ & $2\\nu$ (obs.) & raw inverse p95 / p99 & constrained p95 / p99 & failures / clipped at $R$ [\\%] "
           "& $\\E\\min\\{R^2,e_R^2/t^2\\}$ & violations & bound / MSE")
    cap = f"% common float64 cohort; seeds {sorted(set(seeds))}; four records per head\n"
    write("table_v2_transmission.tex", cap + tabular("lcccccccc", hdr, rows))
else:
    write("table_v2_transmission.tex", "", pending=True)

# ---- 2. the three-member replication (E2a): dkr, dkr_cat, stack per seed ----
e2a = sorted(glob.glob(os.path.join(RES, "e2a", "emit_s*_big3.json")))
if e2a:
    rows = []; agg = {}
    for f in e2a:
        d = load(f); fam = d["families"]
        seed = d["seed"]
        cells = [str(seed)]
        for k in ("dkr", "dkr_cat", "stack", "dnn_corr"):
            m = fam.get(k)
            if m:
                cells.append(f"{100*m['rel_l2_radiance']:.4f} / {100*m['refl_mae_median']:.3f} / {100*m['refl_p95_abs']:.2f}")
                agg.setdefault(k, []).append((m["rel_l2_radiance"], m["refl_mae_median"], m["refl_p95_abs"]))
            else:
                cells.append("--")
        if "dkr" in fam and "dkr_cat" in fam:
            cells.append(f"{100*(fam['dkr']['rel_l2_radiance'] - fam['dkr_cat']['rel_l2_radiance']):+.4f}")
        rows.append(cells)
    mean = ["mean"] + [(f"{msd([a[0] for a in agg[k]], 4, 100)} / {msd([a[1] for a in agg[k]], 3, 100)} / {msd([a[2] for a in agg[k]], 2, 100)}" if k in agg else "--") for k in ("dkr", "dkr_cat", "stack", "dnn_corr")]
    if "dkr" in agg and "dkr_cat" in agg:
        mean.append(msd([100 * (a[0] - b[0]) for a, b in zip(agg["dkr"], agg["dkr_cat"])], 4))
    rows.append(mean)
    hdr = "seed & kernel on features & kernel on concatenated features & convex stack & network + residual kernel & gain of concatenation (radiance points)"
    write("table_v2_replication.tex", "% cells: radiance rel. L2 [%] / median reflectance error [points] / within-split 95th percentile [points]\n" + tabular("lccccc", hdr, rows, mid={len(rows) - 1}))
else:
    write("table_v2_replication.tex", "", pending=True)

# ---- 3. the stacks (E3) ----
st_files = sorted(glob.glob(os.path.join(RES, "stack", "*_stack.json")))
if st_files:
    per = {}
    for f in st_files:
        d = load(f)
        for h, v in d["singles"].items():
            per.setdefault(("single", h), []).append((v["rel_l2_radiance_0.7"], v["epsR"], v["rhos"]["0.7"]["constrained_p95"], v["rhos"]["0.7"]["constrained_p99"], v["rhos"]["0.7"]["raw_p95"]))
        for arm, dd in d["arms"].items():
            for key, v in dd.items():
                per.setdefault((arm, key), []).append((v["rel_l2_radiance_0.7"], v["epsR"], v["rhos"]["0.7"]["constrained_p95"], v["rhos"]["0.7"]["constrained_p99"], v["rhos"]["0.7"]["raw_p95"]))
    rows = []
    order = [k for k in per if k[0] == "single"] + [k for k in per if k[0] == "paper_norm"] + [k for k in per if k[0] == "paper"] + [k for k in per if k[0] == "theorem"]
    for k in order:
        L = per[k]
        if k[0] == "theorem":
            name = ("stack, unweighted component square" if k[1] == "inf"
                    else "stack, $J'_\\tau$, $u=" + k[1] + "$")
        else:
            name = {"single": FAM_NAMES.get(k[1], k[1]),
                    "paper_norm": "stack, mean relative norm",
                    "paper": "stack, row-relative squared error"}[k[0]]
        rows.append([name, msd([a[0] for a in L], 4, 100), msd([a[1] for a in L], 4), msd([a[2] for a in L], 3) + " / " + msd([a[3] for a in L], 2), msd([a[4] for a in L], 3)])
    hdr = "predictor & radiance rel.\\ $L^2$ [\\%] & $\\varepsilon_R$ & constrained p95 / p99 & raw inverse p95"
    write("table_v2_stacks.tex", f"% lanes {len(st_files)}\n" + tabular("lcccc", hdr, rows))
else:
    write("table_v2_stacks.tex", "", pending=True)

# ---- 4. synthetic reference ----
S = load(os.path.join(RES, "sharpness_synthetic.json"))
if S and "profile_family" in S["betas"][0]:
    rows = []
    for r in S["betas"]:
        rows.append([f"{r['beta']:g}", f"{r['slope_mse_vs_eps2_average']:.3f}", f"{r['slope_predicted_average']:.3f}", f"{r['slope_mse_vs_eps_uniform']:.3f}",
                     f"{r['slope_analytic_uniform_same_ladder']:.3f}", f"{r['max_rel_diff_uniform_mc_vs_analytic']:.1e}", f"{r['max_rel_diff_profile_mc_vs_analytic']:.1e}", str(r["pointwise_bound_violations"])])
    hdr = "$\\beta$ & slope, MSE vs $\\varepsilon_R^2$ & $\\beta/(\\beta+2)$ & slope, uniform family & analytic slope, same ladder & MC vs analytic (uniform) & MC vs analytic (profile) & violations"
    write("table_v2_synthetic.tex", tabular("lccccccc", hdr, rows))
else:
    write("table_v2_synthetic.tex", "", pending=True)

# ---- 5. perturbation bounds: state inputs now, features when the dumps arrive ----
K = load(os.path.join(RES, "kernel_perturbation_Y2_n2000.json"))
fp = sorted(glob.glob(os.path.join(RES, "ridge", "*_featpert.json")))
rows = []
if K is None:
    retained = os.path.join(OUT, "table_v2_perturbation.tex")
    if not os.path.exists(retained):
        raise FileNotFoundError("State-input perturbation JSON and retained table are both absent")
    for line in io.open(retained, encoding="utf-8"):
        if line.startswith("state inputs &"):
            rows.append([cell.strip() for cell in line.strip().removesuffix(r"\\").split(" & ")])
    if len(rows) != 14:
        raise ValueError("Expected fourteen archived state-input perturbation rows")
    print("retained state-input perturbation rows: generating JSON absent")
if K and K["systems"] and "bound_b_median_ratio" in K["systems"][0]:
    for s in K["systems"]:
        if s["mean"] != "zero":
            continue
        f = lambda k: "--" if s.get(k) is None else f"{s[k]:.3g}"
        rows.append(["state inputs", s["system"].replace("_", "\\_"), f"{s['lambda_min_B']:+.3f}", f"{s['delta']:.3f}", f"{s['median_move']:.3g}", f("bound_b_median_ratio"), f("bound_c_median_ratio"), f("bound_v3_median_ratio"), f("bound_d_median_ratio")])
for fpath in fp:
    d = load(fpath)
    for s in d["systems"]:
        f = lambda k: "--" if s.get(k) is None else f"{s[k]:.3g}"
        rows.append([f"features {d['component']}", s["system"].replace("_", "\\_"), f"{s['lambda_min_B']:+.3f}", f"{s['delta']:.3f}", f"{s['median_move']:.3g}", f("bound_b_median_ratio"), f("bound_c_median_ratio"), f("bound_v3_median_ratio"), f("bound_d_median_ratio")])
if rows:
    hdr = "kernel & perturbed system & $\\lambda_{\\min}(B)$ & $\\|B\\|$ & median movement & residual-action / movement & signed-spectrum / movement & $\\delta<1$ bound / movement & ridge bound / movement"
    write("table_v2_perturbation.tex", tabular("llccccccc", hdr, rows))
else:
    write("table_v2_perturbation.tex", "", pending=True)

# ---- 6. ridge path (E6) ----
rp = sorted(glob.glob(os.path.join(RES, "ridge", "*_ridge.json")))
if rp:
    rows = []
    for f in rp:
        d = load(f)
        for kname, rec in d["kernels"].items():
            for L in rec["ladder"]:
                if L["gamma"] in (1e-8, 1e-6, 1e-4, 1e-2):
                    rows.append([d["component"], kname, f"{L['gamma']:g}", f"{L['dof']:.0f}", f"{L['bias_term']:.3g}", f"{L['noise_factor']:.3g}", f"{L['a2_median']:.3g}", f"{L['power_fn_median']:.3g}",
                                 f"{100*L['corrected_test_rel_phys']:.4f}", f"{100*L['kernel_alone_test_rel_phys']:.4f}", f"{100*L['same_fit_halfB_rel_pca']:.3f} / {100*L['split_fit_halfB_rel_pca']:.3f}"])
    hdr = "component & kernel & $\\gamma$ & $\\operatorname{tr}S_\\lambda$ & bias term & noise factor & median $\\|a_\\lambda(x)\\|^2$ & median $\\widetilde P_\\lambda$ & corrected test [\\%] & kernel alone [\\%] & half-sample, same / split fit [\\%]"
    write("table_v2_ridge.tex", tabular("llccccccccc", hdr, rows))
else:
    write("table_v2_ridge.tex", "", pending=True)

def _lane_sigma(tag, heads):
    """Per-coefficient target standard deviations of a benchmark lane, recovered from the heads.

    Each head records its per-coefficient RMSE and the mean normalised RMSE
    nrmse_mean = mean_j rmse_j / sigma_j, and every head of a lane shares the same sigma_j, so the
    heads over-determine 1/sigma_j. The solution is checked against the variance-weighted R^2 the
    same records store, which was computed independently of nrmse."""
    import numpy as np
    rec = load(os.path.join(RES, "pkanrtm", tag + ".json"))
    if not rec or "results" not in rec:
        return None
    hs = [h for h in heads if h in rec["results"] and "rmse_by_coef" in heads[h]
          and "test_nrmse_mean" in rec["results"][h]]
    if len(hs) < 3:
        return None
    coef = list(heads[hs[0]]["rmse_by_coef"])
    A = np.array([[heads[h]["rmse_by_coef"][c] for c in coef] for h in hs]) / float(len(coef))
    b = np.array([rec["results"][h]["test_nrmse_mean"] for h in hs])
    u, *_ = np.linalg.lstsq(A, b, rcond=None)
    if np.max(np.abs(A @ u - b) / np.abs(b)) > 1e-4:
        return None
    sigma = 1.0 / u
    for h in hs:
        mse = np.array([heads[h]["rmse_by_coef"][c] for c in coef]) ** 2
        if abs(1.0 - mse.sum() / (sigma ** 2).sum() - heads[h]["r2"]) > 1e-4:
            return None
    return coef, sigma


def _uniform_r2(sig, m):
    """R^2 averaged over the coefficients with equal weight, the benchmark scorer's convention."""
    import numpy as np
    coef, sigma = sig
    mse = np.array([m["rmse_by_coef"][c] for c in coef]) ** 2
    return float(1.0 - np.mean(mse / sigma ** 2))


# ---- 7. pKANrtm matched benchmark (E5) ----
P = load(os.path.join(RES, "pkanrtm", "rescored_pkanrtm.json"))
# the standard-split lanes were rerun in block E5b (their first run was killed for memory) and rescored separately;
# the two rescore files carry disjoint lane sets and the same published rows, so their lanes are merged here
P_std = load(os.path.join(RES, "pkanrtm", "rescored_pkanrtm_std.json"))
if P_std and P_std.get("lanes"):
    if not P:
        P = P_std
    else:
        P = dict(P); P["lanes"] = dict(P.get("lanes", {})); P["lanes"].update(P_std["lanes"])
        P.setdefault("published", {}).update(P_std.get("published", {}))
if P and P.get("lanes"):
    rows = []; groups = {}
    for tag, lane in P["lanes"].items():
        split = "OOD" if "_ood" in tag else "standard"; lowfi = "with 6S inputs" if lane["lowfi"] else "state only"
        sig = _lane_sigma(tag, lane["heads"])
        if sig is None or not np.all(np.isfinite(sig[1]) & (sig[1] > 0)):
            raise ValueError(f"Cannot reconstruct uniform-average R2 for {tag}; refusing convention fallback")
        for h, m in lane["heads"].items():
            r2 = _uniform_r2(sig, m)
            groups.setdefault((split, lowfi, h), []).append((m["rmse"], m["mae"], r2, m["smape_pct"]))
    for (split, lowfi, h), L in sorted(groups.items()):
        rows.append([split, lowfi, h.replace("_", "\\_"), msd([a[0] for a in L], 5), msd([a[1] for a in L], 5), msd([a[2] for a in L], 5), msd([a[3] for a in L], 3)])
    pub = P.get("published", {})
    for k in ("pKANrtm_standard", "pKANrtm_OOD"):
        if k in pub:
            v = pub[k]; rows.append([k.split("_")[1], "published", "pKANrtm", f"{v['rmse']:.5f}", f"{v['mae']:.5f}", f"{v['r2']:.5f}", f"{v['smape_pct']:.3f}"])
    hdr = "split & inputs & head & RMSE & MAE & $R^2$ & SMAPE [\\%]"
    write("table_v2_pkanrtm.tex", tabular("lllcccc", hdr, rows))
else:
    write("table_v2_pkanrtm.tex", "", pending=True)

# ---- 8. OCO-2 selections (E4) ----
oc = sorted(glob.glob(os.path.join(RES, "oco2", "*_select.json")))
if oc:
    groups = {}
    for f in oc:
        d = load(f)
        for rule, v in d["rules"].items():
            groups.setdefault((d["band"], rule), []).append((v["reduced"], v["radiance"]))
        for h, v in d["singles"].items():
            groups.setdefault((d["band"], "single " + h), []).append((v["reduced"], v["radiance"]))
    rows = [[b, r.replace("_", "\\_"), msd([a[0] for a in L], 2), msd([a[1] for a in L], 4)] for (b, r), L in sorted(groups.items())]
    hdr = "band & rule & reduced rel.\\ $L^2$ [\\%] & radiance rel.\\ $L^2$ [\\%]"
    write("table_v2_oco2.tex", tabular("llcc", hdr, rows))
else:
    write("table_v2_oco2.tex", "", pending=True)

# ---- 9. the multi-kernel ablation (E2b): one lane per component and seed, test error in physical units [%] ----
MK_NAMES = [(0, "krr_matern12", "Mat\\'ern-1/2"), (0, "krr_matern32", "Mat\\'ern-3/2"), (0, "krr_matern52", "Mat\\'ern-5/2"), (0, "krr_rbf", "Gaussian"),
            (0, "krr_sm4", "spectral mixture, four components"), (0, "krr_nngp", "NNGP, ReLU"), (0, "krr_ntk", "NTK, ReLU"), (0, "krr_add", "additive Mat\\'ern-5/2"),
            (0, "krr_ard_kf", "ARD by kernel flow"), (0, "krr_ard_eb", "ARD by marginal likelihood"), (1, "mkl_sum", "convex kernel sum"),
            (2, "mlp", "network"), (2, "mlp_resid_matern52", "network + residual kernel"), (2, "dkr_matern52", "kernel on features, Mat\\'ern-5/2"),
            (2, "dkr_rbf", "kernel on features, Gaussian"), (2, "dkr_ntk", "kernel on features, NTK"), (2, "dkr_ard_kf", "kernel on features, ARD"),
            (3, "select", "coordinatewise selection"), (3, "mkl_stack", "convex stack of the heads"), (3, "mkl_ridge", "ridge stack of the heads")]
mk = sorted(glob.glob(os.path.join(RES, "e2bc", "mk_emit_Y*_s*.json")))
if mk:
    per = {}; seeds = set()
    for f in mk:
        d = load(f); m = re.match(r"mk_emit_(Y\d)_s(\d+)$", d["tag"])
        if not m:
            continue
        seeds.add(d["seed"])
        for h, v in d["methods"].items():
            per.setdefault((h, m.group(1)), []).append(v["test"])
    comps = sorted({k[1] for k in per})
    rows = []; mid = set(); last_group = None
    for group, key, name in MK_NAMES:
        if not any((key, c) in per for c in comps):
            continue
        if last_group is not None and group != last_group:
            mid.add(len(rows))
        last_group = group
        rows.append([name] + [msd(per[(key, c)], 3) if (key, c) in per else "--" for c in comps])
    hdr = "predictor & " + " & ".join(f"${c[0]}_{c[1]}$" for c in comps)
    write("table_v2_multikernel.tex", f"% seeds {sorted(seeds)}; test relative L2 error in physical units [%]; a dash marks a lane that has not run\n" + tabular("l" + "c" * len(comps), hdr, rows, mid=mid))
else:
    write("table_v2_multikernel.tex", "", pending=True)

# ---- 10. the regulariser suite (E2c): three network arms, each alone, with the residual kernel and with the kernel on its features ----
rs = sorted(glob.glob(os.path.join(RES, "e2bc", "rs_emit_s*.json")))
if rs:
    per = {}; seeds = set()
    for f in rs:
        d = load(f); seeds.add(d["seed"])
        for h, v in d["methods"].items():
            per.setdefault(h, []).append((v["test"], v["test_rel_l2_radiance"]))
    def cell(h):
        L = per.get(h)
        return "--" if not L else msd([a[0] for a in L], 3, 100) + " / " + msd([a[1] for a in L], 3, 100)
    rows = [["isotropic kernel", cell("krr"), "--", "--"]]
    for arm, name in (("mlp_base", "network, control"), ("mlp_wd", "network, weight decay"), ("mlp_earlystop", "network, early stopping")):
        if arm in per:
            rows.append([name, cell(arm), cell(arm + "_resid"), cell(arm + "_dkr")])
    hdr = "predictor & alone & + residual kernel & kernel on features"
    write("table_v2_regsuite.tex", f"% seeds {sorted(seeds)}; cells: mean component rel. L2 [%] / radiance rel. L2 [%] on the test rows\n" + tabular("lccc", hdr, rows, mid={1}))
else:
    write("table_v2_regsuite.tex", "", pending=True)
