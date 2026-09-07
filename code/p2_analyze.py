"""Aggregate the paper-2 DGX results (EMIT campaign + OCO-2 curve JSONs) into tables.
  python p2_analyze.py <results_dir> [out.json]
Prints mean +- sd over seeds per (family, metric) for the EMIT full-block lanes, the EMIT
learning curve, the rank ablation and the large-budget network, and per (band, n, head) for OCO-2.
"""
import glob, json, os, statistics as st, sys
from collections import defaultdict

R = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else None
emit, oco = [], []
for f in sorted(glob.glob(os.path.join(R, "*.json"))):
    d = json.load(open(f, encoding="utf-8"))
    if d.get("smoke"):
        continue
    (emit if d.get("kind") == "emit_campaign" else oco if d.get("kind") == "oco2_curve" else []).append(d)


def ms(v):
    v = list(v)
    return (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0, len(v))


summary = {"emit": {}, "oco2": {}}
EM_METRICS = ("mean_rel_l2_components", "rel_l2_Y1", "rel_l2_Y2", "rel_l2_Y3", "rel_l2_Y4", "rel_l2_radiance",
              "refl_mae_median", "refl_p95_abs_wellposed")


def emit_group(rows, label):
    fams = defaultdict(lambda: defaultdict(list))
    for d in rows:
        for fam, m in d["families"].items():
            for k in EM_METRICS:
                if k in m:
                    fams[fam][k].append(100 * m[k])
    if not fams:
        return
    print(f"\n== EMIT {label}: {len(rows)} lanes, seeds {sorted(d['seed'] for d in rows)}")
    print("%-10s %-22s %-14s %-14s %-14s %-14s %-14s %-14s" % ("family", "comps", "Y1", "Y2", "Y3", "Y4", "radiance", "refl_med"))
    g = {}
    for fam, mm in fams.items():
        cells = []
        g[fam] = {}
        for k in ("mean_rel_l2_components", "rel_l2_Y1", "rel_l2_Y2", "rel_l2_Y3", "rel_l2_Y4", "rel_l2_radiance", "refl_mae_median"):
            mu, sd, n = ms(mm[k]); g[fam][k] = dict(mean=mu, sd=sd, n=n)
            cells.append("%.3f+-%.3f" % (mu, sd))
        print("%-10s %s" % (fam, " ".join("%-14s" % c for c in cells)))
    summary["emit"][label] = g


full = [d for d in emit if d["ntrain"] > 15000 and d["pca_rank"] == 64 and d["widths"] == [512, 512, 512]]
emit_group(full, "full block, rank 64, 3x512")
for n in sorted({d["ntrain"] for d in emit if d["ntrain"] <= 15000}):
    emit_group([d for d in emit if d["ntrain"] == n and d["pca_rank"] == 64 and d["widths"] == [512, 512, 512]], f"n={n}")
for r in sorted({d["pca_rank"] for d in emit if d["pca_rank"] != 64}):
    emit_group([d for d in emit if d["pca_rank"] == r], f"rank {r}")
big = [d for d in emit if d["widths"] != [512, 512, 512]]
emit_group(big, "3x2000, 500 epochs")

if oco:
    tab = defaultdict(lambda: defaultdict(list))
    for d in oco:
        key = (d["band"], d["ntrain"], d["members"])
        for head, r in d["results"].items():
            tab[key][head + "|reduced"].append(r["reduced"])
            tab[key][head + "|radiance"].append(r["radiance"])
    print("\n== OCO-2 (reduced % / radiance %), mean+-sd over seeds")
    for key in sorted(tab):
        band, n, M = key
        seeds = sorted(d["seed"] for d in oco if (d["band"], d["ntrain"], d["members"]) == key)
        print(f"-- {band} n={n} members={M} seeds={len(seeds)}")
        g = {}
        for head in ("kernel_flow", "kernel_raw", "kernel_ard", "mean_flat", "dkr_flat", "mean_ens", "dkr_ens", "dkr_avg", "combined"):
            if head + "|reduced" not in tab[key]:
                continue
            mr, sr, _ = ms(tab[key][head + "|reduced"]); ma, sa, _ = ms(tab[key][head + "|radiance"])
            g[head] = dict(reduced=mr, reduced_sd=sr, radiance=ma, radiance_sd=sa)
            print("   %-12s %7.3f+-%.3f   %.4f+-%.4f" % (head, mr, sr, ma, sa))
        summary["oco2"]["%s|%d|%d" % key] = g
if OUT:
    json.dump(summary, open(OUT, "w"), indent=1)
    print("wrote", OUT)
