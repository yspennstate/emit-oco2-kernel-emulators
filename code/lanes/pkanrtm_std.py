# pKANrtm, the six STANDARD-split lanes only, rerun with a memory-lean kernel tuner (plan v4, block E5b, 2026-09-21).
# Why: in block E5 every official-split lane died with rc -9 (the container's out-of-memory kill) right after the
# ensemble MLP, while every out-of-distribution lane completed. The term that differs between the splits is the
# kernel tuner's validation-by-subsample distance matrix: krr_fit_predict materializes sqd(Fva, Fs) and m52 of it
# in float64, 2 x n_va x 6000 x 8 bytes, which is 6.1 GB at the official split's ~64,000 validation rows against
# 4.4 GB at the OOD split's ~46,000, on top of the networks' features (426,761 x 384 x 8 bytes = 1.3 GB, twice
# during the member loop) and the CUDA host context. The patch below computes the validation kernel rows in
# 4,000-row chunks against the per-nugget solves, so that term becomes 0.4 GB transient and the peak stays under
# half of a 13 GB session; the arithmetic is row-wise identical to the unchunked product. The peak RSS of every
# lane is printed so the next run is sized from a measurement, not from this estimate.
# Downloads the two paired jsonls from Hugging Face (public dataset mazid-rafee/pKANrtm, Mazid and Rishe 2026,
# arXiv 2605.10958), lets bench_data build paired_arrays.npz, then runs bench_run.py on (a) the release's standard
# state split (35,000 / 7,500 / 7,500 states, --pkan_split official) at three seeds and (b) the paper's
# out-of-distribution split ood_aod_cwv, reconstructed from the state variables as the paper defines it (test =
# states in the upper 0.85 quantile of aod550 OR of cwv_cm, the fallback the paper reports as used; the rest 85/15
# train/val by a fixed draw) at three initialisations, each with the paper's categorical inputs (--pkan_cats 1) and,
# in a second arm, the 6S prediction as an input and as the mean of a residual kernel (--lowfi 1). Kernel families
# use the chunked Nystrom path (6,000 landmarks; the exact solve is capped at 20,000 rows). The published
# RMSE/MAE/R2/SMAPE are compared by rescore_saved.py on the saved predictions. Every child failure fails the session.
import glob, os, subprocess, sys, time, json, shutil, urllib.request

BUDGET_H = float(os.environ.get("PK_BUDGET_H", "4.0"))
SEEDS = [int(s) for s in os.environ.get("PK_SEEDS", "0,1,2").split(",")]
FAM = os.environ.get("PK_FAMILIES", "ridge,krr,mlp,mlp_ens,mlp_resid,dkr,select,stack")
t_all = time.time()

def find(pattern):
    h = sorted(glob.glob(pattern, recursive=True)); return h[0] if h else None
code_root = os.path.dirname(os.path.dirname(find("/kaggle/input/**/bench_run.py")))
work = "/kaggle/working/nmkc"
if not os.path.exists(work):
    shutil.copytree(code_root, work)
dgx = os.path.join(work, "dgx")
out = "/kaggle/working/results"; os.makedirs(out, exist_ok=True)
data_new = "/kaggle/working/data_new"; pk = os.path.join(data_new, "pkanrtm"); os.makedirs(pk, exist_ok=True)
base = "https://huggingface.co/datasets/mazid-rafee/pKANrtm/resolve/main/qavalid_intersection_libradtran_6s_50k_13b/"
for f in ("dataset_rows_6s.jsonl", "dataset_rows_libradtran.jsonl"):
    dst = os.path.join(pk, f)
    if not os.path.exists(dst):
        t0 = time.time(); print("downloading", f, flush=True)
        urllib.request.urlretrieve(base + f, dst)
        print(f, round(os.path.getsize(dst) / 1e6, 1), "MB in %.0fs" % (time.time() - t0), flush=True)
env = dict(os.environ); env.update(dict(P2_OUT=out, NMKC_THREADS="4", DATA_NEW=data_new, CUDA_VISIBLE_DEVICES="0", PYTHONUNBUFFERED="1"))
py = sys.executable


def patch(path, reps, marker):
    s = open(path, encoding="utf-8", newline="").read().replace("\r\n", "\n")
    if marker in s:
        return "already patched"
    for old, new in reps:
        n = s.count(old)
        if n != 1:
            return f"PATCH ABORTED: pattern seen {n} times: {old[:70]!r}"
        s = s.replace(old, new)
    s = f"# {marker} applied at runtime on Kaggle\n" + s
    open(path, "w", encoding="utf-8").write(s)
    return "patched"


# the out-of-distribution split of the paper, reconstructed from the state variables (aod550 = column 4, cwv_cm =
# column 5 of the numeric inputs); state-level, all bands of a state travel together
OOD_DATA = [
    ('    if cats or split == "official":\n', '    if cats or split in ("official", "ood"):\n'),
    ('    if split == "official":\n        # the release\'s own state-level assignment (all bands of a state carry the same label)\n'
     '        is_tr, is_va, is_te = rel_split == "train", rel_split == "val", rel_split == "test"\n',
     '    if split == "official":\n        # the release\'s own state-level assignment (all bands of a state carry the same label)\n'
     '        is_tr, is_va, is_te = rel_split == "train", rel_split == "val", rel_split == "test"\n'
     '    elif split == "ood":\n'
     '        # ood_aod_cwv as the paper defines it: candidate test states above the 0.85 quantile of aod550 OR cwv_cm (the\n'
     '        # fallback the paper reports as used); the remaining states 85/15 train/val by a fixed draw, so --seed varies\n'
     '        # only the initialisation\n'
     '        states, first = np.unique(sid, return_index=True)\n'
     '        aod, cwv = X[first, 4], X[first, 5]\n'
     '        te_states = set(states[(aod >= np.quantile(aod, 0.85)) | (cwv >= np.quantile(cwv, 0.85))])\n'
     '        rest = np.array([s for s in states if s not in te_states]); rp = np.random.RandomState(12345).permutation(len(rest))\n'
     '        va_states = set(rest[rp[:int(round(0.15 * len(rest)))]])\n'
     '        is_te = np.array([s in te_states for s in sid]); is_va = np.array([s in va_states for s in sid]); is_tr = ~(is_te | is_va)\n'),
    ('("_off" if split == "official" else "")', '("_off" if split == "official" else "_ood" if split == "ood" else "")'),
]
OOD_RUN = [('choices=["seeded", "official"]', 'choices=["seeded", "official", "ood"]')]
# rescore_saved.py rebuilds the truth table from the tag; it must rebuild the SAME split and inputs as the lane
RESCORE = [
    ("Ahmed et al. 2026", "Mazid and Rishe 2026"),
    ('        lowfi = 1 if "lowfi" in tag else 0\n        key = (seed, lowfi)\n        if key not in cache:\n            cache[key] = bench_data.pkanrtm(seed=seed, lowfi=lowfi)\n',
     '        lowfi = 1 if "lowfi" in tag else 0\n        cats = 1 if "_cats" in tag else 0\n'
     '        split = "official" if "_off" in tag else "ood" if "_ood" in tag else "seeded"\n'
     '        key = (seed, lowfi, cats, split)\n        if key not in cache:\n            cache[key] = bench_data.pkanrtm(seed=seed, lowfi=lowfi, cats=cats, split=split)\n'),
    ('    D = cache.get((0, 1)) or cache.get((0, 0))\n', '    D = next(iter(cache.values()), None)\n'),
]
print("bench_data ood ->", patch(os.path.join(dgx, "bench_data.py"), OOD_DATA, "PKAN_OOD_PATCH"), flush=True)
print("bench_run ood ->", patch(os.path.join(dgx, "bench_run.py"), OOD_RUN, "PKAN_OOD_PATCH"), flush=True)

# the kernel tuner without the validation-by-subsample matrices: per scale, the per-nugget solves on the 6,000-row
# subsample, then ONE pass over the validation rows in 4,000-row chunks; row-wise the same product as before.
MEM_TUNER = [
    ("    Fs = Ftr[sub_tune]; D2s, D2vs = sqd(Fs, Fs), sqd(Fva, Fs)\n",
     "    Fs = Ftr[sub_tune]; D2s = sqd(Fs, Fs)\n"),
    ("        Ks, Kvs = m52(D2s, sc * med), m52(D2vs, sc * med)\n"
     "        for nug in NUG_GRID:\n"
     "            try:\n"
     "                e = val_fn(Kvs @ solve(Ks, Ytr_[sub_tune], nug))\n"
     "            except np.linalg.LinAlgError:\n"
     "                continue\n"
     "            if e < best[0]:\n"
     "                best = (e, (sc * med, nug))\n",
     "        Ks = m52(D2s, sc * med)\n"
     "        alphas = {}\n"
     "        for nug in NUG_GRID:\n"
     "            try:\n"
     "                alphas[nug] = solve(Ks, Ytr_[sub_tune], nug)\n"
     "            except np.linalg.LinAlgError:\n"
     "                continue\n"
     "        if not alphas:\n"
     "            continue\n"
     "        pvs = {nug: np.empty((len(Fva), Ytr_.shape[1])) for nug in alphas}\n"
     "        for k in range(0, len(Fva), 4000):\n"
     "            Kv = m52(sqd(Fva[k:k + 4000], Fs), sc * med)\n"
     "            for nug, a_s in alphas.items():\n"
     "                pvs[nug][k:k + 4000] = Kv @ a_s\n"
     "        for nug in alphas:\n"
     "            e = val_fn(pvs[nug])\n"
     "            if e < best[0]:\n"
     "                best = (e, (sc * med, nug))\n"),
]
print("bench_run mem ->", patch(os.path.join(dgx, "bench_run.py"), MEM_TUNER, "PKAN_MEM_TUNER_PATCH"), flush=True)
print("rescore ->", patch(os.path.join(dgx, "rescore_saved.py"), RESCORE, "PKAN_RESCORE_PATCH"), flush=True)
os.makedirs(os.path.join(out, "preds"), exist_ok=True)      # bench_run saves the test heads only if this exists

lanes = []
for s in SEEDS:
    lanes.append((f"pkanrtm_s{s}_cats_off", [py, "bench_run.py", "--corpus", "pkanrtm", "--seed", str(s), "--pkan_cats", "1", "--pkan_split", "official",
                                             "--families", FAM, "--tag", f"pkanrtm_s{s}_cats_off"]))
    lanes.append((f"pkanrtm_s{s}_lowfi_cats_off", [py, "bench_run.py", "--corpus", "pkanrtm", "--seed", str(s), "--pkan_cats", "1", "--pkan_split", "official",
                                                   "--lowfi", "1", "--families", FAM + ",lowfi_resid", "--tag", f"pkanrtm_s{s}_lowfi_cats_off"]))
timings = {}; done = 0; skipped = 0; failed = []
for tag, cmd in lanes:
    if os.path.exists(os.path.join(out, tag + ".json")):
        skipped += 1; continue
    if (time.time() - t_all) / 3600 > BUDGET_H:
        print(f"budget reached before {tag}; stopping launches", flush=True); break
    t0 = time.time(); print(f"\n===== {tag}: {' '.join(cmd[1:])}", flush=True)
    r = subprocess.run(cmd, cwd=dgx, env=env, capture_output=True, text=True)
    dt = time.time() - t0; timings[tag] = dict(seconds=round(dt, 1), rc=r.returncode)
    print(r.stdout[-2500:])
    if r.returncode != 0:
        failed.append(tag); print("STDERR:", r.stderr[-2500:])
    else:
        done += 1
    try:
        import resource
        rss_gb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1e6
    except Exception:
        rss_gb = float("nan")
    timings[tag]["children_max_rss_gb_so_far"] = round(rss_gb, 2)
    print(f"===== {tag} rc={r.returncode} in {dt:.0f}s (elapsed {((time.time()-t_all)/3600):.2f} h; children max RSS so far {rss_gb:.1f} GB)", flush=True)
    json.dump(timings, open(os.path.join(out, "lane_timings.json"), "w"), indent=1)
r = subprocess.run([py, "rescore_saved.py", "--what", "pkanrtm", "--roots", out, "--out", os.path.join(out, "rescored_pkanrtm.json")], cwd=dgx, env=env, capture_output=True, text=True)
print("rescore rc", r.returncode, r.stdout[-2000:], r.stderr[-800:] if r.returncode else "")
print(f"\nDONE lanes={done} skipped={skipped} failed={failed} total {(time.time()-t_all)/3600:.2f} h")
print("results:", sorted(os.listdir(out)))
if failed or r.returncode != 0:
    sys.exit(1)
