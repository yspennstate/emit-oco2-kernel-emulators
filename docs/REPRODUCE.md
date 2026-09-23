# How each table and figure is produced

Every number in the paper comes from a record in `results/`. A training run writes one JSON record with the SHA-256 of
its input arrays, the split sizes and, for the runs of Section 6, the digest of its split indices, the selected
hyperparameters and the test metrics. The tables are regenerated from these records with numpy alone. The figures
need matplotlib as well, and four of them read the EMIT arrays or saved test predictions, which are not distributed
here (see the data statement of the paper). Retraining needs the arrays and scipy, torch, scikit-learn, h5py and
psutil.

## Regenerating the tables

From the repository root, with Python 3.11 or later:

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s code -p 'test_*.py' -v
python code/make_width_numbers.py
python code/make_scaling_tables.py
python code/make_retune_table.py
python code/make_emit_tables.py
python code/make_tables.py
python code/make_v2_tables.py
python code/format_publication_tables.py
python code/make_domain_table.py
python code/make_tq_tables.py
python code/make_lrt_tables.py
python code/make_lrt_tables.py lrt
python code/make_confirmation_table.py
(cd paper && pdflatex -interaction=nonstopmode supplement.tex)
latexmk -pdf -interaction=nonstopmode -halt-on-error -cd paper/emit_kernel_dnn.tex
latexmk -g -pdf -interaction=nonstopmode -halt-on-error -cd paper/supplement.tex
```

Each regenerated table is byte-identical to the file in `paper/`. The workflow in `.github/workflows/` runs these
commands on every push and fails if a table changes or either document has an undefined reference or an overfull
box. The article and the supplement refer to each other through `xr-hyper`, so the supplement is compiled once
before the article and again after it.

| Table | File in `paper/` | Script | Records |
| --- | --- | --- | --- |
| `tab:results` | `table_emit_seeds.tex` | `make_emit_tables.py` | `results/emit/emit_s101.json` ... `emit_s110.json` |
| `tab:paired`, `tab:tails` | `table_emit_paired.tex`, `table_emit_tails.tex` | `make_emit_tables.py` | the same, with `results/scaling/per_seed/emit_s*_wide.json` |
| `tab:wide` | `table_emit_widepipe.tex` | `make_emit_tables.py` | the same, with `emit_s*_cat.json` (concatenated features) |
| `tab:curves`, `tab:slopes` | `table_emit_curves.tex`, `table_emit_slopes.tex` | `make_scaling_tables.py` | `results/emit/` and `per_seed/emit_s*_n<rows>.json`, `emit_s*_n<rows>_ard.json` |
| `tab:rank` | `table_emit_rank.tex` | `make_scaling_tables.py` | `per_seed/emit_s*_r<rank>.json` and the rank-64 records of the same splits |
| `tab:retune` | `table_emit_retune.tex` | `make_retune_table.py` | `per_seed/emit_s*_ft.json`, `_ph05.json`, `_ph025.json` |
| `tab:domain` | `table_domain_audit.tex` | `make_domain_table.py` | `results/confirmation/domain_audit.json` |
| `tab:tq-domain`, `tab:tq-policy`, `tab:tq-reversal`, `tab:tq-runs` | `table_tq_*.tex` | `make_tq_tables.py` | `results/target_quality/tq_s*_*.json` and their `_conditioned.json` |
| `tab:confirm` | `table_confirmation.tex` | `make_confirmation_table.py` | `results/confirmation/confirmation_report.json` |
| `tab:lrt` | `table_lrt.tex` | `make_lrt_tables.py` | `results/libradtran/lrtc_s*_w512.json` and their `_conditioned.json` |
| quoted in `sec:lrt` | `table_lrt_numeric.tex` | `make_lrt_tables.py lrt` | `results/libradtran/lrt_s*_w512.json` (seven numeric inputs) |
| `tab:v2-transmission` ... `tab:v2-oco2` | `table_v2_*.tex`, laid out as `layout_v2_*.tex` | `make_v2_tables.py`, `format_publication_tables.py` | `results/tc`, `e2a`, `stack`, `e2bc`, `ridge`, `oco2`, `pkanrtm`, `kernel_perturbation_Y2_n2000.json`, `sharpness_synthetic.json` |
| `tab:corpora` | `table_corpora.tex`, laid out as `layout_corpora.tex` | `make_tables.py`, `format_publication_tables.py` | `results/corpora/` |

Standard deviations are sample standard deviations over splits (`ddof=1`); they describe the spread across splits
and are not confidence intervals. A quantile in a table is the mean over splits of the within-split quantile, not a
quantile of the pooled errors. `make_emit_tables.py` also writes `results/emit_tables.json` with the paired contrasts
and the digests of the records it read; `make_tables.py` prints the paired counts and replication figures quoted in
the text.

The residual correlations of Section 5 are in `results/emit/emit_secmom.json`, written from the test predictions of
the ten main runs by

```sh
EMIT_DATA=/path/to/emit python code/residual_moments.py /path/to/predictions
```

The round-trip errors on the exact components in Section 6.1 are printed by `results/confirmation/domain_audit.py`;
its output is `domain_audit.log` beside it.

## Figures

| Figure | File in `figures/` | Script | Inputs |
| --- | --- | --- | --- |
| `fig:examples`, `fig:structure` | `emit_data_examples.png`, `emit_structure.png` | `make_data_figures.py` | the EMIT arrays; the numbers behind the second figure and the structure paragraph of Section 3 (principal-component ranks, adjacent-band correlations) are in `results/emit_structure.json` |
| `fig:band-anatomy` | `emit_band_anatomy.png` | `make_data_figures.py --predictions DIR` | the arrays and the test predictions of three families on one split |
| `fig:curves` | `emit_curves.png` | `make_scaling_tables.py` | the records |
| `fig:bands` | `emit_quantiles_top_models.png` | `make_band_figure.py` | the arrays and the float64 test predictions of `tq_s101_raw_w512` |
| `fig:tq-profile` | `tq_profile_w512.png` (and `tq_profile_w2000.png`) | `make_profile_figure.py w512` | `results/target_quality/profile_by_arm_s*_w512.json`, written by `profile_by_arm.py` |
| `fig:lrt-bands` | `lrt_band_p95.png` | `make_lrt_band_figure.py` | the libRadtran arrays and the float64 test predictions of the `lrtc_*` runs; the values are in `results/libradtran/lrt_band_p95.json` |

```sh
EMIT_DATA=/path/to/emit python code/make_data_figures.py --predictions /path/to/predictions
EMIT_DATA=/path/to/emit python code/make_band_figure.py tq_s101_raw_w512
```

`make_data_figures.py` uses the split of the notebook distributed with the table (scikit-learn's
`train_test_split(test_size=0.1, random_state=42)`), not the seeded splits of the campaign. The predictions read by the
two figure scripts are available on request.

## The experiments of Section 6

### Training on admissible targets

The protocol is `results/target_quality/PREREGISTRATION_TQ_20260923.md` with its first addendum; their SHA-256 are in
`PREREGISTRATION.sha256` beside them, as are the second addendum and its withdrawal. Each of the sixty runs (ten
splits, three training policies, two configurations) runs `code/emit_campaign.py` with `--training-policy raw`,
`admissible` or `matched-unfiltered`, then scores the float64 test predictions with `code/conditioned_reflectance.py`
and `code/band_transfer_check.py`. Seeds 101 to 105 ran with `code/lanes/tq_lane_dgx.py` and seeds 106 to 110 with
`code/lanes/tq_kaggle_lane.py` (one Kaggle kernel per run, written by `make_tq_kernels.py`). The float64
predictions, about 150 MB per run, are available on request.

`code/check_tq_reproduction.py` compares each raw-arm record with the record of the same split in the main campaign:
the kernel and cubic families are deterministic given the split and agree to rounding, and the networks and every
family built on them agree to within run-to-run variation. `code/profile_by_arm.py`, `make_profile_figure.py` and
`admissible_filter_profile.py` describe where the filter acts; they are not part of the protocol.

### A second radiative-transfer code

The protocol is `results/libradtran/PREREGISTRATION_LRT_20260923.md` with its addendum, both with their SHA-256 in
`PREREGISTRATION.sha256`. The arrays are built from the paired 6S/libRadtran release, the files
`dataset_rows_libradtran.jsonl` and `dataset_rows_6s.jsonl` of the folder `qavalid_intersection_libradtran_6s_50k_13b`
of the Hugging Face dataset `mazid-rafee/pKANrtm`. With the two files in `/path/to/data/pkanrtm`:

```sh
DATA_NEW=/path/to/data python -c "import sys; sys.path.insert(0, 'code'); import bench_data; bench_data.pkanrtm(cats=1)"
python code/make_libradtran_arrays.py /path/to/data/pkanrtm/paired_arrays.npz /path/to/output \
  --cats /path/to/data/pkanrtm/paired_cats.npz
```

The first command caches the numeric rows (`paired_arrays.npz`) and the aerosol model and atmosphere profile of every
row (`paired_cats.npz`). The digests of the sources and of every array are in `results/libradtran/MANIFEST_cats.json`.

The sixteen-input runs use `code/lanes/lrtc_kaggle_lane.py` through `make_lrtc_kernels.py` on Kaggle and
`code/lanes/lrtc_lane_dgx.py` on the Caltech DGX, with the same commit and package versions;
`results/libradtran/MACHINES.json` records which seed ran where. The seven-input runs use `lrt_kaggle_lane.py` through
`make_lrt_kernels.py`. `make_lrt_tables.py` evaluates the three hypotheses of the protocol on every seed and writes
`results/libradtran/lrt_summary.json` (or `lrt_summary_numeric.json`). The transmission statistics quoted in the
section (the median of band B10, the share of its entries below 1e-6, the smallest transmission of the other
bands) are written to `results/libradtran/transmission_by_band.json` by
`EMIT_DATA=/path/to/arrays python code/lrt_transmission_stats.py`.

### The fresh partition

Everything for Table `tab:confirm` is in `results/confirmation/`, and the scripts there run in that folder, in this
order:

1. `fit_rest.py dnn`, `refine_ard.py` and `fit_rest.py kernels` select every hyperparameter on the development split
   (seed 101 of the main protocol) and write `results_dnn_512.json`, `refined_ard.json`, `results_top_models.json`
   and `dev_hyperparams.json`;
2. `write_freeze.py` copies the selected values into `freeze.json` and its SHA-256 into `freeze.sha256`;
3. `confirm.py` draws the confirmation block with `RandomState(20260921)`, fits the six families on the remaining
   states with the frozen values, reads the block once and writes `confirmation_report.json`; its output is
   `confirm.log`. It refuses to run unless `freeze.json` has the registered digest.

`make_bands.py` scores the same six families on the test block of the development split and writes
`development_split_tails.json`; the factors 6.13 and 4.99 that Section 6.5 compares with the fresh partition are
ratios of its 95th percentiles (stack over feature kernel, over all bands and on the screen).

`domain_audit.py` measures the table against the physical domain and fixed the constants of the screened columns
before the confirmation run. The protocol is `PREREGISTRATION.md`. The scripts that read the arrays name their folder
in a constant `D` near the top, and those that save predictions name the output folder in `OUT`; point them at local
copies before running. The test suite checks the digest of `freeze.json` and recomputes the hypothesis of the
protocol from the report.

## Retraining the main campaign

```sh
EMIT_DATA=/path/to/emit P2_OUT=/path/to/output NMKC_THREADS=4 python code/emit_campaign.py --seed 101
```

`EMIT_DATA` holds `X.npy` and `Y1.npy` to `Y4.npy`. The defaults are the main configuration (three hidden layers of
512 units, 150 epochs, five networks, output rank 64); `--widths 2000,2000,2000 --epochs 500 --members 1` gives the
width-2000 pipeline, `--ntrain` a rung of the learning curves and `--pca_rank` the rank ablation. A run fits every
family on 18,884 rows, selects on the 2,098 validation rows of its split and reads the 2,331 test rows once at the end.
The float64 Gram matrix of the training block alone takes 2.85 GB.

The retrieval on the physical domain is scored from a run's predictions:

```sh
python code/conditioned_reflectance.py --data-dir /path/to/emit --predictions /path/to/emit_s101.npz \
  --record results/emit/emit_s101.json --thresholds 1e-12 1e-3 1e-2 --output conditioned_s101.json
```

The mask is fixed by the truth and a flux scale from the training rows, and is common to all families. The default
`--domain physical` requires `0 <= Y4 < 1`; `--domain algebraic` keeps only the conditions under which the inverse is
defined. A failed inversion is counted, never dropped or set to zero, and every error summary is reported with the
coverage and the failure count. The scorer checks the recorded data digests against the supplied arrays, rebuilds the
split and refuses float32 predictions unless `--allow-float32` is given.

## Other corpora

The learned-metric comparison of the supplement on OCO-2 is in `results/oco2_metrics/`, one record per band and split
from `code/kf_kernels.py --problem oco2`; `python code/oco2_metric_summary.py results/oco2_metrics
results/oco2_ensembles` prints the means quoted there beside the baseline runs of the same splits and writes
`summary.json`.

`make_tables.py` reads the records of the other configurations. Their drivers are `code/jpl_seeded.py` (OCO-2 losses
and readouts), `code/oco2_curve.py` (OCO-2 ensembles) and `code/bench_run.py` with the corpus loaders in
`code/bench_data.py` and `code/well_data.py`; `bench_run.py` also accepts corpora outside this paper, whose loaders are
not included. The two correction-coefficient configurations of the corpora table, `pkanrtm_s*_bench.json` and
`pkanrtm_s*_lowfi_bench.json`, are different experiments and are not pooled.

The correction-coefficient table `tab:v2-pkanrtm` follows the benchmark's own protocol: the categorical inputs
(`--pkan_cats 1`), the release's standard split (`--pkan_split official`) and its out-of-distribution split. It was
run on Kaggle with `code/lanes/pkanrtm_bench.py` and `code/lanes/pkanrtm_std.py`, which download the release, add the
out-of-distribution split to `bench_data.py` at run time and call `bench_run.py`; `code/rescore_saved.py` computes the
benchmark's metrics from the saved predictions into `results/pkanrtm/rescored_pkanrtm.json`. The turbulent-radiative-layer row uses
the three rank-256 runs. The runs of The Well share the supplied split, so their spread measures initialization rather
than resampling, and two of its configurations have one run each. Kernel fits with more than 20,000 training rows use
6,000 Nyström landmarks. The ridge of the corpora table is linear and is not a member of that table's combiner.

The correction-coefficient benchmark uses seven state coordinates and the wavelength (`wvl_nm`, `sza_deg`,
`vza_deg`, `raa_deg`, `aod550`, `cwv_cm`, `o3_cm`, `elev_km`), eleven inputs once the three 6S coefficients are added,
and reports the relative error with the denominator `maximum(norm(Y_true, axis=1), 0.05)`, the same floor as in
validation and in the kernel-flow objective.

## Tests

`code/test_identities.py` checks the residual identity of the kernel correction, the convention `lambda = n * nugget`,
the semidefinite Gram matrix, the two counterexamples on norm-error ordering, the alignment criterion, the distinct
frozen operators, PCA truncation with affine decoding, the forward and inverse identities with the error bound, and
the accounting of masks and failures. `code/test_stacking_contract.py` checks the stacking weights and the surrogate
bound of the constrained retrieval. `code/test_target_quality.py` checks the target filter, the mask semantics and the
records of the fresh partition; with `EMIT_INTEGRATION_TESTS=1` (and scipy and torch installed) it also trains and
scores the three policies end to end on a small synthetic table.
