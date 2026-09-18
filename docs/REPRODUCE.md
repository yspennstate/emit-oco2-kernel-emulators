# Reproduction map

This document distinguishes three operations: regenerating numbers from archived
metrics, checking mathematical/diagnostic code on synthetic inputs, and rerunning
a training campaign from raw data. Only the first two were executed for the
September 18, 2026 revision. The main manuscript's table labels are used below
because inserted tables change their printed numbers.

## 1. Public-record analysis and manuscript build

From the repository root, using Python 3.11 or later:

```sh
python -m pip install -r requirements-revision.txt
python -m unittest discover -s code -p 'test_publication_revision.py' -v
python code/make_revision_tables.py
python code/make_tables.py
latexmk -pdf -interaction=nonstopmode -halt-on-error -cd paper/emit_kernel_dnn.tex
```

The new analysis/test environment is pinned in `requirements-revision.txt`.
`latexmk` and a TeX distribution with the manuscript's standard packages are
required for the PDF. The original training environment was not recorded in a
lockfile; do not treat the revision requirements as that environment.

### Fully regenerable primary EMIT tables

- `tab:results`: `paper/table_emit_seeds.tex`, from `results/emit/emit_s101.json`
  through `emit_s110.json`.
- `tab:wide`: `paper/table_emit_widepipe.tex`, from those ten records and
  `results/scaling/per_seed/emit_s101_wide.json` through `emit_s110_wide.json`.
- `tab:paired`: `paper/table_emit_paired.tex`, paired radiance differences from
  the same twenty records, matching each candidate and reference by split seed.
- `tab:tails`: `paper/table_emit_tails.tex`, all-band inversion summaries from
  the same records. Each percentile is computed within a split in the original
  campaign; the new table averages those percentiles, not pooled predictions.

`code/make_revision_tables.py` regenerates all four tables and writes
`results/revision_20260918/reanalysis.json`. It checks seed IDs, widths, training,
validation and test counts, the absence of smoke runs, agreement of the recorded
raw-array SHA-256 digests, and equality of common deterministic metric rows
between the width campaigns. It hashes the actual JSON source files it reads.
It does not possess or independently hash the raw data. Standard deviations use
`ddof=1`; they and the ten-split win counts are descriptive, not confidence
intervals for independent validation datasets.

Original model driver: `code/emit_campaign.py`. The primary splits use 2,331
test, 2,098 validation and 18,884 fitting rows. Model choices and stack weights
use validation rather than the held-out block within a run. Prior exploration
of the same table and overlap between random splits limit confirmatory claims.
The archived `results/emit/emit_secmom.json` records the additional residual
second-moment diagnostics discussed in the original main table analysis.

### Secondary EMIT evidence: retain its actual resolution

`tab:curves`, `tab:slopes`, `tab:rank` and `tab:retune` refer to the secondary
learning-size, rank and tuning experiments. The aggregate files are
`results/scaling/scaling_numbers.json`, `retune_numbers.json` and
`width_numbers.json`; the individual records available are in `per_seed/`.
The existing aggregate-based tables and figure are retained. They are not
silently recomputed on fewer seeds when a record is absent.

The current archive is missing these individual records among the stated
experiments:

- `emit_s106_n8000.json` and `emit_s110_n8000_ard.json`.
- Width 1000: only seeds 101, 102 and 103 are present; 104–110 are absent.
- Full-tuning records (`_ft`): 101, 102, 103, 104 and 110 are present;
  105–109 are absent.

The reanalysis JSON gives the exact path inventory. Other secondary prose
findings, including the three-member wide-feature and width-4000 observations,
are reported at their original limited replication level. The complete
secondary generation drivers, raw predictions and all individual runs are not
publicly available here. Therefore the aggregate claims can be read and checked
for arithmetic consistency, but the public archive does not independently
reproduce every underlying run. Restoring these artifacts is a separate
reproducibility requirement, not something implied by regenerating the primary
tables.

### OCO-2 and further corpora

`code/make_tables.py` reads the existing JSON records and regenerates
`table_oco2_losses.tex`, `table_oco2_seeds.tex` and `table_corpora.tex`. The first
two are companion/supporting tables; `tab:corpora` is included in this paper.
The script prints the paired counts that accompany these records. Drivers are
`code/jpl_seeded.py` for OCO-2 loss/readout comparisons, `code/oco2_curve.py` for
the OCO-2 ensemble campaign, and `code/bench_run.py` with the corpus loaders for
other configurations.

The paired correction-coefficient corpus has two distinct configurations:
`pkanrtm_s*_bench.json` and `pkanrtm_s*_lowfi_bench.json`. Do not pool them. The
turbulent-radiative-layer table uses the three rank-256 runs; the rank-1024
seed-0 rerun is not one of them. `make_tables.py` selects these explicitly.
The Well's repeated runs share the supplied split and measure initialization,
not independent resampling. Two of its configurations have only one run.

Kernel fits with training size above 20,000 use 6,000 Nyström landmarks. The
corpora table's ridge is linear, not the cubic EMIT reference, and is excluded
from its combiner pool. The stored reference OCO-2 retrieval states are not a
new independent test set, as discussed in the manuscript.

## 2. Mathematical checks

`code/test_publication_revision.py` tests the frozen residual identity,
`lambda = n * nugget` convention, a semidefinite Gram matrix, both source-norm
counterexamples, the actual squared-error alignment criterion, two differently
frozen kernel operators, PCA truncation/affine decoding, the finite forward and
inverse identities, and common-mask handling of invalid denominators. These are
finite numerical checks in addition to the proofs in the paper; they are not
experiments on the original EMIT observations.

## 3. Raw-data work that the public records do not replace

The repository does not distribute the raw EMIT `X.npy` and `Y1.npy`–`Y4.npy`
arrays, model weights or per-sample predictions. The supplied materials attribute
the table to JPL but do not specify the simulator version, generation
configuration or redistribution permission. Those details and access to the
actual arrays are needed for full independent reproduction. The original README
states that arrays, weights and prediction dumps are available from the author
on request; the JSON summaries are not a substitute for access.

The existing primary driver can be invoked with appropriately supplied data:

```sh
EMIT_DATA=/path/to/emit P2_OUT=/path/to/output NMKC_THREADS=4 \
  python code/emit_campaign.py --seed 101
```

This is a potentially expensive training run, not part of the revision's CI.
A 18,884-by-18,884 float64 Gram matrix alone occupies about 2.85 GB, before
factorization copies and other arrays. Exact training/inference throughput and
operational deployment suitability were not measured in the revision.

### Common-mask reflectance diagnostic

The new diagnostic uses only a common truth-based mask with a threshold fixed
before model comparison; a positive flux scale is derived from training rows.
Passbands should be physically justified independently of candidate test errors.
The mask includes true positive flux and a forward-denominator lower bound.
Nonfinite or nonpositive predicted inverse denominators count as failures, not
zero errors or exclusions from coverage. Error summaries on successful entries
are reported alongside that failure rate and retained coverage.

Example with supplied raw arrays, a corresponding prediction archive and record:

```sh
python code/conditioned_reflectance.py \
  --data-dir /path/to/emit \
  --predictions /path/to/emit_s101.npz \
  --record results/emit/emit_s101.json \
  --thresholds 0.001 0.01 0.1 \
  --output /path/to/conditioned-s101.json
```

The threshold values above illustrate sensitivity reporting, not a validated
mission passband or an empirically optimized recommendation. An optional
`--band-mask` accepts a Boolean NumPy wavelength mask. Error magnitudes in the
JSON are reflectance units; multiply by 100 for percentage points. The tool
checks recorded data-file hashes against supplied arrays and reconstructs the
split before using the stored `idx_te`.

Earlier `emit_campaign.py` exported predictions as float32 after scoring in
float64. Reusing those arrays cannot exactly reproduce the original inversion
metrics near cancellation. The revised driver preserves float64 for subsequent
exports. The diagnostic rejects float32 archives unless `--allow-float32` is
explicitly specified, in which case the result is labelled a precision-altered
analysis rather than a reproduction of the scored predictions. Precision lost
in old archives cannot be restored by casting them back to float64.

**No common-mask EMIT results or full-campaign reruns are claimed in this
revision.** Independent blocked/untouched evaluation, physical passband error
budgets, and runtime comparisons remain necessary for operational conclusions.
