# Reproducing the tables

Every number in the manuscript comes from a record in `results/`. This file maps each table to
the records and the driver that produced them. The tables in `paper/` that are generated rather
than hand-written are rebuilt by `code/make_tables.py`, which needs only Python and NumPy and
reads `results/` alone.

## Table 1, the EMIT table

Driver: `code/emit_campaign.py`. One record per split, `results/emit/emit_s101.json` through
`emit_s110.json`, each holding every family's component errors, radiance error and reflectance
statistics for that split, together with the selected hyperparameters and the split's own
principal-component ranks. Paired per-split differences between families are in
`results/emit/emit_secmom.json`, which is what the win counts in the text are computed from.

Protocol: the table is split into a 2,331-row test block and a 20,982-row training block, of
which 2,098 rows are held out for validation, leaving 18,884 for fitting. The split is drawn ten
times. Every hyperparameter, every combination weight and every per-coordinate pick is chosen on
that split's validation rows; the test block is read once per family.

## Table 2, OCO-2 by training loss and readout

Driver: `code/jpl_seeded.py`. Thirty records, `results/oco2_losses/oco_<band>_s<0-9>.json`, three
bands by ten splits. Each holds, for each of three training losses, the network's own error, the
error of a ridge readout refitted on its frozen last-layer features, and the error of an exact
Matérn regression on the same features; then three per-coordinate selections; then the reference
emulator's stored predictions rescored on the same test points. The `winners` field records which
member won each of the forty coefficients, and `boot_ci_pct` holds paired bootstrap intervals in
percentage points for the contrasts quoted in the text.

The three losses are the per-sample relative error on the reduced coefficients (`flat`), the same
error under the weights the reconstruction applies to each coefficient (`wnum`), and the relative
error of the reconstructed radiance itself (`radx`).

## Table 3, the OCO-2 ensemble campaign

Driver: `code/oco2_curve.py`. Sixty records, `results/oco2_ensembles/oco_<band>_s<0-9>_n18000.json`
for the single-network protocol and `..._n18000_m3.json` for the three-member protocol. The two
campaigns share their splits, which is why their deterministic rows agree to the stored precision
and their network rows do not: the networks were trained independently, so the rows the two
campaigns have in common are a replication rather than a copy.

## Table 5, the further emulation configurations

Drivers: `code/bench_run.py` with the loaders in `code/bench_data.py` and `code/well_data.py`.
Records under `results/corpora/`, one per configuration and run, with the metadata (`n`, `d`, `q`,
and whether the kernel solve was exact or capped) in the same file. The solve is exact when the
training block has at most 20,000 rows and otherwise uses 6,000 Nyström landmarks; runs with a
capped solve are marked `"exact": false`.

Two rows are easy to pool by accident and must not be. The correction-coefficient corpus has two
configurations, from the atmospheric state alone (`pkanrtm_s*_bench.json`, d = 8) and with the
low-fidelity 6S coefficients added as inputs (`pkanrtm_s*_lowfi_bench.json`, d = 11); they are
different problems and have a row each. The turbulent-radiative-layer row is the three rank-256
runs (`trl2d_s0..s2`); `trl2d_s0_r1024_bench.json` is a rank-1024 rerun of seed 0 and belongs to
neither. `code/make_tables.py` selects these files explicitly.

The reference column of this table is a ridge regression linear in whatever inputs the models
receive, not the cubic reference of Table 1, and it is not a member of the convex stack or of the
per-coordinate selection, whose pool is the kernels, the networks, the residual correction and
the feature kernel.

## The weighting study

The result tables are under `results/stacking/`; each was produced from the member predictions of
the EMIT and OCO-2 campaigns without retraining anything. `code/rmt_stack.py` is the estimator
itself: the minimum-variance stack on the error covariance in the reported metric, with an
intercept and nonnegative weights, plus the blend that chooses between it and the least-squares
stack by cross-validation on the calibration rows alone. The tables report gains relative to a named baseline, in per cent, per cell,
where a cell is one split and one output group.

## Rebuilding the generated tables

```sh
python code/make_tables.py            # writes table_oco2_losses.tex, table_oco2_seeds.tex, table_corpora.tex
```

The script also prints the paired counts and replication figures quoted in the text, so a reader
can check the prose against the records without reading the LaTeX.

## What is not here

The raw EMIT arrays, the trained network weights and the per-sample prediction dumps are not
redistributed. Nothing in the manuscript depends on them beyond the records above; they are
available from the author on request.
