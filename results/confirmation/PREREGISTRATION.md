# Pre-registration of the confirmation experiment

Written 2026-09-21, before any confirmation-block result was computed. The SHA-256 of
`freeze.json` recorded in `freeze.sha256` fixes the content of this plan; `confirm.py`
refuses to run if that hash does not match.

## What this is for

The development experiments for this manuscript chose model families, ablations and the
presentation of results while inspecting test results from the same table, across ten
overlapping random partitions (seeds 101-110). Those comparisons are exploratory. This
experiment is a single confirmatory evaluation with everything fixed in advance.

## The confirmation block

A block of 3,497 states (15% of the 23,313) is drawn from the table by

    numpy.random.RandomState(20260921).permutation(23313)[:3497]

That seed is declared here and appears in no development run. The block is not read,
summarised or plotted before the single evaluation below. The remaining 19,816 states are
the confirmation training pool; 1,982 of them (10%, `RandomState(20260921).permutation`
of the pool) serve only as the early-stopping set for the network, which is a training
device and not a selection among families.

## The frozen model list

Six families, in this order, with no additions and no substitutions:

1. `cubic_ridge` — degree-3 polynomial ridge on the six standardized inputs.
2. `fc_dnn_512` — fully connected 3x512 GELU network on the 285 standardized bands.
3. `dnn_plus_residual_krr` — Matern-5/2 kernel regression of that network's residuals.
4. `krr_ard_matern` — exact Matern-5/2 with one length scale per input.
5. `dkr_feature_kernel` — exact Matern-5/2 on the network's 512-dimensional penultimate features.
6. `convex_stack` — per-component convex combination of 1, 3, 4, 5.

## The frozen hyperparameters

Every hyperparameter is copied from the development selection at seed 101 and is **not**
re-tuned on the confirmation pool. The values are written into `freeze.json` by
`write_freeze.py` from `results_top_models.json` before the confirmation run.

## The one pre-declared hypothesis

H1. **The forward-inverse ranking reversal reproduces out of sample.** Concretely: on the
confirmation block, the model with the lowest radiance relative L2 error does **not** have
the lowest 95th-percentile absolute reflectance error over all bands; and the model with
the lowest radiance error has a strictly larger all-band 95th-percentile reflectance error
than `dnn_plus_residual_krr`.

H1 is judged on a single read of the confirmation block. Failure to reproduce is reported
with the same prominence as success.

## Reported quantities, fixed in advance

Per family, on the confirmation block only:

* relative L2 error per component and for the radiance at rho = 0.7;
* reflectance error over all bands: RMSE, median, 95th and 99th percentiles of the absolute
  error, and the count of non-finite inversions;
* reflectance error on the admissible domain of Proposition 1,
  A = { t = Y2+Y3 >= 1e-12 and q = 1 - rho*Y4 >= 0.3 }, with retained coverage, the count
  of excluded entries, and the same four statistics.

The admissible-domain constants tau = 1e-12 and kappa = 0.3 are fixed here, before the
confirmation run, from the truth-table audit in `domain_audit.json`, which used no model
predictions.

## What this experiment does not establish

The confirmation block is drawn from the same tabulated table. It removes selection on the
evaluated rows and re-tuning on them; it does not create states outside the table, and it
does not make the choice of model families independent of the table. Closing that would
require fresh radiative-transfer runs at new states with the generating simulator, which is
not in our possession. This limitation is stated in the manuscript rather than papered over.
