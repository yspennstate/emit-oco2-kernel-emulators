# Training-target sensitivity on EMIT: protocol fixed before any run

Written 2026-09-23 06:4x (UTC+3), before a single lane of this campaign was launched. Its SHA-256 is recorded in
~/p23/results_tq/PREREGISTRATION.sha256 on the Caltech DGX and beside this file on the author's machine.

## Question

Does removing physically inadmissible training states change the forward and inverse errors of the model families, and
does the forward-inverse reversal between the wide feature kernel and its convex stack survive training on admissible
targets only?

## Fixed inputs

- Data: the EMIT NumPy export in /home/yitz/p2/data/emit. X.npy, Y1.npy..Y4.npy verified 2026-09-23 against the
  data_sha of the campaign records (X 3745bd59c5f8782d.., Y1 3bc568bd7d7aa5bb.., Y2 d7536cbce94e9908..,
  Y3 eb118f3590c3fbaa.., Y4 43f315155486edd4..).
- Code: github.com/yspennstate/emit-oco2-kernel-emulators at commit 1f5d4df (main, 2026-09-21), deployed unmodified
  as ~/p23/tq_repo_1f5d4df (archive sha256 bc51c0ca91dfbc25..). All 34 repository tests pass there, including the
  synthetic end-to-end training and scoring test.
- Environment: ~/nmkc_venv (Python 3.10.12, numpy 2.2.6, scipy 1.15.3, torch 2.13.0+cpu); full pip freeze in
  ~/p23/tq_env_freeze_nmkc_venv.txt. CPU only, four threads per lane.

## Arms

Training policy, applied before standardization, PCA and fitting (emit_campaign.py --training-policy):
`raw` (all candidate training states), `admissible` (whole states with any out-of-domain band removed, no clipping or
imputation), `matched-unfiltered` (a deterministic unfiltered subset of the same size as the admissible set).
Validation states and the validation loss are the original ones in every arm; test blocks are the seeded 10 percent.

Configurations, identical across arms:

- w512: --widths 512,512,512 --epochs 150 --members 5 --pca_rank 64
- w2000: --widths 2000,2000,2000 --epochs 500 --members 1 --pca_rank 64

Families in both: ridge3, krr, ard, dnn, dnn_corr, dkr, stack.

Seeds: 101 is the pilot; 101 through 110 form the campaign. Every (seed, arm, configuration) cell is run once.

## Scoring, fixed now

conditioned_reflectance.py on each lane's record and float64 predictions: --domain physical (0 <= Y4 < 1, positive
transmission), --rho 0.7, --q-min 0.3, all 285 bands, --thresholds 1e-12 1e-3 1e-2 (relative to the median positive
training flux). Every conditional error is reported with its coverage and its failure counts.

## What is reported

For each seed, arm and configuration: original training count, screened count, mean relative radiance error per
family, the within-split 95th-percentile reflectance error on all bands and on the physical domain at each threshold,
coverage and failures. Paired differences (admissible minus raw, matched-unfiltered minus raw) by seed. For w2000,
the stack-versus-feature-kernel comparison in each arm: radiance difference and reflectance-tail ratio.

## Decision rules

The pilot is checked for integrity only: record fields present (policy and audit counts, source-code hash, split-index
hashes, reference flux scale); prediction indices agree with the record; the scorer's consistency checks pass; the raw
arm at seed 101 agrees with the historical seed-101 record to within run-to-run network variation. The campaign then
expands to seeds 102-110 whatever the pilot's test errors show. No arm, threshold or family is chosen from test errors.
A lane that fails is rerun once; a second failure is reported as a failure, not dropped.
