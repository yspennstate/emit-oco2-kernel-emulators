# Review fixes — 21 September 2026

This revision starts from `e334bcbf18c93c8beeda1a4cb00cf356dffb457c`
(`revision/raw-arrays-recovered`) and addresses the review of the uploaded
*Learned representations and kernel corrections for radiative-transfer emulation*.
It does not claim a clean-target EMIT training run or new simulator evaluations.

## Manuscript changes

Section 6 now separates physical admissibility, algebraic existence,
non-identifiability at zero flux, ill-conditioning and floating-point cancellation.
Negative albedo, for example, can lie outside the physical hypothesis while leaving
the rational inverse well-defined. A fixed absolute transmission floor is described
as a diagnostic in the archive's units, not a universal precision or accuracy limit.
The exact-component round-trip RMSE is not described as nine-digit accuracy for
every entry. The abstract, data protocol, albedo discussion, conclusions and data
statement carry the same distinctions.

The six-row refit is explicitly a retained one-split aggregate. Its seed, test
indices, selected hyperparameters, checkpoints, prediction arrays, test-mask
coverage and failure counts are not recoverable from the released table. The
0.1741% network radiance error is not equated with the main 0.387 +/- 0.012%
comparison, nor attributed to an undocumented configuration change. Two similar
baseline summaries do not reproduce every model. The archive-to-NumPy export,
including affine metadata, also requires verification against the recorded hashes.

The stack ranks fourth on the retained all-band p95 and second on its screened
p95, not fifth. The stack/feature ratios are 6.12649 and 4.98845. Only four displayed
medians are unchanged: cubic and input-scaled medians do change. The noncubic p95
reductions range from about 26% to 39%; the cubic change is much smaller. These
are arithmetic checks on rounded published values, not sample-level rescoring.
`code/audit_retained_refit.py` writes their source hash and calculations to
`results/review_20260921/retained_refit_audit.json`.

The archived quantile figure is retained, without attributing every off-scale
curve to a model-independent denominator effect or interpreting display clipping
as a failure count. The earlier exact band-count claims are not carried forward
without the generating sample-level records. The wide stack's six implementation
slots contain only four distinct predictors when `members=1`; this is now explicit.
Its weights are component-dependent, not uniformly 0.88--0.96 on the feature head.
The long archive hash is typeset without overflow and the final corpus table is
flushed before acknowledgements rather than drifting past the bibliography.

## Executable changes and their scope

`code/conditioned_reflectance.py` defaults to the physical truth domain
`0 <= Y4 < 1`, positive transmission and the requested q/flux floors. Earlier
versions tested q and positive flux but omitted that albedo interval. The previous
looser domain remains available only as explicit `--domain algebraic`; it is not
labeled physically admissible. Historical tables are not silently regenerated on
the changed domain. Schema version 2 records the domain and nonfinite/nonpositive
denominator failure categories separately. Coverage remains truth-based and does
not improve when a model fails. Successful-inversion summaries remain conditional
and must always be read alongside failure counts.

`code/emit_campaign.py --training-policy` now accepts `raw` (the default),
`admissible`, and `matched-unfiltered`. The policy is applied before input/output
standardization, PCA and fitting. Admissible filtering drops whole training states
with any out-of-domain band; it does not clip or impute targets and does not exclude
small positive transmission by an arbitrary training floor. The size-matched arm
uses a deterministic unfiltered subset with the same number of rows. Original
validation states and loss are common to all arms: this isolates a training-stage
change and does not claim to clean the validation procedure. Whole-row removal
changes the training distribution, so it is a sensitivity policy, not a uniquely
justified repair of the unknown generating process.

New records contain the policy/audit, source-code hash, exact split-index hashes
and a reference flux scale from the original candidate training block. Prediction
NPZs retain float64 predictions plus train/validation/test indices. The scorer
checks these against the matching record, rather than incorrectly reconstructing
filtered training as the first `ntrain` rows. The common scale is independently
recomputed from the original candidate block when scoring each arm.

Tests cover domain-versus-algebra distinctions, overlapping audit counts,
training-only selection, size matching, no mutation/imputation, ordered index
hashes, unit-consistent thresholding, failure accounting, and retained-table
arithmetic. An optional end-to-end test fits six heads on an 80-state synthetic
fixture for all three policies and runs the scoring CLI. It is not EMIT evidence.

## Remaining empirical work

The training-target sensitivity is **implemented but not run on EMIT**. The main
twenty campaign summaries and the retained refit cells are unchanged. Access to
the raw arrays or a verified export is still required. The missing refit manifest
cannot be replaced by guessing that it used a particular seed or optimizer.
No new radiative-transfer simulator campaign is intrinsically needed for the
training comparison below.

### Run the training-only sensitivity on the existing NumPy export

Use a new output directory; do not overwrite historical run records. The training
code needs NumPy, SciPy and PyTorch in addition to the lightweight analysis setup.
Record the actual environment (`python -m pip freeze`) used for these new runs;
the repository's analysis lockfile is not the original training environment.
Verify `X.npy` and `Y1.npy` through `Y4.npy` against `data_sha` in the main JSONs.
An unverified `.jld2` conversion is not a substitute for this check.

```sh
export EMIT_DATA=/path/to/verified/emit-numpy-export
export P2_OUT=/path/to/new/target-quality-output
export NMKC_THREADS=4

# A single split is a pilot. Use seeds 101 through 110 for the matched campaign.
for seed in 101; do
  for policy in raw admissible matched-unfiltered; do
    tag="quality_s${seed}_${policy}_w512"
    python code/emit_campaign.py --seed "$seed" --training-policy "$policy" \
      --widths 512,512,512 --epochs 150 --members 5 --pca_rank 64 \
      --families ridge3,krr,ard,dnn,dnn_corr,dkr,stack --tag "$tag"

    # Example fixed relative threshold; choose the threshold/passband before
    # examining these test scores. It is not the historical absolute 1e-12 floor.
    python code/conditioned_reflectance.py --data-dir "$EMIT_DATA" \
      --record "$P2_OUT/$tag.json" --predictions "$P2_OUT/preds/$tag.npz" \
      --domain physical --rho 0.7 --q-min 0.3 --thresholds 1e-12 \
      --output "$P2_OUT/${tag}_conditioned.json"
  done
done
```

Repeat with `--widths 2000,2000,2000 --epochs 500 --members 1` and a different tag
to assess the central wide-feature/stack comparison. Keep all other protocol
choices common. Report original training count, screened count, paired errors,
coverage and failures. Since each exact fit is expensive, run and inspect the
pilot's integrity checks before expanding to all seeds; do not select a preferred
policy from its test errors and then call the same pilot independent validation.
The original validation criterion includes unfiltered targets. A separate,
clearly labeled validation-screening study is needed to evaluate changing that
part of the workflow too.

### Checks without EMIT data

```sh
python -m unittest discover -s code -p 'test_*.py' -v
python code/audit_retained_refit.py
# Optional, with SciPy and PyTorch installed:
EMIT_INTEGRATION_TESTS=1 python -m unittest discover -s code -p 'test_*.py' -v
```

The first command runs the mathematical/domain tests and skips the optional
training smoke test. The second is a retained-table audit. The third includes
synthetic fits and scoring; none produces new EMIT performance measurements.
