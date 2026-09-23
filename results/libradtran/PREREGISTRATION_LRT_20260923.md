# The forward-inverse comparison on a second radiative-transfer code: protocol fixed before the runs

Written 2026-09-23 07:06Z. No fit of this campaign has been made. One smoke run of the driver on these arrays (seed 101,
the driver's --smoke settings: 3,000 training rows, 2 epochs, 2 members) was made to check that the pipeline and the
scorer run, and its scores were seen; at those settings they say nothing about the questions below.

## Question

The EMIT comparison found that a small forward gain can coexist with a much larger reflectance tail, and the
transfer theorem locates the cause at entries with small transmission. Does the same hold for emulators of a
different radiative-transfer code, fitted by the same pipeline with no change to any family, on states that played
no part in developing it?

## Data

The libRadtran half of the paired 6S/libRadtran Sentinel-2 corpus released with the physics-guided emulation work
cited in the paper as [pkan]: the release's rows (dataset_rows_libradtran.jsonl, paired with dataset_rows_6s.jsonl),
as cached by the benchmark loader code/bench_data.py in paired_arrays.npz (sha256 recorded in MANIFEST.json as
source_sha256). States with all thirteen bands are kept (9,722 of 50,000; band B10 at 1372 nm is missing from the others). The arrays are made by
code/make_libradtran_arrays.py: inputs are the seven state variables (solar and view zenith, relative azimuth,
aerosol optical depth, water vapour, ozone, elevation); Y1 = path reflectance, Y2 = total transmittance, Y3 = 1e-6
times the total transmittance (a negligible second transmission term so that the four-component driver runs
unmodified), Y4 = spherical albedo, thirteen bands ordered by wavelength. The digests of X.npy and Y1.npy-Y4.npy are
those in MANIFEST.json (X f02293c7.., Y1 6789c57d.., Y2 faf0af8e.., Y3 15c24d29.., Y4 68701724..). Top-of-atmosphere
reflectance is L(rho) = Y1 + rho t / (1 - rho Y4) with t = Y2 + Y3, which is equation (1) of the paper.

## Fixed pipeline

- Code: repository commit 1f5d4df, unmodified: code/emit_campaign.py for fitting, code/conditioned_reflectance.py for
  scoring (the same commit as the training-target experiment).
- Split: the driver's own (RandomState(seed) permutation, 10 percent test, 10 percent of the rest for validation):
  7,875 training, 875 validation and 972 test states.
- Configuration: --widths 512,512,512 --epochs 150 --members 5 --pca_rank 13 (all thirteen bands; the principal
  components are a rotation) --families ridge3,krr,ard,dnn,dnn_corr,dkr,stack, training policy raw, four threads.
- Seeds: 101-110. Every seed is run once; a lane that fails is rerun once, and a second failure is reported.
- Machines: all ten lanes on one kind of machine, Kaggle sessions (CPU fitting; the driver hides any GPU), or, if
  that is not available, whole seeds split between Kaggle and the Caltech DGX, with the assignment recorded.

## Scoring

From each record: the mean relative radiance error at rho = 0.7 and the all-band 95th percentile of the absolute
reflectance error (the unrestricted inverse, non-finite values set to zero, as in the paper). From
conditioned_reflectance.py with --domain physical --rho 0.7 --q-min 0.3 --thresholds 1e-12 1e-3 1e-2 (relative to the
median positive training transmission): coverage, failed inversions and the 95th percentile of the successful
inversions at each floor.

## Hypotheses, each tested separately on the ten seeds

H1. The family with the lowest radiance error does not have the lowest all-band 95th-percentile reflectance error.
    Confirmed if this holds on at least 9 of the 10 seeds (a one-sided sign test, p = 11/1024).

H2. The convex stack's radiance error is at most that of the kernel on features, while its all-band 95th-percentile
    reflectance error is larger. Confirmed if both hold on at least 9 of the 10 seeds.

H3. Excluding the smallest transmissions brings the inverse ordering back to the forward one: the Kendall rank
    correlation between the seven families' radiance errors and their 95th-percentile reflectance errors on the
    physical domain at the floor 1e-3 is larger than the same correlation over all bands. Confirmed if this holds on
    at least 9 of the 10 seeds.

A hypothesis that is not confirmed is reported as not confirmed, with its count.

## Reported in addition, without a decision rule

Radiance and component errors, the tails at every floor with coverage and failures, the per-band 95th percentiles,
and the band-wise form of the transfer theorem (code/band_transfer_check.py applied to each lane's predictions).
