# Addendum 1 to the libRadtran protocol: the aerosol model and the atmosphere profile as inputs

Written 2026-09-23 08:4xZ, before the first fit of the lanes it defines. It amends PREREGISTRATION_LRT_20260923.md
(sha256 bf105fb61877cc74..) in one respect, the input columns, and in nothing else.

## What had been run and seen

Seeds 101 and 102 of the protocol's lanes had completed and seeds 103 and 104 were running; seeds 105-110 had not
been started. The record of seed 102 had been read in full, including its values for the three hypotheses (H1 held,
H2 and H3 did not). The record of seed 101 had been downloaded and not read.

## Why the inputs change

In the seed-102 record the seven families' mean relative radiance errors lie between 3.19 and 3.22 percent, and the
cubic ridge is level with every kernel and with the network. That points to an error floor the inputs cannot
explain. The release describes each state by nine variables: the seven numeric ones of the protocol and two
categorical ones, the aerosol model (continental, desert, maritime, urban) and the atmosphere profile (midlatitude
summer, midlatitude winter, subarctic summer, subarctic winter, tropical). The benchmark loader keeps them in
paired_cats.npz (sha256 9f587780..), aligned row by row with paired_arrays.npz; the protocol's arrays omitted them.

Measured on the 9,722 full-band states, five-fold, with a cubic ridge on the log transmittance at bands B2, B4, B8A
and B11: with the seven numeric inputs the relative error is 2.1-8.7 percent for libRadtran and 7.6-39 percent for 6S;
fitted separately within each of the twenty aerosol-profile classes it is 0.74-3.03 and 0.09-0.23 percent. The two
omitted variables therefore set a floor that is common to every family, and on the seven-input arrays the forward
comparison of families that the hypotheses rely on is not informative.

## The change

The inputs are the seven numeric variables followed by the one-hot aerosol model (4 columns) and the one-hot
atmosphere profile (5 columns), sixteen in all, made by code/make_libradtran_arrays.py with --cats paired_cats.npz.
X.npy has sha256 c403bb22..; Y1.npy-Y4.npy are unchanged (6789c57d.., faf0af8e.., 15c24d29.., 68701724..). The
driver's split depends only on the seed and the number of states, so every seed has the same training, validation
and test states as under the protocol, and the records' split digests will show it.

Everything else stands as fixed: code commit 1f5d4df unmodified, the configuration, the seven families and their
grids, rank 13, seeds 101-110 run once each with one rerun on failure, Kaggle sessions with CPU fitting, the scoring,
the three hypotheses and their decision rule. The lanes are tagged lrtc_s<seed>_w512.

One smoke run of the driver and the scorer on the sixteen-input arrays (seed 101, --smoke, 800 training rows, width
32) was made to check that both run; its scores were seen and say nothing about the hypotheses.

## The seven-input lanes

Seeds 101-104 of the seven-input arrays (the two completed and the two running when this was written) are kept in the
repository and reported as runs on the seven numeric inputs. Seeds 105-110 of those arrays are not fitted. The
hypotheses are decided on the sixteen-input lanes.
