# Addendum 2 to the training-target sensitivity protocol (PREREGISTRATION_TQ_20260923.md, sha256 47e61ba922e19a53..; Addendum 1, sha256 0aab57aa0c1cabe3..)

Written 2026-09-23 06:2xZ. At this time the three seed-101 lanes at width 512 have finished on the DGX and one Kaggle
lane (seed 106, admissible, width 2000) has finished, and their scores had been seen. Nothing below depends on them.
The change concerns where one seed runs, not what is fitted, scored or reported.

## Change: seed 105 runs on Kaggle

The measured lane times differ more than Addendum 1 assumed: a width-2000 lane takes about 90 minutes in a Kaggle
session and about four hours on the DGX at four threads. With seeds 101-105 on the DGX, the DGX half of the campaign
would end about five hours after the Kaggle half. Seed 105, none of whose lanes has started, therefore moves to
Kaggle as a whole seed, under the procedure of Addendum 1: commit 1f5d4df, the same three package versions, the same
dataset digests and the same two commands. Its six lines are removed from the DGX queue before any of them starts.

Seeds 101-104 run on the DGX and seeds 105-110 on Kaggle. All six lanes of every seed still run on one kind of
machine, so every paired difference stays within one environment, and summaries across seeds report which seeds ran
where.

## Unchanged

Everything else in the protocol and in Addendum 1.
