# Addendum 1 to the training-target sensitivity protocol (PREREGISTRATION_TQ_20260923.md, sha256 47e61ba922e19a53..)

Written 2026-09-23 04:5xZ. At this time no lane of the campaign has finished. The six pilot lanes (seed 101) are
training; their logs print the test scores of the first fitted families (cubic ridge and the isotropic kernel), and
those lines had been seen. Nothing below depends on them. The change concerns where lanes run, not what is fitted,
scored or reported.

## Change: where the lanes run

At five concurrent lanes of four threads (the Caltech DGX under its 50 percent share rule) the sixty lanes need about
thirty hours. To finish sooner, the campaign is divided by seed:

- seeds 101-105: the Caltech DGX, ~/nmkc_venv, exactly as in the protocol;
- seeds 106-110: Kaggle CPU sessions, one lane per session, four threads (NMKC_THREADS=4).

The Kaggle lanes clone github.com/yspennstate/emit-oco2-kernel-emulators and check out commit
1f5d4df632c3bd6214757981c168a424ff1a0b56 (the protocol's code, verified by `git rev-parse HEAD` before any fit);
install numpy 2.2.6, scipy 1.15.3 and torch 2.13.0 (CPU wheel) in a fresh virtual environment (the three packages
the two drivers import; the DGX versions); verify the five arrays of the private dataset jpl-emit-reg-data against the
SHA-256 values in the campaign records (X 3745bd59.., Y1 3bc568bd.., Y2 d7536cbc.., Y3 eb118f35.., Y4 43f31515..);
and run the same two commands as tq_lane.py with the same arguments. Each lane writes its pip freeze, Python version
and CPU description beside its record.

All six lanes of a seed run on one kind of machine, so every paired difference (admissible minus raw,
matched-unfiltered minus raw) is taken within one environment. Summaries across seeds pool the two environments and
report which seeds ran where. Python (3.10 on the DGX) and the CPU instruction set may differ between the two
machines; the resulting differences are of floating-point order in the kernel solves and of run-to-run order in
network training.

## Unchanged

Arms, configurations, families, seeds, scoring (domain, rho, q floor, thresholds), what is reported and the decision
rules. The integrity checks of the pilot apply to seed 101 on the DGX. A Kaggle lane that fails is rerun once, on
Kaggle.
