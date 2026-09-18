# Neural means and kernel corrections for radiative-transfer emulators

Manuscript, model drivers and stored evidence for an empirical comparison on a
six-dimensional radiative-transfer table over the 285-band EMIT grid, with OCO-2
and other emulation configurations as contrasts.

**Paper:** [compiled PDF](paper/emit_kernel_dnn.pdf),
[LaTeX source](paper/emit_kernel_dnn.tex), [short abstract](paper/abstract.txt).
The September 18 revision corrects the theoretical interpretation and adds a
paired analysis of radiance gains and reflectance tails. It does not report a
new training campaign or validated operational retrieval.

## Main findings

Across ten stored EMIT splits, with 18,884 training rows and validation-based
selection, the input-scaled Matérn kernel has 0.095% mean relative radiance error,
compared with 0.760% for cubic regression and 0.387% for the neural network.
Residual correction reduces the network error to 0.141%. A kernel on the
width-2000 network's features reaches 0.079%; its convex stack reaches 0.078%.
The width comparison also changes the permitted training duration and does not
establish an intrinsic advantage of width independently of optimization.

The smallest radiance number is not an unqualified retrieval winner. Across the
same ten splits the wide feature kernel has a mean within-split 95th-percentile
absolute reflectance error of 7.57 percentage points; the wide stack has 22.71.
The stack has a worse tail at every split, while its mean radiance gain is only
0.000584 percentage points. These are all-band inverse evaluations, including
ill-conditioned absorption entries, not passband-qualified operational errors.
The manuscript reports coverage limitations instead of hiding these tails behind
a small median.

The mathematical section gives the frozen-kernel residual identity, an exact
Hilbert-space error-alignment criterion, the correct principal-component
implementation, and a finite-error conditional bound for reflectance inversion.
These are algebraic results with stated hypotheses. They do not prove universal
learning rates or guarantee that residual correction improves either parent.

## Reproduce the public-record analysis

Python 3.11 or later and NumPy are sufficient for the new analysis and tests:

```sh
python -m pip install -r requirements-revision.txt
python -m unittest discover -s code -p 'test_publication_revision.py' -v
python code/make_revision_tables.py
python code/make_tables.py
latexmk -pdf -interaction=nonstopmode -halt-on-error -cd paper/emit_kernel_dnn.tex
```

The last command also requires a TeX installation and latexmk. The GitHub
manuscript workflow performs the checks and build; its artifacts contain the
source and stored evidence. The mathematical tests use finite synthetic designs,
not the missing raw EMIT arrays.

## Evidence and limits

The main width-512 campaign and the width-2000 pipeline each have all ten per-seed
JSON records. `code/make_revision_tables.py` checks their configuration, equality
of recorded data hashes, and common deterministic rows, regenerates the two main
EMIT tables, and produces the paired/tail tables plus a source-hashed summary in
`results/revision_20260918/reanalysis.json`. Equality of recorded data hashes is
not independent verification of the raw arrays.

Some secondary learning-curve, width and retuning experiments have aggregate
summaries but incomplete individual archives. They are identified explicitly in
[REPRODUCE.md](docs/REPRODUCE.md); missing records are not reconstructed from
averages. The raw EMIT arrays, trained weights and per-sample predictions are not
in this repository. Dataset generation details and redistribution permission
remain necessary for independent full-campaign reproduction.

A new [conditioning diagnostic](code/conditioned_reflectance.py) can evaluate
prespecified common passband/flux masks when the original arrays and predictions
are supplied. It reports retained coverage and denominator failures separately.
It has been tested on synthetic cases, but no masked EMIT results are claimed.
Older prediction dumps were saved as float32 after float64 scoring; the revised
campaign preserves float64 for future exports, and the diagnostic rejects older
float32 dumps unless explicitly allowed as a different precision experiment.

Ten overlapping random splits from the same explored table provide descriptive
replication, not ten independent external confirmations. OCO-2 and the further
corpora delimit the empirical comparison; they do not establish operational EMIT
retrieval performance or a universal winner across model families.

## Layout

```
paper/       manuscript, included mathematical sections and generated tables
code/        original model drivers, table generators and new diagnostics/tests
results/     original per-run/aggregate records and source-hashed reanalysis
figures/     the original manuscript figures
docs/        reproduction map and September 18 revision record
```

## Companion and data attribution

The coupling method and companion experiments are in
[neural-means-kernel-corrections](https://github.com/yspennstate/neural-means-kernel-corrections).
The supplied study attributes the EMIT arrays to the Jet Propulsion Laboratory.
OCO-2 data and reference predictions come from Lamminpää et al. (2025); further
corpora use the releases cited in the manuscript. Cite these data sources along
with the manuscript. Code is provided under the [MIT license](LICENSE); that
license does not confer rights to raw data that are not distributed here.
