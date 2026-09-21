# Neural means and kernel corrections for radiative-transfer emulators

Manuscript, code and recorded results for a comparison of neural networks, exact
kernel regression and their combinations on a six-dimensional radiative-transfer
table over the 285-band EMIT wavelength grid.

[Paper (PDF)](paper/emit_kernel_dnn.pdf) · [LaTeX source](paper/emit_kernel_dnn.tex) ·
[Abstract](paper/abstract.txt) · [Reproduction instructions](docs/REPRODUCE.md)

## Findings

Across ten EMIT splits, each with 18,884 training rows and validation-based
selection, the input-scaled Matérn kernel has 0.095% mean relative radiance error,
compared with 0.760% for cubic regression and 0.387% for the neural network.
Residual correction reduces the network error to 0.141%. A kernel on the
width-2000 network's features reaches 0.079%; its convex stack reaches 0.078%.

The forward and inverse rankings differ. The wide feature kernel has a mean
within-split 95th-percentile absolute reflectance error of 7.57 percentage points,
compared with 22.71 for its convex stack. The stack has a larger tail at every
split, despite a mean radiance gain of 0.000584 percentage points. The inverse
metrics include all bands and their ill-conditioned absorption entries.

The mathematical analysis gives exact residual-error identities, a Hilbert-space
alignment criterion, the principal-component error decomposition, and conditional
finite-error bounds for reflectance inversion. The empirical comparisons include
learning curves, rank and width experiments, OCO-2, and ten additional emulation
configurations. The correction-coefficient corpus uses eight state/wavelength
inputs (eleven with low-fidelity coefficients) and a 0.05 denominator floor in
its relative-error metric.

## Publication revision of 21 September 2026

The revised paper separates the raw quadratic used to fit stacks from projection during
retrieval scoring, derives the corresponding factor-four bound, and documents the
historical approximate weighting solver. It preserves the principal ten-split model
results, narrows the finite-profile interpretation, and makes the reproducibility limits
explicit. The transmission comparison now uses the common four-record float64 cohort;
the separate float32 experiment is not pooled. Long tables have readable multipage layouts.

[Detailed revision notes](docs/PUBLICATION_REVISION_2026-09-21.md) ·
[Record-level inventory](results/revision_20260921/reanalysis.json).
The earlier additions are described in [the original 21 September notes](docs/REVISION_2026-09-21.md).

## Reproduction

Python 3.11, the analysis requirements and a TeX installation are needed for the
checks, table generation and PDF build:

```sh
python -m pip install -r requirements-revision.txt
python -m unittest discover -s code -p 'test_*.py' -v
python code/make_revision_tables.py
python code/make_tables.py
python code/make_v2_tables.py
python code/format_publication_tables.py
python code/revision_manifest.py
latexmk -pdf -interaction=nonstopmode -halt-on-error -cd paper/emit_kernel_dnn.tex
```

These commands run 22 synthetic regression tests and regenerate the tables supported
by public records. The concatenation and joint-stacking aggregates, and the state-input
perturbation rows, are explicitly retained because their generating JSONs are absent.
They do not retrain the models. The GitHub workflow runs the same
checks and archives the compiled paper, sources and build provenance.

## Data and scope

The main and wide EMIT comparisons each have ten per-split records. Some secondary
experiments have complete aggregate summaries but incomplete individual archives;
[the reproduction manifest](docs/REPRODUCE.md) lists them. Raw EMIT arrays, trained
weights and per-sample predictions are not distributed in this repository.
Simulator-generation details and data access are required for full reproduction.

Model development and evaluation used the same finite table. The overlapping
partitions describe split sensitivity, not independent external validation. The
conditioned-reflectance diagnostic reports common-mask coverage and failures when
sample-level arrays are provided. The primary EMIT results are all-band errors. Supplementary transmission summaries
include entries outside the theorem's physical domain and are not certified all-band
retrieval bounds. No operational retrieval or matched-throughput benchmark is supplied.

## Contributions and sources

Claude Code (Anthropic) and Codex (OpenAI) implemented and ran the experiments,
diagnostics and figures and drafted the manuscript. ChatGPT (OpenAI) developed the
residual-error and inversion-stability analyses, checked the finite identities,
analysed the recorded results, implemented conditioning diagnostics and tests,
and edited the manuscript. Funding and competing interests are stated in the paper.

The method and companion experiments are in
[neural-means-kernel-corrections](https://github.com/yspennstate/neural-means-kernel-corrections).
The EMIT arrays are attributed to the Jet Propulsion Laboratory. The OCO-2 data
and reference predictions come from Lamminpää et al. (2025); the other data sources
are cited in the manuscript. The [MIT license](LICENSE) covers the repository
code, not rights to data absent from the repository.
