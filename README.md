# Learned representations and kernel corrections for radiative-transfer emulation

Manuscript, code and run records for a comparison of neural networks, exact Matérn kernel regression and their
combinations as emulators of a six-dimensional radiative-transfer table on the 285-band EMIT wavelength grid, and for
an analysis of how their forward accuracy carries over to reflectance retrieval.

[Paper (PDF)](paper/emit_kernel_dnn.pdf) · [Supplement (PDF)](paper/supplement.pdf) ·
[LaTeX source](paper/emit_kernel_dnn.tex) ·
[How each table is produced](docs/REPRODUCE.md)

## Findings

On ten partitions of 23,313 tabulated states, each with 18,884 training rows and validation-based selection, a
Matérn kernel with one length scale per input reaches 0.095% mean relative radiance error, against 0.760% for cubic
regression and 0.387% for a 3 × 512 network. A residual kernel brings the network to 0.141%. An exact kernel on the
hidden features of a width-2,000 network reaches 0.079%, and one kernel on the concatenated features of five width-512
networks reaches 0.076%.

Forward and inverse rankings disagree. The convex stack built on the width-2,000 network improves on its own feature
kernel by 0.0006 radiance percentage points, while its mean within-split 95th-percentile reflectance error is 22.71
points against 7.57, larger at every partition. [[README-DOMAIN]]

The analysis proves that on the physical domain the constrained retrieval error is at most the retrieval-weighted
component error divided by the transmission, with constant one; that the rate this implies is sharp; and that
component errors proportional to the transmission lose no rate. Applied band by band, the bound marks in advance the
bands where the retrieval error is large.

[[README-TQ]]

[[README-LRT]]

## Layout

- `paper/`: the article (`emit_kernel_dnn.tex`), its supplement (`supplement.tex`: the proofs, the further
  experiments and the extended tables), the tables of both (generated from the records) and the two compiled PDFs.
- `code/`: drivers, scoring, table and figure scripts, and tests. `code/lanes/` holds the scripts that ran each fit of
  the training-target experiment, the libRadtran comparison and the correction-coefficient benchmark.
- `results/`: one JSON record per run, with the digests of its data and split indices. `results/target_quality/` and
  `results/libradtran/` also hold the protocols of those two experiments with the SHA-256 of each, recorded before the
  first fit; `results/confirmation/` holds the protocol, the frozen hyperparameters with their SHA-256, the scripts
  and the report of the fresh-partition evaluation.
- `figures/`: the figures of the paper.

## Reproduction

With Python 3.11 or later and a TeX installation:

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s code -p 'test_*.py' -v
python code/make_width_numbers.py
python code/make_scaling_tables.py
python code/make_retune_table.py
python code/make_emit_tables.py
python code/make_tables.py
python code/make_v2_tables.py
python code/format_publication_tables.py
python code/make_domain_table.py
python code/make_tq_tables.py
python code/make_lrt_tables.py
python code/make_lrt_tables.py lrt
python code/make_confirmation_table.py
(cd paper && pdflatex -interaction=nonstopmode supplement.tex)
latexmk -pdf -interaction=nonstopmode -halt-on-error -cd paper/emit_kernel_dnn.tex
latexmk -g -pdf -interaction=nonstopmode -halt-on-error -cd paper/supplement.tex
```

These commands run the tests and regenerate every table of the paper from the records in `results/`, byte for byte;
they do not retrain any model. [docs/REPRODUCE.md](docs/REPRODUCE.md) maps every table and figure to its records
and scripts and gives the commands that redraw the figures and rerun the training.

## Data

The EMIT table was provided by the Jet Propulsion Laboratory for EMIT atmospheric correction and is identified in the
paper by size and SHA-256; the arrays are available from the author on request, subject to JPL's permission. The
libRadtran arrays are built from the public release of the paired 6S/libRadtran Sentinel-2 corpus with
`code/make_libradtran_arrays.py`. The OCO-2 arrays and reference predictions come from the release accompanying
Lamminpää et al. (2025); the other corpora are the cited public releases.

## Contributions and license

Claude Code (Anthropic) and Codex (OpenAI) implemented and ran the experiments and drafted the manuscript; ChatGPT
(OpenAI) developed parts of the analysis. Funding and competing interests are stated in the paper. The companion
method and experiments are in
[neural-means-kernel-corrections](https://github.com/yspennstate/neural-means-kernel-corrections). The
[MIT license](LICENSE) covers the code in this repository, not rights to data that are not distributed here.
