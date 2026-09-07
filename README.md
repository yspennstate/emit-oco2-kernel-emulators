# Neural means and kernel corrections for radiative-transfer emulators

Code, per-run records and manuscript for a study of neural networks combined with exact kernel
regression on two atmospheric radiative-transfer emulation problems: a tabulated model on the
285-band EMIT wavelength grid, and the OCO-2 reduced-radiance problem of Lamminpää et al. The
question throughout is which of the two model families a given emulation problem favours, and
whether coupling them keeps the advantage of each.

The manuscript is [`paper/emit_kernel_dnn.pdf`](paper/emit_kernel_dnn.pdf); its LaTeX sources,
tables and figures are beside it.

## What the study finds

On the EMIT table the map is smooth, six-dimensional and low rank, and an exact Matérn
regression fitted on all 18,884 training rows with one length scale per input is the strongest
single model: 0.095% relative radiance error against 0.76% for a cubic polynomial ridge and
0.39% for a fully connected network. Its fitted metric recovers the physics — the transmittances and the
spherical albedo get a relative-azimuth length scale sixteen times longer than the other
coordinates, at every split, and the path radiance does not. A convex stack of the heads is the
best row of the table at 0.087%.

On OCO-2 the input is higher-dimensional and the map rougher, a kernel on the state trails the
network by an order of magnitude, and the useful construction is the kernel fitted inside the
network's representation. A ridge readout refitted on the same frozen features is the control:
the kernel head beats it at all ten splits on every band and under every training loss, while
the ridge readout is worse than the network it reads out from. The gain is the kernel, not the
refitting.

The two reported metrics on OCO-2 — relative error on the forty reduced coefficients, and on the
reconstructed monochromatic radiance — are not proxies for each other, and which one a model wins
is decided by the loss it was trained under. The reconstruction basis is orthogonal, so a
per-coefficient choice acts on both metrics through the same per-coefficient errors rather than
trading one against the other; whether it improves both is then a question for measurement, and
here it does. The choice made on validation assigns the leading coefficient to the kernel head of
the network trained on the weighted coefficients, at every split on every band, and the remaining
thirty-nine to the kernel head of the coefficient-trained network. The result is more accurate
than the release's stored kernel-flow emulator on both metrics on all three bands.

A separate study of how the heads should be weighted finds that random-matrix estimators which
clean the covariance of the member predictions lose, because the leading eigenvalue of that
matrix is the shared signal; the object to clean is the covariance of the member errors, and
once the stack is written as a minimum-variance portfolio on it with nonnegative weights, the
constraint rather than the cleaning is what does the work.

## Layout

```
paper/       LaTeX sources, tables, figures and the compiled manuscript
code/        the drivers that produced every number
results/     one JSON record per run, the evidence behind every table
  emit/               EMIT campaign, seeds 101-110, all model families
  oco2_losses/        OCO-2, three training losses x {network, ridge readout, kernel head}
  oco2_ensembles/     OCO-2, single-network and three-member campaigns, ten splits per band
  corpora/            ten further emulation configurations under one protocol
  stacking/           the weighting study's result tables
docs/        how to reproduce each table
```

## Reproducing the tables

`docs/REPRODUCE.md` maps each table and each quoted number to the records that produce it.
`code/make_tables.py` regenerates the two OCO-2 tables and the corpora table from `results/`
alone; it needs only the standard library.

## Data

The EMIT dataset consists of tabulated radiative-transfer evaluations generated at the Jet
Propulsion Laboratory and is not redistributed here. The OCO-2 data and the reference emulator's
stored predictions come from the release accompanying Lamminpää et al. (2025). The ten
further configurations use public releases: the operator suite of de Hoop et al. (2022),
PDEBench, The Well, ClimSim, and the paired correction-coefficient corpus of Mazid and Rishe
(2026), which is run in both of its configurations. The per-run
summaries in `results/` are the numbers behind every table; the raw arrays, trained weights and
prediction dumps are available from the author on request.

## Companion work

The method and its taxonomy are developed in
[neural-means-kernel-corrections](https://github.com/yspennstate/neural-means-kernel-corrections),
which studies the same construction on structural mechanics and on OCO-2.

## Citing

Please cite the manuscript together with the sources of the data: Lamminpää, Susiluoto, Hobbs,
McDuffie, Braverman and Owhadi (2025) for the OCO-2 emulation problem, and the releases named
above for the additional corpora.

## License

MIT, see [LICENSE](LICENSE).
