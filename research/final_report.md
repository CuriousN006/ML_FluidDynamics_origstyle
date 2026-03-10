# Final Report

## Problem Definition

Describe the cylinder wake modeling objective, the data source, and the required
linear and nonlinear deliverables. Reference the project PDF and the primary data file.

## Data Understanding

Summarize the `CYLINDER_ALL.mat` contents, reshape conventions, time indexing,
and the rationale for using `VORTALL` as the baseline field.

## Linear Baseline

Report the singular spectrum, leading modes, rank-10 reconstruction, linear
dynamics rollout, and DMD sweep. Cite the figures and metrics files in `output/linear/`.

## Nonlinear Baseline

Report the autoencoder setup, 10% reconstruction test split, latent dynamics
training, and `t=100/150` rollout results. Cite the metrics in `output/nonlinear/`.

## Agentic Research Timeline

Build a narrative from the experiment markdown files:

- Which hypotheses were attempted
- Which changes improved `primary_score`
- Which changes were discarded and why
- What failures were informative

## Final Model Selection

Explain why the chosen experiment became the final best run. Cite `results.tsv`,
the relevant experiment notes, and the saved metrics.

## Limitations And Next Steps

List unresolved weaknesses, potential multi-field extensions, and the highest
value follow-up experiments.

