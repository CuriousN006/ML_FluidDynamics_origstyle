# Final Report

## Problem Definition

The project models the 2D cylinder wake snapshots in `CYLINDER_ALL.mat` and
matches the deliverables in `Final_project_DL2025-5.pdf`: a linear reduced-order
baseline (`SVD`, truncated linear dynamics, `DMD`) and a nonlinear
autoencoder-plus-latent-dynamics baseline.

The nonlinear objective used for agentic search is the fixed `primary_score`
written by `python -m mlfd.run_nonlinear`. Lower is better. The score combines
reconstruction quality and rollout quality at `t=100` and `t=150`.

## Data Understanding

The canonical input is `VORTALL` from `CYLINDER_ALL.mat`. The data loader reads
`nx=199`, `ny=449`, reshapes the raw snapshot matrix to `(151, 199, 449)`, and
uses `dt=0.2`. The full summary is stored in `output/linear/baseline/metrics.json`
and repeated in each nonlinear metrics file.

`VORTALL` remains the primary field because it aligns directly with the project
questions and avoids extra ambiguity from multi-field coupling. Existing PNG
dumps are treated as derived visualizations, not the canonical training source.

## Linear Baseline

The linear baseline is stable and serves as a fixed reference. The metrics are in
`output/linear/baseline/metrics.json`.

- The leading singular values decay sharply from `3575.79` to `1299.34`, `1278.52`,
  then into a lower-energy tail.
- The truncated rank-10 reconstruction stays accurate:
  `rmse(step_50)=0.03067`, `rmse(step_100)=0.02503`, `rmse(step_150)=0.02804`.
- The fitted rank-10 linear latent dynamics gives:
  `rmse(step_50)=0.03389`, `rmse(step_100)=0.03521`, `rmse(step_150)=0.03575`.
- The DMD sweep selects rank `15` as the best compact setting with mean
  `nrmse=0.000508` and no spurious eigenvalues outside the unit circle.

The corresponding figures and comparison panels live under `output/linear/baseline/`.

## Nonlinear Baseline

The nonlinear work started from the original baseline in
`output/nonlinear/exp-0003/metrics.json` and improved through targeted raw-only
search on `VORTALL`.

- `exp-0003` established the first full baseline:
  `primary_score=0.027961`, `recon_rmse=0.724065`,
  `rmse_t100=0.922945`, `rmse_t150=1.109648`.
- `exp-0004` showed that `latent_dim=24` is materially better than the original
  latent size:
  `primary_score=0.024610`.
- `exp-0010` showed that longer training is more valuable than deeper dynamics:
  `ae_epochs=160`, `dyn_epochs=260` reached `primary_score=0.023211`.
- `exp-0013` is the current best run:
  `primary_score=0.023163`, `recon_rmse=0.628368`,
  `rmse_t100=0.807242`, `rmse_t150=0.882237`,
  `peak_memory_gb=2.517`.

The current best nonlinear configuration is therefore:

- `latent_dim=24`
- `ae_epochs=160`
- `dyn_epochs=260`
- `rollout_loss_weight=0.15`
- `dynamics_depth=2`
- deterministic seeding enabled
- rollout-aware dynamics validation enabled

The best metrics file is `output/nonlinear/exp-0013/metrics.json`.

## Agentic Research Timeline

The experiment ledger in `results.tsv` and the append-only notes in
`research/experiments/` show a clear progression.

1. The initial nonlinear baseline (`exp-0003`) worked but left obvious rollout
   error, especially at `t=150`.
2. A small sweep around the raw baseline showed:
   - `latent_dim=24` is a true gain (`exp-0004`, keep).
   - increasing rollout-loss weight to `0.20` hurt (`exp-0005`, discard).
   - increasing dynamics depth to `3` hurt (`exp-0006`, discard).
   - widening latent size to `32` was nearly tied but not decisively better
     under the keep policy (`exp-0007`, discard).
   - increasing hidden width to `96` clearly hurt (`exp-0008`, discard).
3. Longer training was then explored:
   - `exp-0009` (`ae160 dyn200`) improved to `0.023994`.
   - `exp-0010` (`ae160 dyn260`) improved again to `0.023211`.
   - `exp-0011` (`ae160 dyn320`) collapsed to `0.029603`, showing that the old
     one-step validation criterion was misaligned with long rollout quality.
4. The code was then updated to improve research quality:
   - deterministic seeding was enabled;
   - data loader randomness was seeded explicitly;
   - dynamics checkpoint selection was changed to use a mixed validation signal
     with both one-step and short-rollout loss;
   - final rollout decoding was moved to CPU to avoid a deterministic-mode CUDA OOM.
5. Those changes produced:
   - `exp-0012` crash, which documented the GPU decode OOM under deterministic mode.
   - `exp-0013` keep, which became the new best run.
   - `exp-0014` discard, which revisited `dyn320` and showed that rollout-aware
     selection prevents full collapse but still does not beat the `dyn260` regime.

## Final Model Selection

`exp-0013` is the final best model because it improves the fixed objective while
also materially lowering peak memory. The evidence is consistent across the
ledger and experiment notes:

- `results.tsv` marks `exp-0013` as a keep with the lowest current score,
  `0.023163`.
- `research/experiments/exp-0013.md` records the exact hypothesis and run
  context.
- `output/nonlinear/exp-0013/metrics.json` confirms the strongest overall
  combination of reconstruction quality, rollout quality, and memory footprint.

Compared with the prior best `exp-0010`, `exp-0013` trades a slightly worse
`t=150` RMSE for a stronger reconstruction term, a better `t=100` RMSE, and a
large GPU-memory reduction (`9.669 GB -> 2.517 GB`). Under the current keep
policy, that combination is the best overall checkpoint.

## Limitations And Next Steps

The nonlinear score is still much larger than the linear DMD reconstruction
errors, so there is room to improve the learned latent dynamics.

Highest-value next steps:

- Add a light learning-rate scheduler before changing architecture again.
- Keep the rollout-aware validation selector, but test whether full deterministic
  algorithms are worth the runtime cost or whether seeded data loading alone is
  enough.
- Explore a narrow band around the current dynamics budget (`dyn_epochs` near
  `220-280`) instead of pushing to `320+`.
- Revisit AE-side refinements only after the dynamics training policy stabilizes.
- Leave image-only and hybrid branches for a later phase; raw `VORTALL` remains
  the canonical mainline.
