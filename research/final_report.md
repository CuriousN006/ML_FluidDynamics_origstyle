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
- `exp-0013` validated rollout-aware selection and the CPU decode fix:
  `primary_score=0.023163`, `peak_memory_gb=2.517`.
- `exp-0018` restored parity after replacing non-deterministic encoder pooling:
  `primary_score=0.020998`.
- `exp-0021` is the current best run:
  `primary_score=0.020250`, `recon_rmse=0.615190`, `recon_mse=0.378470`,
  `rmse_t100=0.730262`, `rmse_t150=0.730221`,
  `peak_memory_gb=1.781`.

The current best nonlinear configuration is therefore:

- `latent_dim=24`
- `ae_epochs=160`
- `dyn_epochs=260`
- `ae_scheduler=plateau`
- `dyn_scheduler=plateau`
- `rollout_loss_weight=0.15`
- `dynamics_depth=2`
- `latent_l1_weight=1e-4`
- `dyn_l2_weight=1e-5`
- `deterministic=False`
- rollout-aware dynamics validation enabled

The best metrics file is `output/nonlinear/exp-0021/metrics.json`.

## 10% Test Reconstruction

The AE evaluation uses the fixed random `90/10` split produced by seed `42`.
The reconstruction comparison panels come from the held-out test set and are
saved in `output/nonlinear/exp-0021/ae_reconstruction_preview_*.png`.

For the current best run `exp-0021`:

- test reconstruction `MSE = 0.378470`
- test reconstruction `RMSE = 0.615190`
- test reconstruction `NRMSE = 0.017552`

These numbers satisfy the project requirement to compare reconstructed snapshots
from the AE against the original snapshots on the 10% test dataset.

## Future Prediction At t=100 And t=150

The future rollout panels are saved in:

- `output/nonlinear/exp-0021/rollout_step_100.png`
- `output/nonlinear/exp-0021/rollout_step_150.png`

For `exp-0021`:

- `t=100`: `MSE = 0.533282`, `RMSE = 0.730262`, `NRMSE = 0.020698`
- `t=150`: `MSE = 0.533222`, `RMSE = 0.730221`, `NRMSE = 0.020285`

The current best nonlinear model therefore improves both project forecast
targets relative to the older raw baselines.

## Latent Dynamics Interpretation

The nonlinear pipeline now saves two explicit latent-dynamics views for each run:

- `latent_pca_trajectory.png`
- `latent_time_series.png`

For `exp-0021`, the PCA trajectory plot shows a smooth time-ordered path rather
than a noisy scatter, which is consistent with a low-dimensional oscillatory
wake cycle. The first three latent coordinates also evolve smoothly over time
instead of jumping abruptly, which supports the interpretation that the latent
state is tracking a structured dynamical system rather than memorizing frames.

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
6. A clean replay and local budget check then refined the conclusion:
   - `exp-0015` cleanly replayed the committed baseline and improved the score
     again to `0.021425`.
   - `exp-0016` (`dyn280`) lost to `exp-0015`, which suggests the current
     optimum sits very near `dyn260`.
7. The next phase targeted explicit project reporting requirements:
   - `exp-0017` showed that parity was still unstable because the encoder used
     `AdaptiveAvgPool2d`, whose CUDA backward path was not truly deterministic.
   - `exp-0018` replaced that encoder pooling path with a deterministic-friendly
     resize and immediately improved the baseline to `0.020998`.
   - `exp-0019` and `exp-0020` turned on plateau schedulers, but the learning
     rate never actually stepped, so the score stayed tied with `exp-0018`.
   - `exp-0021` kept the same training policy but set `deterministic=False`,
     which reduced runtime from roughly `389s` to `143s` and improved the score
     again to `0.020250`.
8. Regularizer ablation then isolated the role of `latent_l1_weight` and
   `dyn_l2_weight` under the locked Stage B training policy. The summary is in
   `research/regularizer_ablation.md`.

## Final Model Selection

`exp-0021` is the final best model because it improves the fixed objective while
also materially lowering peak memory. The evidence is consistent across the
ledger and experiment notes:

- `results.tsv` marks `exp-0021` as a keep with the lowest current score,
  `0.020250`.
- `research/experiments/exp-0021.md` records the exact hypothesis and run
  context.
- `output/nonlinear/exp-0021/metrics.json` confirms the strongest overall
  combination of reconstruction quality, rollout quality, and memory footprint.

Compared with the earlier longer-training best `exp-0010`, `exp-0021` improves
both long-horizon rollout terms while staying in the reduced-memory regime:

- `rmse_t100`: `0.844911 -> 0.730262`
- `rmse_t150`: `0.828501 -> 0.730221`
- `peak_memory_gb`: `9.669 -> 1.781`

## Effects Of Regularizer

The dedicated regularizer ablation confirms that regularization matters, but not
all regularizers help equally.

- `Reg-0` (no regularizer) is competitive in reconstruction but loses in rollout.
- `Reg-1` (AE `latent_l1_weight` only) is clearly worse, which suggests the L1
  term by itself is not enough and may over-constrain the latent code.
- `Reg-2` (dynamics `dyn_l2_weight` only) stays close to the no-regularizer run,
  so the dynamics-side regularizer appears more useful than the AE-side one.
- `Reg-3` (default pair) is the best overall setting and gives the strongest
  combined rollout result.
- `Reg-4` (stronger pair) underfits, hurting both reconstruction and rollout.

In short, the default combined regularization improves long-horizon stability,
while overly strong regularization hurts performance.

## Limitations And Next Steps

The nonlinear score is still much larger than the linear DMD reconstruction
errors, so there is room to improve the learned latent dynamics.

Highest-value next steps:

- Keep `deterministic=False` as the current best runtime/performance setting and
  only revisit strict determinism if exact replay becomes more important than score.
- Treat `dyn_epochs=260` as the current sweet spot. `dyn280` and `dyn320`
  already suggest that simply training longer is no longer the right lever.
- If scheduler exploration resumes, make it more aggressive. In the current
  plateau setup the learning rate never stepped.
- Revisit AE-side refinements only after the training-policy search is exhausted.
- Leave image-only and hybrid branches for a later phase; raw `VORTALL` remains
  the canonical mainline.
