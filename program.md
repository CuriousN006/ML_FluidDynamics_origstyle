# Fluid Autoresearch Program

This file is the human-authored charter for autonomous research in this repo.

## Mission

Improve nonlinear prediction quality for the cylinder wake project while keeping
the fixed data contract and evaluation logic intact. The primary objective is to
minimize the `primary_score` written by `python -m mlfd.run_nonlinear`.

## Fixed Rules

1. Use `CYLINDER_ALL.mat` as the primary data source.
2. Do not change the meaning of `primary_score` or the ledger columns in `results.tsv`.
3. Preserve the append-only research memory:
   - `results.tsv`
   - `research/experiments/*.md`
   - `research/state.md`
   - `research/runs/20260311-fluid-rtx3070-latent-sweep/index.md`
4. Keep the linear pipeline stable unless a bug blocks baseline artifact generation.
5. Keep the portrait layout and portrait comparison figures as the nonlinear default.

## Current Working Baseline

Treat `exp-0051` as the reference model in this branch:

- `layout=portrait`
- `ae_architecture=residual_refine`
- `ae_width_mult=1.0`
- `coordconv=False`
- `dynamics_model=residual_linear`
- `latent_dim=32`
- `coarse_loss_weight=0.25`
- `coarse_blur_kernel=9`
- `coarse_blur_sigma=2.0`
- `refine_blocks=1`
- `refine_channels_mult=1.25`
- `gradient_loss_weight=0.10`
- `fft_loss_weight=0.0`
- `latent_l1_weight=1e-4`
- `dyn_l2_weight=0`
- `rollout_loss_weight=0.15`
- `train_rollout_horizon=16`
- `train_rollout_stride=8`
- `validation_rollout_horizon=24`
- `validation_rollout_stride=4`
- `deterministic=True`

Reference metrics:

- `primary_score=0.000565`
- `recon_rmse=0.020886`
- `rmse_t100=0.020372`
- `rmse_t150=0.018893`
- `ae_floor_rmse_t150=0.018927`

## Locked Lessons

1. Portrait layout is not optional for the nonlinear mainline.
2. Residual linear dynamics is better than the old MLP-only dynamics on the portrait baseline.
3. The coarse-to-fine `residual_refine` AE is the current best family in this budget.
4. `latent_dim=32` beats `latent_dim=24` on the recovered pipeline.
5. `latent_dim=48` is a regression on this architecture.
6. `rollout_loss_weight=0.15` remains the best value around the latent-32 winner.
7. `latent_l1_weight=1e-4` still matters more than `dyn_l2_weight`.
8. Strong regularization still destroys long-horizon rollout even when one-step reconstruction looks acceptable.
9. The remaining DMD gap is now mostly an AE-floor problem, not a dynamics instability problem.

## Suggested Experiment Order

1. Preserve `exp-0051` as the reference baseline and compare every new run directly against it.
2. Do not revisit the old landscape MLP stack except for debugging.
3. If more improvement is needed, target the AE floor directly:
   - lightweight decoder-side ideas
   - AE-only screening before full rollout promotion
   - narrow hyperparameter searches around the winning family
4. Only keep changes that improve `primary_score`, or that preserve the score while clearly lowering VRAM or runtime.

## Runtime Budget

- Hard timeout per experiment: 12 minutes.
- Target runtime per experiment: 10 minutes.
- Campaign budget: 1 baseline plus up to 24 additional trials.

## Keep/Discard Policy

- Better `primary_score`: keep.
- Worse `primary_score`: discard.
- Within 0.25%: keep only if code is simpler or GPU memory is at least 5% lower.
- Crash or timeout: log it and move on.

## Logging Requirements

Every experiment must update:

1. `results.tsv`
2. `research/experiments/exp-XXXX.md`
3. `research/state.md`
4. `research/runs/20260311-fluid-rtx3070-latent-sweep/index.md`

Failures are first-class results and must stay recorded.
