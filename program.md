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
   - `research/runs/20260310-fluid-rtx3070/index.md`
4. Keep the linear pipeline stable unless a bug blocks baseline artifact generation.
5. Keep the portrait layout and portrait comparison figures as the nonlinear default.

## Current Working Baseline

Treat `exp-0028` as the reference model:

- `layout=portrait`
- `ae_architecture=residual`
- `dynamics_model=residual_linear`
- `latent_dim=24`
- `gradient_loss_weight=0.10`
- `train_rollout_horizon=16`
- `train_rollout_stride=8`
- `validation_rollout_horizon=24`
- `validation_rollout_stride=4`
- `deterministic=True`

Reference metrics:

- `primary_score=0.000710`
- `recon_rmse=0.025058`
- `rmse_t100=0.026003`
- `rmse_t150=0.023945`

## Locked Lessons

1. Portrait layout is not optional for the nonlinear mainline. It dramatically improves both interpretability and model quality.
2. Residual linear dynamics is better than the old MLP-only dynamics on the portrait baseline.
3. The residual AE is the main improvement over the portrait baseline AE.
4. `gradient_loss_weight=0.05` is a bad regression. It preserves low AE loss but destroys rollout.
5. `train_rollout_horizon=12` is slightly worse than `16`.
6. `deterministic=False` is not safe for the portrait residual architecture, even if it lowers runtime.

## Suggested Experiment Order

1. Preserve `exp-0028` as the reference baseline and compare every new run directly against it.
2. Do not revisit the old landscape MLP stack except for debugging.
3. If more improvement is needed, try only small follow-ups on top of the portrait residual baseline:
   - latent width
   - scheduler that actually steps
   - minor rollout-weight tuning
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
4. `research/runs/20260310-fluid-rtx3070/index.md`

Failures are first-class results and must stay recorded.
