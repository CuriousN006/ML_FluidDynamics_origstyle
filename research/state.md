# Research State

## Current Best

- Experiment: exp-0051
- Commit: 26e4179-dirty
- Primary score: 0.000565
- Notes: Latent sweep Stage 2 | residual_refine latent=32 full rollout

## Recent Experiments

- exp-0054: discard, score=0.000569, Latent sweep Stage 3 R3 | residual_refine latent=32 rollout_loss_weight=0.30
- exp-0055: discard, score=0.001127, Reg-0 | latent32 residual_refine regularizer ablation
- exp-0056: discard, score=0.000979, Reg-2 | latent32 residual_refine regularizer ablation
- exp-0057: discard, score=0.000566, Reg-3 | latent32 residual_refine regularizer ablation
- exp-0058: discard, score=0.242610, Reg-4 | latent32 residual_refine regularizer ablation

## Avoid Repeating

- Do not retry `latent_dim=48` on the current `residual_refine` branch.
- Do not reopen `rollout_loss_weight` around `0.15` unless another architectural change shifts the optimum.
- Strong regularization (`Reg-4`) is clearly off the table.

## Next Priorities

1. Preserve `exp-0051` as the new nonlinear reference for future branches.
2. Target the remaining DMD gap with a decoder-side AE-floor improvement, not another dynamics-only tweak.
3. Keep future screening cheap: AE-only first, then full rollout only for candidates that improve the floor.
