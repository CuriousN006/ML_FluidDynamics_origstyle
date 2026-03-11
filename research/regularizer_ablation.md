# Regularizer Ablation

This ablation was rerun on the latent-32 branch winner so that the assignment
discussion matches the actual final model in this worktree.

Locked baseline for this study:

- `layout=portrait`
- `ae_architecture=residual_refine`
- `ae_width_mult=1.0`
- `coordconv=False`
- `latent_dim=32`
- `coarse_loss_weight=0.25`
- `coarse_blur_kernel=9`
- `coarse_blur_sigma=2.0`
- `refine_blocks=1`
- `refine_channels_mult=1.25`
- `gradient_loss_weight=0.10`
- `fft_loss_weight=0.0`
- `dynamics_model=residual_linear`
- `rollout_loss_weight=0.15`
- `deterministic=True`

| Label | Experiment | latent_l1_weight | dyn_l2_weight | primary_score | recon_rmse | rmse_t100 | rmse_t150 | peak_memory_gb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Reg-0 | exp-0055 | 0 | 0 | 0.001127 | 0.023670 | 0.040122 | 0.045211 | 1.964 |
| Reg-1 | exp-0051 | 1e-4 | 0 | 0.000565 | 0.020886 | 0.020372 | 0.018893 | 1.964 |
| Reg-2 | exp-0056 | 0 | 1e-5 | 0.000979 | 0.023670 | 0.031120 | 0.040230 | 1.964 |
| Reg-3 | exp-0057 | 1e-4 | 1e-5 | 0.000566 | 0.020886 | 0.020433 | 0.018917 | 1.964 |
| Reg-4 | exp-0058 | 5e-4 | 5e-5 | 0.242610 | 0.028399 | 5.320233 | 13.742015 | 1.981 |

## Takeaways

- `latent_l1_weight=1e-4` is still the best regularizer. `Reg-1` remains the
  overall winner.
- `dyn_l2_weight` is not helpful on its own for the latent-32 model.
  `Reg-2` improves over `Reg-0`, but both are much worse than `Reg-1`.
- The combined setting `Reg-3` is stable and nearly tied with `Reg-1`, but it
  is still slightly worse at both `primary_score` and `rmse_t150`.
- Strong regularization in `Reg-4` clearly over-constrains the model and causes
  severe long-horizon failure.
- In this branch, regularization is still mainly about preserving stable future
  prediction rather than minimizing one-step reconstruction alone.
