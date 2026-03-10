# Regularizer Ablation

This ablation was rerun on the new coarse-to-fine nonlinear baseline so that
the assignment discussion matches the actual final model instead of the older
portrait residual AE.

Locked baseline for this study:

- `layout=portrait`
- `ae_architecture=residual_refine`
- `ae_width_mult=1.0`
- `coordconv=False`
- `latent_dim=24`
- `coarse_loss_weight=0.25`
- `coarse_blur_kernel=9`
- `coarse_blur_sigma=2.0`
- `refine_blocks=1`
- `refine_channels_mult=1.25`
- `gradient_loss_weight=0.10`
- `fft_loss_weight=0.0`
- `dynamics_model=residual_linear`
- `deterministic=True`

| Label | Experiment | latent_l1_weight | dyn_l2_weight | primary_score | recon_rmse | rmse_t100 | rmse_t150 | peak_memory_gb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Reg-0 | exp-0045 | 0 | 0 | 0.000734 | 0.022264 | 0.021074 | 0.029685 | 1.957 |
| Reg-1 | exp-0046 | 1e-4 | 0 | 0.000613 | 0.024214 | 0.022138 | 0.019868 | 1.957 |
| Reg-2 | exp-0047 | 0 | 1e-5 | 0.000720 | 0.022264 | 0.021016 | 0.028791 | 1.957 |
| Reg-3 | exp-0048 | 1e-4 | 1e-5 | 0.000670 | 0.024214 | 0.023006 | 0.023289 | 1.957 |
| Reg-4 | exp-0049 | 5e-4 | 5e-5 | 0.092935 | 0.021139 | 2.788586 | 4.809601 | 1.957 |

## Takeaways

- `latent_l1_weight=1e-4` is now the best regularizer. `Reg-1` is the overall
  winner and becomes the new nonlinear baseline.
- `dyn_l2_weight` is not helpful on its own for the coarse-to-fine decoder.
  `Reg-2` is only marginally better than `Reg-0`, and both are much worse than
  `Reg-1` at `t=150`.
- The tradeoff changed compared with the older residual AE: `Reg-0` and
  `Reg-2` slightly improve raw reconstruction RMSE, but they hurt long-horizon
  rollout stability. In this architecture, regularization matters more for
  stable future prediction than for lowering the AE floor.
- The combined setting `Reg-3` is still safe and much better than `Reg-0`, but
  it is no longer the best choice. The extra dynamics L2 term slightly hurts
  the final rollout relative to `Reg-1`.
- Stronger regularization in `Reg-4` is clearly too much. Reconstruction stays
  numerically low, but the rollout diverges and the wake dynamics collapse,
  which is consistent with severe over-constraint and underfitting.
