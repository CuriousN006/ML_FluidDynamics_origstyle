# Regularizer Ablation

This ablation was rerun on the recovered portrait residual AE architecture so
that the result matches the final nonlinear mainline instead of the older
pre-recovery model.

Locked baseline for this study:

- `layout=portrait`
- `ae_architecture=residual`
- `ae_width_mult=1.0`
- `coordconv=False`
- `latent_dim=24`
- `gradient_loss_weight=0.10`
- `fft_loss_weight=0.0`
- `dynamics_model=residual_linear`
- `deterministic=True`

| Label | Experiment | latent_l1_weight | dyn_l2_weight | primary_score | recon_rmse | rmse_t100 | rmse_t150 | peak_memory_gb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Reg-0 | exp-0039 | 0 | 0 | 0.013191 | 0.091986 | 0.540997 | 0.500955 | 1.724 |
| Reg-1 | exp-0040 | 1e-4 | 0 | 0.000709 | 0.025058 | 0.026003 | 0.023907 | 1.724 |
| Reg-2 | exp-0041 | 0 | 1e-5 | 0.013191 | 0.091986 | 0.540997 | 0.500955 | 1.724 |
| Reg-3 | exp-0037 | 1e-4 | 1e-5 | 0.000710 | 0.025058 | 0.026003 | 0.023945 | 1.724 |
| Reg-4 | exp-0042 | 5e-4 | 5e-5 | 0.000764 | 0.025903 | 0.026463 | 0.024683 | 1.724 |

## Takeaways

- Some regularization is necessary. Removing both regularizers in `Reg-0`
  causes the rollout to collapse even though the architecture is otherwise
  unchanged.
- `latent_l1_weight=1e-4` is the key regularizer in the recovered model.
  `Reg-1` nearly exactly matches the recovered baseline and slightly edges the
  combined setting at `t=150`.
- `dyn_l2_weight` alone is not enough. `Reg-2` falls back to the same bad
  behavior as `Reg-0`, so the AE-side latent sparsity matters more than the
  dynamics-side L2 term here.
- The combined default `Reg-3` is still a safe, conservative choice because it
  reproduces the current best configuration almost exactly.
- Stronger regularization in `Reg-4` underfits mildly. Reconstruction and
  rollout both worsen, which is consistent with excessive smoothing of the wake.
