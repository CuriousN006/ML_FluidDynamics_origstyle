# AE-Only Screening Summary

This branch tested whether the recovered `residual_refine` nonlinear baseline
is still bottlenecked by `latent_dim=24`.

Reference AE-only floor from the parity replay `exp-0050`:

- `ae_score=0.022144`
- `recon_rmse=0.024214`
- `ae_floor_rmse_t100=0.020707`
- `ae_floor_rmse_t150=0.019441`

| Label | Output Tag | latent_dim | recon_rmse | ae_floor_rmse_t100 | ae_floor_rmse_t150 | ae_score | peak_memory_gb |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L1 | `latent32-ae` | 32 | 0.020886 | 0.019860 | 0.018927 | 0.020140 | 1.964 |
| L2 | `latent48-ae` | 48 | 0.023869 | 0.022861 | 0.022483 | 0.023270 | 1.981 |

## Conclusion

- `latent_dim=32` is the clear AE-only winner.
- `latent_dim=32` lowered the AE floor at both required forecast targets and
  earned promotion to the full nonlinear pipeline.
- `latent_dim=48` made the floor worse and was discarded before rollout
  reevaluation.
