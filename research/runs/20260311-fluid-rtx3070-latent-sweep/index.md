# Run 20260311-fluid-rtx3070-latent-sweep

## Goal

Validate whether the `exp-0046` bottleneck is still limited by `latent_dim=24`
and whether a larger latent can beat the mainline nonlinear baseline without
changing the architecture family.

Reference mainline baseline from `D:\Projects\ML_FluidDynamics`:

- `exp-0046`
- `primary_score=0.000613`
- `recon_rmse=0.024214`
- `rmse_t100=0.022138`
- `rmse_t150=0.019868`
- `ae_floor_rmse_t150=0.019441`

## Timeline

- `exp-0050` | discard | `score=0.000613` | separate worktree parity replay of `exp-0046`
- `latent32-ae` | screen winner | `ae_score=0.020140` | AE-only latent sweep candidate, improved AE floor to `0.018927`
- `latent48-ae` | screen discard | `ae_score=0.023270` | AE-only latent sweep candidate, worse than reference
- `exp-0051` | keep | `score=0.000565` | latent sweep Stage 2 full rollout with `latent_dim=32`
- `exp-0052` | discard | `score=0.000567` | rollout weight `0.10`
- `exp-0053` | discard | `score=0.000566` | rollout weight `0.20`
- `exp-0054` | discard | `score=0.000569` | rollout weight `0.30`
- `exp-0055` | discard | `score=0.001127` | Reg-0 on the latent-32 winner
- `exp-0056` | discard | `score=0.000979` | Reg-2 on the latent-32 winner
- `exp-0057` | discard | `score=0.000566` | Reg-3 on the latent-32 winner
- `exp-0058` | discard | `score=0.242610` | Reg-4 on the latent-32 winner

## Best Model Changes

- Branch best: `exp-0051`
- New branch best metrics:
  - `primary_score=0.000565`
  - `recon_rmse=0.020886`
  - `rmse_t100=0.020372`
  - `rmse_t150=0.018893`
  - `ae_floor_rmse_t150=0.018927`
- Relative to mainline `exp-0046`:
  - `primary_score`: `0.000613 -> 0.000565`
  - `recon_rmse`: `0.024214 -> 0.020886`
  - `rmse_t100`: `0.022138 -> 0.020372`
  - `rmse_t150`: `0.019868 -> 0.018893`

## Decisions

- `latent_dim=32` is a real improvement and is now the branch winner.
- `latent_dim=48` is not worth promoting.
- `rollout_loss_weight=0.15` remains the best setting around the latent-32
  winner.
- `latent_l1_weight=1e-4`, `dyn_l2_weight=0` remains the best regularizer
  choice on the latent-32 model.

## Open Risks

- The nonlinear model still trails the best DMD baseline:
  - DMD `rmse_t150=0.018166`
  - branch best `rmse_t150=0.018893`
- The remaining gap is much smaller now, but the AE floor still sits above DMD.
