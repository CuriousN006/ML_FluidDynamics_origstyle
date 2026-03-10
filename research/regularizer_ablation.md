# Regularizer Ablation

The regularizer study was run under the locked Stage B training policy:

- `ae_scheduler=plateau`
- `dyn_scheduler=plateau`
- `deterministic=False`
- same seed and same 10% test split as the current baseline

| Label | Experiment | latent_l1_weight | dyn_l2_weight | primary_score | recon_rmse | rmse_t100 | rmse_t150 | peak_memory_gb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Reg-0 | exp-0022 | 0 | 0 | 0.020768 | 0.613836 | 0.760325 | 0.748900 | 1.781 |
| Reg-1 | exp-0023 | 1e-4 | 0 | 0.022191 | 0.682136 | 0.799338 | 0.797563 | 1.781 |
| Reg-2 | exp-0024 | 0 | 1e-5 | 0.020955 | 0.624919 | 0.766877 | 0.753601 | 1.781 |
| Reg-3 | exp-0021 | 1e-4 | 1e-5 | 0.020250 | 0.615190 | 0.730262 | 0.730221 | 1.781 |
| Reg-4 | exp-0025 | 5e-4 | 5e-5 | 0.022281 | 0.664550 | 0.778349 | 0.823510 | 1.781 |

## Takeaways

- The default combined regularization (`Reg-3`) is the best overall setting.
- Removing all regularization (`Reg-0`) slightly improves reconstruction but hurts rollout enough to lose overall.
- `dyn_l2_weight` contributes more than `latent_l1_weight` on its own. `Reg-2` stays near the no-regularizer result, while `Reg-1` degrades more clearly.
- Stronger regularization (`Reg-4`) underfits both reconstruction and rollout.
