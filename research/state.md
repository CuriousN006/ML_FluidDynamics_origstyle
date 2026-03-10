# Research State

## Current Best

- Experiment: exp-0028
- Commit: 4b4effd-dirty
- Primary score: 0.000710
- Notes: Stage 3: portrait residual autoencoder with residual linear dynamics

## Recent Experiments

- exp-0027: keep, score=0.001736, Stage 2: portrait baseline AE with residual linear dynamics
- exp-0028: keep, score=0.000710, Stage 3: portrait residual autoencoder with residual linear dynamics
- exp-0029: discard, score=0.154054, Stage 4A: residual AE + residual linear dynamics with lower gradient loss weight
- exp-0030: discard, score=0.000835, Stage 4B: residual AE + residual linear dynamics with shorter train rollout horizon
- exp-0031: discard, score=0.013512, Stage 4C: portrait residual AE + residual linear dynamics with deterministic=false

## Avoid Repeating

- `gradient_loss_weight=0.05` caused rollout collapse in exp-0029.
- `train_rollout_horizon=12` was slightly worse than the default in exp-0030.
- `deterministic=False` destabilized the portrait residual stack in exp-0031.

## Next Priorities

1. Preserve `exp-0028` as the reference portrait baseline.
2. If more optimization is needed, explore only small changes on top of the portrait residual stack.
3. Keep the old landscape MLP pipeline closed unless it is needed for debugging.
