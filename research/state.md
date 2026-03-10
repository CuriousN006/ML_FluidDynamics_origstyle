# Research State

## Current Best

- Experiment: exp-0021
- Commit: 25ec4d3-dirty
- Primary score: 0.020250
- Notes: Sched-3: dual plateau with deterministic=false runtime check

## Recent Experiments

- exp-0021: keep, score=0.020250, Sched-3: dual plateau with deterministic=false runtime check
- exp-0022: discard, score=0.020768, Reg-0: no regularizer under dual plateau deterministic=false
- exp-0023: discard, score=0.022191, Reg-1: latent L1 only under dual plateau deterministic=false
- exp-0024: discard, score=0.020955, Reg-2: dynamics L2 only under dual plateau deterministic=false
- exp-0025: discard, score=0.022281, Reg-4: strong AE+dyn regularizers under dual plateau deterministic=false

## Avoid Repeating

- Review discarded runs before retrying the same idea.

## Next Priorities

1. Preserve `exp-0021` as the new reference baseline in the report and research charter.
2. Use the regularizer ablation to justify why the default combined regularizer stays on.
3. If more optimization is needed, try a scheduler that actually lowers the learning rate before changing architecture.
