# Research State

## Current Best

- Experiment: exp-0013
- Commit: 77a8a03
- Primary score: 0.023163
- Notes: raw baseline refresh: deterministic seed + rollout-aware dynamics selection (cpu decode fix)

## Recent Experiments

- exp-0010: keep, score=0.023211, raw sweep: latent_dim=24, ae160 dyn260
- exp-0011: discard, score=0.029603, raw sweep: latent_dim=24, ae160 dyn320
- exp-0012: crash, score=0.000000, raw baseline refresh: deterministic seed + rollout-aware dynamics selection
- exp-0013: keep, score=0.023163, raw baseline refresh: deterministic seed + rollout-aware dynamics selection (cpu decode fix)
- exp-0014: discard, score=0.024582, raw sweep: rollout-aware selection with dyn320

## Avoid Repeating

- Review discarded runs before retrying the same idea.

## Next Priorities

1. Inspect the latest discarded run and extract the lesson.
2. Try the next highest-value architecture or regularization change.
3. Update the long-form report after a meaningful improvement.
