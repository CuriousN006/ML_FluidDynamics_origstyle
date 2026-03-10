# Research State

## Current Best

- Experiment: exp-0015
- Commit: 6022104
- Primary score: 0.021425
- Notes: clean baseline replay: rollout-aware deterministic raw baseline

## Recent Experiments

- exp-0013: keep, score=0.023163, raw baseline refresh: deterministic seed + rollout-aware dynamics selection (cpu decode fix)
- exp-0014: discard, score=0.024582, raw sweep: rollout-aware selection with dyn320
- exp-0015: keep, score=0.021425, clean baseline replay: rollout-aware deterministic raw baseline
- exp-0016: discard, score=0.022834, raw sweep: clean baseline with dyn280
- exp-0017: discard, score=0.024719, parity run: exp-0015 baseline with expanded metrics

## Avoid Repeating

- Review discarded runs before retrying the same idea.

## Next Priorities

1. Inspect the latest discarded run and extract the lesson.
2. Try the next highest-value architecture or regularization change.
3. Update the long-form report after a meaningful improvement.
