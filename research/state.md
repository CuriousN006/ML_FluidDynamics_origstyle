# Research State

## Current Best

- Experiment: exp-0004
- Commit: 19b9a93
- Primary score: 0.024610
- Notes: raw sweep: latent_dim=24

## Recent Experiments

- exp-0004: keep, score=0.024610, raw sweep: latent_dim=24
- exp-0005: discard, score=0.026242, raw sweep: latent_dim=24, rollout_loss_weight=0.20
- exp-0006: discard, score=0.027935, raw sweep: latent_dim=24, dynamics_depth=3
- exp-0007: discard, score=0.024609, raw sweep: latent_dim=32
- exp-0008: discard, score=0.028337, raw sweep: baseline latent_dim=24, dynamics_hidden_dim=96

## Avoid Repeating

- Review discarded runs before retrying the same idea.

## Next Priorities

1. Inspect the latest discarded run and extract the lesson.
2. Try the next highest-value architecture or regularization change.
3. Update the long-form report after a meaningful improvement.
