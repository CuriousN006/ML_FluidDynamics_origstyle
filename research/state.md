# Research State

## Current Best

- Experiment: exp-0010
- Commit: 3cf7f7b
- Primary score: 0.023211
- Notes: raw sweep: latent_dim=24, ae160 dyn260

## Recent Experiments

- exp-0007: discard, score=0.024609, raw sweep: latent_dim=32
- exp-0008: discard, score=0.028337, raw sweep: baseline latent_dim=24, dynamics_hidden_dim=96
- exp-0009: keep, score=0.023994, raw sweep: latent_dim=24, longer training (ae160 dyn200)
- exp-0010: keep, score=0.023211, raw sweep: latent_dim=24, ae160 dyn260
- exp-0011: discard, score=0.029603, raw sweep: latent_dim=24, ae160 dyn320

## Avoid Repeating

- Review discarded runs before retrying the same idea.

## Next Priorities

1. Inspect the latest discarded run and extract the lesson.
2. Try the next highest-value architecture or regularization change.
3. Update the long-form report after a meaningful improvement.
