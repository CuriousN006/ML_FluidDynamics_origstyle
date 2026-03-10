# Research State

## Current Best

- Experiment: exp-0028
- Commit: 4b4effd-dirty
- Primary score: 0.000710
- Notes: Stage 3: portrait residual autoencoder with residual linear dynamics

## Recent Experiments

- exp-0038: discard, score=0.068061, AE search top-2 reevaluation | family=residual | latent=24 | width=1.000 | coordconv=True
- exp-0039: discard, score=0.013191, Reg-0 | residual AE regularizer ablation
- exp-0040: discard, score=0.000709, Reg-1 | latent-L1 only
- exp-0041: discard, score=0.013191, Reg-2 | dyn-L2 only
- exp-0042: discard, score=0.000764, Reg-4 | strong AE and dynamics regularization

## Avoid Repeating

- Review discarded runs before retrying the same idea.

## Next Priorities

1. Inspect the latest discarded run and extract the lesson.
2. Try the next highest-value architecture or regularization change.
3. Update the long-form report after a meaningful improvement.
