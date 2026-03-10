# Fluid Autoresearch Program

This file is the human-authored charter for autonomous research in this repo.

## Mission

Improve nonlinear prediction quality for the cylinder wake project while keeping
the fixed data contract and evaluation logic intact. The primary objective is to
minimize the `primary_score` written by `python -m mlfd.run_nonlinear`.

## Fixed Rules

1. Use `CYLINDER_ALL.mat` as the primary data source.
2. Do not change the meaning of `primary_score` or the ledger columns in `results.tsv`.
3. Preserve the append-only research memory:
   - `results.tsv`
   - `research/experiments/*.md`
   - `research/state.md`
   - `research/runs/20260310-fluid-rtx3070/index.md`
4. Keep the linear pipeline stable unless a bug blocks baseline artifact generation.

## Suggested Experiment Order

1. Establish the baseline nonlinear run and log it.
2. Explore latent dimension, regularization, and rollout loss changes.
3. Explore encoder/decoder depth and residual connections.
4. Explore latent dynamics width/depth and multi-step losses.
5. Only keep changes that improve `primary_score`, or that preserve the score while reducing VRAM or complexity.

## Runtime Budget

- Hard timeout per experiment: 12 minutes.
- Target runtime per experiment: 10 minutes.
- Campaign budget: 1 baseline plus up to 24 additional trials.

## Keep/Discard Policy

- Better `primary_score`: keep.
- Worse `primary_score`: discard.
- Within 0.25%: keep only if code is simpler or GPU memory is at least 5% lower.
- Crash or timeout: log it and move on.

## Logging Requirements

Every experiment must update:

1. `results.tsv`
2. `research/experiments/exp-XXXX.md`
3. `research/state.md`
4. `research/runs/20260310-fluid-rtx3070/index.md`

Failures are first-class results and must stay recorded.

