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

1. Treat `exp-0021` as the working baseline: `latent_dim=24`, `ae_epochs=160`, `dyn_epochs=260`, dual plateau scheduler configured, `latent_l1_weight=1e-4`, `dyn_l2_weight=1e-5`, `deterministic=False`.
2. Keep the raw `VORTALL` mainline fixed and preserve the new reporting artifacts for 10% test reconstruction, future prediction, latent plots, and regularizer analysis.
3. Do not spend more time on larger dynamics budgets until a different learning-rate policy actually steps. `dyn280` and `dyn320` are already worse than the current baseline.
4. Treat the default combined regularizer as the current winner. `latent_l1_weight` alone is weak, `dyn_l2_weight` matters more, and stronger regularization underfits.
5. Only keep changes that improve `primary_score`, or that preserve the score while reducing VRAM or runtime.

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
