# Fluid Autoresearch Program

This workspace follows the original `karpathy/autoresearch` style: the agent is
the researcher. It should keep reading the current code, editing the nonlinear
pipeline, running experiments, judging the result, and continuing until the
human interrupts it.

## Worktree Scope

- This is an isolated autonomous sandbox cloned from the 2026-03-11 global reference.
- Imported reference: `exp-0051` (`primary_score=0.000565`).
- Keep local wins here until they are reviewed and deliberately promoted back into the sibling worktrees.

## Setup

Before starting a new loop:

1. Read `README.md`, `AGENTS.md`, `program.md`, `research/state.md`, and `research/final_report.md`.
2. Verify the environment with `python -m mlfd.setup`.
3. Confirm `CYLINDER_ALL.mat` exists in the repo root.
4. Confirm the current best imported reference is still `exp-0051`.
5. If this sandbox has no local experiment history yet, run one clean baseline with:

```powershell
python -m mlfd.autoresearch run-current --description "baseline import check" --next-hypothesis "Start the first real candidate change."
```

After setup, enter the experiment loop and do not stop unless interrupted by the human.

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
   - `research/runs/20260311-fluid-rtx3070-latent-sweep/index.md`
4. Keep the linear pipeline stable unless a bug blocks baseline artifact generation.
5. Keep the portrait layout and portrait comparison figures as the nonlinear default.

## Current Working Baseline

Treat `exp-0051` as the reference model for this worktree and the current overall campaign:

- `layout=portrait`
- `ae_architecture=residual_refine`
- `ae_width_mult=1.0`
- `coordconv=False`
- `dynamics_model=residual_linear`
- `latent_dim=32`
- `coarse_loss_weight=0.25`
- `coarse_blur_kernel=9`
- `coarse_blur_sigma=2.0`
- `refine_blocks=1`
- `refine_channels_mult=1.25`
- `gradient_loss_weight=0.10`
- `fft_loss_weight=0.0`
- `latent_l1_weight=1e-4`
- `dyn_l2_weight=0`
- `rollout_loss_weight=0.15`
- `train_rollout_horizon=16`
- `train_rollout_stride=8`
- `validation_rollout_horizon=24`
- `validation_rollout_stride=4`
- `deterministic=True`

Reference metrics:

- `primary_score=0.000565`
- `recon_rmse=0.020886`
- `rmse_t100=0.020372`
- `rmse_t150=0.018893`
- `ae_floor_rmse_t150=0.018927`

## In-Scope Files

Read these before major changes:

- `README.md`
- `AGENTS.md`
- `program.md`
- `research/state.md`
- `research/final_report.md`
- `src/mlfd/config.py`
- `src/mlfd/run_nonlinear.py`
- `src/mlfd/models.py`
- `src/mlfd/nonlinear.py`
- `src/mlfd/autoresearch.py`

## Allowed Changes

- Nonlinear model architecture
- Decoder or representation design
- Latent dynamics implementation
- Training schedule and hyperparameters
- AE-only screening flow
- Autoresearch helper code in this sandbox if it helps the loop run more cleanly

## Forbidden Changes

- Changing the semantics of `primary_score`
- Changing the fixed dataset split logic to make results incomparable
- Quietly rewriting or deleting past experiment history
- Treating imported results as if they were discovered locally

## Core Loop

Loop until interrupted by the human:

1. Read the latest research memory and identify the current best result.
2. Form one concrete hypothesis.
3. Edit the nonlinear code directly under `src/` or `tests/`.
4. Run a single measured experiment with `apply-candidate`.
5. Check the result against the current best.
6. Keep the change if it wins, otherwise let `apply-candidate` restore the previous code and move on.
7. Record the outcome in the normal research memory.
8. Continue immediately to the next idea.

The primary command for a single fully logged candidate experiment is:

```powershell
python -m mlfd.autoresearch apply-candidate --description "<what changed>" --next-hypothesis "<next idea>"
```

`apply-candidate` is the closest equivalent to the original `autoresearch` philosophy in this sandbox:

- it commits only the current candidate code edits
- runs one experiment
- records the result
- keeps the commit if it wins
- restores the previous code automatically if it loses or crashes

Use `python -m mlfd.autoresearch search-ae ...` or `run-campaign ...` only as helper tools for cheap AE-floor screening. They are not the main research loop. The main loop is still agent-driven code editing plus keep/discard judgment.

## Keep/Discard Policy

- Better `primary_score`: keep.
- Worse `primary_score`: discard.
- Within 0.25%: keep only if the code is simpler or GPU memory is at least 5% lower.
- Crash or timeout: log it and move on.

## Runtime Budget

- Hard timeout per experiment: 12 minutes.
- Target runtime per experiment: 10 minutes.
- Prefer cheap AE-only screening before full rollout promotion.

## Safety Valves

This sandbox intentionally supports bounded helper loops even though the main mode is open-ended agentic research.

- `run-campaign --max-rounds N` stops after `N` rounds.
- `run-campaign --max-hours H` stops after `H` wall-clock hours.
- `run-campaign` also stops when its stop file appears.

These are guardrails for unattended helper runs, not replacements for the main autoresearch workflow.

## Logging Requirements

Every experiment must update:

1. `results.tsv`
2. `research/experiments/exp-XXXX.md`
3. `research/state.md`
4. `research/runs/20260311-fluid-rtx3070-latent-sweep/index.md`

Failures are first-class results and must stay recorded.
