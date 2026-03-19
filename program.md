# autoresearch

This is an experiment to have the LLM do its own research on the cylinder-wake
nonlinear model.

## Setup

To set up a new run, work with the user to:

1. **Agree on a run tag**: use a short tag based on today's date or objective
   (for example `mar16`, `mar16-decoder`, `mar16-rtx3070`).
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from the current
   branch. This run should happen on its own fresh branch.
3. **Read the in-scope files**:
   - `README.md`
   - `prepare.py`
   - `train.py`
4. **Verify data exists**: confirm `CYLINDER_ALL.mat` is present in the repo
   root.
5. **Verify the environment**: run `python prepare.py`.
6. **Initialize results.tsv**: keep `results.tsv` with only the header row
   before the first run.
7. **Confirm and go**: the first run should be the current baseline as-is.

Once setup is done, kick off the experimentation.

## Experimentation

Each experiment runs on a single local GPU.

**What you CAN do:**

- Modify `train.py`.
- Only inspect or modify files under `src/mlfd/` if you are blocked and the
  wrapper in `train.py` is not enough.
- Modify hyperparameters, losses, training schedule, model structure, and
  rollout behavior as long as the evaluation stays comparable under the fixed
  `assignment_temporal_holdout_v1` protocol.

**What you CANNOT do:**

- Modify `prepare.py`. It is the fixed setup/data check script for this clone.
- Change the meaning of `primary_score`.
- Change the fixed `assignment_temporal_holdout_v1` split logic to make results
  incomparable.
- Quietly rewrite past experiment history.
- Touch unrelated tooling just because it is available.

**The goal is simple: get the lowest `primary_score`.**

VRAM is a soft constraint. Some increase is acceptable for a meaningful gain,
but do not explode memory for tiny wins.

**Simplicity criterion**: all else equal, simpler is better. A small win that
adds ugly complexity is not automatically worth keeping. A similarly good result
from deleting complexity is a strong win.

**The first run**: always establish the baseline first by running the current
code without changes.

## Current baseline

All new runs in this worktree now use the fixed `assignment_temporal_holdout_v1`
evaluation contract:

- snapshots `[0,120)` train, `[120,135)` val, `[135,151)` test
- transitions `[0,119)` train, `[119,134)` val, `[134,150)` test

Interpret the required forecast targets honestly under this protocol:

- `t=100` is a seen-region forecast inside the training segment
- `t=150` is a held-out future forecast inside the test segment

Active branch baseline:

- commit `d8c36d7`
- output `output/nonlinear/mar18-control-ae3600-dyn1800-01/metrics.json`
- `primary_score=0.000255791203`
- `recon_nrmse=0.000256861808`
- `nrmse_t100=0.000252927063`
- `nrmse_t150=0.000257081445`

Active ledger and archive rules:

- `results.tsv` is the active full-run scoreboard
- `proxy_results.tsv` is the active screening ledger for short proxy runs
- `speed_results.tsv` is the active ledger for the 90-minute same-score-faster campaign
- archived pre-reset notes live under `archive/`
- do not compare against archived nonlinear or imported `DMD` numbers in active decisions

Active local `DMD` reference under the same protocol:

- output `output/linear/mar17-assignment-dmd-baseline-01/metrics.json`
- best rank `15`
- `primary_score=0.000466643474`
- `recon_nrmse=0.000163697573`
- `nrmse_t100=0.000496426162`
- `nrmse_t150=0.000569952222`

This repo now ships a local `DMD` runner via `run_linear.py`, but `results.tsv`
still tracks only nonlinear experiments. Use the regenerated local `DMD`
baseline above as the active fixed reference when deciding whether a nonlinear
run is truly competitive.

## Current campaign

The active campaign is now `speed90_same_score_faster_v1`.

- start from the current proxy-champion code in `train.py`
- keep the long-run full champion as the quality anchor
- first success target: `primary_score <= 0.000268580764` within training-time `wall_seconds <= 5400`
- pause the proxy loop for this campaign; do not log 90-minute runs to `results.tsv`

## Run command

Launch one experiment like this:

```powershell
python train.py --profile speed90 --output-tag candidate90 > run.log 2>&1
```

Use `--profile speed90` for the active 90-minute campaign, `--profile proxy` for
short screening when the proxy loop is resumed, and `--profile full` for the
long confirmation run. The exact output tag is up to you. Keep it simple and
unique per run.

When the run finishes, read:

- `output/nonlinear/<output-tag>/metrics.json`
- `run.log`

Key fields to look at:

- `primary_score`
- `peak_memory_gb`
- `wall_seconds`
- `wall_seconds_end_to_end`
- `recon_rmse`
- `rmse_t100`
- `rmse_t150`

`wall_seconds` is the training-time budget metric used for campaign gating.
`wall_seconds_end_to_end` is a diagnostic that includes post-training artifact
generation and evaluation.

If `metrics.json` is missing, the run crashed.

## Logging results

When a proxy experiment is done, log it to `proxy_results.tsv` as tab-separated
values. When a speed90 experiment is done, log it to `speed_results.tsv`. When
a long full experiment is done, log it to `results.tsv`. Do not commit any of
the ledgers.

The TSV has a header row and 5 columns:

```text
commit	primary_score	memory_gb	status	description
```

1. short git commit hash
2. `primary_score` achieved
3. peak GPU memory in GB
4. status: `keep`, `discard`, or `crash`
5. short description of what the experiment tried

Example:

```text
commit	primary_score	memory_gb	status	description
abc1234	0.000565	6.1	keep	baseline imported reference
def5678	0.000552	6.4	keep	add decoder-side conditioning
9876fed	0.000580	6.2	discard	increase latent dim to 48
6543cba	0.000000	0.0	crash	double decoder width
```

## The experiment loop

The experiment runs on a dedicated branch such as `autoresearch/mar16`.

LOOP FOREVER:

1. Look at the current git state and current best result.
2. Form one concrete hypothesis.
3. Hack `train.py` directly. Only open `src/mlfd/` if the idea cannot be
   expressed in `train.py`.
4. `git commit` the candidate code.
5. For the active same-score-faster campaign, run speed90 first:

   ```powershell
   python train.py --profile speed90 --output-tag candidate90 > run.log 2>&1
   ```

6. Inspect `output/nonlinear/<output-tag>/metrics.json`. If it is missing, read
   the end of `run.log` for the failure.
7. Record the result in `speed_results.tsv`.
8. Keep the current speed90 campaign leader if it improves the best speed90
   score seen so far. Declare campaign success only when:

   - `primary_score <= 0.000268580764`
   - training-time `wall_seconds <= 5400`

9. Only use a long full rerun after the speed90 campaign finds a clear winner
   that deserves long-run confirmation:

   ```powershell
   python train.py --profile full --output-tag candidate-full > run.log 2>&1
   ```

10. Record long full outcomes only in `results.tsv`.
11. If a long full rerun improves the long-run champion, keep the commit and advance.
12. If a long full rerun is equal or worse, `git reset` back to where you started.

When this speed campaign ends, tighten the target in stages:

- stage 2 target: within `2%` of the long-run champion
- stage 3 target: match or beat the long-run champion

If a run crashes because of a small bug, fix it and try again. If the idea
itself is broken, log it as `crash`, revert, and move on.

**Profile budgets**:

- `--profile proxy`: `10` minutes AE + `3` minutes dynamics, stopping on batch boundaries
- `--profile speed90`: `85` minutes AE + `5` minutes dynamics, stopping on batch boundaries
- `--profile full`: long-run epoch-driven setup

**NEVER STOP**: once the loop begins, do not ask the human whether to continue.
Keep iterating until manually interrupted.
