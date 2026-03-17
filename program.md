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
  `strict_temporal_holdout_v1` protocol.

**What you CANNOT do:**

- Modify `prepare.py`. It is the fixed setup/data check script for this clone.
- Change the meaning of `primary_score`.
- Change the fixed `strict_temporal_holdout_v1` split logic to make results
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

All new runs in this worktree now use the fixed `strict_temporal_holdout_v1`
evaluation contract:

- snapshots `[0,80)` train, `[80,100)` val, `[100,151)` test
- transitions `[0,79)` train, `[79,99)` val, `[99,150)` test

Historical context only:

- imported `exp-0051` nonlinear reference
- `ae_architecture=residual_refine`
- `latent_dim=32`
- `refine_channels_mult=1.25`
- legacy `primary_score=0.000565`

The imported sibling-worktree `DMD` baseline is also historical context only:

The fixed best `DMD` baseline was imported from the sibling
`D:\Projects\ML_FluidDynamics_Project\ML_FluidDynamics` worktree:

- `DMD` rank `15`
- `rmse_t100=0.017423`
- `rmse_t150=0.018166`

These legacy nonlinear and `DMD` numbers were not produced under
`strict_temporal_holdout_v1` and are not comparable to new runs.

The current goal is to lower strict `primary_score` and regenerate any linear
comparison baseline under the same protocol before making "beats DMD" claims.

## Run command

Launch one experiment like this:

```powershell
python train.py --output-tag candidate > run.log 2>&1
```

The exact output tag is up to you. Keep it simple and unique per run.

When the run finishes, read:

- `output/nonlinear/<output-tag>/metrics.json`
- `run.log`

Key fields to look at:

- `primary_score`
- `peak_memory_gb`
- `wall_seconds`
- `recon_rmse`
- `rmse_t100`
- `rmse_t150`

If `metrics.json` is missing, the run crashed.

## Logging results

When an experiment is done, log it to `results.tsv` as tab-separated values.
Do not commit `results.tsv`.

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
5. Run the experiment:

   ```powershell
   python train.py --output-tag candidate > run.log 2>&1
   ```

6. Inspect `output/nonlinear/<output-tag>/metrics.json`. If it is missing, read
   the end of `run.log` for the failure.
7. Record the result in `results.tsv`.
8. If `primary_score` improved, keep the commit and advance.
9. If `primary_score` is equal or worse, `git reset` back to where you started.

If a run crashes because of a small bug, fix it and try again. If the idea
itself is broken, log it as `crash`, revert, and move on.

**Timeout**: a single experiment should normally stay around the existing local
runtime envelope. If it goes clearly off the rails, kill it and treat it as a
failure.

**NEVER STOP**: once the loop begins, do not ask the human whether to continue.
Keep iterating until manually interrupted.
