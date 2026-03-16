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
   - `src/mlfd/nonlinear.py`
   - `src/mlfd/models.py` only if needed for an architecture change
   - `src/mlfd/run_nonlinear.py` for the CLI surface
4. **Verify data exists**: confirm `CYLINDER_ALL.mat` is present in the repo
   root.
5. **Verify the environment**: run `python -m mlfd.setup`.
6. **Initialize results.tsv**: keep `results.tsv` with only the header row
   before the first run.
7. **Confirm and go**: the first run should be the current baseline as-is.

Once setup is done, kick off the experimentation.

## Experimentation

Each experiment runs on a single local GPU.

**What you CAN do:**

- Modify `src/mlfd/nonlinear.py`.
- Modify `src/mlfd/models.py` if the nonlinear idea genuinely needs an
  architectural change there.
- Modify hyperparameters, losses, training schedule, model structure, and
  rollout behavior as long as the evaluation stays comparable.

**What you CANNOT do:**

- Change the meaning of `primary_score`.
- Change the fixed train/validation/test split logic to make results
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

This clone starts from the imported `exp-0051` nonlinear reference:

- `ae_architecture=residual_refine`
- `latent_dim=32`
- `refine_channels_mult=1.25`
- `rollout_loss_weight=0.15`
- `primary_score=0.000565`

The current overall goal is still to beat the fixed best `DMD` baseline.

## Run command

Launch one experiment like this:

```powershell
python -m mlfd.run_nonlinear --output-tag candidate > run.log 2>&1
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
3. Hack `src/mlfd/nonlinear.py` directly. Touch `src/mlfd/models.py` only if
   required.
4. `git commit` the candidate code.
5. Run the experiment:

   ```powershell
   python -m mlfd.run_nonlinear --output-tag candidate > run.log 2>&1
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
