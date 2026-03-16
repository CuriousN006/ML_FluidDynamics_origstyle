# Agent Instructions

This clone is intentionally bare-bones. Treat it more like the original
`karpathy/autoresearch` repo than the richer fluid sandbox.

## Read order

1. `README.md`
2. `program.md`
3. `src/mlfd/nonlinear.py`
4. `src/mlfd/models.py` only if the candidate idea really needs it

## Working style

- Default to editing `src/mlfd/nonlinear.py`.
- Keep the main loop simple: edit, commit, run, log, keep/discard, continue.
- Log every finished run in `results.tsv`, but do not commit `results.tsv`.
- If a candidate loses, revert the code and move on.
- If the machine is interrupted mid-run, treat that run as unfinished, inspect
  `git status` and `run.log`, and continue from a clean git state.

## Scope

- This folder is the minimal original-style clone.
- Use `ML_FluidDynamics_autonomous` if you want idea registry, generated
  research memory, or helper automation.
