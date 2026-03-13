# Agent Instructions

This file is the onboarding note for first-time agents in this worktree.

## Read Order

1. `README.md` for the quick summary of this worktree's role.
2. `program.md` for the branch charter, reference baseline, and runtime budget.
3. `research/idea_registry.md` for cross-worktree blocked and paused idea families.
4. `research/state.md` for the short external memory.
5. `research/final_report.md` for the long-form narrative.

Always trust the scope/status notes at the top of those files before reading the rest.

## Documentation Contract

- Treat `results.tsv`, `research/experiments/*.md`, and `research/runs/*/index.md` as append-only history.
- Correct factual mistakes when needed, but do not rewrite experiment history or renumber runs.
- Keep `README.md`, `program.md`, `research/state.md`, and `research/final_report.md` aligned in the same turn when the branch role, baseline, or best-result story changes.
- If a result is imported from a sibling worktree, say so explicitly. Do not present an imported winner as if it was discovered locally.
- Every `research/final_report.md` should begin with a status note that says whether it is the active global reference or a frozen inherited report.
- If a new overall winner is confirmed, update the onboarding notes in sibling worktrees before starting another branch from stale docs.

## Project Problem

- Implement the cylinder-wake modeling assignment described in `Final_project_DL2025-5.pdf` using `CYLINDER_ALL.mat`.
- Maintain both required tracks:
  - a linear reduced-order baseline built around `SVD`, truncated linear rollout, and `DMD`
  - a nonlinear autoencoder-plus-latent-dynamics model
- The fixed deliverables are:
  - held-out reconstruction on the fixed `10%` test split
  - future prediction at `t=100` and `t=150`
  - `MSE`, `RMSE`, and `NRMSE` reporting
  - latent-dynamics visualization and interpretation
  - regularizer-effect discussion

## Current Progress

- The data contract, portrait layout, linear pipeline, nonlinear pipeline, experiment ledger, and reporting flow are all implemented.
- Best linear comparison target: `DMD` rank `15` with `rmse_t100=0.017423` and `rmse_t150=0.018166`.
- Best nonlinear result as of 2026-03-11: `exp-0051` in this worktree with `primary_score=0.000565`, `rmse_t100=0.020372`, and `rmse_t150=0.018893`.
- The nonlinear line already beats the rank-10 linear rollout and truncated reconstruction, but it does not yet beat the best `DMD` baseline.

## Current Objective

- Primary research goal: produce a nonlinear method that beats the fixed `DMD` baseline on the required evaluation.
- The current evidence says the remaining gap is mostly an AE-floor and decoder-side representation problem, not just a latent-dynamics problem.

## What To Do Next

- Preserve `exp-0051` as the nonlinear reference unless a branch-local instruction says otherwise.
- Prefer cheap `AE`-only screening before full rollout promotion.
- Focus next effort on decoder-side or representation-side ideas that can lower the AE floor at `t=100` and `t=150`.
- Do not spend another campaign on pure dynamics complexity unless a branch has evidence that the representation bottleneck is no longer the limiter.
- Run this clone in original-style agentic loops: edit code under `src/` or `tests/`, execute `python -m mlfd.autoresearch apply-candidate --idea-key ...`, judge the result, and continue.
- Treat `run-campaign` only as a bounded helper for unattended AE screening.

## Current Worktree

- Role: isolated autonomous sandbox cloned from the active global-reference worktree.
- Imported reference as of 2026-03-11: `exp-0051` with `primary_score=0.000565`.
- Keep local experiments here until they are reviewed and explicitly promoted back into the sibling worktrees.

## Environment Notes

- `CYLINDER_ALL.mat` must remain in the repo root.
- This worktree may reuse `D:\Projects\ML_FluidDynamics_Project\ML_FluidDynamics\.venv` if no local `.venv` is present; use `PYTHONPATH=<worktree>\src` in that case.

