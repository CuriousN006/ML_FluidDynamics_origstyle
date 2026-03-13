# Agent Instructions

This file is the onboarding note for first-time agents in this worktree.

## Read Order

1. `README.md` for the quick summary of this worktree's role.
2. `program.md` for the authoritative branch charter, exact imported baseline, and runtime budget.
3. `research/idea_registry.md` for cross-worktree blocked and paused idea families.
4. `research/state.md` for the short external memory.
5. Open `research/final_report.md` only after you have a candidate direction or if the short memory is not enough.

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
- The authoritative branch role, imported baseline, and exact metrics live in `program.md`.
- The nonlinear line already beats the weaker linear references, but it does not yet beat the best `DMD` baseline.

## Current Objective

- Primary research goal: produce a nonlinear method that beats the fixed `DMD` baseline on the required evaluation.
- The current evidence says the remaining gap is mostly an AE-floor and decoder-side representation problem, not just a latent-dynamics problem.

## What To Do Next

- Preserve `exp-0051` as the nonlinear reference unless a branch-local instruction says otherwise.
- Prefer cheap `AE`-only screening before full rollout promotion.
- Focus next effort on decoder-side or representation-side ideas that can lower the AE floor at `t=100` and `t=150`.
- Do not spend another campaign on pure dynamics complexity unless a branch has evidence that the representation bottleneck is no longer the limiter.
- Run this clone in original-style agentic loops: edit code under `src/` or `tests/`, execute `python -m mlfd.autoresearch apply-candidate --idea-key ...`, judge the result, and continue.
- Use `python -m mlfd.autoresearch baseline-check` or `run-current` only for baseline/debug checks, not for normal candidate iteration.
- If `apply-candidate` was interrupted by a crash, forced stop, or power loss, run `python -m mlfd.autoresearch recover-candidate` before starting a new candidate.
- Idea keys should use `lower_snake_case`; use `_vN` when the formulation changes materially.
- Keep winner code history on the active campaign branch; checkpoint `results.tsv` and `research/` into the separate log worktree with `snapshot-logs`.
- Snapshot logs after every kept result, after about five experiments, and before shutting down a session.
- Treat `run-campaign` only as a bounded helper for unattended AE screening.

## Current Worktree

- Role: isolated autonomous sandbox cloned from the active global-reference worktree.
- Exact imported-reference details are authoritative in `program.md`.
- Keep local experiments here until they are reviewed and explicitly promoted back into the sibling worktrees.

## Environment Notes

- `CYLINDER_ALL.mat` must remain in the repo root.
- This worktree may reuse `D:\Projects\ML_FluidDynamics_Project\ML_FluidDynamics\.venv` if no local `.venv` is present; use `PYTHONPATH=<worktree>\src` in that case.

