# Legacy Archive

This folder holds pre-reset notes and ledgers that are kept only for audit and
recovery. It is not part of the active experiment control surface.

Use the repo root for all current work:

- `README.md`
- `program.md`
- `results.tsv`

Use this archive only when you explicitly need historical context.

## Archived Contents

- `results_legacy.tsv`
  - pre-reset and superseded experiment ledger
  - includes the contaminated mixed-split campaign
  - includes the interim `ab78b20` strict-reset baseline that was later replaced
- legacy imported baseline notes
  - the old sibling-worktree `DMD` rank-15 numbers
  - old nonlinear headline scores from before `assignment_temporal_holdout_v1`

## Legacy Status

The archived nonlinear and `DMD` numbers are not comparable to the current
`assignment_temporal_holdout_v1` protocol.

Current active interpretation:

- `t=100` is a seen-region forecast
- `t=150` is a held-out future forecast

Current active `DMD` status:

- no valid `DMD` baseline has been rerun yet under `assignment_temporal_holdout_v1`
- do not make active `DMD` comparison claims until that rerun exists
