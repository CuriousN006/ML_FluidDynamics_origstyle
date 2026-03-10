# Run 20260310-fluid-rtx3070

## Timeline

- exp-0001 | keep | score=0.031904 | baseline smoke validation
- exp-0002 | discard | score=0.031896 | autoresearch numbering validation
- exp-0003 | keep | score=0.027961 | full baseline nonlinear run
- exp-0004 | keep | score=0.024610 | raw sweep: latent_dim=24
- exp-0005 | discard | score=0.026242 | raw sweep: latent_dim=24, rollout_loss_weight=0.20
- exp-0006 | discard | score=0.027935 | raw sweep: latent_dim=24, dynamics_depth=3
- exp-0007 | discard | score=0.024609 | raw sweep: latent_dim=32
- exp-0008 | discard | score=0.028337 | raw sweep: baseline latent_dim=24, dynamics_hidden_dim=96
- exp-0009 | keep | score=0.023994 | raw sweep: latent_dim=24, longer training (ae160 dyn200)
- exp-0010 | keep | score=0.023211 | raw sweep: latent_dim=24, ae160 dyn260
- exp-0011 | discard | score=0.029603 | raw sweep: latent_dim=24, ae160 dyn320

## Best Model Changes

- Current best: exp-0010 (0.023211)

## Open Risks

- Continue searching for lower rollout error without increasing complexity too much.
