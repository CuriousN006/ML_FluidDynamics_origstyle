# Research State

## Current Best

- Experiment: exp-0051
- Commit: 26e4179-dirty
- Primary score: 0.000565
- Notes: Latent sweep Stage 2 | residual_refine latent=32 full rollout

## Recent Experiments

- exp-0054: discard, score=0.000569, Latent sweep Stage 3 R3 | residual_refine latent=32 rollout_loss_weight=0.30
- exp-0055: discard, score=0.001127, Reg-0 | latent32 residual_refine regularizer ablation
- exp-0056: discard, score=0.000979, Reg-2 | latent32 residual_refine regularizer ablation
- exp-0057: discard, score=0.000566, Reg-3 | latent32 residual_refine regularizer ablation
- exp-0058: discard, score=0.242610, Reg-4 | latent32 residual_refine regularizer ablation

## Blocked Ideas

- pixelshuffle_film_groupnorm_v1 (blocked): current PixelShuffle/FiLM/GroupNorm decoder line is negative evidence. [ML_FluidDynamics_decoder_combo: exp-0001]
- ffl_wake_decoder_ft_v1 (blocked): current FFL/wake/decoder-FT recipe is negative evidence. [ML_FluidDynamics_loss_combo: exp-0001]
- phase_refine_phase_residual_v1 (blocked): explicit phase splitting alone was not enough. [ML_FluidDynamics_phase_amplitude: exp-0080, exp-0082]
- joint_v2_teacher_anchor_v1 (blocked): first teacher-anchor setting collapsed. [ML_FluidDynamics_joint_v2: exp-0073]
- residual_target_supervision_v2 (blocked): promoted full rollout collapsed after AE gate pass. [ML_FluidDynamics_residual_target_v2: exp-0060]

## Avoid Repeating

- Review discarded runs before retrying the same idea.

## Next Priorities

1. Inspect the latest discarded run and extract the lesson.
2. Try the next highest-value architecture or regularization change.
3. Update the long-form report after a meaningful improvement.
