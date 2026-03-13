# Idea Registry

This file is the compressed long-horizon memory for idea families.
Read this before opening a new candidate line.

## pixelshuffle_film_groupnorm_v1

- Status: blocked
- Family: decoder
- Source Worktree: ML_FluidDynamics_decoder_combo
- Source Experiments: exp-0001
- Summary: The current PixelShuffle/FiLM/GroupNorm decoder line is negative evidence in its tested form.
- Revisit Only If: Only if the decoder formulation changes materially instead of retuning the same stack.

## ffl_wake_decoder_ft_v1

- Status: blocked
- Family: loss_recipe
- Source Worktree: ML_FluidDynamics_loss_combo
- Source Experiments: exp-0001
- Summary: The tested FFL/wake/decoder-FT recipe is negative evidence in its current form.
- Revisit Only If: Only if the loss contract changes materially.

## phase_refine_phase_residual_v1

- Status: blocked
- Family: phase_amplitude
- Source Worktree: ML_FluidDynamics_phase_amplitude
- Source Experiments: exp-0080, exp-0082
- Summary: Explicit phase splitting alone was not enough; the v1 phase_refine + phase_residual line is negative evidence.
- Revisit Only If: Only if the formulation changes materially, such as decoder-side phase conditioning or a different phase objective.

## joint_v2_decoder_curriculum_v1

- Status: blocked
- Family: joint_training
- Source Worktree: ML_FluidDynamics_joint_v2
- Source Experiments: exp-0068
- Summary: Decoder-only curriculum 8 -> 16 -> 32 regressed badly enough to close the current formulation.
- Revisit Only If: Only if the curriculum objective changes materially.

## joint_v2_teacher_anchor_v1

- Status: blocked
- Family: joint_training
- Source Worktree: ML_FluidDynamics_joint_v2
- Source Experiments: exp-0073
- Summary: The first teacher-anchor setting collapsed and should not be retried as a near miss.
- Revisit Only If: Only if the teacher-anchor formulation changes materially.

## joint_v2_alternating_refresh_v1

- Status: blocked
- Family: joint_training
- Source Worktree: ML_FluidDynamics_joint_v2
- Source Experiments: exp-0075
- Summary: The first linear-only alternating-refresh setting is negative evidence in its current form.
- Revisit Only If: Only if refresh mechanics change materially.

## joint_v2_ema_latent_stats_v1

- Status: blocked
- Family: joint_training
- Source Worktree: ML_FluidDynamics_joint_v2
- Source Experiments: exp-0077
- Summary: EMA latent-stat stabilization regressed relative to the corrected control.
- Revisit Only If: Only if the EMA formulation changes materially.

## residual_target_supervision_v2

- Status: blocked
- Family: decoder_supervision
- Source Worktree: ML_FluidDynamics_residual_target_v2
- Source Experiments: exp-0060
- Summary: Residual-target supervision v2 cleared the AE gate but collapsed on promoted full rollout, so it is negative evidence.
- Revisit Only If: Only if the supervision target or decoder contract changes materially.
