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
- exp-0012 | crash | score=0.000000 | raw baseline refresh: deterministic seed + rollout-aware dynamics selection
- exp-0013 | keep | score=0.023163 | raw baseline refresh: deterministic seed + rollout-aware dynamics selection (cpu decode fix)
- exp-0014 | discard | score=0.024582 | raw sweep: rollout-aware selection with dyn320
- exp-0015 | keep | score=0.021425 | clean baseline replay: rollout-aware deterministic raw baseline
- exp-0016 | discard | score=0.022834 | raw sweep: clean baseline with dyn280
- exp-0017 | discard | score=0.024719 | parity run: exp-0015 baseline with expanded metrics
- exp-0018 | keep | score=0.020998 | parity rerun: deterministic resize baseline
- exp-0019 | discard | score=0.020998 | Sched-1: dyn-only plateau scheduler
- exp-0020 | discard | score=0.020998 | Sched-2: dual plateau scheduler
- exp-0021 | keep | score=0.020250 | Sched-3: dual plateau with deterministic=false runtime check
- exp-0022 | discard | score=0.020768 | Reg-0: no regularizer under dual plateau deterministic=false
- exp-0023 | discard | score=0.022191 | Reg-1: latent L1 only under dual plateau deterministic=false
- exp-0024 | discard | score=0.020955 | Reg-2: dynamics L2 only under dual plateau deterministic=false
- exp-0025 | discard | score=0.022281 | Reg-4: strong AE+dyn regularizers under dual plateau deterministic=false
- exp-0026 | keep | score=0.001830 | Stage 1: portrait layout parity with baseline AE + MLP dynamics
- exp-0027 | keep | score=0.001736 | Stage 2: portrait baseline AE with residual linear dynamics
- exp-0028 | keep | score=0.000710 | Stage 3: portrait residual autoencoder with residual linear dynamics
- exp-0029 | discard | score=0.154054 | Stage 4A: residual AE + residual linear dynamics with lower gradient loss weight
- exp-0030 | discard | score=0.000835 | Stage 4B: residual AE + residual linear dynamics with shorter train rollout horizon
- exp-0031 | discard | score=0.013512 | Stage 4C: portrait residual AE + residual linear dynamics with deterministic=false
- exp-0032 | discard | score=234299.921330 | Stage 1 | wider residual AE with FFT loss
- exp-0033 | crash | score=0.000000 | Stage 1 | residual_multiscale AE with CoordConv and light FFT loss
- exp-0034 | discard | score=0.012891 | Stage 1 | residual AE with CoordConv and light FFT loss
- exp-0035 | discard | score=1.526203 | AE search top-1 reevaluation | family=residual | latent=24 | width=1.375 | coordconv=False
- exp-0036 | discard | score=0.000710 | AE search top-1 reevaluation | family=residual | latent=24 | width=1.000 | coordconv=False
- exp-0037 | discard | score=0.000710 | AE search top-1 reevaluation | family=residual | latent=24 | width=1.000 | coordconv=False
- exp-0038 | discard | score=0.068061 | AE search top-2 reevaluation | family=residual | latent=24 | width=1.000 | coordconv=True
- exp-0039 | discard | score=0.013191 | Reg-0 | residual AE regularizer ablation
- exp-0040 | discard | score=0.000709 | Reg-1 | latent-L1 only
- exp-0041 | discard | score=0.013191 | Reg-2 | dyn-L2 only
- exp-0042 | discard | score=0.000764 | Reg-4 | strong AE and dynamics regularization
- exp-0043 | keep | score=0.000670 | coarse-to-fine Stage 3 R5 | residual_refine latent=24 refine_blocks=1 refine_channels_mult=1.25 coarse_loss=0.25
- exp-0044 | discard | score=0.001285 | coarse-to-fine Stage 3 R1 | residual_refine latent=24 refine_blocks=1 refine_channels_mult=1.00 coarse_loss=0.25
- exp-0045 | discard | score=0.000734 | Reg-0 | residual_refine regularizer ablation
- exp-0046 | keep | score=0.000613 | Reg-1 | residual_refine regularizer ablation
- exp-0047 | discard | score=0.000720 | Reg-2 | residual_refine regularizer ablation
- exp-0048 | discard | score=0.000670 | Reg-3 | residual_refine regularizer ablation
- exp-0049 | discard | score=0.092935 | Reg-4 | residual_refine regularizer ablation
- exp-0050 | discard | score=0.000613 | Latent sweep Stage 0 parity | residual_refine latent=24 baseline replay
- exp-0051 | keep | score=0.000565 | Latent sweep Stage 2 | residual_refine latent=32 full rollout
- exp-0052 | discard | score=0.000567 | Latent sweep Stage 3 R1 | residual_refine latent=32 rollout_loss_weight=0.10
- exp-0053 | discard | score=0.000566 | Latent sweep Stage 3 R2 | residual_refine latent=32 rollout_loss_weight=0.20
- exp-0054 | discard | score=0.000569 | Latent sweep Stage 3 R3 | residual_refine latent=32 rollout_loss_weight=0.30
- exp-0055 | discard | score=0.001127 | Reg-0 | latent32 residual_refine regularizer ablation
- exp-0056 | discard | score=0.000979 | Reg-2 | latent32 residual_refine regularizer ablation
- exp-0057 | discard | score=0.000566 | Reg-3 | latent32 residual_refine regularizer ablation
- exp-0058 | discard | score=0.242610 | Reg-4 | latent32 residual_refine regularizer ablation

## Best Model Changes

- Current best: exp-0051 (0.000565)

## Open Risks

- Recent crashes or timeouts need follow-up.
