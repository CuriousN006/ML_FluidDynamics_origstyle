from __future__ import annotations

from argparse import Namespace

from mlfd.config import NonlinearConfig
from mlfd.run_nonlinear import apply_cli_overrides


def test_apply_cli_overrides_updates_scheduler_and_regularizers() -> None:
    args = Namespace(
        smoke=False,
        layout="portrait",
        ae_architecture="residual_refine",
        ae_width_mult=1.0,
        coordconv=False,
        coarse_loss_weight=0.4,
        coarse_blur_kernel=11,
        coarse_blur_sigma=1.5,
        refine_blocks=2,
        refine_channels_mult=1.25,
        dynamics_model="residual_linear",
        ae_epochs=None,
        dyn_epochs=320,
        latent_dim=None,
        ae_learning_rate=5e-4,
        dyn_learning_rate=2e-4,
        rollout_loss_weight=None,
        gradient_loss_weight=0.05,
        fft_loss_weight=0.03,
        dynamics_depth=None,
        dynamics_hidden_dim=None,
        train_rollout_stride=6,
        train_rollout_horizon=12,
        validation_rollout_stride=3,
        validation_rollout_horizon=18,
        ae_scheduler="plateau",
        dyn_scheduler="plateau",
        ae_scheduler_factor=0.4,
        dyn_scheduler_factor=0.3,
        ae_scheduler_patience=7,
        dyn_scheduler_patience=11,
        ae_min_learning_rate=1e-6,
        dyn_min_learning_rate=2e-6,
        latent_l1_weight=5e-4,
        dyn_l2_weight=5e-5,
        deterministic=False,
        save_checkpoint=True,
        device=None,
    )
    config = apply_cli_overrides(NonlinearConfig(), args)
    assert config.layout == "portrait"
    assert config.ae_architecture == "residual_refine"
    assert config.ae_width_mult == 1.0
    assert config.coordconv is False
    assert config.coarse_loss_weight == 0.4
    assert config.coarse_blur_kernel == 11
    assert config.coarse_blur_sigma == 1.5
    assert config.refine_blocks == 2
    assert config.refine_channels_mult == 1.25
    assert config.dynamics_model == "residual_linear"
    assert config.dyn_epochs == 320
    assert config.ae_learning_rate == 5e-4
    assert config.dyn_learning_rate == 2e-4
    assert config.gradient_loss_weight == 0.05
    assert config.fft_loss_weight == 0.03
    assert config.train_rollout_stride == 6
    assert config.train_rollout_horizon == 12
    assert config.validation_rollout_stride == 3
    assert config.validation_rollout_horizon == 18
    assert config.ae_scheduler == "plateau"
    assert config.dyn_scheduler == "plateau"
    assert config.ae_scheduler_factor == 0.4
    assert config.dyn_scheduler_factor == 0.3
    assert config.latent_l1_weight == 5e-4
    assert config.dyn_l2_weight == 5e-5
    assert config.deterministic is False
    assert config.save_checkpoint is True
