from __future__ import annotations

import numpy as np
import torch
from torch import nn

from mlfd.config import NonlinearConfig
from mlfd.nonlinear import _dynamics_validation_terms, _latent_rollout_loss


class IdentityDynamics(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class StepDynamics(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 1.0


def test_latent_rollout_loss_zero_for_constant_sequence() -> None:
    latents = torch.ones(6, 3)
    loss = _latent_rollout_loss(IdentityDynamics(), latents, start_index=0, horizon=4)
    assert torch.isclose(loss, torch.tensor(0.0))


def test_dynamics_validation_terms_include_rollout_component() -> None:
    latents = torch.arange(0, 8, dtype=torch.float32).unsqueeze(1)
    val_idx = np.array([2, 3, 4, 5])
    config = NonlinearConfig(validation_rollout_weight=0.25, validation_rollout_horizon=3)
    one_step, rollout, selection = _dynamics_validation_terms(StepDynamics(), latents, val_idx, config)
    assert torch.isclose(one_step, torch.tensor(0.0))
    assert torch.isclose(rollout, torch.tensor(0.0))
    assert torch.isclose(selection, torch.tensor(0.0))
