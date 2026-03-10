from __future__ import annotations

import torch

from mlfd.models import build_autoencoder


def test_residual_multiscale_autoencoder_preserves_shape_and_latent_dim() -> None:
    model = build_autoencoder(
        input_shape=(64, 32),
        latent_dim=32,
        architecture="residual_multiscale",
        width_mult=1.25,
        coordconv=True,
    )
    batch = torch.randn(2, 1, 64, 32)
    recon, latent = model(batch)
    assert recon.shape == batch.shape
    assert latent.shape == (2, 32)


def test_residual_refine_autoencoder_exposes_coarse_and_final_fields() -> None:
    model = build_autoencoder(
        input_shape=(64, 32),
        latent_dim=24,
        architecture="residual_refine",
        refine_blocks=2,
        refine_channels_mult=1.25,
    )
    batch = torch.randn(2, 1, 64, 32)
    latent = model.encode(batch)
    final, auxiliary = model.decode_with_aux(latent)
    assert final.shape == batch.shape
    assert auxiliary["coarse"].shape == batch.shape
    assert auxiliary["refine_residual"].shape == batch.shape
