from __future__ import annotations

import numpy as np

from mlfd.plots import save_latent_time_series, save_latent_trajectory_pca


def test_latent_plot_helpers_create_files(tmp_path) -> None:
    true_latents = np.stack([np.linspace(0.0, 1.0, 12), np.linspace(1.0, 0.0, 12), np.sin(np.linspace(0, 1, 12))], axis=1)
    pred_latents = true_latents + 0.05
    pca_path = tmp_path / "latent_pca.png"
    time_path = tmp_path / "latent_time.png"
    save_latent_trajectory_pca(true_latents, pred_latents, pca_path, "Latent PCA")
    save_latent_time_series(true_latents, pred_latents, time_path, "Latent time")
    assert pca_path.exists()
    assert time_path.exists()
