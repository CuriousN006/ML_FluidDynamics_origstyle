from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvAutoencoder(nn.Module):
    def __init__(self, input_shape: tuple[int, int], latent_dim: int) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.latent_shape = (8, 4)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.encoder_head = nn.Linear(64 * self.latent_shape[0] * self.latent_shape[1], latent_dim)
        self.decoder_head = nn.Linear(latent_dim, 64 * self.latent_shape[0] * self.latent_shape[1])
        self.decoder_blocks = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        encoded = F.interpolate(encoded, size=self.latent_shape, mode="bilinear", align_corners=False)
        return self.encoder_head(encoded.flatten(start_dim=1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        hidden = self.decoder_head(z).view(-1, 64, self.latent_shape[0], self.latent_shape[1])
        hidden = F.interpolate(hidden, scale_factor=2.0, mode="bilinear", align_corners=False)
        hidden = F.interpolate(hidden, scale_factor=2.0, mode="bilinear", align_corners=False)
        hidden = F.interpolate(hidden, scale_factor=2.0, mode="bilinear", align_corners=False)
        hidden = F.interpolate(hidden, size=self.input_shape, mode="bilinear", align_corners=False)
        return self.decoder_blocks(hidden)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(x)
        recon = self.decode(latent)
        return recon, latent


class LatentDynamicsMLP(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int = 64, depth: int = 2) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_features = latent_dim
        for _ in range(max(1, depth)):
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            in_features = hidden_dim
        layers.append(nn.Linear(in_features, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
