from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        return self.act(x + residual)


class BaselineConvAutoencoder(nn.Module):
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


class ResidualConvAutoencoder(nn.Module):
    def __init__(self, input_shape: tuple[int, int], latent_dim: int) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(16),
        )
        self.down1 = nn.Sequential(nn.Conv2d(16, 24, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.block1 = ResidualBlock(24)
        self.down2 = nn.Sequential(nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.block2 = ResidualBlock(48)
        self.down3 = nn.Sequential(nn.Conv2d(48, 72, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.block3 = ResidualBlock(72)
        self.down4 = nn.Sequential(nn.Conv2d(72, 96, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.block4 = ResidualBlock(96)

        bottleneck_shape = self._infer_bottleneck_shape(input_shape)
        self.bottleneck_shape = bottleneck_shape
        flattened = 96 * bottleneck_shape[0] * bottleneck_shape[1]
        self.encoder_head = nn.Linear(flattened, latent_dim)
        self.decoder_head = nn.Linear(latent_dim, flattened)

        self.up1 = nn.Sequential(
            nn.Conv2d(96, 72, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(72),
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(72, 48, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(48),
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(48, 24, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(24),
        )
        self.up4 = nn.Sequential(
            nn.Conv2d(24, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(16),
        )
        self.output_head = nn.Conv2d(16, 1, kernel_size=3, padding=1)

    def _encode_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block1(self.down1(x))
        x = self.block2(self.down2(x))
        x = self.block3(self.down3(x))
        x = self.block4(self.down4(x))
        return x

    def _infer_bottleneck_shape(self, input_shape: tuple[int, int]) -> tuple[int, int]:
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_shape[0], input_shape[1])
            encoded = self._encode_features(dummy)
        return int(encoded.shape[-2]), int(encoded.shape[-1])

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self._encode_features(x)
        return self.encoder_head(encoded.flatten(start_dim=1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        hidden = self.decoder_head(z).view(-1, 96, self.bottleneck_shape[0], self.bottleneck_shape[1])
        hidden = F.interpolate(hidden, scale_factor=2.0, mode="bilinear", align_corners=False)
        hidden = self.up1(hidden)
        hidden = F.interpolate(hidden, scale_factor=2.0, mode="bilinear", align_corners=False)
        hidden = self.up2(hidden)
        hidden = F.interpolate(hidden, scale_factor=2.0, mode="bilinear", align_corners=False)
        hidden = self.up3(hidden)
        hidden = F.interpolate(hidden, size=self.input_shape, mode="bilinear", align_corners=False)
        hidden = self.up4(hidden)
        return self.output_head(hidden)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(x)
        recon = self.decode(latent)
        return recon, latent


def build_autoencoder(input_shape: tuple[int, int], latent_dim: int, architecture: str) -> nn.Module:
    if architecture == "baseline":
        return BaselineConvAutoencoder(input_shape, latent_dim)
    if architecture == "residual":
        return ResidualConvAutoencoder(input_shape, latent_dim)
    raise ValueError(f"Unsupported autoencoder architecture: {architecture}")


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


class ResidualLinearDynamics(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 64,
        depth: int = 2,
        linear_init: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(latent_dim, latent_dim, bias=False)
        if linear_init is not None:
            self.linear.weight.data.copy_(linear_init)

        layers: list[nn.Module] = []
        in_features = latent_dim
        for _ in range(max(1, depth)):
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            in_features = hidden_dim
        layers.append(nn.Linear(in_features, latent_dim))
        self.residual = nn.Sequential(*layers)
        final = self.residual[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.residual(x)


def build_dynamics_model(
    kind: str,
    latent_dim: int,
    hidden_dim: int,
    depth: int,
    linear_init: torch.Tensor | None = None,
) -> nn.Module:
    if kind == "mlp":
        return LatentDynamicsMLP(latent_dim, hidden_dim, depth)
    if kind == "residual_linear":
        return ResidualLinearDynamics(latent_dim, hidden_dim, depth, linear_init=linear_init)
    raise ValueError(f"Unsupported dynamics model: {kind}")
