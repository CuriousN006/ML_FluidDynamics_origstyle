from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _scaled_channels(base: int, width_mult: float) -> int:
    return max(8, int(round(base * width_mult)))


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


class CoordInputMixin:
    def __init__(self, input_shape: tuple[int, int], coordconv: bool) -> None:
        self.input_shape = input_shape
        self.coordconv = coordconv

    @property
    def encoder_input_channels(self) -> int:
        return 3 if self.coordconv else 1

    def _prepare_encoder_input(self, x: torch.Tensor) -> torch.Tensor:
        if not self.coordconv:
            return x
        batch, _, height, width = x.shape
        y_coords = torch.linspace(-1.0, 1.0, height, device=x.device, dtype=x.dtype).view(1, 1, height, 1)
        x_coords = torch.linspace(-1.0, 1.0, width, device=x.device, dtype=x.dtype).view(1, 1, 1, width)
        y_coords = y_coords.expand(batch, 1, height, width)
        x_coords = x_coords.expand(batch, 1, height, width)
        return torch.cat([x, x_coords, y_coords], dim=1)


class BaselineConvAutoencoder(nn.Module, CoordInputMixin):
    def __init__(
        self,
        input_shape: tuple[int, int],
        latent_dim: int,
        width_mult: float = 1.0,
        coordconv: bool = False,
    ) -> None:
        nn.Module.__init__(self)
        CoordInputMixin.__init__(self, input_shape, coordconv)
        self.latent_dim = latent_dim
        self.latent_shape = (8, 4)
        channels = [
            _scaled_channels(16, width_mult),
            _scaled_channels(32, width_mult),
            _scaled_channels(64, width_mult),
            _scaled_channels(64, width_mult),
        ]
        self.encoder = nn.Sequential(
            nn.Conv2d(self.encoder_input_channels, channels[0], kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[0], channels[1], kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[1], channels[2], kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[2], channels[3], kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        flattened = channels[3] * self.latent_shape[0] * self.latent_shape[1]
        self.encoder_head = nn.Linear(flattened, latent_dim)
        self.decoder_head = nn.Linear(latent_dim, flattened)
        self.decoder_blocks = nn.Sequential(
            nn.Conv2d(channels[3], channels[3], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[3], channels[1], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[1], channels[0], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[0], 1, kernel_size=3, padding=1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(self._prepare_encoder_input(x))
        encoded = F.interpolate(encoded, size=self.latent_shape, mode="bilinear", align_corners=False)
        return self.encoder_head(encoded.flatten(start_dim=1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        hidden = self.decoder_head(z).view(-1, self.decoder_blocks[0].in_channels, self.latent_shape[0], self.latent_shape[1])
        hidden = F.interpolate(hidden, scale_factor=2.0, mode="bilinear", align_corners=False)
        hidden = F.interpolate(hidden, scale_factor=2.0, mode="bilinear", align_corners=False)
        hidden = F.interpolate(hidden, scale_factor=2.0, mode="bilinear", align_corners=False)
        hidden = F.interpolate(hidden, size=self.input_shape, mode="bilinear", align_corners=False)
        return self.decoder_blocks(hidden)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(x)
        recon = self.decode(latent)
        return recon, latent


class ResidualConvAutoencoder(nn.Module, CoordInputMixin):
    def __init__(
        self,
        input_shape: tuple[int, int],
        latent_dim: int,
        width_mult: float = 1.0,
        coordconv: bool = False,
    ) -> None:
        nn.Module.__init__(self)
        CoordInputMixin.__init__(self, input_shape, coordconv)
        self.latent_dim = latent_dim
        stem_channels = _scaled_channels(16, width_mult)
        c1 = _scaled_channels(24, width_mult)
        c2 = _scaled_channels(48, width_mult)
        c3 = _scaled_channels(72, width_mult)
        c4 = _scaled_channels(96, width_mult)
        self.stem = nn.Sequential(
            nn.Conv2d(self.encoder_input_channels, stem_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(stem_channels),
        )
        self.down1 = nn.Sequential(nn.Conv2d(stem_channels, c1, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.block1 = ResidualBlock(c1)
        self.down2 = nn.Sequential(nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.block2 = ResidualBlock(c2)
        self.down3 = nn.Sequential(nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.block3 = ResidualBlock(c3)
        self.down4 = nn.Sequential(nn.Conv2d(c3, c4, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.block4 = ResidualBlock(c4)

        bottleneck_shape = self._infer_bottleneck_shape(input_shape)
        self.bottleneck_shape = bottleneck_shape
        flattened = c4 * bottleneck_shape[0] * bottleneck_shape[1]
        self.decoder_channels = c4
        self.encoder_head = nn.Linear(flattened, latent_dim)
        self.decoder_head = nn.Linear(latent_dim, flattened)

        self.up1 = nn.Sequential(
            nn.Conv2d(c4, c3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(c3),
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(c3, c2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(c2),
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(c2, c1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(c1),
        )
        self.up4 = nn.Sequential(
            nn.Conv2d(c1, stem_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(stem_channels),
        )
        self.output_head = nn.Conv2d(stem_channels, 1, kernel_size=3, padding=1)

    def _encode_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(self._prepare_encoder_input(x))
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
        hidden = self.decoder_head(z).view(-1, self.decoder_channels, self.bottleneck_shape[0], self.bottleneck_shape[1])
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


class ResidualMultiscaleAutoencoder(nn.Module, CoordInputMixin):
    def __init__(
        self,
        input_shape: tuple[int, int],
        latent_dim: int,
        width_mult: float = 1.0,
        coordconv: bool = False,
    ) -> None:
        nn.Module.__init__(self)
        CoordInputMixin.__init__(self, input_shape, coordconv)
        self.latent_dim = latent_dim
        stem_channels = _scaled_channels(24, width_mult)
        c1 = _scaled_channels(40, width_mult)
        c2 = _scaled_channels(72, width_mult)
        c3 = _scaled_channels(104, width_mult)
        c4 = _scaled_channels(144, width_mult)

        self.stem = nn.Sequential(
            nn.Conv2d(self.encoder_input_channels, stem_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(stem_channels),
            ResidualBlock(stem_channels),
        )
        self.down1 = nn.Sequential(nn.Conv2d(stem_channels, c1, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.block1 = nn.Sequential(ResidualBlock(c1), ResidualBlock(c1))
        self.down2 = nn.Sequential(nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.block2 = nn.Sequential(ResidualBlock(c2), ResidualBlock(c2))
        self.down3 = nn.Sequential(nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.block3 = nn.Sequential(ResidualBlock(c3), ResidualBlock(c3))
        self.down4 = nn.Sequential(nn.Conv2d(c3, c4, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.block4 = nn.Sequential(ResidualBlock(c4), ResidualBlock(c4))

        bottleneck_shape = self._infer_bottleneck_shape(input_shape)
        self.bottleneck_shape = bottleneck_shape
        flattened = c4 * bottleneck_shape[0] * bottleneck_shape[1]
        self.decoder_channels = c4
        self.encoder_head = nn.Linear(flattened, latent_dim)
        self.decoder_head = nn.Linear(latent_dim, flattened)

        self.up1 = nn.Sequential(
            nn.Conv2d(c4, c3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(c3),
            nn.Conv2d(c3, c3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(c3),
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(c3, c2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(c2),
            nn.Conv2d(c2, c2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(c2),
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(c2, c1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(c1),
            nn.Conv2d(c1, c1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(c1),
        )
        self.up4 = nn.Sequential(
            nn.Conv2d(c1, stem_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(stem_channels),
            nn.Conv2d(stem_channels, stem_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(stem_channels),
        )
        self.scale2_head = nn.Conv2d(c2, 1, kernel_size=3, padding=1)
        self.scale3_head = nn.Conv2d(c1, 1, kernel_size=3, padding=1)
        self.output_head = nn.Conv2d(stem_channels, 1, kernel_size=3, padding=1)

    def _encode_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(self._prepare_encoder_input(x))
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
        hidden = self.decoder_head(z).view(-1, self.decoder_channels, self.bottleneck_shape[0], self.bottleneck_shape[1])
        hidden = F.interpolate(hidden, scale_factor=2.0, mode="bilinear", align_corners=False)
        hidden = self.up1(hidden)
        hidden = F.interpolate(hidden, scale_factor=2.0, mode="bilinear", align_corners=False)
        hidden = self.up2(hidden)
        scale2 = F.interpolate(self.scale2_head(hidden), size=self.input_shape, mode="bilinear", align_corners=False)
        hidden = F.interpolate(hidden, scale_factor=2.0, mode="bilinear", align_corners=False)
        hidden = self.up3(hidden)
        scale3 = F.interpolate(self.scale3_head(hidden), size=self.input_shape, mode="bilinear", align_corners=False)
        hidden = F.interpolate(hidden, size=self.input_shape, mode="bilinear", align_corners=False)
        hidden = self.up4(hidden)
        return self.output_head(hidden) + scale3 + scale2

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(x)
        recon = self.decode(latent)
        return recon, latent


def build_autoencoder(
    input_shape: tuple[int, int],
    latent_dim: int,
    architecture: str,
    *,
    width_mult: float = 1.0,
    coordconv: bool = False,
) -> nn.Module:
    if architecture == "baseline":
        return BaselineConvAutoencoder(input_shape, latent_dim, width_mult=width_mult, coordconv=coordconv)
    if architecture == "residual":
        return ResidualConvAutoencoder(input_shape, latent_dim, width_mult=width_mult, coordconv=coordconv)
    if architecture == "residual_multiscale":
        return ResidualMultiscaleAutoencoder(input_shape, latent_dim, width_mult=width_mult, coordconv=coordconv)
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
