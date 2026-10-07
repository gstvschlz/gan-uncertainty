"""Fully convolutional WGAN-GP pair (spatial GAN, Jetchev et al. 2016; Laloy et al. 2018).

The latent is a grid, not a vector: a (B, z_ch, h, w) latent gives a (B, 1, 32h, 32w)
image. Training uses 4x4 latents (128x128 crops of the TI at its own pixel scale);
sampling uses an 8x8 latent (256x256) cropped to the 250x250 size of the reference TI.
"""

import torch.nn as nn

SCALE = 32  # output pixels per latent cell (five 2x upsamplings)


class GeneratorModel(nn.Module):
    def __init__(self, z_ch: int = 64, dims: tuple = (256, 128, 64, 32, 32)):
        super().__init__()

        def block(dim_in, dim_out, up=True):
            # Nearest upsampling + 3x3 conv instead of transposed convs: no checkerboard
            # artifacts, which a binary facies image shows as grain.
            return nn.Sequential(
                *([nn.Upsample(scale_factor=2)] if up else []),
                nn.Conv2d(dim_in, dim_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(dim_out),
                nn.ReLU(),
            )

        self.z_ch = z_ch
        self.net = nn.Sequential(
            block(z_ch, dims[0], up=False),
            block(dims[0], dims[0]),
            *[block(a, b) for a, b in zip(dims, dims[1:])],
            nn.Conv2d(dims[-1], 1, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)


class CriticModel(nn.Module):
    """Five stride-2 convs (receptive field ~125 px), then the mean of the score map.
    No normalization layers: the gradient penalty is per sample."""

    def __init__(self, dims: tuple = (32, 64, 128, 256, 256)):
        super().__init__()
        layers, dim_in = [], 1
        for d in dims:
            layers += [nn.Conv2d(dim_in, d, 5, stride=2, padding=2), nn.LeakyReLU(0.2)]
            dim_in = d
        self.net = nn.Sequential(*layers, nn.Conv2d(dim_in, 1, 1))

    def forward(self, x):
        return self.net(x).mean(dim=(1, 2, 3))
