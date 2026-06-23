from __future__ import annotations
import torch
import torch.nn as nn


def _center_crop_like(src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    _, _, d, h, w = ref.shape
    sd, sh, sw = src.shape[2:]
    d0 = max((sd - d) // 2, 0)
    h0 = max((sh - h) // 2, 0)
    w0 = max((sw - w) // 2, 0)
    return src[:, :, d0:d0+d, h0:h0+h, w0:w0+w]


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet3DReg(nn.Module):
    """3D U-Net regressor used by DTI-SHNet."""

    def __init__(self, in_channels: int, out_channels: int, base: int = 64):
        super().__init__()
        b = int(base)
        self.e1 = ConvBlock(in_channels, b)
        self.p1 = nn.MaxPool3d(2)
        self.e2 = ConvBlock(b, b * 2)
        self.p2 = nn.MaxPool3d(2)
        self.e3 = ConvBlock(b * 2, b * 4)
        self.p3 = nn.MaxPool3d(2)
        self.mid = ConvBlock(b * 4, b * 8)
        self.u3 = nn.ConvTranspose3d(b * 8, b * 4, 2, stride=2)
        self.d3 = ConvBlock(b * 8, b * 4)
        self.u2 = nn.ConvTranspose3d(b * 4, b * 2, 2, stride=2)
        self.d2 = ConvBlock(b * 4, b * 2)
        self.u1 = nn.ConvTranspose3d(b * 2, b, 2, stride=2)
        self.d1 = ConvBlock(b * 2, b)
        self.out = nn.Conv3d(b, out_channels, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.p1(e1))
        e3 = self.e3(self.p2(e2))
        m = self.mid(self.p3(e3))
        u3 = self.u3(m)
        u3 = self.d3(torch.cat([u3, _center_crop_like(e3, u3)], dim=1))
        u2 = self.u2(u3)
        u2 = self.d2(torch.cat([u2, _center_crop_like(e2, u2)], dim=1))
        u1 = self.u1(u2)
        u1 = self.d1(torch.cat([u1, _center_crop_like(e1, u1)], dim=1))
        return self.out(u1)
