"""
Encoder–decoder CNN for salient object segmentation (scratch, no pretrained backbones).

- Baseline: conv + ReLU + max pool encoder, convtranspose + ReLU decoder, sigmoid mask.
- Improved: batch normalization + dropout in the encoder for regularization experiments.
- UNet: encoder–decoder with skip connections (best IoU for this project; still no pretrained backbone).
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn


def _dbl_conv(ci: int, co: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(ci, co, kernel_size=3, padding=1),
        nn.BatchNorm2d(co),
        nn.ReLU(inplace=True),
        nn.Conv2d(co, co, kernel_size=3, padding=1),
        nn.BatchNorm2d(co),
        nn.ReLU(inplace=True),
    )


class SODUNet(nn.Module):
    """
    U-Net style encoder–decoder with skip concatenation (scratch, no pretrained weights).
    Typically reaches much higher IoU on ECSSD than the plain bottleneck CNN.
    """

    def __init__(self, in_ch: int = 3, base: int = 32) -> None:
        super().__init__()
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8

        self.enc1 = _dbl_conv(in_ch, c1)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = _dbl_conv(c1, c2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = _dbl_conv(c2, c3)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = _dbl_conv(c3, c4)
        self.pool4 = nn.MaxPool2d(2)

        self.mid = _dbl_conv(c4, c4)

        self.up4 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec4 = _dbl_conv(c3 + c4, c3)
        self.up3 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec3 = _dbl_conv(c2 + c3, c2)
        self.up2 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec2 = _dbl_conv(c1 + c2, c1)
        self.up1 = nn.ConvTranspose2d(c1, c1, kernel_size=2, stride=2)
        self.dec1 = _dbl_conv(c1 + c1, c1)

        self.out_conv = nn.Conv2d(c1, 1, kernel_size=1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.enc1(x)
        p1 = self.pool1(s1)
        s2 = self.enc2(p1)
        p2 = self.pool2(s2)
        s3 = self.enc3(p2)
        p3 = self.pool3(s3)
        s4 = self.enc4(p3)
        p4 = self.pool4(s4)

        b = self.mid(p4)

        x = self.up4(b)
        x = self.dec4(torch.cat([x, s4], dim=1))
        x = self.up3(x)
        x = self.dec3(torch.cat([x, s3], dim=1))
        x = self.up2(x)
        x = self.dec2(torch.cat([x, s2], dim=1))
        x = self.up1(x)
        x = self.dec1(torch.cat([x, s1], dim=1))

        return torch.sigmoid(self.out_conv(x))


class SODBaseline(nn.Module):
    """Baseline UNet-style bottleneck without skip connections (minimal spec-compliant CNN)."""

    def __init__(self, in_ch: int = 3, base: int = 32) -> None:
        super().__init__()

        def enc_block(c_in: int, c_out: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(c_out, c_out, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        def dec_block(c_in: int, c_out: int) -> nn.Sequential:
            return nn.Sequential(
                nn.ConvTranspose2d(c_in, c_out, kernel_size=2, stride=2),
                nn.Conv2d(c_out, c_out, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(c_out, c_out, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )

        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8
        self.enc1 = enc_block(in_ch, c1)
        self.enc2 = enc_block(c1, c2)
        self.enc3 = enc_block(c2, c3)
        self.enc4 = enc_block(c3, c4)

        self.mid = nn.Sequential(
            nn.Conv2d(c4, c4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.dec4 = dec_block(c4, c3)
        self.dec3 = dec_block(c3, c2)
        self.dec2 = dec_block(c2, c1)
        self.dec1 = dec_block(c1, c1)

        self.out_conv = nn.Conv2d(c1, 1, kernel_size=1)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.mid(x)
        x = self.dec4(x)
        x = self.dec3(x)
        x = self.dec2(x)
        x = self.dec1(x)
        x = self.out_conv(x)
        return torch.sigmoid(x)


class SODImproved(nn.Module):
    """Deeper regularized variant: batch norm + dropout on encoder conv stacks."""

    def __init__(self, in_ch: int = 3, base: int = 32, dropout_p: float = 0.1) -> None:
        super().__init__()

        def enc_block(c_in: int, c_out: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout_p),
                nn.Conv2d(c_out, c_out, kernel_size=3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        def dec_block(c_in: int, c_out: int) -> nn.Sequential:
            return nn.Sequential(
                nn.ConvTranspose2d(c_in, c_out, kernel_size=2, stride=2),
                nn.Conv2d(c_out, c_out, kernel_size=3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.Conv2d(c_out, c_out, kernel_size=3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
            )

        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8
        self.enc1 = enc_block(in_ch, c1)
        self.enc2 = enc_block(c1, c2)
        self.enc3 = enc_block(c2, c3)
        self.enc4 = enc_block(c3, c4)

        self.mid = nn.Sequential(
            nn.Conv2d(c4, c4, kernel_size=3, padding=1),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),
        )

        self.dec4 = dec_block(c4, c3)
        self.dec3 = dec_block(c3, c2)
        self.dec2 = dec_block(c2, c1)
        self.dec1 = dec_block(c1, c1)

        self.out_conv = nn.Conv2d(c1, 1, kernel_size=1)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.mid(x)
        x = self.dec4(x)
        x = self.dec3(x)
        x = self.dec2(x)
        x = self.dec1(x)
        x = self.out_conv(x)
        return torch.sigmoid(x)


def build_model(
    variant: Literal["baseline", "improved", "unet"] = "baseline", **kwargs
) -> nn.Module:
    if variant == "baseline":
        return SODBaseline(**kwargs)
    if variant == "improved":
        return SODImproved(**kwargs)
    if variant == "unet":
        return SODUNet(**kwargs)
    raise ValueError(f"Unknown variant: {variant}")


if __name__ == "__main__":
    for name, ctor in [
        ("baseline", SODBaseline),
        ("improved", SODImproved),
        ("unet", SODUNet),
    ]:
        m = ctor()
        t = torch.randn(2, 3, 128, 128)
        y = m(t)
        print(name, y.shape, y.min().item(), y.max().item())
