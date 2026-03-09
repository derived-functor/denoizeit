"""U-Net model"""

import torch
from torch import nn
import torch.nn.functional as F
from complexPyTorch.complexLayers import (
    ComplexBatchNorm2d,
    ComplexConv2d,
    ComplexConvTranspose2d,
)
from .activation_function import ComplexGELU
from .attention import BottleneckAttention


class UNet(nn.Module):  # pylint: disable=C0116,R0902
    """Implementation of U-Net architecture"""

    def __init__(self):
        super().__init__()

        default_params = {"kernel_size": 3, "stride": 2, "padding": 1}

        self.encoder_block1 = nn.Sequential(
            ComplexConv2d(1, 16, **default_params),
            ComplexBatchNorm2d(16),
            ComplexGELU(),
        )
        self.encoder_block2 = nn.Sequential(
            ComplexConv2d(16, 32, **default_params),
            ComplexBatchNorm2d(32),
            ComplexGELU(),
        )
        self.encoder_block3 = nn.Sequential(
            ComplexConv2d(32, 64, **default_params),
            ComplexBatchNorm2d(64),
            ComplexGELU(),
        )
        self.encoder_block4 = nn.Sequential(
            ComplexConv2d(64, 128, **default_params),
            ComplexBatchNorm2d(128),
            ComplexGELU(),
        )
        self.encoder_block5 = nn.Sequential(
            ComplexConv2d(128, 256, **default_params),
            ComplexBatchNorm2d(256),
            ComplexGELU(),
        )
        self.encoder_block6 = nn.Sequential(
            ComplexConv2d(256, 512, **default_params),
            ComplexBatchNorm2d(512),
            ComplexGELU(),
        )
        self.bottleneck_attn = BottleneckAttention()

        upscale_params = {
            "kernel_size": 3,
            "stride": 2,
            "padding": 1,
            "output_padding": 1,
        }
        out_conv_params = {"kernel_size": 3, "padding": 1}

        self.up1 = ComplexConvTranspose2d(512, 256, **upscale_params)
        self.decoder_block1 = nn.Sequential(
            ComplexConv2d(512, 256, **out_conv_params),
            ComplexBatchNorm2d(256),
            ComplexGELU(),
        )

        self.up2 = ComplexConvTranspose2d(256, 128, **upscale_params)
        self.decoder_block2 = nn.Sequential(
            ComplexConv2d(256, 128, **out_conv_params),
            ComplexBatchNorm2d(128),
            ComplexGELU(),
        )

        self.up3 = ComplexConvTranspose2d(128, 64, **upscale_params)
        self.decoder_block3 = nn.Sequential(
            ComplexConv2d(128, 64, **out_conv_params),
            ComplexBatchNorm2d(64),
            ComplexGELU(),
        )

        self.up4 = ComplexConvTranspose2d(64, 32, **upscale_params)
        self.decoder_block4 = nn.Sequential(
            ComplexConv2d(64, 32, **out_conv_params),
            ComplexBatchNorm2d(32),
            ComplexGELU(),
        )

        self.up5 = ComplexConvTranspose2d(32, 16, **upscale_params)
        self.decoder_block5 = nn.Sequential(
            ComplexConv2d(32, 16, **out_conv_params),
            ComplexBatchNorm2d(16),
            ComplexGELU(),
        )

        self.up6 = ComplexConvTranspose2d(16, 16, **upscale_params)
        self.decoder_block6 = nn.Sequential(
            ComplexConv2d(17, 16, **out_conv_params),
            ComplexBatchNorm2d(16),
            ComplexGELU(),
        )

        self.mask = ComplexConv2d(16, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, F, T)
        """
        orig_t = x.shape[-1]
        x = x[:, :, :256, :]

        pad_t = (64 - orig_t % 64) % 64

        if pad_t > 0:
            # Добавляем pad_t нулей в конец измерения времени
            x = F.pad(x, (0, pad_t))  # pad задается с конца: (left, right, top, bottom)

        e1 = self.encoder_block1(x)
        e2 = self.encoder_block2(e1)
        e3 = self.encoder_block3(e2)
        e4 = self.encoder_block4(e3)
        e5 = self.encoder_block5(e4)
        e6 = self.encoder_block6(e5)

        b = self.bottleneck_attn(e6)

        d = self.up1(b)
        d = torch.cat([d, e5], dim=1)
        d = self.decoder_block1(d)

        d = self.up2(d)
        d = torch.cat([d, e4], dim=1)
        d = self.decoder_block2(d)

        d = self.up3(d)
        d = torch.cat([d, e3], dim=1)
        d = self.decoder_block3(d)

        d = self.up4(d)
        d = torch.cat([d, e2], dim=1)
        d = self.decoder_block4(d)

        d = self.up5(d)
        d = torch.cat([d, e1], dim=1)
        d = self.decoder_block5(d)

        d = self.up6(d)
        d = torch.cat([d, x], dim=1)
        d = self.decoder_block6(d)

        mask = self.mask(d)

        if pad_t > 0:
            mask = mask[:, :, :, :-pad_t]

        return mask
