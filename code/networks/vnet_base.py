import torch
from torch import nn
import torch.nn.functional as F

"""
Two-output VNet for M0 (DTC baseline reproduction).

Identical encoder/decoder to vnet_sdf.py; the boundary head (out_conv3)
and auxiliary deep-supervision head (out_conv_aux) are absent, so the
parameter count exactly matches the original DTC paper.

Forward returns:
    out_tanh (Tensor): SDF regression, shape (B, n_classes, H, W, D), in [-1, 1].
    out_seg  (Tensor): segmentation logits, shape (B, n_classes, H, W, D).
"""


# ---------------------------------------------------------------------------
# Shared building blocks (identical to vnet_sdf.py)
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out,
                 normalization='none'):
        super().__init__()
        ops = []
        for i in range(n_stages):
            in_ch = n_filters_in if i == 0 else n_filters_out
            ops.append(nn.Conv3d(in_ch, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(16, n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            ops.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        return self.conv(x)


class ResidualConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out,
                 normalization='none'):
        super().__init__()
        ops = []
        for i in range(n_stages):
            in_ch = n_filters_in if i == 0 else n_filters_out
            ops.append(nn.Conv3d(in_ch, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(16, n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            if i != n_stages - 1:
                ops.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*ops)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x) + x)


class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2,
                 normalization='none'):
        super().__init__()
        ops = [nn.Conv3d(n_filters_in, n_filters_out, stride,
                         padding=0, stride=stride)]
        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(16, n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        ops.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        return self.conv(x)


class UpsamplingDeconvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2,
                 normalization='none'):
        super().__init__()
        ops = [nn.ConvTranspose3d(n_filters_in, n_filters_out, stride,
                                  padding=0, stride=stride)]
        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(16, n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        ops.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        return self.conv(x)


class Upsampling(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2,
                 normalization='none'):
        super().__init__()
        ops = [
            nn.Upsample(scale_factor=stride, mode='trilinear',
                        align_corners=False),
            nn.Conv3d(n_filters_in, n_filters_out, 3, padding=1),
        ]
        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(16, n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        ops.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        return self.conv(x)


# ---------------------------------------------------------------------------
# Two-head VNet (M0 baseline)
# ---------------------------------------------------------------------------

class VNet(nn.Module):
    """
    VNet with exactly two output heads:
        out_tanh — Task 2: SDF regression via Tanh, range [-1, 1].
        out_seg  — Task 1: segmentation logits (apply sigmoid externally).

    The boundary head (out_conv3) and auxiliary deep-supervision head
    (out_conv_aux) present in vnet_sdf.py are intentionally omitted so that
    M0 faithfully reproduces the original DTC network topology and parameter
    count (9.44 M params with n_filters=16, n_classes=1).
    """

    def __init__(self, n_channels=1, n_classes=1, n_filters=16,
                 normalization='none', has_dropout=False,
                 has_residual=False):
        super().__init__()
        self.has_dropout = has_dropout
        Block = ConvBlock if not has_residual else ResidualConvBlock

        # --- Encoder ---
        self.block_one    = Block(1, n_channels,      n_filters,      normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters,     n_filters * 2,  normalization=normalization)

        self.block_two    = Block(2, n_filters * 2,  n_filters * 2,  normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2,  n_filters * 4,  normalization=normalization)

        self.block_three    = Block(3, n_filters * 4,  n_filters * 4,  normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4,  n_filters * 8,  normalization=normalization)

        self.block_four    = Block(3, n_filters * 8,  n_filters * 8,  normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8,  n_filters * 16, normalization=normalization)

        # --- Bottleneck ---
        self.block_five = Block(3, n_filters * 16, n_filters * 16, normalization=normalization)

        # --- Decoder ---
        self.block_five_up  = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8,  normalization=normalization)
        self.block_six      = Block(3, n_filters * 8,  n_filters * 8,  normalization=normalization)
        self.block_six_up   = UpsamplingDeconvBlock(n_filters * 8,  n_filters * 4,  normalization=normalization)

        self.block_seven    = Block(3, n_filters * 4,  n_filters * 4,  normalization=normalization)
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4,  n_filters * 2,  normalization=normalization)

        self.block_eight    = Block(2, n_filters * 2,  n_filters * 2,  normalization=normalization)
        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2,  n_filters,      normalization=normalization)

        self.block_nine = Block(1, n_filters, n_filters, normalization=normalization)

        # --- Output heads (2 only) ---
        self.out_conv  = nn.Conv3d(n_filters, n_classes, 1)   # Task 2: SDF
        self.out_conv2 = nn.Conv3d(n_filters, n_classes, 1)   # Task 1: seg logits

        self.tanh    = nn.Tanh()
        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    # ------------------------------------------------------------------
    def encoder(self, x):
        x1    = self.block_one(x)
        x1_dw = self.block_one_dw(x1)

        x2    = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3    = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4    = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)
        if self.has_dropout:
            x5 = self.dropout(x5)

        return [x1, x2, x3, x4, x5]

    def decoder(self, features):
        x1, x2, x3, x4, x5 = features

        x5_up = self.block_five_up(x5) + x4
        x6    = self.block_six(x5_up)
        x6_up = self.block_six_up(x6) + x3
        x7    = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7) + x2
        x8    = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8) + x1
        x9    = self.block_nine(x8_up)
        if self.has_dropout:
            x9 = self.dropout(x9)

        out_tanh = self.tanh(self.out_conv(x9))   # SDF in [-1, 1]
        out_seg  = self.out_conv2(x9)              # segmentation logits

        return out_tanh, out_seg   # 2 outputs — no boundary/aux

    def forward(self, x, turnoff_drop=False):
        if turnoff_drop:
            _saved = self.has_dropout
            self.has_dropout = False
        out = self.decoder(self.encoder(x))
        if turnoff_drop:
            self.has_dropout = _saved
        return out   # (out_tanh, out_seg)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    model = VNet(n_channels=1, n_classes=1, normalization='batchnorm',
                 has_dropout=False)
    x = torch.randn(2, 1, 112, 112, 80)
    out_tanh, out_seg = model(x)
    print("out_tanh:", out_tanh.shape)   # (2, 1, 112, 112, 80)
    print("out_seg: ", out_seg.shape)    # (2, 1, 112, 112, 80)
    print("Params:  ", sum(p.numel() for p in model.parameters()))
