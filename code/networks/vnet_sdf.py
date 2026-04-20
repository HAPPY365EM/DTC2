import torch
from torch import nn
import torch.nn.functional as F

"""
Modified VNet with four output heads:
  - out_tanh    : SDF/LSF regression head (Task 2), output in [-1, 1].
                  Implemented via out_conv + Tanh in the decoder.
  - out_seg     : pixel-wise segmentation head (Task 1), raw logits.
                  Implemented via out_conv2 in the decoder.
  - out_boundary: lightweight boundary prediction head (Task 3), raw logits.
  - out_aux     : auxiliary deep supervision head at half spatial resolution,
                  attached to x7_up (56x56x40). Upsampled to full resolution
                  in the training loop for loss computation. Discarded at inference.

Head-to-layer mapping (decoder output names match __init__ layer names):
    out_tanh     ← self.tanh(self.out_conv(x9))    Task 2: SDF regression
    out_seg      ← self.out_conv2(x9)              Task 1: seg logits
    out_boundary ← self.out_conv3(x9)              Task 3: boundary logits
    out_aux      ← self.out_conv_aux(x7_up)        Aux:    mid-decoder Dice
"""


class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlock, self).__init__()
        ops = []
        for i in range(n_stages):
            input_channel = n_filters_in if i == 0 else n_filters_out
            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False
            ops.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        return self.conv(x)


class ResidualConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ResidualConvBlock, self).__init__()
        ops = []
        for i in range(n_stages):
            input_channel = n_filters_in if i == 0 else n_filters_out
            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False
            if i != n_stages - 1:
                ops.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*ops)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x) + x)


class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(DownsamplingConvBlock, self).__init__()
        ops = []
        ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride,
                             padding=0, stride=stride))
        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != 'none':
            assert False
        ops.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        return self.conv(x)


class UpsamplingDeconvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(UpsamplingDeconvBlock, self).__init__()
        ops = []
        ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride,
                                      padding=0, stride=stride))
        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != 'none':
            assert False
        ops.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        return self.conv(x)


class Upsampling(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(Upsampling, self).__init__()
        ops = []
        ops.append(nn.Upsample(scale_factor=stride, mode='trilinear',
                               align_corners=False))
        ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))
        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != 'none':
            assert False
        ops.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        return self.conv(x)


class VNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16,
                 normalization='none', has_dropout=False, has_residual=False):
        super(VNet, self).__init__()
        self.has_dropout = has_dropout
        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        # Encoder
        self.block_one = convBlock(1, n_channels, n_filters,
                                   normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters,
                                                  normalization=normalization)

        self.block_two = convBlock(2, n_filters * 2, n_filters * 2,
                                   normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4,
                                                  normalization=normalization)

        self.block_three = convBlock(3, n_filters * 4, n_filters * 4,
                                     normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8,
                                                    normalization=normalization)

        self.block_four = convBlock(3, n_filters * 8, n_filters * 8,
                                    normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16,
                                                   normalization=normalization)

        # Bottleneck
        self.block_five = convBlock(3, n_filters * 16, n_filters * 16,
                                    normalization=normalization)

        # Decoder
        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8,
                                                   normalization=normalization)
        self.block_six = convBlock(3, n_filters * 8, n_filters * 8,
                                   normalization=normalization)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4,
                                                  normalization=normalization)

        self.block_seven = convBlock(3, n_filters * 4, n_filters * 4,
                                     normalization=normalization)
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2,
                                                    normalization=normalization)

        self.block_eight = convBlock(2, n_filters * 2, n_filters * 2,
                                     normalization=normalization)
        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters,
                                                    normalization=normalization)

        self.block_nine = convBlock(1, n_filters, n_filters,
                                    normalization=normalization)

        # FIX: comments below now match actual usage in decoder().
        # Previously out_conv was documented as "Task 1: segmentation" but was
        # actually used with Tanh for SDF (Task 2), and out_conv2 was documented
        # as "Task 2: SDF" but used for segmentation logits (Task 1).
        # The weights themselves are unchanged — only the comments are corrected.

        # Task 2: SDF/LSF regression head — reads from x9, output passed through
        # Tanh in decoder() to produce values in [-1, 1].
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        # Task 1: pixel-wise segmentation head — reads from x9, outputs raw logits.
        # Apply sigmoid externally (e.g. torch.sigmoid(out_seg)) for probabilities.
        self.out_conv2 = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        # Task 3: boundary prediction head — reads from x9, single channel,
        # raw logits. Apply sigmoid externally for boundary probabilities.
        self.out_conv3 = nn.Conv3d(n_filters, 1, 1, padding=0)

        # Auxiliary deep supervision head — reads from x7_up (n_filters*2 channels)
        # at half spatial resolution (56x56x40). Upsampled to full resolution in the
        # training loop before computing Dice loss. Discarded at inference time.
        self.out_conv_aux = nn.Conv3d(n_filters * 2, n_classes, 1, padding=0)

        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def encoder(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)
        if self.has_dropout:
            x5 = self.dropout(x5)

        return [x1, x2, x3, x4, x5]

    def decoder(self, features):
        x1, x2, x3, x4, x5 = features

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2
        # x7_up shape: (B, n_filters*2, H/2, W/2, D/2) = (B, 32, 56, 56, 40)

        # Auxiliary deep supervision output at half resolution.
        # Applied before block_eight so it captures mid-decoder semantics.
        out_aux = self.out_conv_aux(x7_up)    # (B, n_classes, 56, 56, 40)

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1

        x9 = self.block_nine(x8_up)
        if self.has_dropout:
            x9 = self.dropout(x9)

        # All three full-resolution heads share x9.
        # out_conv  → Tanh  → Task 2: SDF in [-1, 1]
        # out_conv2 →       → Task 1: segmentation logits (sigmoid externally)
        # out_conv3 →       → Task 3: boundary logits (sigmoid externally)
        out_tanh     = self.tanh(self.out_conv(x9))   # Task 2: SDF [-1, 1]
        out_seg      = self.out_conv2(x9)             # Task 1: seg logits
        out_boundary = self.out_conv3(x9)             # Task 3: boundary logits

        return out_tanh, out_seg, out_boundary, out_aux

    def forward(self, input, turnoff_drop=False):
        if turnoff_drop:
            has_dropout = self.has_dropout
            self.has_dropout = False
        features = self.encoder(input)
        out_tanh, out_seg, out_boundary, out_aux = self.decoder(features)
        if turnoff_drop:
            self.has_dropout = has_dropout
        return out_tanh, out_seg, out_boundary, out_aux


if __name__ == '__main__':
    from thop import profile
    from thop import clever_format
    model = VNet(n_channels=1, n_classes=1, normalization='batchnorm',
                 has_dropout=False)
    input = torch.randn(2, 1, 112, 112, 80)
    out_tanh, out_seg, out_boundary, out_aux = model(input)
    print("out_tanh shape:    ", out_tanh.shape)     # Task 2: SDF
    print("out_seg shape:     ", out_seg.shape)      # Task 1: seg logits
    print("out_boundary shape:", out_boundary.shape) # Task 3: boundary logits
    print("out_aux shape:     ", out_aux.shape)      # Aux: mid-decoder
    print("VNet have {} parameters in total".format(
        sum(x.numel() for x in model.parameters())))
