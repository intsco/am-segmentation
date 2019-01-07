import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from torchvision import models


def conv3x3(in_ch, out_ch):
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, dilation=1)


class ConvRelu(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = conv3x3(in_ch, out_ch)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class Interpolate(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=False):
        super().__init__()
        self.interp = F.interpolate
        self.size = size
        self.mode = mode
        self.scale_factor = scale_factor
        self.align_corners = align_corners

    def forward(self, x):
        x = self.interp(x, size=self.size, scale_factor=self.scale_factor,
                        mode=self.mode, align_corners=self.align_corners)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, upsampling=False):
        super().__init__()
        self.in_ch = in_ch

        if not upsampling:
            self.block = nn.Sequential(
                ConvRelu(in_ch, mid_ch),
                nn.ConvTranspose2d(mid_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                Interpolate(scale_factor=2, mode='bilinear'),
                ConvRelu(in_ch, mid_ch),
                ConvRelu(mid_ch, out_ch)
            )

    def forward(self, x):
        return self.block(x)


class UNet11(nn.Module):
    def __init__(self, num_filters=32, pretrained=True):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)

        encoder = models.vgg11(pretrained=pretrained).features

        self.conv1 = encoder[0]  # 3 -> 64
        self.conv2 = encoder[3]  # 64 -> 128
        self.conv3_1 = encoder[6]  # 128 -> 256
        self.conv3_2 = encoder[8]  # 256 -> 256
        self.conv4_1 = encoder[11]  # 256 -> 512
        self.conv4_2 = encoder[13]  # 512 -> 512
        self.conv5_1 = encoder[16]  # 512 -> 512
        self.conv5_2 = encoder[18]  # 512 -> 512

        self.center = DecoderBlock(num_filters * 16, num_filters * 16, num_filters * 8)

        self.dec5 = DecoderBlock(num_filters * (8 + 16), num_filters * 16, num_filters * 8)
        self.dec4 = DecoderBlock(num_filters * (8 + 16), num_filters * 16, num_filters * 4)
        self.dec3 = DecoderBlock(num_filters * (4 + 8), num_filters * 8, num_filters * 2)
        self.dec2 = DecoderBlock(num_filters * (2 + 4), num_filters * 4, num_filters)
        self.dec1 = ConvRelu(num_filters * (1 + 2), num_filters)

        self.final = nn.Conv2d(num_filters, 1, kernel_size=1)

    def forward(self, x):
        conv1 = self.relu(self.conv1(x))
        conv2 = self.relu(self.conv2(self.pool(conv1)))
        conv3_1 = self.relu(self.conv3_1(self.pool(conv2)))
        conv3_2 = self.relu(self.conv3_2(conv3_1))
        conv4_1 = self.relu(self.conv4_1(self.pool(conv3_2)))
        conv4_2 = self.relu(self.conv4_2(conv4_1))
        conv5_1 = self.relu(self.conv5_1(self.pool(conv4_2)))
        conv5_2 = self.relu(self.conv5_2(conv5_1))

        center = self.center(self.pool(conv5_2))

        dec5 = self.dec5(torch.cat([center, conv5_2], dim=1))
        dec4 = self.dec4(torch.cat([dec5, conv4_2], dim=1))
        dec3 = self.dec3(torch.cat([dec4, conv3_2], dim=1))
        dec2 = self.dec2(torch.cat([dec3, conv2], dim=1))
        dec1 = self.dec1(torch.cat([dec2, conv1], dim=1))

        return self.final(dec1)

