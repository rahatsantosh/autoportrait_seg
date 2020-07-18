import torch
from torch import nn

def get_same_padding(kernel_size):
    """Calculate padding size for same padding,
    assuming stride of 1 and square kernel"""

    if type(kernel_size) is tuple:
        kernel_size = kernel_size[0]
    pad_size = (kernel_size-1)//2
    if kernel_size%2 == 0:
        padding = (pad_size, pad_size+1)
    else:
        padding = pad_size

    return padding


class Conv_block(nn.Sequential):
    def __init__(self, conv_type, in_channels, out_channels, inter_channels = None):
        if conv_type == "1x1":
            super(Conv_block, self).__init__(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size = 1,
                    stride = 1,
                    padding = get_same_padding(1)
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU6(inplace=True)
            )
        elif conv_type == "3x3":
            super(Conv_block, self).__init__(
                nn.Conv2d(
                    in_channels,
                    in_channels//2,
                    kernel_size = 1,
                    stride = 1,
                    padding = get_same_padding(1)
                ),
                nn.ReLU6(inplace=True),
                nn.Conv2d(
                    in_channels//2,
                    out_channels,
                    kernel_size = 3,
                    stride = 1,
                    padding = get_same_padding(3)
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU6(inplace=True)
            )
        else:
            super(Conv_block, self).__init__(
                nn.Conv2d(
                    in_channels,
                    in_channels//2,
                    kernel_size = 1,
                    stride = 1,
                    padding = get_same_padding(1)
                ),
                nn.ReLU6(inplace=True),
                nn.Conv2d(
                    in_channels//2,
                    out_channels,
                    kernel_size = 5,
                    stride = 1,
                    padding = get_same_padding(5)
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU6(inplace=True)
            )

class Deconv_block(nn.Sequential):
    def __init__(self, conv_type, in_channels, out_channels):
        if conv_type == "1x1":
            super(Deconv_block, self).__init__(
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size = 1,
                    stride = 1,
                    padding = get_same_padding(1)
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU6(inplace=True)
            )
        elif conv_type == "3x3":
            super(Deconv_block, self).__init__(
                nn.ConvTranspose2d(
                    in_channels,
                    in_channels//2,
                    kernel_size = 1,
                    stride = 1,
                    padding = get_same_padding(1)
                ),
                nn.ReLU6(inplace=True),
                nn.ConvTranspose2d(
                    in_channels//2,
                    out_channels,
                    kernel_size = 3,
                    stride = 1,
                    padding = get_same_padding(3)
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU6(inplace=True)
            )
        else:
            super(Deconv_block, self).__init__(
                nn.ConvTranspose2d(
                    in_channels,
                    in_channels//2,
                    kernel_size = 1,
                    stride = 1,
                    padding = get_same_padding(1)
                ),
                nn.ReLU6(inplace=True),
                nn.ConvTranspose2d(
                    in_channels//2,
                    out_channels,
                    kernel_size = 5,
                    stride = 1,
                    padding = get_same_padding(5)
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU6(inplace=True)
            )

class Inception_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Inception_block, self).__init__()

        self.conv1x1 = Conv_block("1x1", in_channels, out_channels)
        self.conv3x3 = Conv_block("3x3", in_channels, out_channels)
        self.conv5x5 = Conv_block("5x5", in_channels, out_channels)

    def forward(self, x):

        a = self.conv1x1(x)
        b = self.conv3x3(x)
        c = self.conv5x5(x)

        x = a + b + c
        return x


class Inception_block_r(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Inception_block_r, self).__init__()

        self.deconv1x1 = Deconv_block("1x1", in_channels, out_channels)
        self.deconv3x3 = Deconv_block("3x3", in_channels, out_channels)
        self.deconv5x5 = Deconv_block("5x5", in_channels, out_channels)

    def forward(self, x):

        a = self.deconv1x1(x)
        b = self.deconv3x3(x)
        c = self.deconv5x5(x)

        x = a + b + c

        return x

class Autoencoder1(nn.Module):
    def __init__(self):
        super(Autoencoder1, self).__init__()

        self.encode1 = Inception_block(3, 64)
        self.conv1 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=2)
        self.relu6 = nn.ReLU6(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=2, output_padding = 1)
        self.decode1 = Inception_block_r(64, 3)

    def forward(self, x):

        x = self.encode1(x)
        x = self.conv1(x)
        x = self.relu6(x)
        x = self.deconv1(x)
        x = self.relu6(x)
        x = self.decode1(x)
        x = torch.sigmoid(x)

        return x
