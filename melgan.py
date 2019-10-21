from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm


def calc_padding(kernel_size, stride, dilation=1):
    return (dilation * (kernel_size - 1) - stride + 2) // 2


def wnconv1d(
    in_channel, out_channel, kernel_size, stride=1, dilation=1, groups=1, act='postact'
):
    padding = calc_padding(kernel_size, stride, dilation)
    conv = nn.Conv1d(
        in_channel,
        out_channel,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    nn.init.kaiming_uniform_(
        conv.weight,
        a=0.2,
        mode='fan_out',
        nonlinearity='leaky_relu' if act else 'linear',
    )
    nn.init.zeros_(conv.bias)
    conv = weight_norm(conv)

    if act == 'preact':
        layers = [nn.LeakyReLU(0.2), conv]

    elif act == 'postact':
        layers = [conv, nn.LeakyReLU(0.2)]

    elif not act:
        layers = [conv]

    return nn.Sequential(*layers)


def wnconvtranspose1d(
    in_channel, out_channel, kernel_size, stride=1, dilation=1, act='postact'
):
    padding = calc_padding(kernel_size, stride, dilation)
    conv = nn.ConvTranspose1d(
        in_channel,
        out_channel,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    nn.init.kaiming_uniform_(
        conv.weight, a=0.2, mode='fan_out', nonlinearity='leaky_relu'
    )
    nn.init.zeros_(conv.bias)
    conv = weight_norm(conv)

    if act == 'preact':
        layers = [nn.LeakyReLU(0.2), conv]

    elif act == 'postact':
        layers = [conv, nn.LeakyReLU(0.2)]

    elif not act:
        layers = [conv]

    return nn.Sequential(*layers)


class ResidualStack(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.conv1 = nn.Sequential(
            wnconv1d(channel, channel, 3, act='preact'),
            wnconv1d(channel, channel, 3, act='preact'),
        )

        self.conv2 = nn.Sequential(
            wnconv1d(channel, channel, 3, dilation=3, act='preact'),
            wnconv1d(channel, channel, 3, act='preact'),
        )

        self.conv3 = nn.Sequential(
            wnconv1d(channel, channel, 3, dilation=9, act='preact'),
            wnconv1d(channel, channel, 3, act='preact'),
        )

    def forward(self, input):
        out = self.conv1(input) + input
        out = self.conv2(out) + out
        out = self.conv3(out) + out

        return out


class Generator(nn.Sequential):
    def __init__(self, n_mels):
        layers = [
            wnconv1d(n_mels, 512, 7, act=None),
            wnconvtranspose1d(512, 256, 16, stride=8, act='preact'),
            ResidualStack(256),
            wnconvtranspose1d(256, 128, 16, stride=8, act='preact'),
            ResidualStack(128),
            wnconvtranspose1d(128, 64, 4, stride=2, act='preact'),
            ResidualStack(64),
            wnconvtranspose1d(64, 32, 4, stride=2, act='preact'),
            ResidualStack(32),
            wnconvtranspose1d(32, 1, 7, act='preact'),
            nn.Tanh(),
        ]

        super().__init__(*layers)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.ModuleList(
            [
                wnconv1d(1, 16, 15),
                wnconv1d(16, 64, 41, stride=4, groups=4),
                wnconv1d(64, 256, 41, stride=4, groups=16),
                wnconv1d(256, 1024, 41, stride=4, groups=64),
                wnconv1d(1024, 1024, 41, stride=4, groups=256),
                wnconv1d(1024, 1024, 5),
            ]
        )

        self.out = nn.Sequential(
            wnconv1d(1024, 1, 3, act=None), nn.AdaptiveAvgPool1d(1), nn.Flatten()
        )

    def forward(self, input):
        feats = []

        out = input
        for conv in self.conv:
            out = conv(out)
            feats.append(out)

        out = self.out(out)

        return out, feats


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, n_scales=3):
        super().__init__()

        self.n_scales = n_scales

        self.discriminators = nn.ModuleList()
        for i in range(self.n_scales):
            self.discriminators.append(Discriminator())

    def forward(self, input):
        feats = []
        outs = []

        for i, discriminator in enumerate(self.discriminators):
            if i != 0:
                input = F.avg_pool1d(input, kernel_size=4, stride=2, padding=1)

            out, feat = discriminator(input)
            outs.append(out)
            feats.extend(feat)

        return outs, feats
