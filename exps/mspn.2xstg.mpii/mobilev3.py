import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

def conv_bn(inp, oup, stride, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(inp, oup, 3, stride, 1, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)
    )


def conv_1x1_bn(inp, oup, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(inp, oup, 1, 1, 0, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)
    )

class Hswish(nn.Module):
    def __init__(self, inplace = True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace = self.inplace) / 6.

class Hsigmoid(nn.Module):
    def __init__(self, inplace = True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace = self.inplace) / 6.

class SEModule(nn.Module):
    def __init__(self, channel, reduction = 4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias = False),
            nn.ReLU(inplace = True),
            nn.Linear(channel // reduction, channel, bias = False),
            Hsigmoid()
            # nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class Identity(nn.Module):
    def __init__(self, channel):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class MobileBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, latent_channels, kernel, stride, se = False, nl = 'RE'):
        super(MobileBottleneck, self).__init__()
        assert stride in [1, 2]
        assert kernel in [3, 5]
        padding = (kernel - 1) // 2
        self.use_res_connect = stride == 1 and in_channels == out_channels

        if nl == 'RE':
            nonlinear_layer = nn.ReLU # or ReLU6
        elif nl == 'HS':
            nonlinear_layer = Hswish
        else:
            raise NotImplementedError
        if se:
            SELayer = SEModule
        else:
            SELayer = Identity

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(in_channels, latent_channels, 1, 1, 0, bias = False),
            nn.BatchNorm2d(latent_channels),
            nonlinear_layer(inplace = True),
            # dw
            nn.Conv2d(latent_channels, latent_channels, kernel, stride, padding, groups = latent_channels, bias = False),
            nn.BatchNorm2d(latent_channels),
            SELayer(latent_channels),
            nonlinear_layer(inplace = True),
            # pw-linear
            nn.Conv2d(latent_channels, out_channels, 1, 1, 0, bias = False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)