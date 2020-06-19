import torch
import torch.nn as nn
import math
from torchsummary import summary
from torch.nn.parameter import Parameter
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]

class conv_bn_relu(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding):
        super(conv_bn_relu, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_planes, in_planes, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_planes),
            nn.Conv2d(in_planes, out_planes, 1, 1, 0, 1, 1)
        )

    def forward(self, x):
        return self.conv(x)

inputs = torch.rand(8,64,16,16)

gos = GhostModule(inp=64, oup=64, kernel_size=1, ratio=2, dw_size=3, stride=2).to(device)

dw = conv_bn_relu(in_planes=64, out_planes=64, kernel_size=3, stride=1, padding=1).to(device)
#outputs = avg(inputs)
summary(gos, input_size=(64, 16, 16))
#print(outputs.shape)