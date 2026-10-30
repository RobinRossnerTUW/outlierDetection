import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

def _weights_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)

class DepthwiseSeparableConv(nn.Module):
    """Drop-in replacement for Conv2d(in, out, 3, stride, padding=1).
    Depthwise conv operates per-channel; pointwise mixes channels."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=3, stride=stride, padding=1,
            groups=in_channels, bias=False
        )
        self.bn_dw = nn.BatchNorm2d(in_channels)   # BN+ReLU between DW and PW
        self.pointwise = nn.Conv2d(                 # (MobileNet-style)
            in_channels, out_channels,
            kernel_size=1, bias=False
        )

    def forward(self, x):
        return self.pointwise(F.relu(self.bn_dw(self.depthwise(x))))


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlockDWS(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(in_planes, planes, stride=stride)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = DepthwiseSeparableConv(planes, planes, stride=1)
        self.bn2   = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                self.shortcut = LambdaLayer(lambda x:
                    F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNetDWS(nn.Module):
    def __init__(self, block, num_blocks):
        super().__init__()
        self.in_planes = 16
        self.conv1  = DepthwiseSeparableConv(3, 16, stride=1)
        self.bn1    = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        layers = [block(self.in_planes, planes, stride)]
        self.in_planes = planes
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, planes, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        return out.view(out.size(0), -1)


def resnet56_dws(**kwargs):
    return ResNetDWS(BasicBlockDWS, [9, 9, 9], **kwargs)