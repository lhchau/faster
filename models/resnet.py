'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        self.activate = nn.ReLU(inplace=True)
        # self.activate = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        out = self.activate(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.activate(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        self.activate = nn.ReLU(inplace=True)
        # self.activate = nn.LeakyReLU(negative_slope=0.1, inplace=True)
            

    def forward(self, x):
        out = self.activate(self.bn1(self.conv1(x)))
        out = self.activate(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.activate(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, widen_factor=1):
        super(ResNet, self).__init__()
        self.in_planes = int(64 * widen_factor)
        
        """
        # ImageNet downscaling
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        """
        self.conv1 = nn.Conv2d(3, int(64 * widen_factor), kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(int(64 * widen_factor))
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, int(64 * widen_factor), num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, int(128 * widen_factor), num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, int(256 * widen_factor), num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, int(512 * widen_factor), num_blocks[3], stride=2)
        self.avgpool2d = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(int(512*block.expansion*widen_factor), num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        # out = self.maxpool(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avgpool2d(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        
        return out


def resnet18(num_classes=10, widen_factor=1):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, widen_factor=widen_factor)


def resnet34(num_classes=10, widen_factor=1):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, widen_factor=widen_factor)


def resnet50(num_classes=10, widen_factor=1):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, widen_factor=widen_factor)


def resnet101(num_classes=10, widen_factor=1):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes, widen_factor=widen_factor)


def resnet152(num_classes=10, widen_factor=1):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes, widen_factor=widen_factor)


def test():
    net = resnet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()

if __name__ == "__main__":
    widths = [1, 1.5, 2]
    for w in widths:
        net = resnet18(num_classes=100, widen_factor=w)
        print(f"Rn18_wi={w}", sum(p.numel() for p in net.parameters()))
    net = resnet34(num_classes=100)
    print(f"Rn34_wi=1", sum(p.numel() for p in net.parameters()))
    net = resnet50(num_classes=100)
    print(f"Rn50_wi=1", sum(p.numel() for p in net.parameters()))
    net = resnet101(num_classes=100)
    print(f"Rn101_wi=1", sum(p.numel() for p in net.parameters()))

