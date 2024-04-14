import torch
from torch import nn
from models.ModelsUtils import Model


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.double_conv(x)
        out += self.shortcut(x)
        return out


class Covid(Model):
    @property
    def y_field_name(self) -> str:
        return "labels"

    @property
    def x_field_name(self) -> str:
        return "original_image"

    def __init__(self, num_classes=3):
        super().__init__()
        self.in_channels = 64
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1, in_channels=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2, in_channels=64)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2, in_channels=128)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2, in_channels=256)
        self.linear = nn.Linear(512, num_classes)

        self.model = nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            nn.Flatten(),
            self.linear
        )

    def _make_layer(self, block, out_channels, num_blocks, stride, in_channels=None):
        if in_channels is None:
            in_channels = self.in_channels
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(in_channels, out_channels, stride))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)