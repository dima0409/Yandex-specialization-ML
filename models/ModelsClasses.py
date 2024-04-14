from torch import nn
import torch.nn.functional as F
from models.ModelsUtils import Model

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class Covid(Model):
    def __init__(self):
        super(Covid, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(32, 64, stride=1),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(64, 128, stride=2),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(128, 128, stride=1),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(128, 256, stride=2),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(256, 256, stride=1),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(256, 512, stride=2),
            nn.ReLU(inplace=True),
            *[
                nn.Sequential(
                    DepthwiseSeparableConv(512, 512, stride=1),
                    nn.ReLU(inplace=True)
                ) for _ in range(5)
            ],
            DepthwiseSeparableConv(512, 1024, stride=2),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(1024, 1024, stride=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1024, 3)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)
