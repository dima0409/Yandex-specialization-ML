from torch import nn
import torch.nn.functional as F
from models.ModelsUtils import Model


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)


class Covid(Model):
    @property
    def y_field_name(self) -> str:
        return "labels"

    @property
    def x_field_name(self) -> str:
        return "original_image"

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            DoubleConv(1, 64),
            nn.MaxPool2d(2),
            DoubleConv(64, 128),
            nn.MaxPool2d(2),
            DoubleConv(128, 256),
            nn.MaxPool2d(2),
            DoubleConv(256, 512),
            nn.MaxPool2d(2),
            DoubleConv(512, 512),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            DoubleConv(512, 256),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            DoubleConv(256, 128),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            DoubleConv(128, 64),
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            DoubleConv(128, 64),
            nn.Conv2d(64, 3, kernel_size=1)
        )

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        return F.log_softmax(x, dim=1)
