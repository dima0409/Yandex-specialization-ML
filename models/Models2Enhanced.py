from torch import nn
import torch.nn.functional as F
from models.ModelsUtils import Model
from models.ModelsClasses import DoubleConv


class XRayClassifier(Model):
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
            nn.Dropout(0.25),  # Add dropout after the first downsampling
            DoubleConv(64, 64),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),  # Add dropout after the second downsampling
            DoubleConv(64, 128),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),  # Add dropout after the third downsampling
            DoubleConv(128, 256),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),  # Add dropout after the fourth downsampling
            DoubleConv(256, 256),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.Dropout(0.25),  # Add dropout after the first upsampling
            DoubleConv(128, 128),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.Dropout(0.25),  # Add dropout after the second upsampling
            DoubleConv(64, 64),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.Dropout(0.25),  # Add dropout after the third upsampling
            DoubleConv(32, 32),
            nn.Flatten(),
            nn.Linear(16 * 128 * 256, 3)
        )

    def forward(self, x):
        x = self.model(x)
        return F.softmax(x, dim=1)
