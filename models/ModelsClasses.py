from torch import nn

from models.ModelsUtils import Model


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
            nn.Flatten(),
            nn.Linear(256 * 256, 100),
            nn.ReLU(),
            nn.Linear(100, 3)
        )

    def forward(self, x):
        return self.model(x)
