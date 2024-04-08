from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Any
from torch import nn


class Model(nn.Module):
    __metaclass__ = ABCMeta

    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def x_field_name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def y_field_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError


@dataclass
class ModelInfo:
    model: Any
    loss_fn: Any
    optimizer: Any
    lr: float
    epochs: int


def create_model_info(model,
                      optimizer,
                      loss_fn=nn.CrossEntropyLoss(),
                      lr: float = 1e-3,
                      epochs: int = 100
                      ):
    return ModelInfo(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        lr=lr,
        epochs=epochs
    )
