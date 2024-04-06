import torch
from torch import nn
from dataTranforms.DataLoaders import DataLoadersGenerator
from utils.Utils import fit

dataLoadersGenerator = DataLoadersGenerator("Укажи путь до файла csv",
                                            "укажи путь до папки train_images",
                                            "укажи путь до папки с масками")


class CovidModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # архитектура модели
        )

    def forward(self, X):
        return self.model(X)


covidModel = CovidModel()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(covidModel.parameters(), lr=1e-3)
dataloaders = dataLoadersGenerator.data_loaders['first without augmentation']

device = torch.device("cuda")

fit(covidModel, dataloaders.train_dl, dataloaders.test_dl, optimizer, loss_fn, device, 100, "first model")
