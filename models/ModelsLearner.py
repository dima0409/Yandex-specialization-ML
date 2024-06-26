import torch

from dataTranforms.DataLoaders import DataLoadersGenerator
from models.ModelsClasses import Covid
from models.ModelsUtils import create_model_info
from utils.Utils import fit


class CovidModels:
    def __init__(self):
        self.models = dict()
        self.__init_models()

    def __init_models(self):
        # Here you can add new models version
        self.models = {
            "firstModel": create_model_info(Covid(), optimizer=torch.optim.Adam, epochs=3)
        }


# testing
if __name__ == "__main__":
    dataLoadersGenerator = DataLoadersGenerator("~/Downloads/data/train_answers.csv", "~/Downloads/data/train_images",
                                                "~/Downloads/data/train_lung_masks")
    models = CovidModels()
    device = torch.device('mps')
    dataloaders = dataLoadersGenerator.data_loaders['first without augmentation']
    for i in models.models.keys():
        model_info = models.models[i]
        print(i)
        model_info.model.to(device)
        fit(model_info.model, train_loader=dataloaders.train_dl, valid_loader=dataloaders.test_dl,
            x_field_name=model_info.model.x_field_name, y_field_name=model_info.model.y_field_name,
            loss_fn=model_info.loss_fn, optimizer=model_info.optimizer, num_epochs=model_info.epochs,
            title="Model", device=device)
