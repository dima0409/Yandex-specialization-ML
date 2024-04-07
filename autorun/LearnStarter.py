import openpyxl
import torch

from dataTranforms.DataLoaders import DataLoadersGenerator
from models.ModelsLearner import CovidModels
from utils.Utils import fit

if __name__ == "__main__":
    path = "ModelsLearning.xlsx"

    wb_obj = openpyxl.load_workbook(path)

    sheet_obj = wb_obj.active

    row = sheet_obj.max_row

    # set dataset paths
    dataLoadersGenerator = DataLoadersGenerator("~/Downloads/data/train_answers.csv",
                                                "~/Downloads/data/train_images",
                                                "~/Downloads/data/train_lung_masks")
    models = CovidModels()
    device = torch.device('mps')  # set learning device (Apple Silicon - "mps"; GPU Cuda cores - "cuda"; CPU - "cpu")

    if row >= 2:
        for i in range(2, row + 1):
            version_name = sheet_obj.cell(row=i, column=1).value
            data_loader_name = sheet_obj.cell(row=i, column=2).value
            model_name = sheet_obj.cell(row=i, column=3).value
            status = sheet_obj.cell(row=i, column=4)

            if status.value in ("", None, "trainable"):
                print(f"Start learning {version_name}")

                status.value = "progress"
                wb_obj.save(path)

                dataloaders = dataLoadersGenerator.data_loaders[data_loader_name]
                model_info = models.models[model_name]
                model_info.model.to(device)
                fit(model_info.model, train_loader=dataloaders.train_dl, valid_loader=dataloaders.test_dl,
                    x_field_name=model_info.model.x_field_name, y_field_name=model_info.model.y_field_name,
                    loss_fn=model_info.loss_fn, optimizer=model_info.optimizer, num_epochs=model_info.epochs,
                    title=version_name, device=device)

                wb_obj = openpyxl.load_workbook(path)
                wb_obj.active.cell(row=i, column=4).value = "done"

                print(f"Finished learning {version_name}")

    else:
        print("Empty xlsx")

    wb_obj.save(path)
