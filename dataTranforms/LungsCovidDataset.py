import os
import torch
import pandas as pd
from skimage import io
import numpy as np
from torch.utils.data import Dataset


# Custom dataset model from pytorch docs (https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)
class LungsCovidDataset(Dataset):

    def __init__(self, csv_file, original_root_dir, cropped_root_dir, transformation=None):
        self.labes_frame = pd.read_csv(csv_file)
        self.original_root_dir = original_root_dir
        self.cropped_root_dir = cropped_root_dir
        self.transform = transformation

    def __len__(self):
        return len(self.labes_frame)  # В качестве размера датасета отдаем количество элементов в таблице

    def __getitem__(self, idx):

        # Если на вход приходит тензор нужно преобразовать в список, т.к. другие библиотеки не умеют работать с
        # тензорами pytorch
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Формируем пути до картинок (оригинальной и маски легких)
        original_img_name = os.path.join(self.original_root_dir,
                                         "img_" + str(self.labes_frame.iloc[idx, 0]) + ".png")
        cropped_img_name = os.path.join(self.cropped_root_dir,
                                        "img_" + str(self.labes_frame.iloc[idx, 0]) + ".png")

        # Записыываем картинки в переменные
        original_image = io.imread(original_img_name)
        cropped_image = io.imread(cropped_img_name)

        # Подготавливаем label с типом картинки из таблицы ответов
        labels = self.labes_frame.iloc[idx, 1]

        # Формируем элемент датасета
        sample = {'original_image': original_image, 'cropped_image': cropped_image, 'labels': labels}

        if self.transform:
            sample = self.transform(sample)

        return sample


