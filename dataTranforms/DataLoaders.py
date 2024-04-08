from dataclasses import dataclass

import matplotlib.pyplot as plt
from torch.utils.data import random_split, DataLoader

from dataTranforms.LungsCovidDataset import LungsCovidDataset


class DataLoadersGenerator:
    def __init__(self, csv_file: str, original_root_dir: str, cropped_root_dir: str):
        self.data_loaders: dict = dict()
        self.providers_info: dict = dict()
        self.csv_file = csv_file
        self.original_root_dir = original_root_dir
        self.cropped_root_dir = cropped_root_dir
        self.__init_data_providers()
        self.__init_dataloaders()

    def __init_data_providers(self):
        self.providers_info = {
            'first without augmentation': self.__create_provider()
        }

    def __init_dataloaders(self):
        def train_test_dataset_split(dataset, test_split: float):
            test_len = int(len(dataset) * test_split)
            train_len = len(dataset) - test_len
            return random_split(dataset, [train_len, test_len])

        for i in self.providers_info.keys():
            provider_info = self.providers_info[i]
            train_set, test_set = train_test_dataset_split(provider_info.dataset, provider_info.test_split)

            train_dl = DataLoader(train_set, batch_size=provider_info.train_batch_size, shuffle=True)
            test_dl = DataLoader(test_set, batch_size=provider_info.test_batch_size, shuffle=False)

            self.data_loaders.update({i: TrainTestDataLoaders(train_dl, test_dl)})

    def __create_provider(self, transform=None, test_split=0.2, train_batch_size=16, test_batch_size=16):
        return DatasetInfo(
            LungsCovidDataset(self.csv_file, self.original_root_dir,
                              self.cropped_root_dir, transformation=transform),
            test_split,
            train_batch_size,
            test_batch_size
        )


def show_images(i, original_image, cropped_image, labels):
    ax = plt.subplot(2, 4, i + 1)
    plt.tight_layout()
    ax.set_title('Sample №{}\n\nanswer - {}'.format(i, labels))
    ax.axis('off')
    plt.imshow(original_image)
    plt.subplot(2, 4, i + 5).axis('off')
    plt.imshow(cropped_image)


def preview_dataset(dataset):
    for i, sample in enumerate(dataset):
        show_images(i, **sample)
        if i == 3:
            plt.show()
            break


@dataclass
class DatasetInfo:
    dataset: LungsCovidDataset
    test_split: float
    train_batch_size: int
    test_batch_size: int


@dataclass
class TrainTestDataLoaders:
    train_dl: DataLoader
    test_dl: DataLoader


# testing
if __name__ == "__main__":
    dataloadersGenerator = DataLoadersGenerator("~/Downloads/data/train_answers.csv", "~/Downloads/data/train_images",
                                                "~/Downloads/data/train_lung_masks")
    dataloaders = dataloadersGenerator.data_loaders['first without augmentation']
    for element in dataloaders.train_dl:
        ax = plt.subplot(2, 1, 1)
        plt.tight_layout()
        ax.set_title('answer - {}'.format(element['labels'][0]))
        ax.axis('off')
        plt.imshow(element["original_image"][0])
        plt.subplot(2, 1, 2).axis('off')
        plt.imshow(element["cropped_image"][0])
        plt.show()
        break
