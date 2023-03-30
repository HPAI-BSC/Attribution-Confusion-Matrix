import os

import pandas as pd
from consts.paths import MosaicPaths


class MosaicDataset:
    def __init__(self, csv_path, data_folder = None):
        self.data_folder = data_folder
        self.df = pd.read_csv(csv_path)

    def get_subset(self):
        list_image_filenames, list_image_labels = [], []
        for index, row in self.df.iterrows():
            mosaic_filenames = [row.loc[f'filename_{i}'] for i in range(4)]
            mosaic_labels = [row.loc[f'label_{i}'] for i in range(4)]

            list_image_filenames.append(mosaic_filenames)
            list_image_labels.append(mosaic_labels)

        mosaic_filenames = self.df['filename'].tolist()
        mosaic_filepaths = self.get_filepaths(mosaic_filenames)
        target_classes = self.df['target_class'].tolist()

        return mosaic_filepaths, target_classes, list_image_filenames, list_image_labels

    def get_n_outputs(self):
        _, target_classes, _, _ = self.get_subset()
        unique_labels = len(set(target_classes))
        return unique_labels

    def get_filepaths(self, image_filenames):
        return [os.path.join(self.data_folder, img_filename) for img_filename in image_filenames]

    @classmethod
    def load_dataset(cls):
        raise NotImplementedError


class Mit67Mosaic(MosaicDataset):
    @classmethod
    def load_dataset(cls):
        csv_path = MosaicPaths.Mit67Mosaic.csv_path
        data_folder = MosaicPaths.Mit67Mosaic.images_folder
        return cls(csv_path, data_folder=data_folder)


class CatsDogsMosaic(MosaicDataset):
    @classmethod
    def load_dataset(cls):
        csv_path = MosaicPaths.CatsDogsMosaic.csv_path
        data_folder = MosaicPaths.CatsDogsMosaic.images_folder
        return cls(csv_path, data_folder=data_folder)


class MameMosaic(MosaicDataset):
    @classmethod
    def load_dataset(cls):
        csv_path = MosaicPaths.MameMosaic.csv_path
        data_folder = MosaicPaths.MameMosaic.images_folder
        return cls(csv_path, data_folder=data_folder)