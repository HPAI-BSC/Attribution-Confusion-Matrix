import numpy as np
from torchvision import transforms
from PIL import Image
import torch

class Explainability:
    transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    def __init__(self, name):
        self.name = name

    def get_name(self):
        return self.name

    def filename2tensor(self, mosaic_path: str):
        img = Image.open(mosaic_path)
        img = img.convert('RGB')
        img_tensor = self.transformation(img)
        img_tensor = torch.unsqueeze(img_tensor, 0)
        return img_tensor

    def get_transformation(self):
        crop_size = (224, 224)
        resize_size = (int(crop_size[0] / 0.875), int(crop_size[1] / 0.875))
        transformations = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(crop_size),
        ])
        return transformations

    def create_mosaic(self, mosaic_image_paths):
        transform = self.get_transformation()
        images = []
        for image_path in mosaic_image_paths:
            img = Image.open(image_path)
            img = img.convert('RGB')
            img = np.asarray(transform(img))
            images.append(img)

        imgs_comb = np.vstack((np.hstack((images[0], images[1])), np.hstack((images[2], images[3]))))
        imgs_comb = Image.fromarray(imgs_comb)
        imgs_comb = imgs_comb.convert('RGB')

        img_tensor = self.transformation(imgs_comb)
        img_tensor = torch.unsqueeze(img_tensor, 0)
        return img_tensor

    def explain(self, mosaic_filepath: str, target_class: int, baseline: str = None, images_filepaths: list = []) -> np.ndarray:
        if images_filepaths:
            mosaic_tensor = self.create_mosaic(images_filepaths)
        else:
            mosaic_tensor = self.filename2tensor(mosaic_filepath)
        mosaic_explanation = self.eval_image(mosaic_tensor, target_class, baseline=baseline)
        return mosaic_explanation

    def eval_image(self, img: torch.Tensor, target_class: int, baseline: str = None) -> np.ndarray:
        raise NotImplementedError

    def visualization(self, heatmap: np.ndarray, images_filepaths: list = []) -> np.ndarray:
        mosaic_tensor = self.create_mosaic(images_filepaths)
        heatmap_image = self.heatmap_visualization(heatmap, mosaic_tensor)
        return heatmap_image, mosaic_tensor

    def heatmap_visualization(self, heatmap: np.ndarray, img: torch.Tensor) -> np.ndarray:
        raise NotImplementedError


