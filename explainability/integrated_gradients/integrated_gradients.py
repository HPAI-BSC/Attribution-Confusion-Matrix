import torch
import numpy as np
from torch.autograd import Variable
from torchvision import transforms

from explainability.explainability_class import Explainability
from explainability.integrated_gradients.integrated_gradients_original import IntegratedGradients as IntGrad

class IntegratedGradients(Explainability):
    def __init__(self, model):
        self.model = model
        self.ig = IntGrad(self.model)
        super().__init__('integratedgradients')

    def eval_image(self, img: torch.Tensor, target_class: int, baseline: str = None) -> np.ndarray:
        baseline = self.create_baseline(baseline, img)
        torch.cuda.empty_cache()
        cuda = torch.cuda.is_available()
        img = img.cuda() if cuda else img
        img = Variable(img, volatile=False, requires_grad=True)
        baseline = baseline.cuda() if cuda else img
        baseline = Variable(baseline, volatile=False, requires_grad=True)

        attributions_ig, delta = self.ig.attribute(img, target=target_class, n_steps=30, baselines=baseline,
                                                   return_convergence_delta=True, internal_batch_size=1)
        heatmap = np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1, 2, 0))
        heatmap = np.sum(heatmap, axis=2)
        return heatmap

    def heatmap_visualization(self, heatmap: np.ndarray, img: torch.Tensor) -> np.ndarray:
        cmap_name = 'seismic'
        cmap = eval(f'matplotlib.cm.{cmap_name}')
        heatmap = heatmap / np.max(np.abs(heatmap))
        heatmap = (heatmap + 1.) / 2.

        rgb = cmap(heatmap.flatten())[..., 0:3].reshape([heatmap.shape[0], heatmap.shape[1], 3])
        img = img.numpy()[0, :, :, :]
        img = np.moveaxis(img, 0, 2)
        img = np.float32(img)

        out_img = np.zeros(img.shape,dtype=img.dtype)
        alpha = 0.2
        out_img[:,:,:] = (alpha * img[:,:,:]) + ((1-alpha) * rgb[:,:,:])
        return out_img

    def create_baseline(self, baseline_type: str, img: torch.Tensor) -> torch.Tensor:
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        baseline_transformation = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        if baseline_type == 'black' or baseline_type is None:
            baseline = np.zeros_like(img.squeeze().numpy())
            baseline = np.transpose(baseline, (1,2,0))

        baseline = baseline_transformation(baseline)
        return torch.unsqueeze(baseline, 0)