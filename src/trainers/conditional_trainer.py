from tqdm import tqdm
import torch
from torchvision.utils import make_grid

from .base_trainer import BaseTrainer


class CondTrainer(BaseTrainer):
    def model_forward_pass(self, batch) -> torch.Tensor:
        """With the conditional trainer, send the classes into
        the model as well as the images"""
        img, classes = batch
        return self.model(img, classes=classes)

    def imagine(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Samples from the diffusion model and returns a grid of images"""

        # Set the model to eval mode
        self.ema.ema_model.eval()  # type: ignore

        # Sample 9 classes
        classes = torch.arange(9).to(self.accelerator.device)

        # Sample 9 images and make a grid out of them
        images = self.ema.ema_model.sample(classes=classes, cond_scale=self.cfg.cond_scale)  # type: ignore
        fake_images = make_grid(images, nrow=3)

        # Select random images from the validation set
        real_images = make_grid(
            [self.val_set[i][0] for i in torch.randint(len(self.val_set), (9,))], nrow=3  # type: ignore
        )

        return fake_images, real_images
