# Standard Library Imports
from typing import Union

# Third Party Imports
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import VisionDataset
from torchvision.utils import make_grid

from accelerate import Accelerator
from tqdm import tqdm
import wandb

# Local Imports
from diffusion.base_diffusion import BaseDiffusion

class BaseTrainer:
    def __init__(
        self,
        train_set: Union[Dataset, VisionDataset],
        val_set: Union[Dataset, VisionDataset],
        diffuser : BaseDiffusion,
        cfg : DictConfig,
        **_
    ):
        # Accelerate handles device placement and distributed training
        self.accelerator = Accelerator(log_with="wandb")

        # Turn the config into a format suitable for logging
        log_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

        # Start experiment tracking
        self.accelerator.init_trackers(cfg.project_name, log_cfg)  # type: ignore

        # We only care about the trainer config
        self.cfg = cfg.trainer
        self.train_set = train_set
        self.val_set = val_set
        self.diffuser = diffuser

        # The diffuser is the wrapper around the model with potentially other learnable parameters
        self.optimizer : Optimizer = instantiate(self.cfg.optimizer, params=diffuser.parameters())

        self.train_loader = DataLoader(train_set, **self.cfg.dataloader)
        self.val_loader = DataLoader(val_set, **self.cfg.dataloader)

        # Wrap everything with accelerate
        (
            self.diffuser,
            self.optimizer,
            self.train_loader,
            self.val_loader,
        ) = self.accelerator.prepare(
            self.diffuser, self.optimizer, self.train_loader, self.val_loader
        )

    def train(self):
        """Main training loop"""

        training_loss = 0.0
        update_steps = 0

        for epoch in range(self.cfg.epochs):
            for batch in tqdm(self.train_loader):
                # Single batch of training
                training_loss += self.training_step(batch)
                update_steps += 1

                # Every N batches, run validation and log losses
                if update_steps % self.cfg.log_every == 0:
                    validation_loss = self.validation_loop()
                    val_samples = self.imagine()

                    self.accelerator.log(
                        {
                            "Training Loss": training_loss / self.cfg.log_every,
                            "Validation Loss": validation_loss,
                            "Epoch": epoch,
                            "Update Steps": update_steps,
                            "Validation Samples": wandb.Image(val_samples),
                        }
                    )

                    # Reset metrics
                    training_loss = 0.0

    def training_step(self, batch) -> float:
        """ Given a single batch, sends it through the diffuser
        to obtain a loss and then backpropagates the loss."""

        # Sets the model to train mode
        self.diffuser.train()

        # Send the batch through the diffuser
        img, label = batch
        loss = self.diffuser(img)

        # Gradient Descent
        self.optimizer.zero_grad()
        self.accelerator.backward(loss)
        self.optimizer.step()

        # Only care about the float now
        return loss.item()

    def validation_loop(self) -> float:
        """ Runs through the entire validation set and obtains the loss"""

        self.diffuser.eval()

        validation_loss = 0.0

        for img, label in tqdm(self.val_loader):
            loss = self.diffuser(img)
            validation_loss += loss.item()

        # Return the average validation loss
        return validation_loss / len(self.val_loader)

    def imagine(self) -> torch.Tensor:
        """ Samples from the diffusion model and returns a grid of images"""

        # Set the model to eval mode
        self.diffuser.eval()

        # Sample 9 images and make a grid out of them
        images = self.diffuser.sample(batch_size=9)
        images = make_grid(images, nrow=3)

        return images
