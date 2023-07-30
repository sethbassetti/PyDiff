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

from ema_pytorch import EMA

# Local Imports
from diffusion.base_diffusion import BaseDiffusion


class BaseTrainer:
    def __init__(
        self,
        train_set: Union[Dataset, VisionDataset],
        val_set: Union[Dataset, VisionDataset],
        model: BaseDiffusion,
        cfg: DictConfig,
        **_
    ):
        # Accelerate handles device placement and distributed training
        self.accelerator = Accelerator(log_with="wandb")  # type: ignore

        # Turn the config into a format suitable for logging
        log_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

        # Start experiment tracking
        self.accelerator.init_trackers(cfg.project_name, log_cfg)  # type: ignore

        # We only care about the trainer config
        self.cfg = cfg.trainer
        self.train_set = train_set
        self.val_set = val_set
        self.model = model
        self.device = self.accelerator.device

        # Create ema wrapper to update
        self.ema: EMA = instantiate(self.cfg.ema, model=model)
        self.ema = self.ema.to(self.device)

        # The diffuser is the wrapper around the model with potentially other learnable parameters
        self.optimizer: Optimizer = instantiate(
            self.cfg.optimizer, params=model.parameters()
        )

        self.train_loader = DataLoader(train_set, **self.cfg.dataloader)
        self.val_loader = DataLoader(val_set, **self.cfg.dataloader)

        # Wrap everything with accelerate
        (
            self.model,
            self.optimizer,
            self.train_loader,
            self.val_loader,
        ) = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.val_loader
        )

    def train(self):
        """Main training loop"""

        training_loss = 0.0
        update_steps = 0

        for epoch in range(self.cfg.epochs):
            for batch in tqdm(
                self.train_loader, disable=not self.accelerator.is_main_process
            ):
                # Single batch of training
                training_loss += self.training_step(batch)
                update_steps += 1

                # Every N batches, run validation and log losses
                if update_steps % self.cfg.log_every == 0:
                    validation_loss = self.validation_loop()
                    fake_samples, real_samples = self.imagine()

                    self.accelerator.log(
                        {
                            "Training Loss": training_loss / self.cfg.log_every,
                            "Validation Loss": validation_loss,
                            "Epoch": epoch,
                            "Update Steps": update_steps,
                            "Validation Samples": wandb.Image(fake_samples),
                            "Real Samples": wandb.Image(real_samples),
                        }
                    )

                    # Reset metrics
                    training_loss = 0.0

    def model_forward_pass(self, batch) -> torch.Tensor:
        """Defines what a forward pass through the model looks like.
        i.e. how to deconstruct the batch and what to pass through to the
        model"""

        # Deconstruct the batch
        img, _ = batch

        # With unconditional model, only send image through
        return self.model(img)

    def training_step(self, batch) -> float:
        """Given a single batch, sends it through the diffuser
        to obtain a loss and then backpropagates the loss."""

        # Sets the model to train mode
        self.model.train()

        # Send the batch through the diffuser
        loss = self.model_forward_pass(batch)

        # Gradient Descent
        self.optimizer.zero_grad()
        self.accelerator.backward(loss)
        self.optimizer.step()

        # Update our running averages
        self.ema.update()

        # Only care about the float now
        return loss.item()

    def validation_loop(self) -> float:
        """Runs through the entire validation set and obtains the loss"""

        self.model.eval()

        validation_loss = 0.0

        for batch in tqdm(
            self.val_loader, disable=not self.accelerator.is_main_process
        ):
            loss = self.model_forward_pass(batch)
            validation_loss += loss.item()

        # Return the average validation loss
        return validation_loss / len(self.val_loader)

    def imagine(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Samples from the diffusion model and returns a grid of images"""

        # Set the model to eval mode
        self.ema.ema_model.eval()  # type: ignore

        # Sample 9 images and make a grid out of them
        images = self.ema.ema_model.sample(batch_size=9)  # type: ignore
        fake_images = make_grid(images, nrow=3)

        # Select random images from the validation set
        real_images = make_grid(
            [self.val_set[i][0] for i in torch.randint(len(self.val_set), (9,))], nrow=3  # type: ignore
        )

        return fake_images, real_images
