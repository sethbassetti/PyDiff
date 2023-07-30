# Standard Library Imports
from typing import Union

# Third Party Imports
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from accelerate.utils import set_seed

from torch.utils.data import Dataset
from torch import nn
from torchvision.datasets import VisionDataset

# Local Imports
from diffusion.base_diffusion import BaseDiffusion
from trainers.base_trainer import BaseTrainer


@hydra.main(config_path="../config", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    """Initializes the model, dataset, and trainer and starts the training loop.

    Args:
        cfg (DictConfig): Provided by hydra. Contains all configurations
    """

    # Sets random seed
    set_seed(42)

    # Datasets have shared configs and train/val specific configs
    train_cfg = dict(cfg.dataset.shared) | dict(cfg.dataset.train)
    val_cfg = dict(cfg.dataset.shared) | dict(cfg.dataset.val)

    # Instantiates the objects specified in the configurations
    train_set: Union[Dataset, VisionDataset] = instantiate(train_cfg)
    val_set: Union[Dataset, VisionDataset] = instantiate(val_cfg)
    model: nn.Module = instantiate(cfg.model)

    # Get a sample from the dataset to determine the height and width of the image
    sample = train_set[0][0]
    height, width = sample.shape[-2:]

    # Diffuser wraps the model and handles diffusion/loss calculation
    diffuser: BaseDiffusion = instantiate(cfg.diffuser, model, height, width)
    trainer: BaseTrainer = instantiate(
        cfg.trainer, train_set, val_set, diffuser, cfg, _recursive_=False
    )

    # Starts main training loop
    trainer.train()


if __name__ == "__main__":
    main()
