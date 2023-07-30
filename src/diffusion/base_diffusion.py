from abc import ABC, abstractmethod

import torch
from torch import nn

class BaseDiffusion(nn.Module, ABC):

    @abstractmethod
    def __init__(self):
        super().__init__()
        pass
    
    @abstractmethod
    def forward(self, x : torch.Tensor, *args, **kwargs) -> torch.Tensor:
        pass

    @abstractmethod
    def sample(self, batch_size: int, return_all_timesteps : bool=False) -> torch.Tensor:
        pass