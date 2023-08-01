from torch.utils.data import DataLoader

from .base_trainer import BaseTrainer
from data.climate_data_utils import ChunkSampler

class ClimateTrainer(BaseTrainer):
    
    def create_dataloaders(self):
        train_sampler = ChunkSampler(self.train_set, self.train_set.chunk_size) #type: ignore
        train_loader = DataLoader(self.train_set, sampler=train_sampler, **self.cfg.dataloader)

        val_sampler = ChunkSampler(self.val_set, self.val_set.chunk_size) #type: ignore
        val_loader = DataLoader(self.val_set, sampler=val_sampler, **self.cfg.dataloader)
        return train_loader, val_loader