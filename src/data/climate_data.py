import json
from typing import Union

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset

from climate_data_utils import process_realization
from climate_norm import min_max_norm, min_max_inv_norm


class ClimateDataset(Dataset):
    def __init__(
        self, json_path: str, norm_stats_path: str, seq_len: int, variables: list[str]
    ):
        """Initialization method for the Dataset object.
        Here, we read the JSON file and store the name of the variable of interest.

        Args:
            json_path (str): The path to the JSON file.
            seq_len (int): The length of the sequence to return.
            variables (list[str]): The variables we want to use
        """
        super().__init__()

        self.seq_len = seq_len
        self.variables = variables

        # Used to only load one realization at a time
        self.chunk_idx: int
        self.current_chunk: Union[xr.Dataset, xr.DataArray, torch.Tensor]
        self.chunk_size: int

        # Open and load the JSON file
        with open(json_path) as f:
            data = json.load(f)

        # Contains info about the normalization statistics
        self.norm_stats = torch.load(norm_stats_path)

        # Get the base directory path and the realizations
        self.base_dir = data["base_dir"]
        self.realizations = list(data["realizations"].values())

        self.prepare()

    def load_realization(self, realization_idx: int):
        """Loads a realization into memory"""

        # Call the function to load the chunk into memory
        chunk = process_realization(self.base_dir, self.realizations[realization_idx])

        # Update which chunk we have loaded currently
        self.chunk_idx = realization_idx

        # Select the variables we want to use
        chunk = chunk[self.variables]

        # Normalize the data
        for var in chunk.data_vars:
            chunk[var] = min_max_norm[var](chunk[var])  # type: ignore

        return chunk

    def unnorm(self, data: torch.Tensor) -> torch.Tensor:
        """Unnormalizes the data"""

        data = data.clone()
        for idx, var in enumerate(self.variables):
            data[..., idx, :, :, :] = min_max_inv_norm[var](data[..., idx, :, :, :])

        return data

    def prepare(self):
        """Prepares the dataset by loading data and calculating norm statistics"""

        # Loads realization #0 into memory with all the metadata
        self.current_chunk = self.load_realization(0)

        # How many samples exist in a single chunk
        self.chunk_size = len(self.current_chunk.time) - self.seq_len + 1

        # Finally, turn the chunk into a tensor for processing speed
        self.current_chunk = torch.Tensor(self.current_chunk.to_array().values)  # type: ignore

    def __len__(self):
        """Returns the total number of samples in the data."""

        return self.chunk_size * len(self.realizations)

    def __getitem__(self, index):
        """Return data for a single sequence.

        Args:
            index (int): The index of the time step to retrieve.

        Returns:
            data_slice (numpy.ndarray): A numpy array of the requested data for a single day.
        """
        # Calculate the realization and time index from the global index
        realization_idx = index // (self.chunk_size)
        time_idx = index % (self.chunk_size)

        # Load the realization if it's not already loaded
        if self.chunk_idx != realization_idx:
            self.current_chunk = self.load_realization(realization_idx)

            # Turn into a tensor for speed
            self.current_chunk = torch.Tensor(self.current_chunk.to_array().values)

        # Select data by indexing into the 'time' dimension
        data = self.current_chunk[:, time_idx : time_idx + self.seq_len]

        # We return the values as a numpy array.
        # This will have the shape [realization x height x width]
        return data, time_idx % 365


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from climate_data_utils import ChunkSampler

    dataset = ClimateDataset(
        "data/IPSL/rcp85_train.json", "data/IPSL/stats.pt", 1, ["tas", "pr"]
    )

    dataloader = DataLoader(
        dataset,
        batch_size=64,
        num_workers=0,
        sampler=ChunkSampler(len(dataset), dataset.chunk_size),
    )

    for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        breakpoint()
        pass
