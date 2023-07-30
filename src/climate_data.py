import json
from typing import Hashable, Callable

import torch
import numpy as np
import xarray as xr
import pandas as pd
from torch.utils.data import Dataset

PREPROCESS_FN: dict[Hashable, Callable] = {
    "tas": lambda x: x - 273.15,
    "pr": lambda x: x * 86400,
}


def preprocess(ds: xr.Dataset):
    # Delete duplicate indices
    ds = drop_vars(ds)

    # Apply variable specific preprocessing
    for var in ds.data_vars:
        ds[var] = PREPROCESS_FN[var](ds[var])

    return ds


def drop_vars(ds: xr.Dataset):
    # Check to see if there are duplicates in dataset's time index
    if pd.Series(ds.time).duplicated().any():
        ds = ds.drop("time")

    # Drop unnecessary variables
    ds = ds.drop_vars(["time_bnds", "lon_bnds", "lat_bnds"])
    return ds


def process_realization(base_dir: str, realization):
    """Process a single realization by opening the associated datasets and merging them.

    Args:
        base_dir (str): The base directory for the paths.
        realization (dict): A dictionary of variable paths.

    Returns:
        merged_ds (xarray.Dataset): The merged Dataset from the realization.
    """
    realization_datasets = []
    for variable_paths in realization.values():
        # Construct the full paths and load them into a dataset
        full_path = [base_dir + path for path in variable_paths]

        # Open each dataset and preprocess it, then add it to the list
        realization_datasets.append(
            xr.concat(
                [preprocess(xr.open_dataset(path)) for path in full_path], dim="time"
            )
        )

    # Merge all datasets in the realization
    merged_ds = xr.merge(realization_datasets)
    return merged_ds


class ClimateDataset(Dataset):
    def __init__(self, json_path: str, seq_len: int):
        """Initialization method for the Dataset object.
        Here, we read the JSON file and store the name of the variable of interest.

        Args:
            json_path (str): The path to the JSON file.
            seq_len (int): The length of the sequence to return.
        """
        super().__init__()

        self.seq_len = seq_len

        # Used to only load one realization at a time
        self.chunk_idx: int
        self.current_chunk: xr.Dataset

        # Normalization stats
        self.mean: dict[Hashable, np.ndarray]
        self.std: dict[Hashable, np.ndarray]

        # Open and load the JSON file
        with open(json_path) as f:
            data = json.load(f)

        # Get the base directory path and the realizations
        self.base_dir = data["base_dir"]
        self.realizations = list(data["realizations"].values())

        # Get the length of the time dimension for a single realization
        # This assumes that all realizations have the same length
        sample_ds = process_realization(self.base_dir, self.realizations[0])
        self.time_len = len(sample_ds.time)

        self.prepare()

    def prepare(self):
        """Prepares the dataset by loading data and calculating norm statistics"""

        self.current_chunk = process_realization(self.base_dir, self.realizations[0])
        self.chunk_idx = 0

        # Calculate the mean and standard deviation for each variable
        ref_period = self.current_chunk.sel(time=slice("1960", "1990"))
        self.mean = {var: ref_period[var].mean().values for var in ref_period.data_vars}
        self.std = {var: ref_period[var].std().values for var in ref_period.data_vars}

    def unnorm(self, data: torch.Tensor):
        """Unnormalize the data. Performs operation in-place (modifies original data)

        Args:
            data (numpy.ndarray): The data to unnormalize.

        Returns:
            data (numpy.ndarray): The unnormalized data.
        """

        # Unormmalize each variable separately
        for i, var in enumerate(self.current_chunk.data_vars):
            # Make sure we always grab the fourth from the end dimension (the variable dimension)
            data[..., i, :, :, :] = (
                data[..., i, :, :, :] * self.std[var] + self.mean[var]
            )

        return data

    def __len__(self):
        """Returns the total number of samples in the data."""
        samples_per_realization = self.time_len - self.seq_len + 1
        return samples_per_realization * len(self.realizations)

    def __getitem__(self, index):
        """Return data for a single sequence.

        Args:
            index (int): The index of the time step to retrieve.

        Returns:
            data_slice (numpy.ndarray): A numpy array of the requested data for a single day.
        """
        # Calculate the realization and time index from the global index
        realization_idx = index // (self.time_len - self.seq_len + 1)
        time_idx = index % (self.time_len - self.seq_len + 1)

        # Load the realization if it's not already loaded
        if self.current_chunk is None or self.chunk_idx != realization_idx:
            self.current_chunk = process_realization(
                self.base_dir, self.realizations[realization_idx]
            )
            self.chunk_idx = realization_idx

        # Select data by indexing into the 'time' dimension
        data = self.current_chunk.isel(time=range(time_idx, time_idx + self.seq_len))

        # Normalize the data
        for var in data.data_vars:
            data[var] = (data[var] - self.mean[var]) / self.std[var]

        # We return the values as a numpy array.
        # This will have the shape [realization x height x width]
        return data.to_array().values


if __name__ == "__main__":
    dataset = ClimateDataset("data/IPSL/rcp85_train.json", 1)
    x = dataset[0]
    breakpoint()
