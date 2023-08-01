from typing import Hashable, Callable
from multiprocessing import Pool
import random

import torch
from torch.utils.data import Sampler
import xarray as xr
import pandas as pd

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


def open_file(path: str):
    """Opens a netCDF file and preprocesses it.

    Args:
        path (str): The path to the netCDF file.

    Returns:
        ds (xarray.Dataset): The opened and preprocessed Dataset.
    """
    ds = xr.open_dataset(path)
    return preprocess(ds)


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

        # Open all of the datasets in the realization (for that variable)
        realization_datasets.append(
            xr.open_mfdataset(
                full_path, combine="by_coords", preprocess=preprocess, chunks=None
            )
        )

    # Merge all datasets in the realization
    merged_ds = xr.merge(realization_datasets)

    return merged_ds


class ChunkSampler(Sampler):
    def __init__(self, num_data, chunk_size):
        """
        num_data: Total number of data points
        chunk_size: Number of data points in a chunk
        """
        self.num_data = num_data
        self.chunk_size = chunk_size
        self.num_chunks = self.num_data // self.chunk_size

    def __iter__(self):
        
        # Generate a random permutation of chunks
        random_chunks = torch.randperm(self.num_chunks)

        for chunk_idx in random_chunks:
            # Calculate start and end of the current chunk
            start_idx = chunk_idx * self.chunk_size
            end_idx = min((chunk_idx + 1) * self.chunk_size, self.num_data)

            # Generate indices within this chunk and shuffle them
            indices = torch.randperm(end_idx - start_idx) + start_idx
            for idx in indices:
                yield idx

    def __len__(self):
        return self.num_data
