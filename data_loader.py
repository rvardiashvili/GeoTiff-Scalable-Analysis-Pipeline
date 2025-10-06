"""
Defines the PyTorch Dataset and data loading functions for BigEarthNet patches.
"""
from pathlib import Path
from typing import List, Tuple

import torch
import rasterio
from torch.utils.data import Dataset
import numpy as np # Needed for array_split safety

from config import PATCH_SIZE
from utils import STANDARD_BANDS, stack_and_interpolate

def load_patch_tensor(patch_path: Path) -> torch.Tensor:
    """
    Loads a single patch, stacks the bands, interpolates, and returns an UN-NORMALIZED tensor.
    Normalization is offloaded to the GPU worker for efficiency.
    """
    data = {}
    bands = STANDARD_BANDS[10]
    for b in bands:
        # Find the first TIFF file matching the band name
        f = next(patch_path.glob(f"*{b}*.tif"), None)
        if f:
            try:
                with rasterio.open(f) as src:
                    data[b] = src.read(1)
            except Exception as e:
                print(f"Error reading {f}: {e}")
                # Return a zero tensor if a file is corrupt or unreadable
                return torch.zeros(len(bands), PATCH_SIZE, PATCH_SIZE)
    
    # Stack bands and interpolate to the target patch size
    img = stack_and_interpolate(data, order=bands, img_size=PATCH_SIZE).squeeze(0)
    return img

class PatchFolderDataset(Dataset):
    """
    A PyTorch Dataset to wrap a folder of BigEarthNet patches.
    """
    def __init__(self, folder_path: Path, max_patches: int = 0):
        """
        Initializes the dataset by scanning for patch subdirectories.
        
        Args:
            folder_path (Path): The path to the scene folder containing patch subdirectories.
            max_patches (int): If > 0, limits the number of patches to process.
        """
        self.patch_paths = [p for p in folder_path.iterdir() if p.is_dir()]
        if max_patches > 0:
            self.patch_paths = self.patch_paths[:max_patches]
        print(f"Dataset initialized with {len(self.patch_paths)} patches from '{folder_path.name}'.")

    def __len__(self) -> int:
        return len(self.patch_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """
        Retrieves a single patch tensor and its name by index.
        """
        patch_path = self.patch_paths[idx]
        patch_tensor = load_patch_tensor(patch_path)
        return patch_tensor, patch_path.name

    def preload_all_patches(self) -> List[Tuple[torch.Tensor, str]]:
        """
        Executes the bulk I/O operation: reads all patch files from the SSD
        and stores the resulting tensors and names in a list in RAM.
        This is the single large read operation to minimize disk I/O bottlenecks.
        """
        preloaded_data = []
        total_patches = len(self)
        print(f"📦 Starting bulk load of {total_patches} patches from SSD to RAM...")
        
        for idx in range(total_patches):
            # Show progress in 10% increments
            if total_patches > 10 and (idx + 1) % (total_patches // 10) == 0:
                 print(f"   ... {((idx + 1) / total_patches) * 100:.0f}% complete.")

            try:
                # Direct call to __getitem__ which performs the disk read
                patch_tensor, patch_name = self.__getitem__(idx)
                preloaded_data.append((patch_tensor, patch_name))
            except Exception as e:
                # Robustness: log error and skip the patch
                print(f"Error during bulk load of patch {self.patch_paths[idx].name}: {e}")
                continue 
                
        print(f"✅ Bulk load complete. {len(preloaded_data)} patches loaded into RAM.")
        return preloaded_data
