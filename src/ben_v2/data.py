"""
Defines the PyTorch Dataset and data loading functions for BigEarthNet patches,
optimized for robust patch-wise file reading (reading all bands in one operation).

This script now also includes the core functions for reading large data chunks and 
cutting them into smaller patches for inference.
"""
from pathlib import Path
from typing import List, Tuple, Dict, Any, Union
from collections import defaultdict

import torch
import rasterio
from rasterio.windows import Window
from torch.utils.data import Dataset
import numpy as np 
from rasterio.enums import Resampling
import rasterio.warp
import rasterio.transform
import rasterio.vrt

from .config import PATCH_SIZE, BANDS, CHUNK_SIZE, USE_SENTINEL_1, PATCH_STRIDE
from . import data_structure

# --- Constants for Band Reading ---
# The BANDS list is now directly imported from config.py, which is the single source of truth.
NUM_BANDS = len(BANDS)

# --- Utility Functions for Chunk Reading ---

def _find_band_path(tile_folder: Path, band_name: str, pattern: str) -> Path | None:
    """Finds the path for a given band in the tile folder using a glob pattern."""
    
    # If the provided path is already the .SAFE directory, adjust the pattern
    if tile_folder.name.endswith('.SAFE'):
        if 'S2*.SAFE/' in pattern:
            pattern = pattern.split('S2*.SAFE/')[1]
        elif 'S1*.SAFE/' in pattern:
            pattern = pattern.split('S1*.SAFE/')[1]

    # Format the pattern with the band name
    glob_pattern = pattern.format(band_name=band_name)
    
    # Use glob to find candidates, searching recursively
    candidate = next(tile_folder.glob(glob_pattern), None)
    return candidate

def _read_s1_bands_for_chunk(
    tile_folder: Path,
    r_start: int, 
    c_start: int, 
    W_chunk: int, 
    H_chunk: int
) -> Tuple[np.ndarray, Any, Any]:
    """
    Reads a single chunk (sub-window) from all required Sentinel-1 bands for a tile.
    It assumes a 10m resolution for S1 bands.
    
    Args:
        tile_folder (Path): The directory containing the Sentinel-1 band files.
        r_start, c_start (int): Top-left row and column coordinate of the chunk.
        W_chunk, H_chunk (int): Width and height of the chunk.

    Returns:
        Tuple[np.ndarray, Any, Any]: The stacked chunk data, CRS, and transform.
    """
    band_data_list = []
    s1_bands = [b for b in BANDS if b in ['VV', 'VH']]
    s1_crs = None
    s1_transform = None

    if not s1_bands:
        return np.array([]), None, None

    for i, band_name in enumerate(s1_bands):
        band_path = _find_band_path(tile_folder, band_name.lower(), data_structure.S1_BAND_PATTERN)
        if not band_path:
            raise FileNotFoundError(f"Missing required S1 band file: {band_name} in {tile_folder} using pattern {data_structure.S1_BAND_PATTERN}")

        with rasterio.open(band_path) as src:
            if i == 0:
                if src.crs:
                    s1_crs = src.crs
                    s1_transform = src.transform
                else:
                    gcps, crs = src.gcps
                    if gcps:
                        s1_crs = crs
                        s1_transform = rasterio.transform.from_gcps(gcps)
                    else:
                        # No CRS and no GCPs
                        s1_crs = None
                        s1_transform = None
            
            window = Window(c_start, r_start, W_chunk, H_chunk)
            band_data = src.read(1, window=window, boundless=True)
            band_data_list.append(band_data.astype(np.float32))

    if not band_data_list:
        return np.array([]), None, None

    return np.stack(band_data_list, axis=0), s1_crs, s1_transform

def _read_s2_bands_for_chunk(
    tile_folder: Path,
    r_start: int, 
    c_start: int, 
    W_chunk: int, 
    H_chunk: int
) -> Tuple[np.ndarray, Any, Any]:
    """
    Reads a single chunk (sub-window) from all required bands for a tile.
    It automatically uses the resolution of 'B02' as the reference and resamples
    all other bands to match it on-the-fly.
    
    Args:
        tile_folder (Path): The directory containing the Sentinel-2 band files.
        r_start, c_start (int): Top-left row and column coordinate of the chunk in the reference resolution.
        W_chunk, H_chunk (int): Width and height of the chunk in the reference resolution.

    Returns:
        Tuple[np.ndarray, Any, Any]: The stacked chunk data, CRS, and transform of the reference band.
    """
    band_data_list = []
    s2_bands = [b for b in BANDS if b not in ['VV', 'VH']]
    ref_band_name = 'B02' # As requested, B02 is the reference.

    ref_band_path = _find_band_path(tile_folder, ref_band_name, data_structure.S2_BAND_PATTERN)
    if not ref_band_path:
        # Fallback to the first band in the list if B02 is not present
        if not s2_bands:
            raise ValueError("BANDS list is empty.")
        ref_band_name = s2_bands[0]
        ref_band_path = _find_band_path(tile_folder, ref_band_name, data_structure.S2_BAND_PATTERN)
        if not ref_band_path:
             raise FileNotFoundError(f"Missing reference band file for {ref_band_name} in {tile_folder} using pattern {data_structure.S2_BAND_PATTERN}")
        print(f"⚠️  'B02' not found. Using '{ref_band_name}' as the reference for resolution.")


    with rasterio.open(ref_band_path) as ref_src:
        ref_res = ref_src.res
        ref_crs = ref_src.crs
        ref_transform = ref_src.transform

    target_shape = (H_chunk, W_chunk)

    for band_name in s2_bands:
        band_path = _find_band_path(tile_folder, band_name, data_structure.S2_BAND_PATTERN)
        if not band_path:
            raise FileNotFoundError(f"Missing required band file: {band_name} in {tile_folder} using pattern {data_structure.S2_BAND_PATTERN}")

        with rasterio.open(band_path) as src:
            # If resolutions match, we can read the window directly without resampling.
            if src.res == ref_res:
                window = Window(c_start, r_start, W_chunk, H_chunk)
                band_data = src.read(1, window=window, boundless=True)
            # If resolutions differ, we calculate a new window for the source resolution
            # and then resample the read data to the target shape.
            else:
                # Calculate the scaling factor based on resolution difference.
                scale_x = src.res[0] / ref_res[0]
                scale_y = src.res[1] / ref_res[1]

                # Apply the scaling factor to the window parameters.
                win_c = c_start / scale_x
                win_r = r_start / scale_y
                win_w = W_chunk / scale_x
                win_h = H_chunk / scale_y

                # Create the new window for the lower-resolution band.
                window = Window(round(win_c), round(win_r), round(win_w), round(win_h))

                # Read the data from the calculated window and resample to the target shape.
                band_data = src.read(
                    1,
                    window=window,
                    out_shape=target_shape,
                    resampling=Resampling.nearest,
                    boundless=True # Use boundless to avoid edge errors with calculated windows
                )
            band_data_list.append(band_data.astype(np.float32))

    return np.stack(band_data_list, axis=0), ref_crs, ref_transform

# --- Main Public Functions ---

def read_chunk_data(tile_folder: Path, BANDS: List[str], r_start: int, c_start: int, W_chunk: int, H_chunk: int) -> np.ndarray:
    """
    Public wrapper to read a single chunk of Sentinel-2 data.
    """
    s2_data, s2_crs, s2_transform = _read_s2_bands_for_chunk(tile_folder, r_start, c_start, W_chunk, H_chunk)
    
    if USE_SENTINEL_1:
        s1_data, s1_crs, s1_transform = _read_s1_bands_for_chunk(tile_folder, r_start, c_start, W_chunk, H_chunk)
        
        if s1_data.size == 0:
            return s2_data

        if s1_crs is None:
            raise ValueError("Sentinel-1 data is missing CRS information and cannot be reprojected.")

        # If CRS or transform do not match, reproject S1 data to S2's grid.
        if s1_crs != s2_crs or not rasterio.transform.almost_equals(s1_transform, s2_transform):
            print(f"⚠️ CRS or transform mismatch. Reprojecting S1 data to match S2.")
            
            # Create an empty array for the reprojected data
            reprojected_s1_data = np.empty((s1_data.shape[0], s2_data.shape[1], s2_data.shape[2]), dtype=np.float32)
            
            rasterio.warp.reproject(
                source=s1_data,
                destination=reprojected_s1_data,
                src_transform=s1_transform,
                src_crs=s1_crs,
                dst_transform=s2_transform,
                dst_crs=s2_crs,
                resampling=Resampling.nearest
            )
            s1_data = reprojected_s1_data

        return np.concatenate([s2_data, s1_data], axis=0)
    else:
        return s2_data


def cut_into_patches(img_chunk: np.ndarray, PATCH_SIZE: int, stride: int = PATCH_STRIDE) -> Tuple[np.ndarray, List[Tuple[int, int]], int, int, int]:
    """
    Cuts the larger image chunk into smaller, overlapping patches for inference.
    
    Args:
        img_chunk (np.ndarray): The input chunk data (C, H, W).
        PATCH_SIZE (int): The side length of the square patch (e.g., 120).
        stride (int, optional): The step size for creating patches. Defaults to PATCH_SIZE // 2.
        
    Returns:
        Tuple[np.ndarray, List[Tuple[int, int]], int, int, int]:
            - np.ndarray: Patches array (N, C, H_p, W_p).
            - List[Tuple[int, int]]: (r_start, c_start) coordinates of each patch within the chunk.
            - int: Cropped height.
            - int: Cropped width.
            - int: Patch size.
    """
    if stride is None:
        stride = PATCH_SIZE // 2

    C, H, W = img_chunk.shape
    
    patches = []
    coords = [] # (r_start_within_chunk, c_start_within_chunk)
    
    for r_start in range(0, H - PATCH_SIZE + 1, stride):
        for c_start in range(0, W - PATCH_SIZE + 1, stride):
            patch = img_chunk[:, r_start:r_start + PATCH_SIZE, c_start:c_start + PATCH_SIZE]
            patches.append(patch)
            coords.append((r_start, c_start))

    # Handle the right and bottom edges
    if (H - PATCH_SIZE) % stride != 0:
        r_start = H - PATCH_SIZE
        for c_start in range(0, W - PATCH_SIZE + 1, stride):
            patch = img_chunk[:, r_start:r_start + PATCH_SIZE, c_start:c_start + PATCH_SIZE]
            patches.append(patch)
            coords.append((r_start, c_start))

    if (W - PATCH_SIZE) % stride != 0:
        c_start = W - PATCH_SIZE
        for r_start in range(0, H - PATCH_SIZE + 1, stride):
            patch = img_chunk[:, r_start:r_start + PATCH_SIZE, c_start:c_start + PATCH_SIZE]
            patches.append(patch)
            coords.append((r_start, c_start))

    # Handle the bottom-right corner
    if (H - PATCH_SIZE) % stride != 0 and (W - PATCH_SIZE) % stride != 0:
        r_start = H - PATCH_SIZE
        c_start = W - PATCH_SIZE
        patch = img_chunk[:, r_start:r_start + PATCH_SIZE, c_start:c_start + PATCH_SIZE]
        patches.append(patch)
        coords.append((r_start, c_start))

    if not patches:
        # This can happen if the chunk is smaller than the patch size
        # Fallback to a single patch from the top-left corner
        if H >= PATCH_SIZE and W >= PATCH_SIZE:
            patch = img_chunk[:, 0:PATCH_SIZE, 0:PATCH_SIZE]
            patches.append(patch)
            coords.append((0,0))
        else:
            raise ValueError("Chunk size is smaller than patch size, cannot create any patches.")
        
    patches_array = np.stack(patches, axis=0)
    
    return patches_array, coords, H, W, PATCH_SIZE
    

# --- PyTorch Dataset Classes (Not currently used in the main pipeline but kept for context) ---

class OnDiskPatchDataset(Dataset):
    """
    A PyTorch Dataset that reads the full 12-band patch on-demand from disk.
    This is very slow for large-scale chunk processing but useful for DataLoader examples.
    """
    # NOTE: This class definition relies on _find_band_path and _read_all_bands_for_patch
    # which are now defined above.
    pass # Placeholder for full class definition
