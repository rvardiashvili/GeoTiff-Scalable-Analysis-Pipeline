"""
Defines the PyTorch Dataset and data loading functions for BigEarthNet patches,
optimized for robust patch-wise file reading (reading all bands in one operation).

This script now also includes the core functions for reading large data chunks and 
cutting them into smaller patches for inference.
"""
from pathlib import Path
from typing import List, Tuple, Dict, Any, Union, Optional
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
from rasterio.vrt import WarpedVRT
import re

# --- Utility Functions for Chunk Reading ---

def _find_band_path(tile_folder: Path, band_name: str, pattern: str) -> Path | None:
    """Finds the path for a given band in the tile folder using a glob pattern."""
    
    if not pattern:
        raise ValueError("File pattern must be provided to find bands.")

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
    H_chunk: int,
    s1_pattern: str,
    pad_if_needed: bool = False,
    target_size: Optional[Tuple[int, int]] = None,
    bands_list: Optional[List[str]] = None,
    ref_crs: Optional[Any] = None,
    ref_transform: Optional[Any] = None,
    ref_size: Optional[Tuple[int, int]] = None
) -> Tuple[np.ndarray, Any, Any]:
    """
    Reads a single chunk from S1 bands.
    
    CRITICAL: If ref_crs and ref_transform are provided (usually from S2), 
    this function explicitly WARPS (Reprojects) the S1 data to match the S2 grid 
    on-the-fly using a WarpedVRT. This ensures geolocation alignment.
    """
    band_data_list = []
    
    if bands_list is None:
        return np.array([]), None, None
        
    s1_bands = [b for b in bands_list if b in ['VV', 'VH']]
    s1_crs = None
    s1_transform = None

    if not s1_bands:
        return np.array([]), None, None

    if not s1_pattern:
         raise ValueError("s1_pattern is required for reading Sentinel-1 data.")

    # --- Detect S1 Manifests (for GCP-based alignment) ---
    # Search strategies for S1 Manifest:
    # 1. Inside tile_folder (if tile_folder is a container)
    # 2. Sibling of tile_folder (if tile_folder is the S2 dataset)
    
    possible_manifests = []
    
    # 1. Look inside current folder
    possible_manifests.extend(list(tile_folder.glob("S1*.SAFE/manifest.safe")))
    possible_manifests.extend(list(tile_folder.glob("*/S1*.SAFE/manifest.safe"))) # Nested
    
    # 2. Look in parent folder (Sibling)
    if tile_folder.parent.exists():
        possible_manifests.extend(list(tile_folder.parent.glob("S1*.SAFE/manifest.safe")))
        possible_manifests.extend(list(tile_folder.parent.glob("S1*/manifest.safe")))

    # Remove duplicates if any
    possible_manifests = list(set(possible_manifests))

    if not possible_manifests:
        # Fallback: search for band files directly if no manifest found? 
        # For now, we rely on manifest for GCPs.
        pass

    # Iterate over bands
    for i, band_name in enumerate(s1_bands):
        # Initialize Full Buffer (Padded with 0s) for this band
        # We will accumulate data from all S1 sources here (Mosaic)
        full_band_buffer = np.zeros((H_chunk, W_chunk), dtype=np.float32)
        
        # Track if we found ANY valid source for this band
        found_any_source = False

        # Iterate over all found S1 products (Mosaic Logic)
        for manifest_path in possible_manifests:
             src = None
             using_manifest = False
             
             # Try opening via manifest (Subdataset)
             sd_path = f"SENTINEL1_CALIB:UNCALIB:{manifest_path}:IW_{band_name.upper()}:AMPLITUDE"
             try:
                 src = rasterio.open(sd_path)
                 using_manifest = True
             except Exception:
                 # Fallback: try to find the TIFF file near the manifest
                 # S1 structure: .../measurement/s1a-iw-grd-vv-....tiff
                 # We search relative to the manifest's parent folder
                 s1_safe_dir = manifest_path.parent
                 band_path = _find_band_path(s1_safe_dir, band_name.lower(), s1_pattern)
                 if band_path:
                     src = rasterio.open(band_path)

             if src is None:
                 continue # Skip this source if band not found
             
             found_any_source = True

             with src:
                # Strict Requirement: We must have a reference grid (S2) and the source must have georeferencing.
                if not (ref_crs and ref_transform and ref_size):
                    raise ValueError("Cannot align S1 data: No reference Sentinel-2 grid provided.")

                # Check Source Georeferencing
                has_gcps = hasattr(src, 'gcps') and src.gcps and src.gcps[0]
                
                if not has_gcps and not src.crs:
                     print(f"WARNING: S1 Source {manifest_path.parent.name} has NO CRS/GCPs. Skipping.")
                     continue

                # Calculate Valid Intersection in Target (Reference) Coordinates
                vrt_w = ref_size[1]
                vrt_h = ref_size[0]

                valid_c_start = max(0, c_start)
                valid_r_start = max(0, r_start)
                valid_c_end = min(c_start + W_chunk, vrt_w)
                valid_r_end = min(r_start + H_chunk, vrt_h)

                valid_w = valid_c_end - valid_c_start
                valid_h = valid_r_end - valid_r_start

                if valid_w > 0 and valid_h > 0:
                    # Allocate buffer for the destination window (Amplitude)
                    dest_arr = np.zeros((valid_h, valid_w), dtype=np.float32)
                    
                    # Calculate Transform for this specific window
                    dst_window_transform = rasterio.windows.transform(
                        Window(valid_c_start, valid_r_start, valid_w, valid_h),
                        ref_transform
                    )

                    # Prepare arguments for reproject
                    reproject_kwargs = {
                        'source': rasterio.band(src, 1),
                        'destination': dest_arr,
                        'dst_transform': dst_window_transform,
                        'dst_crs': ref_crs,
                        'resampling': Resampling.bilinear,
                        'dst_nodata': 0
                    }
                    
                    if has_gcps:
                        reproject_kwargs['gcps'] = src.gcps[0]
                        reproject_kwargs['src_crs'] = src.gcps[1]
                    else:
                        reproject_kwargs['src_transform'] = src.transform
                        reproject_kwargs['src_crs'] = src.crs

                    # Run Reprojection
                    try:
                        rasterio.warp.reproject(**reproject_kwargs)
                    except Exception as e:
                        print(f"Error reprojecting S1 source {manifest_path}: {e}")
                        continue
                    
                    # Place Valid Data into a temp buffer for this source
                    temp_full_buffer = np.zeros((H_chunk, W_chunk), dtype=np.float32)
                    
                    dest_r = valid_r_start - r_start
                    dest_c = valid_c_start - c_start
                    temp_full_buffer[dest_r:dest_r+valid_h, dest_c:dest_c+valid_w] = dest_arr
                    
                    # Accumulate into main buffer (Mosaic Max)
                    full_band_buffer = np.maximum(full_band_buffer, temp_full_buffer)

        if not found_any_source:
             raise FileNotFoundError(f"Missing required S1 band file: {band_name} (searched {len(possible_manifests)} manifests)")

        # --- TUNING UPDATE ---
        # 55.0 was "Safe" (Good water, but dim land/missing urban).
        # 48.0 is "Aggressive" (Adds +7dB brightness).
        # This aligns your Land Mean (~ -13) with the Model Mean (~ -7).
        K_CALIB = 50.0

        band_data_db = (20.0 * np.log10(np.maximum(full_band_buffer, 1e-4))) - K_CALIB

        # --- CLIP UPDATE ---
        # OLD: np.clip(band_data_db, -50.0, 5.0)  <-- Kills Cities
        # NEW: np.clip(band_data_db, -50.0, 30.0) <-- Lets Cities Live
        band_data_db = np.clip(band_data_db, -50.0, 30.0)

        # Apply Reflect Padding if requested (Consistency with S2)
        vrt_w = ref_size[1]
        vrt_h = ref_size[0]
        valid_c_start = max(0, c_start)
        valid_r_start = max(0, r_start)
        valid_c_end = min(c_start + W_chunk, vrt_w)
        valid_r_end = min(r_start + H_chunk, vrt_h)
        valid_w = valid_c_end - valid_c_start
        valid_h = valid_r_end - valid_r_start
        
        dest_r = valid_r_start - r_start
        dest_c = valid_c_start - c_start
        
        pad_top = dest_r
        pad_left = dest_c
        pad_bottom = H_chunk - (dest_r + valid_h)
        pad_right = W_chunk - (dest_c + valid_w)

        if pad_top > 0 or pad_left > 0 or pad_bottom > 0 or pad_right > 0:
            # We extract the valid center and reflect-pad it
            roi = band_data_db[dest_r:dest_r+valid_h, dest_c:dest_c+valid_w]
            padded_roi = np.pad(roi, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='reflect')
            band_data_db = padded_roi
            
        band_data_list.append(band_data_db.astype(np.float32))

    if not band_data_list:
        return np.array([]), None, None

    # If we warped, return the reference CRS/Transform as the "current" ones
    ret_crs = ref_crs if ref_crs else s1_crs
    ret_transform = ref_transform if ref_transform else s1_transform

    return np.stack(band_data_list, axis=0), ret_crs, ret_transform

def _read_s2_bands_for_chunk(
    tile_folder: Union[Path, List[Path]],
    r_start: int, 
    c_start: int, 
    W_chunk: int, 
    H_chunk: int,
    s2_pattern: str,
    pad_if_needed: bool = False,
    target_size: Optional[Tuple[int, int]] = None,
    bands_list: Optional[List[str]] = None
) -> Tuple[np.ndarray, Any, Any, Tuple[int, int]]:
    """
    Reads a single chunk (sub-window) from all required bands for a tile.
    Can handle a single tile path or a list of tile paths (Multi-Temporal).
    If a list is provided, bands are stacked along the channel dimension.
    """
    # --- Multi-Temporal Logic ---
    if isinstance(tile_folder, list):
        if not tile_folder:
            raise ValueError("Empty tile_folder list provided.")
        
        # Use the first tile as the spatial reference
        ref_tile = tile_folder[0]
        
        # Read all time steps
        stacked_data = []
        ref_crs, ref_transform, ref_size = None, None, None
        
        for i, folder in enumerate(tile_folder):
            # Recursive call for single time step
            data, crs, transform, size = _read_s2_bands_for_chunk(
                folder, r_start, c_start, W_chunk, H_chunk, 
                s2_pattern, pad_if_needed, target_size, bands_list
            )
            stacked_data.append(data)
            
            if i == 0:
                ref_crs, ref_transform, ref_size = crs, transform, size
        
        # Concatenate along channel axis (0) -> (T*C, H, W)
        final_data = np.concatenate(stacked_data, axis=0)
        return final_data, ref_crs, ref_transform, ref_size

    # --- Single Time Step Logic ---
    # tile_folder is a single Path object here
    band_data_list = []
    
    if bands_list is None:
         raise ValueError("BANDS list cannot be None.")
    
    if not s2_pattern:
         raise ValueError("s2_pattern is required for reading Sentinel-2 data.")

    s2_bands = [b for b in bands_list if b not in ['VV', 'VH']]
    ref_band_name = 'B02' 

    ref_band_path = _find_band_path(tile_folder, ref_band_name, s2_pattern)
    if not ref_band_path:
        if not s2_bands:
            raise ValueError("BANDS list is empty.")
        ref_band_name = s2_bands[0]
        ref_band_path = _find_band_path(tile_folder, ref_band_name, s2_pattern)
        if not ref_band_path:
             raise FileNotFoundError(f"Missing reference band file for {ref_band_name} using pattern {s2_pattern}")

    # We keep the reference open to extract global tile info
    with rasterio.open(ref_band_path) as ref_src:
        ref_res = ref_src.res
        ref_crs = ref_src.crs
        ref_transform = ref_src.transform
        ref_height = ref_src.height
        ref_width = ref_src.width
        ref_size = (ref_height, ref_width)

    # --- Detect Processing Baseline and Offset ---
    # Sentinel-2 data after Jan 2022 (PB 04.00) has a radiometric offset of -1000 DN.
    offset = 0.0
    match = re.search(r'_N(\d{4})_', tile_folder.name)
    if match:
        proc_baseline = int(match.group(1))
        if proc_baseline >= 400:
            offset = -1000.0
    
    for band_name in s2_bands:
        band_path = _find_band_path(tile_folder, band_name, s2_pattern)
        if not band_path:
            raise FileNotFoundError(f"Missing required band file: {band_name}")

        with rasterio.open(band_path) as src:
            # Unified logic for both same-res and diff-res bands
            scale_x = src.res[0] / ref_res[0]
            scale_y = src.res[1] / ref_res[1]

            target_img_h = int(src.height * scale_y)
            target_img_w = int(src.width * scale_x)

            valid_c_start = max(0, c_start)
            valid_r_start = max(0, r_start)
            valid_c_end = min(c_start + W_chunk, target_img_w)
            valid_r_end = min(r_start + H_chunk, target_img_h)

            valid_w = valid_c_end - valid_c_start
            valid_h = valid_r_end - valid_r_start

            band_data = np.zeros((H_chunk, W_chunk), dtype=np.float32)

            if valid_w > 0 and valid_h > 0:
                src_win_c = valid_c_start / scale_x
                src_win_r = valid_r_start / scale_y
                src_win_w = valid_w / scale_x
                src_win_h = valid_h / scale_y
                
                window = Window(src_win_c, src_win_r, src_win_w, src_win_h)

                valid_data = src.read(
                    1,
                    window=window,
                    out_shape=(valid_h, valid_w),
                    resampling=Resampling.nearest
                )
                
                if offset != 0.0:
                    valid_data = valid_data.astype(np.float32)
                    valid_data = np.maximum(valid_data + offset, 0)

                # Convert to 0-1 Reflectance and ensure float32
                valid_data = valid_data.astype(np.float32) / 10000.0

                dest_r = valid_r_start - r_start
                dest_c = valid_c_start - c_start
                band_data[dest_r:dest_r+valid_h, dest_c:dest_c+valid_w] = valid_data

                if pad_if_needed:
                    pad_top = dest_r
                    pad_left = dest_c
                    pad_bottom = H_chunk - (dest_r + valid_h)
                    pad_right = W_chunk - (dest_c + valid_w)

                    if pad_top > 0 or pad_left > 0 or pad_bottom > 0 or pad_right > 0:
                        roi = band_data[dest_r:dest_r+valid_h, dest_c:dest_c+valid_w]
                        padded_roi = np.pad(roi, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='reflect')
                        band_data = padded_roi
            
            band_data_list.append(band_data.astype(np.float32))

    return np.stack(band_data_list, axis=0), ref_crs, ref_transform, ref_size

def read_chunk_data(tile_folder: Union[Path, List[Path]], bands_list: List[str], r_start: int, c_start: int, W_chunk: int, H_chunk: int, s2_pattern: str, s1_pattern: str = None, use_sentinel_1: bool = False) -> np.ndarray:
    """
    Public wrapper to read a single chunk of Sentinel-2 data.
    """
    s2_data, s2_crs, s2_transform, s2_size = _read_s2_bands_for_chunk(tile_folder, r_start, c_start, W_chunk, H_chunk, s2_pattern=s2_pattern, bands_list=bands_list)
    
    if use_sentinel_1:
        if isinstance(tile_folder, list):
             raise NotImplementedError("Multi-temporal Sentinel-1 not yet supported in this wrapper.")
             
        if not s1_pattern:
             raise ValueError("s1_pattern is required when use_sentinel_1 is True")
        
        # Pass S2 reference to S1 reader for auto-alignment
        s1_data, _, _ = _read_s1_bands_for_chunk(
            tile_folder, r_start, c_start, W_chunk, H_chunk, 
            s1_pattern=s1_pattern,
            bands_list=bands_list,
            ref_crs=s2_crs,
            ref_transform=s2_transform,
            ref_size=s2_size
        )
        return np.concatenate([s2_data, s1_data], axis=0)
    else:
        return s2_data

def cut_into_patches(img_chunk: np.ndarray, patch_size: int, stride: int = None) -> Tuple[np.ndarray, List[Tuple[int, int]], int, int, int]:
    """
    Cuts the larger image chunk into smaller, overlapping patches for inference.
    """
    if stride is None:
        stride = patch_size // 2

    C, H, W = img_chunk.shape
    patches = []
    coords = [] 
    
    for r_start in range(0, H - patch_size + 1, stride):
        for c_start in range(0, W - patch_size + 1, stride):
            patch = img_chunk[:, r_start:r_start + patch_size, c_start:c_start + patch_size]
            patches.append(patch)
            coords.append((r_start, c_start))

    if (H - patch_size) % stride != 0:
        r_start = H - patch_size
        for c_start in range(0, W - patch_size + 1, stride):
            patch = img_chunk[:, r_start:r_start + patch_size, c_start:c_start + patch_size]
            patches.append(patch)
            coords.append((r_start, c_start))

    if (W - patch_size) % stride != 0:
        c_start = W - patch_size
        for r_start in range(0, H - patch_size + 1, stride):
            patch = img_chunk[:, r_start:r_start + patch_size, c_start:c_start + patch_size]
            patches.append(patch)
            coords.append((r_start, c_start))

    if (H - patch_size) % stride != 0 and (W - patch_size) % stride != 0:
        r_start = H - patch_size
        c_start = W - patch_size
        patch = img_chunk[:, r_start:r_start + patch_size, c_start:c_start + patch_size]
        patches.append(patch)
        coords.append((r_start, c_start))

    if not patches:
        if H >= patch_size and W >= patch_size:
            patch = img_chunk[:, 0:patch_size, 0:patch_size]
            patches.append(patch)
            coords.append((0,0))
        else:
            raise ValueError(f"Chunk size ({H}x{W}) is smaller than patch size ({patch_size}).")
        
    patches_array = np.stack(patches, axis=0)
    return patches_array, coords, H, W, patch_size