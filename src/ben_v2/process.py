import numpy as np
import torch
import torch.multiprocessing as mp
import rasterio
import time
import gc
import json
import sys
import logging
import queue
from pathlib import Path
from rasterio.windows import Window
from rasterio.enums import Resampling
from typing import Tuple, List, Optional, Dict, Any
from omegaconf import DictConfig, OmegaConf
import hydra
from tqdm import tqdm

# Import new fusion module
from .fusion import MultiModalInput, DeepLearningRegistrationPipeline
# Reuse existing utilities where possible
from .utils import (
    NEW_LABELS, LABEL_COLOR_MAP, save_color_mask_preview, run_gpu_inference, get_device
)
from .data import _read_s2_bands_for_chunk, _read_s1_bands_for_chunk, _find_band_path
from .generate_viewer import generate_single_node_viewer
from .memory_utils import resolve_zor

log = logging.getLogger(__name__)


class ERFAwareInference:
    """
    Handles the Tiling-Error-Free inference using the Overlap-Tile strategy.
    """
    def __init__(self, model, zor_size: int, halo_size: int, norm_m: torch.Tensor, norm_s: torch.Tensor, device=None):
        self.model = model
        self.zor = zor_size
        self.halo = halo_size
        self.input_size = self.zor + (2 * self.halo)
        self.device = device or get_device()
        self.fusion_pipeline = DeepLearningRegistrationPipeline() # Placeholder
        
        # Ensure norms are on the correct device
        self.norm_m = norm_m.to(self.device)
        self.norm_s = norm_s.to(self.device)

    def infer_tile_region(self, tile_folder: Path, r_start: int, c_start: int, cfg: DictConfig) -> Tuple[np.ndarray, List[Tuple[int,int]], int, int]:
        """
        Reads padded region, runs inference on patches.
        Returns:
            results: (N_patches, N_classes) - raw probabilities from model
            coords: List of (r, c) patch offsets
            H_crop: Height of the chunk to be reconstructed
            W_crop: Width of the chunk to be reconstructed
        """
        from .data import cut_into_patches
        
        # 1. Define Read Window (ZoR + Halo)
        r_read = r_start - self.halo
        c_read = c_start - self.halo
        w_read = self.input_size
        h_read = self.input_size
        
        # Determine bands to read
        if 'bands' in cfg.model:
             bands = list(cfg.model.bands)
        else:
             bands = list(cfg.data_source.bands)

        use_s1 = any(b in ['VV', 'VH'] for b in bands)

        # S2 Data
        s2_bands = [b for b in bands if 'B' in b]
        s2_data, s2_crs, s2_transform, s2_size = _read_s2_bands_for_chunk(tile_folder, r_read, c_read, w_read, h_read, pad_if_needed=True, bands_list=s2_bands)
        
        log.info(f"DEBUG: S2 Data Stats - Shape: {s2_data.shape}, Min: {s2_data.min():.4f}, Max: {s2_data.max():.4f}, Mean: {s2_data.mean():.4f}")

        # S1 Data
        if use_s1:
            s1_bands = [b for b in bands if b in ['VV', 'VH']]
            s1_data, _, _ = _read_s1_bands_for_chunk(
                tile_folder, r_read, c_read, w_read, h_read, 
                pad_if_needed=True, 
                bands_list=s1_bands,
                ref_crs=s2_crs,
                ref_transform=s2_transform,
                ref_size=s2_size
            )
            
            if s1_data.size > 0:
                log.info(f"DEBUG: S1 Data Stats - Shape: {s1_data.shape}, Min: {s1_data.min():.4f}, Max: {s1_data.max():.4f}, Mean: {s1_data.mean():.4f}")
                
                # --- CLIP UPDATE ---
                # Allow values up to 30.0 dB to capture double-bounce (Urban)
                s1_data = np.clip(s1_data, -50.0, 30.0) 
            
            # CRITICAL FIX: Concatenate S1 FIRST, then S2.
            # Matches config: ["VV", "VH", "B02", ...]
            input_data = np.concatenate([s1_data, s2_data], axis=0)
        else:
            input_data = s2_data

        # 2. Cut into patches
        patch_size = cfg.pipeline.tiling.get('patch_size', 120)
        stride = cfg.pipeline.tiling.get('patch_stride', patch_size // 2)
        
        patches, coords, H_crop, W_crop, _ = cut_into_patches(input_data, patch_size, stride=stride)
        
        # 3. Run batch inference (on GPU)
        batch_size = cfg.pipeline.distributed.get('gpu_batch_size', 32)
        use_amp = True # Default to True if possible
        
        results = run_gpu_inference(
            patches, 
            self.model, 
            norm_m=self.norm_m, 
            norm_s=self.norm_s,
            device=self.device,
            batch_size=batch_size,
            use_amp=use_amp
        )
        
        return results, coords, H_crop, W_crop


def writer_process(q: mp.Queue, out_paths: Dict[str, Path], profile_dict: Dict[str, Any], zor: int, halo: int, W_full: int, H_full: int, total_chunks: int, patch_size: int):
    """
    Independent process to reconstruct map, calculate metrics, and write results.
    Receives raw patch probabilities.
    """
    # Pre-calculate sinusoidal window for reconstruction
    window_1d = np.sin(np.linspace(0, np.pi, patch_size))**2
    patch_weight = np.outer(window_1d, window_1d).astype(np.float32)
    patch_weight = patch_weight[np.newaxis, :, :] # (1, P, P)
    
    # Open files
    dsts = {}
    pbar = tqdm(total=total_chunks, desc="Writing  ", position=1, leave=True)
    
    try:
        for key, path in out_paths.items():
            if path:
                p = profile_dict.copy()
                if key == 'class':
                    p.update(dtype='uint8', nodata=255)
                else:
                    p.update(dtype='float32', nodata=None)
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                dsts[key] = rasterio.open(path, 'w', **p)

        while True:
            item = q.get()
            if item is None:
                break
            
            r_chunk_start, c_chunk_start, results_t, coords_t, H_crop, W_crop = item
            
            results = results_t.numpy()
            coords = coords_t 
            
            # --- Reconstruction (CPU Heavy) ---
            n_classes = results.shape[1]
            avg_probs = np.zeros((n_classes, H_crop, W_crop), dtype=np.float32)
            weight_sum = np.zeros((1, H_crop, W_crop), dtype=np.float32)
            
            idx = 0
            for r_p, c_p in coords:
                patch_prob = results[idx][:, np.newaxis, np.newaxis] * patch_weight
                avg_probs[:, r_p:r_p+patch_size, c_p:c_p+patch_size] += patch_prob
                weight_sum[:, r_p:r_p+patch_size, c_p:c_p+patch_size] += patch_weight
                idx += 1
                
            weight_sum[weight_sum == 0] = 1.0
            probs_map = avg_probs / weight_sum

            # --- Crop Center (ZoR) ---
            if probs_map.ndim == 3:
                h, w = probs_map.shape[1], probs_map.shape[2]
                start_y, start_x = halo, halo
                end_y, end_x = h - halo, w - halo
                valid_probs = probs_map[:, start_y:end_y, start_x:end_x]
            else:
                 pass

            # --- Metrics Calculation (CPU Heavy) ---
            dom = np.argmax(valid_probs, axis=0).astype(np.uint8)
            
            if 'conf' in dsts:
                conf = np.max(valid_probs, axis=0).astype(np.float32)
            else:
                conf = None
                
            if 'entr' in dsts:
                entr = -np.sum(valid_probs * np.log(np.clip(valid_probs, 1e-6, 1.0)), axis=0).astype(np.float32)
            else:
                entr = None
                
            if 'gap' in dsts:
                top2 = np.partition(valid_probs, -2, axis=0)[-2:]
                gap = (top2[1] - top2[0]).astype(np.float32)
            else:
                gap = None
            
            # --- Write to Disk ---
            w_width = min(zor, W_full - c_chunk_start)
            w_height = min(zor, H_full - r_chunk_start)
            window = Window(c_chunk_start, r_chunk_start, w_width, w_height)
            
            if 'class' in dsts: dsts['class'].write(dom[:w_height, :w_width], window=window, indexes=1)
            if 'conf' in dsts: dsts['conf'].write(conf[:w_height, :w_width], window=window, indexes=1)
            if 'entr' in dsts and entr is not None: dsts['entr'].write(entr[:w_height, :w_width], window=window, indexes=1)
            if 'gap' in dsts and gap is not None: dsts['gap'].write(gap[:w_height, :w_width], window=window, indexes=1)
            
            pbar.update(1)

    except Exception as e:
        print(f"CRITICAL ERROR IN WRITER PROCESS: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
    finally:
        pbar.close()
        for dst in dsts.values():
            dst.close()


def main_hydra(cfg: DictConfig, model=None):
    try:
        ctx = mp.get_context('spawn')
    except ValueError:
        ctx = mp.get_context('fork')

    t0 = time.time()
    tile_path = Path(cfg.input_path)
    output_path = Path(cfg.output_path) / tile_path.name
    output_path.mkdir(parents=True, exist_ok=True)
    
    log.info(f"Processing tile: {tile_path}")
    log.info(f"Output directory: {output_path}")
    
    # Setup Model
    if model is None:
        model_path_parts = cfg.model.pretrained_model_name_or_path.split('/')
        model_display_name = model_path_parts[-1] if len(model_path_parts) > 1 else cfg.model.pretrained_model_name_or_path
        log.info(f"Initializing model: {model_display_name}")
        model = hydra.utils.instantiate(cfg.model)
        device = get_device()
        model.to(device).eval()
    else:
        log.info("Using pre-loaded model.")
        device = get_device() # Ensure device is set correctly even if model pre-loaded (model.device not reliable if on cpu)
        # Ideally check model.device, but get_device() is safe for new tensors.
    
    # Get Tile Dimensions
    ref_path = _find_band_path(tile_path, 'B02')
    with rasterio.open(ref_path) as src:
        H_full, W_full = src.shape
        profile = src.profile.copy()

    # Determine Means and Stds for Normalization from the config file
    if 'means' in cfg.model and 'stds' in cfg.model:
        means = cfg.model.means
        stds = cfg.model.stds
    else:
        raise ValueError("Model config must contain 'means' and 'stds' for normalization.")
        
    log.info(f"Using fixed normalization MEANS: {means}")
    log.info(f"Using fixed normalization STDS: {stds}")

    norm_m = torch.tensor(means, dtype=torch.float32).view(1, len(means), 1, 1)
    norm_s = torch.tensor(stds, dtype=torch.float32).view(1, len(stds), 1, 1)
    
    # Setup Inference Engine
    zor_config = cfg.pipeline.tiling.zone_of_responsibility_size
    halo = cfg.pipeline.tiling.halo_size_pixels
    patch_size = cfg.pipeline.tiling.get('patch_size', 120)
    
    zor = resolve_zor(zor_config, halo, patch_size=patch_size)
    log.info(f"Inference Configuration: ZoR={zor}, Halo={halo}, ChunkSize={zor + 2*halo}")
    
    erf_engine = ERFAwareInference(model, zor, halo, norm_m, norm_s, device)

    # Prepare Output Profiles
    profile.update(
        driver='GTiff',
        count=1,
        compress='lzw',
        tiled=True,
        blockxsize=256, 
        blockysize=256
    )
    
    # Prepare Paths for Writer
    out_paths = {
        'class': output_path / f"{tile_path.name}_class.tif",
        'conf': output_path / f"{tile_path.name}_maxprob.tif",
    }
    if cfg.pipeline.output.save_entropy:
        out_paths['entr'] = output_path / f"{tile_path.name}_entropy.tif"
    if cfg.pipeline.output.save_gap:
        out_paths['gap'] = output_path / f"{tile_path.name}_gap.tif"

    # Calculate Total Chunks
    coords_list = []
    for r in range(0, H_full, zor):
        for c in range(0, W_full, zor):
            coords_list.append((r,c))
    total_chunks = len(coords_list)

    # --- Initialize Writer Process ---
    write_queue = ctx.Queue(maxsize=4) 
    
    writer_p = ctx.Process(
        target=writer_process,
        args=(write_queue, out_paths, profile, zor, halo, W_full, H_full, total_chunks, patch_size),
        daemon=True
    )
    writer_p.start()

    # Distributed Execution Branch
    if cfg.pipeline.distributed.engine == 'ray':
        log.error("Ray distributed mode not yet updated for decoupled writer.")
        pass
    
    log.info("Starting Inference Loop (ERF-Aware)")
    
    inference_pbar = tqdm(total=total_chunks, desc="Inference", position=0, leave=True)
    
    try:
        for r, c in coords_list:
            # 1. Read -> Cut -> Infer (GPU)
            results, coords, H_crop, W_crop = erf_engine.infer_tile_region(tile_path, r, c, cfg)
            
            # 2. Pass to Writer (CPU)
            results_t = torch.from_numpy(results).share_memory_()
            
            write_queue.put((r, c, results_t, coords, H_crop, W_crop))
            
            inference_pbar.update(1)

    finally:
        inference_pbar.close()
        write_queue.put(None)
        writer_p.join()

    # Metadata
    class_map = {label: {"index": i, "color": c.tolist()} for i, (label, c) in enumerate(zip(NEW_LABELS, LABEL_COLOR_MAP.values()))}
    with open(output_path / f"{tile_path.name}_classmap.json", "w") as f:
        json.dump(class_map, f)

    save_preview = cfg.pipeline.output.get('save_preview', True)
    if save_preview:
        downscale = cfg.pipeline.output.get("preview_downscale_factor", 10)
        class_path = out_paths['class']
        with rasterio.open(class_path) as src:
            save_color_mask_preview(src.read(1), output_path / "preview.png", save_preview=save_preview, downscale_factor=downscale)

    try:
        generate_single_node_viewer(tile_path.name, str(output_path.parent))
    except Exception as e:
        log.error(f"Failed to generate viewer: {e}")

    log.info(f"Finished in {time.time() - t0:.2f}s")
