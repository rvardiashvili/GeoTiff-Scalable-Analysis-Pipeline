import numpy as np, torch, rasterio, time, gc, json, threading, queue, sys
from rasterio.enums import Resampling
from rasterio.windows import Window
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Tuple, List
from collections.abc import Callable
from .config import (
    PATCH_SIZE, REPO_ID, DEVICE, GPU_BATCH_SIZE, USE_AMP, autocast, 
    CONFIDENCE_THRESHOLD, CHUNK_SIZE, BANDS, SAVE_FULL_PROBS, SAVE_PREVIEW_IMAGE, PREVIEW_DOWNSCALE_FACTOR
)
from .utils import (
    NEW_LABELS, LABEL_COLOR_MAP, BigEarthNetv2_0_ImageClassifier,
    NORM_M, NORM_S, STANDARD_BANDS, save_color_mask_preview, run_gpu_inference
)
from .data import read_chunk_data, cut_into_patches

# ------------------------------------------------------------
# Setup
# ------------------------------------------------------------
FALLBACK_LABEL = "No_Dominant_Class"
LABEL_COLOR_MAP[FALLBACK_LABEL] = np.array([128,128,128],dtype=np.uint8)

SAVE_CONFIDENCE = True
SAVE_ENTROPY = True
SAVE_GAP = True # Assuming this was intended to be defined here
PROB_DTYPE = np.float32
PROB_COMPRESS = "lzw"

# ------------------------------------------------------------
def _find_band_path(tile_folder: Path, band_name: str) -> Path | None:
    for ext in ['.jp2','.tif']:
        candidate = next(tile_folder.rglob(f"*{band_name}*{ext}"), None)
        if candidate:
            return candidate
    return None

# ------------------------------------------------------------
def accumulate_probs(results: np.ndarray, coords: List[Tuple[int, int]], H: int, W: int, patch_size: int, n_classes: int, weight_mask: np.ndarray, weight_mask_count: np.ndarray) -> np.ndarray:
    avg = np.zeros((H, W, n_classes), dtype=np.float32)
    count = np.zeros((H, W, 1), dtype=np.float32)
    for i, (r0, c0) in enumerate(coords):
        prob = results[i]
        weighted_prob = prob[np.newaxis, np.newaxis, :] * weight_mask
        avg[r0:r0+patch_size, c0:c0+patch_size, :] += weighted_prob
        count[r0:r0+patch_size, c0:c0+patch_size, :] += weight_mask_count
    count[count == 0] = 1.0
    avg /= count
    return avg

# ------------------------------------------------------------
def main(tile_folder: str, crop_limit=None, output_directory: str | None = None, extra_data_generators: List[Callable] | None = None, model=None):
    t0 = time.time()
    tile = Path(tile_folder)
    # Ensure output_directory is used correctly
    out_base = Path(output_directory or tile.parent)
    out_path = out_base / tile.name
    out_path.mkdir(parents=True, exist_ok=True)

    target_band_path = _find_band_path(tile, 'B02')
    if not target_band_path:
        print("❌ Error: Reference band B02 not found.")
        return

    with rasterio.open(target_band_path) as src:
        H_full_original, W_full_original = src.height, src.width
        # 1. Get the base profile, including geospatial info
        base_profile = src.profile
        full_tile_shape = (H_full_original, W_full_original)

    if crop_limit:
        H_full = min(H_full_original, crop_limit[0])
        W_full = min(W_full_original, crop_limit[1])
    else:
        H_full, W_full = H_full_original, W_full_original
    
    # --- Profile Generation (Fixes RasterBlockError and StripOffsets Error) ---
    
    # Keys that often cause conflict when changing from stripped/tiled or changing driver
    # We remove parameters that define the internal TIFF structure which we are redefining below.
    keys_to_remove = ['tiled', 'blockxsize', 'blockysize', 'interleave', 'compress', 'count', 'dtype', 'nodata', 'driver']
    
    # Start with a clean profile derived from the source
    profile_base = base_profile.copy()
    for key in keys_to_remove:
        profile_base.pop(key, None)
    
    # 2. Define the Tiled GeoTIFF profile for uint8 class masks
    profile_mask = profile_base.copy()
    profile_mask.update(
        driver="GTiff",
        height=H_full,  # Ensure final height is set
        width=W_full,   # Ensure final width is set
        dtype=rasterio.uint8,
        count=1,
        nodata=255, 
        compress='lzw',
        tiled=True, 
        blockxsize=CHUNK_SIZE, # CHUNK_SIZE must be a multiple of 16!
        blockysize=CHUNK_SIZE
    )
    
    # 3. Define the Tiled GeoTIFF profile for float (max_prob, entropy, gap)
    profile_float = profile_base.copy()
    profile_float.update(
        driver="GTiff",
        height=H_full,  # Ensure final height is set
        width=W_full,   # Ensure final width is set
        dtype=rasterio.float32,
        count=1, # For single-band outputs
        compress='lzw',
        tiled=True,
        blockxsize=CHUNK_SIZE,
        blockysize=CHUNK_SIZE
    )

    out_class_path = out_path / f"{tile.name}_class.tif"
    out_conf_path  = out_path / f"{tile.name}_maxprob.tif"
    out_prob_path  = out_path / f"{tile.name}_probs.tif"
    out_entropy_path = out_path / f"{tile.name}_entropy.tif"
    out_gap_path = out_path / f"{tile.name}_gap.tif"

    # Create a detailed class map with labels, indices, and colors
    class_map_data = {
        label: {
            "index": i,
            "color_rgb": LABEL_COLOR_MAP[label].tolist()
        }
        for i, label in enumerate(NEW_LABELS)
    }
    # Also add the fallback label for completeness
    class_map_data[FALLBACK_LABEL] = {
        "index": 255, # Using 255 as it's the nodata value for the mask
        "color_rgb": LABEL_COLOR_MAP[FALLBACK_LABEL].tolist()
    }

    with open(out_path / f"{tile.name}_classmap.json", "w") as f:
        json.dump(class_map_data, f, indent=2)

    if model is None:
        model = BigEarthNetv2_0_ImageClassifier.from_pretrained(REPO_ID).to(DEVICE).eval()

    # --- Pre-calculate weighting masks ---
    n_classes = len(NEW_LABELS)
    window_2d = np.outer(np.sin(np.linspace(0, np.pi, PATCH_SIZE))**2,
                         np.sin(np.linspace(0, np.pi, PATCH_SIZE))**2).astype(np.float32)
    weight_mask = np.tile(window_2d[:,:,np.newaxis], (1, 1, n_classes))
    weight_mask_count = np.tile(window_2d[:,:,np.newaxis], (1, 1, 1))

    result_queue = queue.Queue(maxsize=2)
    stop_signal = object()
    total_chunks = ((H_full + CHUNK_SIZE - 1) // CHUNK_SIZE) * ((W_full + CHUNK_SIZE - 1) // CHUNK_SIZE)
    mosaic_pbar = tqdm(total=total_chunks, desc="Writing", position=1)
    fetch_pbar  = tqdm(total=total_chunks, desc="Inference", position=0)

    def mosaicking_worker(dst_class, dst_conf=None, dst_probs=None, dst_entropy=None, dst_gap=None, extra_data_generators=None, weight_mask=None, weight_mask_count=None):
        try:
            while True:
                item = result_queue.get()
                if item is stop_signal:
                    result_queue.task_done()
                    break
                (results, coords, H_crop, W_crop, patch_size, n_classes_runtime,
                 c_start, r_start, W_chunk, H_chunk) = item
                try:
                    # Accumulate probabilities
                    avg = accumulate_probs(results, coords, H_crop, W_crop, patch_size, n_classes_runtime, weight_mask, weight_mask_count)
                    
                    # Calculate outputs
                    dominant_idx = np.argmax(avg, axis=2).astype(np.uint8)
                    max_prob = np.max(avg, axis=2).astype(np.float32)
                    entropy = -np.sum(avg * np.log(np.clip(avg, 1e-6, 1.0)), axis=2).astype(np.float32)
                    
                    # Calculate Gap (Difference between max and second-max prob)
                    top2 = np.partition(avg, -2, axis=2)[:, :, -2:]
                    gap = (top2[:, :, 1] - top2[:, :, 0]).astype(np.float32)

                    # Write to GeoTIFFs
                    window_cropped = Window(col_off=c_start, row_off=r_start, width=W_chunk, height=H_chunk)
                    
                    dst_class.write(dominant_idx[np.newaxis,:,:], window=window_cropped)
                    if dst_conf is not None:
                        dst_conf.write(max_prob[np.newaxis,:,:], window=window_cropped)
                    if dst_entropy is not None:
                        dst_entropy.write(entropy[np.newaxis,:,:], window=window_cropped)
                    if dst_gap is not None:
                        dst_gap.write(gap[np.newaxis,:,:], window=window_cropped)
                    if dst_probs is not None:
                        # Write all probability bands (n_classes bands)
                        dst_probs.write(avg.transpose(2,0,1), window=window_cropped)

                    if extra_data_generators:
                        for generator_func in extra_data_generators:
                            extra_data = generator_func(probabilities=avg, NEW_LABELS=NEW_LABELS)
                            out_extra_path = out_path / f"{tile.name}_{generator_func.__name__}.tif"
                            with rasterio.open(out_extra_path, "w", **profile_float) as dst_extra:
                                dst_extra.write(extra_data[np.newaxis,:,:], window=window_cropped)
                        
                except Exception as e:
                    print(f"Error in worker: {e}", file=sys.stderr)
                finally:
                    del results, avg
                    gc.collect()
                    mosaic_pbar.update(1)
                    result_queue.task_done()
        except Exception as e:
            print(f"Fatal error in mosaicking_worker: {e}", file=sys.stderr)

    # Open all destination files before starting the threads
    with rasterio.open(out_class_path, "w", **profile_mask) as dst_class:
        
        # Open single-band float files
        dst_conf = rasterio.open(out_conf_path, "w", **profile_float)
        dst_entropy = rasterio.open(out_entropy_path, "w", **profile_float) if SAVE_ENTROPY else None
        dst_gap = rasterio.open(out_gap_path, "w", **profile_float) if SAVE_GAP else None

        # Open multi-band float file
        if SAVE_FULL_PROBS:
            profile_probs = profile_float.copy()
            profile_probs.update(count=len(NEW_LABELS))
            dst_probs = rasterio.open(out_prob_path, "w", **profile_probs)
        else:
            dst_probs = None

        worker_thread = threading.Thread(
            target=mosaicking_worker,
            args=(dst_class, dst_conf, dst_probs, dst_entropy, dst_gap, extra_data_generators, weight_mask, weight_mask_count),
            daemon=True
        )
        worker_thread.start()

        for r_start in range(0, H_full, CHUNK_SIZE):
            for c_start in range(0, W_full, CHUNK_SIZE):
                r_end = min(r_start + CHUNK_SIZE, H_full)
                c_end = min(c_start + CHUNK_SIZE, W_full)
                H_chunk, W_chunk = r_end - r_start, c_end - c_start
                
                img_chunk = read_chunk_data(tile, BANDS, r_start, c_start, W_chunk, H_chunk)
                
                # Check for empty/nodata chunk
                if img_chunk.size == 0 or np.mean(img_chunk < 1.0) > 0.99:
                    dummy_idx = np.full((H_chunk, W_chunk), profile_mask.get('nodata', 255), dtype=np.uint8)
                    dst_class.write(dummy_idx[np.newaxis,:,:], window=window)
                    if dst_conf: dst_conf.write(np.full((H_chunk, W_chunk), 0.0, dtype=np.float32)[np.newaxis,:,:], window=window)
                    if dst_entropy: dst_entropy.write(np.full((H_chunk, W_chunk), 0.0, dtype=np.float32)[np.newaxis,:,:], window=window)
                    if dst_gap: dst_gap.write(np.full((H_chunk, W_chunk), 0.0, dtype=np.float32)[np.newaxis,:,:], window=window)
                    if dst_probs: dst_probs.write(np.full((len(NEW_LABELS), H_chunk, W_chunk), 0.0, dtype=np.float32), window=window)
                    fetch_pbar.update(1)
                    mosaic_pbar.update(1) # Ensure we count this skipped chunk
                    continue
                
                # Inference
                patches, coords, H_crop, W_crop, _ = cut_into_patches(img_chunk, PATCH_SIZE)
                del img_chunk; gc.collect()
                results = run_gpu_inference(patches, model=model)
                n_classes = results.shape[1] if results.ndim == 2 else len(NEW_LABELS)
                del patches; gc.collect()
                
                # Push results to worker queue
                result_queue.put((results, coords, H_crop, W_crop, PATCH_SIZE, n_classes,
                                  c_start, r_start, W_chunk, H_chunk))
                fetch_pbar.update(1)

        result_queue.put(stop_signal)
        result_queue.join()
        worker_thread.join()

        # Close all files
        dst_conf.close()
        if dst_entropy: dst_entropy.close()
        if dst_gap: dst_gap.close()
        if dst_probs: dst_probs.close()
        fetch_pbar.close(); mosaic_pbar.close()

    print(f"\n✅ Finished in {time.time()-t0:.2f}s")
    print(f"Class map: {out_class_path}")
    print(f"Max prob:  {out_conf_path}")
    if SAVE_ENTROPY: print(f"Entropy:  {out_entropy_path}")
    if SAVE_GAP: print(f"Gap:       {out_gap_path}")
    if SAVE_FULL_PROBS: print(f"Full probs: {out_prob_path}")
    if SAVE_PREVIEW_IMAGE:
        # Need to re-open the file as it was closed in the worker thread
        try:
            with rasterio.open(out_class_path) as src:
                class_mask = src.read(1)
                # Calling the new, correctly imported function
                save_color_mask_preview(
                    class_mask, 
                    out_path / "preview.png", 
                    downscale_factor=PREVIEW_DOWNSCALE_FACTOR
                )
        except Exception as e:
            print(f"❌ Error during final preview generation: {e}")

    fetch_pbar.close(); mosaic_pbar.close()