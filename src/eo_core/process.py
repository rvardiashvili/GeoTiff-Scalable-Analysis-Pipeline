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

# Import new inference engine
from .inference_engine import InferenceEngine
# Reuse existing utilities where possible
from .utils import (
    NEW_LABELS, LABEL_COLOR_MAP, generate_low_res_preview, get_device
)
from .data import _find_band_path
from .generate_viewer import generate_single_node_viewer
from .memory_utils import resolve_zor

log = logging.getLogger(__name__)

def writer_process(q: mp.Queue, out_paths: Dict[str, Path], profile_dict: Dict[str, Any], zor: int, halo: int, W_full: int, H_full: int, total_chunks: int, patch_size: int, adapter: Any):
    """
    Independent process to reconstruct map, calculate metrics, and write results.
    Receives raw patch probabilities.
    """
    t_process_start = time.perf_counter()
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

            # Unpack raw model output and metadata
            logits_t, metadata = item

            # --- Postprocessing (CPU) ---
            # Offloaded from Main Process to distribute load
            # adapter.postprocess expects (tensor, metadata_dict)
            # And returns dict with 'probs_tensor'
            t_postprocess_start = time.perf_counter()
            post_result = adapter.postprocess((logits_t, metadata))
            log.debug(f"[{time.perf_counter()-t_postprocess_start:.4f}s] Writer: adapter.postprocess")

            results = post_result['probs_tensor'].numpy()
            coords = post_result['coords']
            H_crop = post_result['H_crop']
            W_crop = post_result['W_crop']
            r_chunk_start = post_result['r_chunk']
            c_chunk_start = post_result['c_chunk']

            # Determine N_classes from results shape
            # Classification: (N, C)
            # Segmentation: (N, C, H, W)
            if results.ndim == 2:
                n_classes = results.shape[1]
            elif results.ndim == 4:
                n_classes = results.shape[1]
            else:
                # Fallback or error
                n_classes = 1

            # --- Reconstruction (CPU Heavy) ---
            t_reconstruction_start = time.perf_counter()
            avg_probs = np.zeros((n_classes, H_crop, W_crop), dtype=np.float32)
            weight_sum = np.zeros((1, H_crop, W_crop), dtype=np.float32)

            idx = 0
            for r_p, c_p in coords:
                # Handle diff shapes
                if results.ndim == 2:
                    # (C,) -> (C, 1, 1)
                    patch_data = results[idx][:, np.newaxis, np.newaxis]
                elif results.ndim == 4:
                    # (C, H, W)
                    patch_data = results[idx]
                else:
                     continue # Skip weird shapes

                # Multiply by weight (broadcasts automatically)
                patch_prob = patch_data * patch_weight

                avg_probs[:, r_p:r_p+patch_size, c_p:c_p+patch_size] += patch_prob
                weight_sum[:, r_p:r_p+patch_size, c_p:c_p+patch_size] += patch_weight
                idx += 1

            weight_sum[weight_sum == 0] = 1.0
            probs_map = avg_probs / weight_sum
            log.debug(f"[{time.perf_counter()-t_reconstruction_start:.4f}s] Writer: Reconstruction")

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

            if 'gradient' in dsts:
                # For binary, save Class 1 probability
                if n_classes == 2:
                    gradient = valid_probs[1].astype(np.float32)
                else:
                    # Fallback/Undefined for multi-class (could be maxprob?)
                    gradient = None
            else:
                gradient = None

            # --- Write to Disk ---
            # If chunk start is negative (halo), the valid data (ZoR) starts at c_chunk_start + halo
            valid_c = c_chunk_start + halo
            valid_r = r_chunk_start + halo

            w_width = min(zor, W_full - valid_c)
            w_height = min(zor, H_full - valid_r)

            # Ensure window is not out of bounds (e.g. last partial chunk)
            if w_width <= 0 or w_height <= 0:
                continue

            window = Window(valid_c, valid_r, w_width, w_height)

            if 'class' in dsts: dsts['class'].write(dom[:w_height, :w_width], window=window, indexes=1)
            if 'conf' in dsts: dsts['conf'].write(conf[:w_height, :w_width], window=window, indexes=1)
            if 'entr' in dsts and entr is not None: dsts['entr'].write(entr[:w_height, :w_width], window=window, indexes=1)
            if 'gap' in dsts and gap is not None: dsts['gap'].write(gap[:w_height, :w_width], window=window, indexes=1)
            if 'gradient' in dsts and gradient is not None: dsts['gradient'].write(gradient[:w_height, :w_width], window=window, indexes=1)

            pbar.update(1)

    except Exception as e:
        print(f"CRITICAL ERROR IN WRITER PROCESS: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
    finally:
        pbar.close()
        for dst in dsts.values():
            dst.close()
    log.info(f"Writer process finished in {time.perf_counter() - t_process_start:.2f}s")


def main_hydra(cfg: DictConfig):
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

    # Get Tile Dimensions
    s2_pattern = cfg.data_source.get('s2_file_pattern', "S2*.SAFE/**/*{band_name}*.jp2")
    ref_path = _find_band_path(tile_path, 'B02', s2_pattern)
    with rasterio.open(ref_path) as src:
        H_full, W_full = src.shape
        profile = src.profile.copy()

    # Determine Means and Stds for Normalization if available (legacy/fallback support)
    means = cfg.model.get('means')
    stds = cfg.model.get('stds')

    # --- 1. Instantiate Adapter First (Dependency Injection) ---
    # This allows us to query the adapter for its requirements (bands, classes, patch_size)
    # BEFORE we calculate memory usage and tiling.

    from importlib import import_module

    if 'adapter' in cfg.model:
        log.info("Using adapter configuration from model config.")
        adapter_cfg = OmegaConf.to_container(cfg.model.adapter, resolve=True)

        # Inject pipeline defaults if missing in adapter params
        if 'params' not in adapter_cfg:
            adapter_cfg['params'] = {}

        # Inject file patterns from data_source
        if 's2_file_pattern' in cfg.data_source:
            adapter_cfg['params']['s2_file_pattern'] = cfg.data_source.s2_file_pattern
        if 's1_file_pattern' in cfg.data_source:
            adapter_cfg['params']['s1_file_pattern'] = cfg.data_source.s1_file_pattern

        # Use pipeline defaults if not set in adapter
        # Note: The adapter might have its own hard defaults if these are None
        if 'patch_size' not in adapter_cfg['params']:
             adapter_cfg['params']['patch_size'] = cfg.pipeline.tiling.get('patch_size', 120)

        if 'stride' not in adapter_cfg['params']:
             adapter_cfg['params']['stride'] = cfg.pipeline.tiling.get('patch_stride', 60)

    else:
        # Legacy / Default behavior for BigEarthNet
        if means is None or stds is None:
            raise ValueError("Model config must contain 'means' and 'stds' for normalization if no explicit adapter is defined.")

        log.info(f"Using fixed normalization MEANS: {means}")
        log.info(f"Using fixed normalization STDS: {stds}")

        adapter_params = {
            'model_config': OmegaConf.to_container(cfg.model, resolve=True),
            'means': means,
            'stds': stds,
            'patch_size': cfg.pipeline.tiling.get('patch_size', 120),
            'stride': cfg.pipeline.tiling.get('patch_stride', 60),
            'gpu_batch_size': cfg.pipeline.distributed.get('gpu_batch_size', 32),
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            's2_file_pattern': cfg.data_source.get('s2_file_pattern'),
            's1_file_pattern': cfg.data_source.get('s1_file_pattern')
        }

        adapter_cfg = {
            'path': 'eo_core.adapters.bigearthnet_adapter',
            'class': 'BigEarthNetAdapter',
            'params': adapter_params
        }

    # Instantiate the Adapter Class
    try:
        module_path = adapter_cfg['path']
        class_name = adapter_cfg['class']
        module = import_module(module_path)
        adapter_class = getattr(module, class_name)
        adapter = adapter_class(adapter_cfg['params'])
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not load adapter '{class_name}' from '{module_path}': {e}")

    # --- 2. Dynamic Configuration based on Adapter ---
    patch_size = adapter.patch_size
    stride = adapter.stride
    num_classes = adapter.num_classes
    num_bands = adapter.num_bands

    stride_ratio = stride / patch_size if patch_size > 0 else 0.5

    # Setup Inference Engine Config Dynamically
    zor_config = cfg.pipeline.tiling.zone_of_responsibility_size
    halo = cfg.pipeline.tiling.halo_size_pixels

    # Determine concurrent chunks for memory safety
    use_prefetcher = cfg.model.get('use_prefetcher', cfg.pipeline.distributed.get('use_prefetcher', True))
    if use_prefetcher:
        pq_size = cfg.pipeline.distributed.get('prefetch_queue_size', 2)
    else:
        pq_size = 0

    writer_q_size = cfg.pipeline.distributed.get('writer_queue_size', 4)

    # Calculate ZoR using the model-specific parameters
    zor = resolve_zor(
        zor_config,
        halo,
        patch_size=patch_size,
        num_bands=num_bands,
        num_classes=num_classes,
        stride_ratio=stride_ratio,
        prefetch_queue_size=pq_size,
        writer_queue_size=writer_q_size,
        is_segmentation=adapter.is_segmentation,
        save_conf=cfg.pipeline.output.get('save_confidence', True),
        save_entr=cfg.pipeline.output.get('save_entropy', True),
        save_gap=cfg.pipeline.output.get('save_gap', True)
    )
    log.info(f"Inference Configuration: ZoR={zor}, Halo={halo}, ChunkSize={zor + 2*halo}")
    log.info(f"Model Params: Patch={patch_size}, Stride={stride}, Bands={num_bands}, Classes={num_classes}")
    log.info(f"Memory Logic: Prefetch Q={pq_size}, Writer Q={writer_q_size}, Segmentation={adapter.is_segmentation}")

    # Prepare bands list
    if 'bands' in adapter.params:
         bands = list(adapter.params['bands'])
    else:
         bands = list(cfg.data_source.bands)

    # Instantiate Inference Engine with the pre-built adapter
    engine_config = {'adapter': adapter_cfg} # Still pass config for reference if needed
    engine = InferenceEngine(engine_config, adapter=adapter)

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

    # Gradient Preview: Saves Class 1 probability for binary models (visualizes detection confidence)
    if cfg.pipeline.output.get('save_gradient_preview', False) and num_classes == 2:
        out_paths['gradient'] = output_path / f"{tile_path.name}_gradient.tif"

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
    # Increase maxsize to decouple Inference (Fast) from Writing (Slow)
    writer_queue_size = cfg.pipeline.distributed.get('writer_queue_size', 4)
    write_queue = ctx.Queue(maxsize=writer_queue_size)

    writer_p = ctx.Process(
        target=writer_process,
        args=(write_queue, out_paths, profile, zor, halo, W_full, H_full, total_chunks, patch_size, engine.adapter),
        daemon=True
    )
    writer_p.start()

    log.info("Starting Inference Loop (InferenceEngine)")

    # --- Initialize Input Stream ---
    def input_stream_generator():
        for r, c in coords_list:
            r_read = r - halo
            c_read = c - halo
            w_read = zor + (2 * halo)
            h_read = zor + (2 * halo)

            raw_input = {
                'tile_folder': tile_path,
                'r_start': r_read,
                'c_start': c_read,
                'w_read': w_read,
                'h_read': h_read,
                'bands': bands
            }

            # Run CPU-bound preprocessing
            t_preprocess_start = time.perf_counter()
            try:
                processed_input = engine.preprocess(raw_input)
                log.debug(f"[{time.perf_counter()-t_preprocess_start:.4f}s] Main: engine.preprocess for chunk ({r},{c})")
                yield processed_input
            except Exception as e:
                log.error(f"Preprocessing error at {r}, {c}: {e}")
                raise e

    if use_prefetcher:
        # Use a thread queue to buffer preprocessed inputs
        prefetch_queue_size = cfg.pipeline.distributed.get('prefetch_queue_size', 3)
        prefetch_queue = queue.Queue(maxsize=prefetch_queue_size)

        def prefetch_worker():
            try:
                for item in input_stream_generator():
                    prefetch_queue.put(item)
                prefetch_queue.put(None) # End of data
            except Exception as e:
                log.error(f"Prefetch error: {e}")
                prefetch_queue.put(None) # Signal error or end

        import threading
        loader_thread = threading.Thread(target=prefetch_worker, daemon=True)
        loader_thread.start()

        def get_next_batch():
            return prefetch_queue.get()
    else:
        # Synchronous execution
        stream_iter = input_stream_generator()
        def get_next_batch():
            try:
                return next(stream_iter)
            except StopIteration:
                return None
            except Exception as e:
                log.error(f"Synchronous error: {e}")
                return None

    inference_pbar = tqdm(total=total_chunks, desc="Inference", position=0, leave=True)

    try:
        while True:
            # Check writer status
            if not writer_p.is_alive():
                log.error(f"Writer process died unexpectedly with exit code {writer_p.exitcode}. Stopping.")
                break

            # Get next batch
            model_input = get_next_batch()
            if model_input is None:
                break # Done or Error # Done or Error

            # 2. Predict (GPU) - RAW Output (logits, metadata)
            t_predict_raw_start = time.perf_counter()
            logits_t, metadata = engine.predict_raw(model_input)
            log.debug(f"[{time.perf_counter()-t_predict_raw_start:.4f}s] Main: engine.predict_raw")

            # 3. Pass to Writer (CPU)
            # We pass the raw logits and the metadata.
            # Postprocessing happens in the writer process.

            logits_t = logits_t.share_memory_()

            # Put in queue (this might block if queue is full, indicating writer is slow)
            write_queue.put((logits_t, metadata))

            inference_pbar.update(1)

    except KeyboardInterrupt:
        log.info("Interrupted by user.")
        write_queue.put(None)
        writer_p.terminate()

    finally:
        inference_pbar.close()
        if writer_p.is_alive():
            write_queue.put(None)
            writer_p.join()

        if writer_p.exitcode is not None and writer_p.exitcode != 0:
            log.error(f"Writer process exited with error code {writer_p.exitcode}. Output files may be incomplete.")

    # Metadata
    # Use adapter labels/colormap if available, else fallback to globals
    if adapter.labels:
        labels = adapter.labels
        color_map = adapter.color_map
    else:
        labels = NEW_LABELS
        color_map = LABEL_COLOR_MAP

    class_map = {
        label: {
            "index": i,
            "color": color_map.get(label, [128,128,128]) if isinstance(color_map.get(label), list) else color_map.get(label, np.array([128,128,128])).tolist()
        }
        for i, label in enumerate(labels)
    }
    with open(output_path / f"{tile_path.name}_classmap.json", "w") as f:
        json.dump(class_map, f)

    save_preview = cfg.pipeline.output.get('save_preview', True)
    if save_preview:
        try:
            downscale = cfg.pipeline.output.get("preview_downscale_factor", 10)
            class_path = out_paths['class']
            if class_path.exists():
                with rasterio.open(class_path) as src:
                    generate_low_res_preview(
                        src.read(1),
                        output_path / "preview.png",
                        save_preview=save_preview,
                        downscale_factor=downscale,
                        labels=labels,
                        color_map=color_map
                    )
            else:
                log.warning(f"Class map file not found at {class_path}. Skipping preview generation.")
        except Exception as e:
            log.error(f"Failed to generate preview: {e}")

    try:
        generate_single_node_viewer(tile_path.name, str(output_path.parent))
    except Exception as e:
        log.error(f"Failed to generate viewer: {e}")

    log.info(f"Finished in {time.time() - t0:.2f}s")
