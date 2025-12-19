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
from omegaconf import DictConfig, OmegaConf, ListConfig
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

# from .reporters import GeoTIFFReporter, PreviewReporter, MetadataReporter

from importlib import import_module

def writer_process(q: mp.Queue, context_config: Dict[str, Any], zor: int, halo: int, W_full: int, H_full: int, total_chunks: int, patch_size: int, adapter: Any, reporter_configs: Any, benchmark_report_path: Optional[str] = None):
    """
    Independent process to reconstruct map and delegate writing to Reporters.
    """
    # Ensure src is in path for subprocesses
    src_path = str(Path(__file__).parent.parent)
    if src_path not in sys.path:
        sys.path.append(src_path)
        
    t_process_start = time.perf_counter()
    
    # Instantiate Reporters via Hydra
    reporters = []
    
    # Normalize input to list of configs
    configs_to_process = []
    if isinstance(reporter_configs, (dict, DictConfig)):
        configs_to_process = list(reporter_configs.values())
    elif isinstance(reporter_configs, (list, ListConfig)):
        configs_to_process = reporter_configs
    
    log.info(f"Writer Process started. Configured reporters: {len(configs_to_process)}")
        
    for r_conf in configs_to_process:
        if r_conf is None: 
            continue
        try:
            # hydra.utils.instantiate creates the object based on _target_
            reporter = hydra.utils.instantiate(r_conf)
            reporters.append(reporter)
            log.info(f"Initialized reporter: {reporter.__class__.__name__}")
        except Exception as e:
            log.error(f"Failed to instantiate reporter config {r_conf}: {e}")

    # Construct Full Context
    # We reconstruct the context object from the config passed
    context = {
        'output_path': Path(context_config['output_path']),
        'tile_name': context_config['tile_name'],
        'profile': context_config['profile'],
        'adapter': adapter,
        'config': context_config['hydra_config'],
        'H_full': H_full,
        'W_full': W_full,
        'benchmark_report_path': benchmark_report_path # Added benchmark report path
    }
    
    # 1. Start Reporters
    for r in reporters:
        try:
            r.on_start(context)
        except Exception as e:
            log.error(f"Error in {r.__class__.__name__}.on_start: {e}")

    # Pre-calculate sinusoidal window for reconstruction
    window_1d = np.sin(np.linspace(0, np.pi, patch_size))**2
    patch_weight = np.outer(window_1d, window_1d).astype(np.float32)
    patch_weight = patch_weight[np.newaxis, :, :] # (1, P, P)
    
    pbar = tqdm(total=total_chunks, desc="Writing  ", position=1, leave=True)
    
    try:
        while True:
            item = q.get()
            if item is None:
                break
            
            # Unpack raw model output and metadata
            logits_t, metadata = item
            
            # --- Postprocessing (CPU) ---
            t_postprocess_start = time.perf_counter()
            post_result = adapter.postprocess((logits_t, metadata))
            
            results = post_result['probs_tensor'].numpy()
            coords = post_result['coords']
            H_crop = post_result['H_crop']
            W_crop = post_result['W_crop']
            r_chunk_start = post_result['r_chunk']
            c_chunk_start = post_result['c_chunk']
            
            if results.ndim == 2:
                n_classes = results.shape[1]
            elif results.ndim == 4:
                n_classes = results.shape[1]
            else:
                n_classes = 1

            # --- Reconstruction (CPU Heavy) ---
            # t_reconstruction_start = time.perf_counter()
            avg_probs = np.zeros((n_classes, H_crop, W_crop), dtype=np.float32)
            weight_sum = np.zeros((1, H_crop, W_crop), dtype=np.float32)
            
            idx = 0
            for r_p, c_p in coords:
                if results.ndim == 2:
                    patch_data = results[idx][:, np.newaxis, np.newaxis]
                elif results.ndim == 4:
                    patch_data = results[idx]
                else:
                     continue 

                patch_prob = patch_data * patch_weight
                
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
                 valid_probs = probs_map

            # --- Delegate to Reporters ---
            # If chunk start is negative (halo), the valid data (ZoR) starts at c_chunk_start + halo
            valid_c = c_chunk_start + halo
            valid_r = r_chunk_start + halo
            
            w_width = min(zor, W_full - valid_c)
            w_height = min(zor, H_full - valid_r)
            
            # Ensure window is not out of bounds
            if w_width <= 0 or w_height <= 0:
                continue
                
            window = Window(valid_c, valid_r, w_width, w_height)
            
            chunk_data = {
                'valid_probs': valid_probs, # (C, H_zor, W_zor)
                'window': window,
                'coords': (valid_r, valid_c)
            }
            
            for r in reporters:
                try:
                    r.on_chunk(chunk_data)
                except Exception as e:
                    log.error(f"Error in {r.__class__.__name__}.on_chunk: {e}")
            
            pbar.update(1)

    except Exception as e:
        print(f"CRITICAL ERROR IN WRITER PROCESS: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
    finally:
        pbar.close()
        # 3. Finish Reporters
        for r in reporters:
            try:
                r.on_finish(context)
            except Exception as e:
                log.error(f"Error in {r.__class__.__name__}.on_finish: {e}")
        
        # 4. Generate Viewer (After all reporters are done)
        try:
            generate_single_node_viewer(context['tile_name'], str(context['output_path'].parent))
        except Exception as e:
            log.error(f"Failed to generate viewer: {e}")
                
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
    
    # --- Input Path Handling & Series Detection ---
    input_is_series = False
    tile_paths_list = []
    
    # Check Model Requirement for Temporal Depth
    # Safe access to num_frames, defaulting to 1 if not present
    model_num_frames = OmegaConf.select(cfg, "model.adapter.params.backbone_params.num_frames", default=1)
    if model_num_frames is None: model_num_frames = 1 # Handle explicit None
    
    if tile_path.name.endswith('.SAFE'):
        # Direct path to a single product
        log.info("Processing as Single Product (Direct Path).")
        ref_tile_path = tile_path
    else:
        # Directory containing products?
        sub_tiles = sorted(list(tile_path.glob("*.SAFE")))
        s2_sub_tiles = [p for p in sub_tiles if p.name.startswith('S2')]
        
        if len(s2_sub_tiles) > 1 and model_num_frames > 1:
            log.info(f"Detected Multi-Temporal S2 Series ({len(s2_sub_tiles)} frames) for Temporal Model (req: {model_num_frames}).")
            input_is_series = True
            tile_paths_list = s2_sub_tiles
            ref_tile_path = s2_sub_tiles[0] # Use first as reference
        elif len(s2_sub_tiles) == 1:
            log.info("Detected Single S2 Product in directory.")
            ref_tile_path = s2_sub_tiles[0]
        else:
            # Fallback: maybe S1 only or just a folder structure we interpret as "the tile"
            # Or Multi-S2 but Model is Single-Frame (ambiguous, default to container path or first?)
            if len(s2_sub_tiles) > 1:
                 log.warning(f"Found {len(s2_sub_tiles)} S2 products but model is Single-Frame. Using first product as target.")
                 ref_tile_path = s2_sub_tiles[0]
            else:
                 log.info("No multiple S2 products found. Treating directory as tile root.")
                 ref_tile_path = tile_path

    log.info(f"Output directory: {output_path}")
    
    # Get Tile Dimensions (from reference)
    s2_pattern = cfg.data_source.get('s2_file_pattern', "S2*.SAFE/**/*{band_name}*.jp2")
    ref_path = _find_band_path(ref_tile_path, 'B02', s2_pattern)
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
            'path': 'ben_v2.adapters.bigearthnet_adapter',
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
    
    # Reporter Configuration
    if 'reporters' in cfg.pipeline and cfg.pipeline.reporters:
        log.info("Loading reporters from configuration.")
        # Pass the DictConfig/ListConfig directly, no need to resolve to container yet
        # as hydra.utils.instantiate expects OmegaConf objects or dicts.
        reporter_configs = cfg.pipeline.reporters
    else:
        log.info("Using default reporters (GeoTIFF, Preview, Metadata).")
        reporter_configs = {
            'geotiff': {'_target_': 'eo_core.reporters.geotiff.GeoTIFFReporter'},
            'preview': {'_target_': 'eo_core.reporters.preview.PreviewReporter'},
            'metadata': {'_target_': 'eo_core.reporters.metadata.MetadataReporter'}
        }
    
    context_config = {
        'output_path': output_path,
        'tile_name': tile_path.name,
        'profile': profile,
        'hydra_config': OmegaConf.to_container(cfg, resolve=True)
    }

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
        args=(write_queue, context_config, zor, halo, W_full, H_full, total_chunks, patch_size, engine.adapter, reporter_configs, context_config.get('benchmark_report_path')),
        daemon=True
    )
    writer_p.start()

    log.info("Starting Inference Loop (InferenceEngine)")
    
    # --- Benchmarker Initialization ---
    from .benchmarker import Benchmarker
    benchmarker = Benchmarker(output_dir=output_path)
    
    # Record Model Config
    bench_config = {
        "adapter_class": adapter_cfg.get('class', 'Unknown'),
        "patch_size": patch_size,
        "stride": stride,
        "zor": zor,
        "halo": halo,
        "gpu_batch_size": adapter.gpu_batch_size if hasattr(adapter, 'gpu_batch_size') else 'unknown',
        "prefetch_queue_size": pq_size,
        "writer_queue_size": writer_queue_size
    }
    benchmarker.set_model_config(bench_config)
    benchmarker.start()
    
    # --- Initialize Input Stream ---
    def input_stream_generator():
        for r, c in coords_list:
            r_read = r - halo
            c_read = c - halo
            w_read = zor + (2 * halo)
            h_read = zor + (2 * halo)
            
            raw_input = {
                'tile_folder': tile_paths_list if input_is_series else tile_path,
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
                dur_prep = time.perf_counter() - t_preprocess_start
                log.debug(f"[{dur_prep:.4f}s] Main: engine.preprocess for chunk ({r},{c})")
                benchmarker.record_event('cpu_preprocess_duration', dur_prep)
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
            t_wait = time.perf_counter()
            item = prefetch_queue.get()
            benchmarker.record_event('wait_for_prefetch_duration', time.perf_counter() - t_wait)
            return item
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
            dur_pred = time.perf_counter() - t_predict_raw_start
            log.debug(f"[{dur_pred:.4f}s] Main: engine.predict_raw")
            benchmarker.record_event('gpu_inference_duration', dur_pred)
            
            # 3. Pass to Writer (CPU)
            # We pass the raw logits and the metadata. 
            # Postprocessing happens in the writer process.
            
            logits_t = logits_t.share_memory_()
            
            # Put in queue (this might block if queue is full, indicating writer is slow)
            t_wait_write = time.perf_counter()
            write_queue.put((logits_t, metadata))
            benchmarker.record_event('wait_for_writer_queue_duration', time.perf_counter() - t_wait_write)
            
            inference_pbar.update(1)

    except KeyboardInterrupt:
        log.info("Interrupted by user.")
        write_queue.put(None)
        writer_p.terminate()
        
    finally:
        benchmarker.stop()
        benchmark_report_path = benchmarker.save_report() # Capture the filename
        
        # Add benchmark report path to context_config for the writer process
        context_config['benchmark_report_path'] = str(benchmark_report_path)

        if writer_p.is_alive():
            write_queue.put(None)
            writer_p.join()
        
        if writer_p.exitcode is not None and writer_p.exitcode != 0:
            log.error(f"Writer process exited with error code {writer_p.exitcode}. Output files may be incomplete.")

        # Explicitly release GPU memory
        if 'engine' in locals():
            engine.cleanup()

    log.info(f"Finished in {time.time() - t0:.2f}s")
