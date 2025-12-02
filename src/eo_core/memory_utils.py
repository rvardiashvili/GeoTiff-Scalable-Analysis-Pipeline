import psutil
import math
import torch
import gc
import logging
from typing import Tuple, Union, Any

log = logging.getLogger(__name__)

def calculate_optimal_zor(
    halo: int = 128, 
    patch_size: int = 120,
    num_bands: int = 12,
    num_classes: int = 19,
    stride_ratio: float = 0.5,
    is_segmentation: bool = False,
    mem_safety_factor: float = 0.80,
    prefetch_queue_size: int = 0,
    writer_queue_size: int = 4,
    save_conf: bool = True,
    save_entr: bool = True,
    save_gap: bool = True
) -> int:
    """
    Calculates the optimal Zone of Responsibility (ZoR) size using a precise memory model.
    
    Formula accounts for:
    1. Input Patches (Float32):
       - Multiplied by (PrefetchQueue + 1 Active Batch)
       - Expansion factor due to overlap: (1/stride_ratio)^2
    2. Model Output / Logits (Float32):
       - Multiplied by (WriterQueue + 2 Active Batches)
       - Size depends on task (Segmentation vs Classification)
    3. Reconstruction Maps (Float32):
       - 1 copy in Writer process (avg_probs + weight_sum)
    4. Metrics Buffers (Float32/UInt8):
       - Conditional buffers for dominant class, confidence, entropy, gap.
    """
    
    # 1. Get Available Memory in Bytes
    mem = psutil.virtual_memory()
    available_bytes = mem.available * mem_safety_factor
    
    BYTES_FLOAT = 4
    BYTES_UINT8 = 1
    
    # --- BPP (Bytes Per Pixel of the CHUNK) Calculation ---
    
    # 1. Patches BPP (Float32)
    # N_patches ~= (H / S)^2. 
    # Patch Size = P^2 * C_in * 4.
    # Total Patch Bytes = (H/S)^2 * P^2 * C_in * 4 = H^2 * (P/S)^2 * C_in * 4.
    # BPP = C_in * 4 * (1/stride_ratio)^2
    patch_overlap_factor = (1.0 / stride_ratio) ** 2
    bpp_patches = num_bands * BYTES_FLOAT * patch_overlap_factor
    
    # 2. Logits BPP (Float32)
    if is_segmentation:
        # Output is (N, C_out, P, P)
        # Similar geometry to input patches, just C_out instead of C_in
        bpp_logits = num_classes * BYTES_FLOAT * patch_overlap_factor
    else:
        # Classification Output is (N, C_out)
        # Total Bytes = (H/S)^2 * C_out * 4
        # BPP = C_out * 4 / S^2
        stride_pixels = patch_size * stride_ratio
        bpp_logits = (num_classes * BYTES_FLOAT) / (stride_pixels ** 2)

    # 3. Reconstruction BPP (Float32)
    # Writer allocates (C_out) * H * W * 4 for avg_probs
    # + 1 * H * W * 4 for weight_sum
    bpp_recon = (num_classes + 1) * BYTES_FLOAT
    
    # 4. Metrics BPP (Conditional)
    bpp_metrics = 0
    bpp_metrics += BYTES_UINT8 # dom (always)
    
    if save_conf:
        bpp_metrics += BYTES_FLOAT # conf
    
    if save_entr:
        bpp_metrics += BYTES_FLOAT # entr
        
    if save_gap:
        # gap output (4 bytes) + top2 intermediate array (2 * 4 bytes)
        bpp_metrics += (BYTES_FLOAT * 3)
    
    # 5. Write Buffer Overhead (Estimate)
    # Rasterio needs to buffer the chunk being written. 
    # Approx equal to input bytes (worst case for uncompressed).
    bpp_io = num_bands * BYTES_FLOAT
    
    # --- Total System Footprint per Pixel ---
    # Weighted by how many copies exist in queues
    
    # Patches: Exist in Prefetch Queue (if any) + Inference Input
    total_bpp_patches = bpp_patches * (prefetch_queue_size + 1)
    
    # Logits: Exist as Inference Output + Writer Queue + Writer Processing
    # We add +2 (1 for Inference Engine output holding, 1 for current Writer item)
    total_bpp_logits = bpp_logits * (writer_queue_size + 2)
    
    # Recon & Metrics: Exist only in Writer
    total_bpp_recon = bpp_recon + bpp_metrics
    
    overhead_bpp = 200 # Python overhead
    
    total_bpp = total_bpp_patches + total_bpp_logits + total_bpp_recon + bpp_io + overhead_bpp
    
    # --- Solve for ZoR ---
    max_pixels = available_bytes / total_bpp
    max_chunk_side = int(math.sqrt(max_pixels))
    
    max_zor = max_chunk_side - (2 * halo)
    
    if max_zor <= 0:
        return patch_size
        
    # Round to multiple of patch_size
    optimal_zor = (max_zor // patch_size) * patch_size
    
    if optimal_zor < patch_size:
        optimal_zor = patch_size
        
    return optimal_zor

def resolve_zor(zor_config: Union[int, str], halo: int, patch_size: int = 120, **kwargs) -> int:
    """
    Resolves the ZoR size from config, handling "auto" or string inputs.
    kwargs are passed to calculate_optimal_zor (num_bands, num_classes, stride_ratio).
    """
    if isinstance(zor_config, str) and zor_config.lower() == "auto":
        print("🧠 Auto-calculating Chunk Size based on available RAM...")
        return calculate_optimal_zor(halo=halo, patch_size=patch_size, **kwargs)
    elif isinstance(zor_config, int):
        return zor_config
    else:
        try:
            return int(zor_config)
        except:
            print(f"⚠️ Invalid ZoR config, defaulting to {patch_size * 10}")
            return patch_size * 10

def estimate_optimal_batch_size(model: torch.nn.Module, input_shape: Tuple[int, ...], device: torch.device, safety_factor: float = 0.80) -> int:
    """
    Estimates the optimal batch size for the given model and input shape by binary search
    or heuristic to fill GPU memory.
    
    Args:
        model: The PyTorch model (should be on 'device').
        input_shape: Shape of a SINGLE sample (C, H, W) or (C, T, H, W).
        device: The target device.
        safety_factor: Fraction of available memory to use (default 0.85).
        
    Returns:
        Recommended batch size (int).
    """
    if device.type == 'cpu':
        return 16 # Conservative default for CPU
        
    log.info("🧠 Starting Binary Search for Optimal Batch Size...")
    
    # Binary Search Parameters
    low = 1
    high = 512 # Hard cap
    optimal_bs = 1
    
    # Move model to device
    model.to(device)
    model.eval() # Ensure eval mode
    
    # Get initial state
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    
    try:
        while low <= high:
            mid = (low + high) // 2
            
            try:
                # Create dummy input
                dummy_input = torch.zeros((mid, *input_shape), device=device, dtype=torch.float32)
                
                # Clean previous run
                torch.cuda.empty_cache()
                
                # Dry Run
                with torch.no_grad():
                     _ = model(dummy_input)
                
                # If we are here, it worked
                optimal_bs = mid
                low = mid + 1
                # log.debug(f"Batch {mid}: OK")
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    # log.debug(f"Batch {mid}: OOM")
                    high = mid - 1
                else:
                    # Some other error? Re-raise
                    raise e
                    
        # Apply Safety Factor
        # We found the absolute maximum 'mid' that didn't crash.
        # Now we back off slightly to account for fragmentation during long runs.
        final_bs = int(optimal_bs * safety_factor)
        final_bs = max(1, final_bs)
        
        log.info(f"🧠 Binary Search Result: MaxBS={optimal_bs} -> SafeBS={final_bs} (Safety={safety_factor})")
        
        return final_bs
        
    except Exception as e:
        log.warning(f"⚠️ Binary search failed ({e}). Defaulting to 1.")
        return 1