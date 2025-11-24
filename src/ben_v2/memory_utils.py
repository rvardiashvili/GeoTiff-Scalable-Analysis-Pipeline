import psutil
import math
from typing import Tuple, Union

def calculate_optimal_zor(
    halo: int = 128, 
    patch_size: int = 120,
    mem_safety_factor: float = 0.85
) -> int:
    """
    Calculates the optimal Zone of Responsibility (ZoR) size based on available system memory.
    
    Heuristic:
    - Available Memory * Safety Factor
    - Estimate bytes per pixel (Input + Patches + Output + Overhead)
    - Calculate Max Side Length
    - Round down to nearest multiple of LCM of core resolutions (10, 20, 60) & Patch Size.
      Actually, just rounding to PATCH_SIZE (120) is usually sufficient for alignment.
      
    Estimated Bytes Per Pixel of CHUNK (ZoR + 2Halo):
    1. Input Data (Float32): 12 bands * 4 bytes = 48 bytes
    2. Patches Tensor (Float32): 
       With stride=PATCH_SIZE/2, we have 4x overlap.
       So roughly 4 * 48 bytes = 192 bytes.
    3. Output Maps (Float32/UInt8):
       Probabilities (19 classes * 4 bytes) = 76 bytes.
       Other metrics ~ 20 bytes.
    4. Overhead (Python objects, intermediate buffers): ~100 bytes.
    
    Total ~ 450 bytes per pixel of the PADDED chunk.
    
    Math:
    Max_Pixels = Available_Bytes / 450
    Max_Chunk_Side = Sqrt(Max_Pixels)
    Max_ZoR = Max_Chunk_Side - 2 * Halo
    """
    
    # 1. Get Available Memory in Bytes
    mem = psutil.virtual_memory()
    available_bytes = mem.available * mem_safety_factor
    
    # 2. Bytes Per Pixel Estimate
    BYTES_PER_PIXEL = 450
    
    max_pixels = available_bytes / BYTES_PER_PIXEL
    max_chunk_side = int(math.sqrt(max_pixels))
    
    # 3. Calculate Max ZoR
    max_zor = max_chunk_side - (2 * halo)
    
    if max_zor <= 0:
        # Fallback for extremely low memory
        return patch_size
        
    # 4. Round down to multiple of PATCH_SIZE (120)
    # This ensures good alignment with patches
    optimal_zor = (max_zor // patch_size) * patch_size
    
    if optimal_zor < patch_size:
        optimal_zor = patch_size
        
    return optimal_zor

def resolve_zor(zor_config: Union[int, str], halo: int, patch_size: int = 120) -> int:
    """
    Resolves the ZoR size from config, handling "auto" or string inputs.
    """
    if isinstance(zor_config, str) and zor_config.lower() == "auto":
        print("🧠 Auto-calculating Chunk Size based on available RAM...")
        return calculate_optimal_zor(halo=halo, patch_size=patch_size)
    elif isinstance(zor_config, int):
        return zor_config
    else:
        try:
            return int(zor_config)
        except:
            print(f"⚠️ Invalid ZoR config, defaulting to {patch_size * 10}")
            return patch_size * 10
