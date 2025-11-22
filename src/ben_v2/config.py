"""
Central configuration file for the BigEarthNet analysis pipeline.
"""
import torch
from functools import reduce
import math
import numpy as np # NEW: Need numpy for PROB_DTYPE definition

# --- Model Configuration ---
MODEL_NAME = "resnet50-s2-v0.2.0"
REPO_ID = f"BIFOLD-BigEarthNetv2-0/{MODEL_NAME}"
PATCH_SIZE = 120
PATCH_STRIDE = PATCH_SIZE // 2

RGB_INDICES = (2,1,0)

# --- Model-Specific Configurations ---
MODEL_CONFIG = {
    "resnet50-s2-v0.2.0": {
        "bands": ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12'],
        "means": [380.17, 439.02, 471.91, 559.72, 984.72, 1198.77, 1221.28, 1104.79, 793.88, 592.76],
        "stds":  [683.47, 727.20, 804.89, 973.80, 1583.50, 1729.62, 1798.24, 1735.71, 1515.26, 1100.17]
    },
    "resnet101-s2-v0.2.0": {
        "bands": ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12'],
        "means": [1354.405, 1118.244, 1042.929, 1177.816, 1294.462, 1373.085, 1321.078, 1291.621, 1139.839, 1009.326],
        "stds": [448.822, 405.820, 393.963, 379.190, 395.953, 409.489, 410.497, 415.933, 379.191, 340.610]
    },
    "resnet101-all-v0.2.0": {
        "bands": ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12', 'VV', 'VH'],
        "means": [1354.405, 1118.244, 1042.929, 1177.816, 1294.462, 1373.085, 1321.078, 1291.621, 1139.839, 1009.326, 0.0, 0.0],
        "stds": [448.822, 405.820, 393.963, 379.190, 395.953, 409.489, 410.497, 415.933, 379.191, 340.610, 1.0, 1.0]
    },

}

# --- Dynamic Configuration from MODEL_NAME ---
_model_config = MODEL_CONFIG.get(MODEL_NAME)
if not _model_config:
    raise ValueError(f"Configuration for model '{MODEL_NAME}' not found.")

BANDS = _model_config["bands"]
MEANS = _model_config["means"]
STDS = _model_config["stds"]

USE_SENTINEL_1 = "all" in MODEL_NAME

# --- Consistency Check ---
if len(BANDS) != len(MEANS) or len(BANDS) != len(STDS):
    raise ValueError(f"Mismatch in configuration for model '{MODEL_NAME}'. Number of bands, means, and stds must be equal.")

# --- Output Configuration ---
SAVE_FULL_PROBS = False
SAVE_ENTROPY = True
SAVE_GAP = True
SAVE_PREVIEW_IMAGE = True
PREVIEW_DOWNSCALE_FACTOR = 10
CONFIDENCE_THRESHOLD = 0.05

# --- Scalable Processing Configuration ---
PREFERRED_CHUNK_SIZE = 5000

# --- Device Setup ---
try:
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        USE_AMP = True
        autocast = torch.amp.autocast
        print(f"✅ CUDA found. Running on {DEVICE} (AMP enabled).")
    else:
        DEVICE = torch.device("cpu")
        USE_AMP = False
        class autocast:
            def __enter__(self): pass
            def __exit__(self, exc_type, exc_val, exc_tb): pass
        print("⚠️ CUDA not found. Running on CPU (AMP disabled).")
except ImportError:
    class autocast:
        def __enter__(self): pass
        def __exit__(self, exc_type, exc_val, exc_tb): pass
    USE_AMP = False
    DEVICE = torch.device("cpu")
    print("⚠️ CUDA not found (ImportError). Running on CPU (AMP disabled).")

# --- Normalization Tensors ---
NORM_M = torch.tensor(MEANS, dtype=torch.float32).view(1, len(MEANS), 1, 1)
NORM_S = torch.tensor(STDS, dtype=torch.float32).view(1, len(STDS), 1, 1)

# --- Data Loading Configuration ---
GPU_BATCH_SIZE = 32 # Default GPU inference batch size
DATA_LOADER_WORKERS = 4 # Number of worker processes for DataLoader (I/O)

# --- Geometry Helper Functions ---
def lcm(a, b):
    """Least Common Multiple."""
    return abs(a*b) // math.gcd(a, b)

def nearest_viable_chunk(preferred_size: int, resolutions=(10, 20, 60, 16, PATCH_SIZE)):
    """
    Ensures the chunk size is a multiple of all relevant resolutions and the patch size.
    It returns the preferred_size, or the nearest smaller multiple of the common LCM.
    It caps the chunk size at the tile dimension itself if the tile is smaller than preferred_size.
    """
    all_res = list(resolutions)
    lcm_val = reduce(lcm, all_res)
    
        
    # Find the nearest multiple of lcm_val that is <= preferred_size
    multiple = preferred_size // lcm_val
    if multiple == 0:
        # If preferred size is less than lcm_val, default to lcm_val (smallest viable)
        return lcm_val
        
    return multiple * lcm_val

CHUNK_SIZE = nearest_viable_chunk(PREFERRED_CHUNK_SIZE)