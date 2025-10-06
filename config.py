"""
Central configuration file for the BigEarthNet analysis pipeline.
"""
import torch

# --- Model Configuration ---
MODEL_NAME = "resnet50-s2-v0.2.0"
REPO_ID = f"BIFOLD-BigEarthNetv2-0/{MODEL_NAME}"

# --- Performance & Pipeline Configuration ---
PATCH_SIZE = 120
# DATA_LOADER_WORKERS: Use these for PyTorch DataLoader's num_workers (multiprocessing for I/O)
# This is where your CPU_THREADS setting is now applied to multiprocessing.
DATA_LOADER_WORKERS = 12 
# Set GPU batch size to a power of 2 for potential memory alignment benefits
GPU_BATCH_SIZE = 128


# --- Analysis Scope ---
# Set to 0 to process all patches/scenes
MAX_PATCHES = 0
MAX_SCENES = 4

# --- Hardware & Acceleration Setup ---
try:
    if torch.cuda.is_available():
        # Attempt to import autocast for Automatic Mixed Precision (AMP)
        from torch.cuda.amp import autocast
        USE_AMP = True
        DEVICE = torch.device("cuda")
        print("✅ CUDA found. Automatic Mixed Precision (AMP) will be used.")
    else:
        USE_AMP = False
        DEVICE = torch.device("cpu")
        print("⚠️ CUDA not found. Running on CPU (AMP disabled).")
except ImportError:
    USE_AMP = False
    DEVICE = torch.device("cpu")
    print("⚠️ PyTorch or torch.cuda.amp not available. Running on CPU (AMP disabled).")
