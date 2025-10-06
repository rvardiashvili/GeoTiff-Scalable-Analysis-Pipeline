import os
import time
import threading
from pathlib import Path
from typing import List, Dict, Any, Tuple

import torch
import numpy as np
from colorama import Fore, Style, init as colorama_init
from torch.utils.data import DataLoader

from config import DEVICE, USE_AMP, GPU_BATCH_SIZE, DATA_LOADER_WORKERS
from data_loader import PatchFolderDataset, InMemoryPatchDataset # Import the new dataset
from utils import NORM_M, NORM_S, NEW_LABELS

colorama_init(autoreset=True)

NUM_WORKERS = DATA_LOADER_WORKERS
PREFETCH_FACTOR = 4

# Toggle monitoring display (clearing console)
ENABLE_MONITOR = True

# ----------------------------------------------------------------------
# NEW MODULAR FUNCTION FOR IN-MEMORY ANALYSIS (replacing complex SceneAnalyzer for this task)
# ----------------------------------------------------------------------

def analyze_in_memory_data(patches: np.ndarray, model: torch.nn.Module) -> List[Dict[str, Any]]:
    """
    Analyzes a batch of in-memory NumPy patches using a PyTorch DataLoader.
    
    This function acts as the core inference engine, taking the prepared data
    and returning the raw model results (probabilities) for mask generation.
    
    Args:
        patches (np.ndarray): NumPy array of 10-band patches (N, H, W, C).
        model (torch.nn.Module): The loaded BigEarthNet classification model.
        
    Returns:
        List[Dict[str, Any]]: List of results, one dict per patch, containing
                              {'name': str, 'probs': np.ndarray}.
    """
    # 1. Setup Dataset and DataLoader
    base_dataset = InMemoryPatchDataset(patches)

    data_loader = DataLoader(
        base_dataset,
        batch_size=GPU_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == 'cuda'),
        prefetch_factor=PREFETCH_FACTOR,
        persistent_workers=False,
        collate_fn=lambda x: (
            torch.stack([item[0] for item in x]), # Stack patch tensors
            [item[1] for item in x]               # Keep patch names (indices)
        )
    )

    all_results: List[Dict[str, Any]] = []
    total_patches = len(base_dataset)
    patches_processed = 0

    print(f"🔥 Starting in-memory patch analysis ({total_patches} patches) on {DEVICE}...")
    start_time = time.time()

    model.eval() # Ensure model is in evaluation mode
    
    with torch.no_grad():
        for current_batch_cpu, current_batch_names in data_loader:
            current_batch_size = len(current_batch_names)
            
            # --- GPU Processing Logic ---
            try:
                tensor_gpu = current_batch_cpu.to(DEVICE, non_blocking=True)
                # Normalize data on the GPU
                tensor_gpu = (tensor_gpu - NORM_M.to(DEVICE)) / (NORM_S.to(DEVICE) + 1e-6)
                
                # Inference
                if USE_AMP and DEVICE.type == 'cuda':
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        logits = model(tensor_gpu)
                else:
                    logits = model(tensor_gpu)
                    
                # Calculate probabilities and move back to CPU
                probs = torch.sigmoid(logits.float()).cpu().numpy()
            except Exception as e:
                print(f"{Fore.RED}GPU inference error on batch: {e}{Style.RESET_ALL}")
                probs = np.zeros((current_batch_size, len(NEW_LABELS)))
                
            # --- Aggregate Results ---
            for i, name in enumerate(current_batch_names):
                prob_arr = probs[i]
                
                result = {
                    "name": name,
                    "probs": prob_arr, # Full probability array is needed by mask generator
                }
                all_results.append(result)
            
            patches_processed += current_batch_size
            print(f"  ... {patches_processed}/{total_patches} patches processed.")

    end_time = time.time()
    print(f"✅ In-memory analysis complete in {end_time - start_time:.2f}s.")
    return all_results


# ----------------------------------------------------------------------
# Original SceneAnalyzer class remains below (commented or truncated for clarity)
# ----------------------------------------------------------------------
# class DebugDataset(...)
# class SceneAnalyzer(...)
# ... (Keep the original SceneAnalyzer definition for the folder-based pipeline)
