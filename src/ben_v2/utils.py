"""
Utility functions, placeholder logic for external libraries, and visualization tools.
"""
import base64
import io
import time 
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from PIL import Image 
from scipy.ndimage import zoom 
from torch.utils.data import DataLoader

# ----------------------------------------------------------------------
# --- Constants & Helpers ---
# ----------------------------------------------------------------------

RGB_INDICES = (2, 1, 0)

def get_device():
    """Returns the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def get_autocast(device):
    """Returns the appropriate autocast context manager."""
    if device.type == 'cuda':
        return torch.amp.autocast
    else:
        class NullAutocast:
            def __enter__(self): pass
            def __exit__(self, exc_type, exc_val, exc_tb): pass
        return NullAutocast

# ----------------------------------------------------------------------
# --- Placeholder/Required Module Variables ---
# ----------------------------------------------------------------------

STANDARD_BANDS: Dict[int, List[str]] = {}
NEW_LABELS: List[str] = []
BigEarthNetv2_0_ImageClassifier = None
FALLBACK_LABEL_KEY = 'No_Dominant_Class' 

# ----------------------------------------------------------------------
# --- Placeholder Logic for Missing Libraries ---
# ----------------------------------------------------------------------

try:
    # Attempt to import the real libraries
    from configilm.extra.BENv2_utils import STANDARD_BANDS, NEW_LABELS, stack_and_interpolate, means, stds
    from .model import BigEarthNetv2_0_ImageClassifier
    print("✅ Successfully loaded BENv2 libraries.")
    
except ImportError as e:
    print(f"⚠️ Warning: External BENv2 libraries not found or failed to load ({e}). Using robust placeholders.")
    
    # Define fallback data structures if import fails
    STANDARD_BANDS = {
        10: ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12'],
        12: ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12'],
    }

    # Fallback Labels (43 classes from BigEarthNet v1)
    NEW_LABELS = [
        'Annual-crops', 'Permanent-crops', 'Pastures', 'Complex-cultivation-patterns', 
        'Agro-forestry-areas', 'Broad-leaved-forest', 'Coniferous-forest', 'Mixed-forest', 
        'Natural-grasslands-and-sparsely-vegetated-areas', 'Moors-and-heathland', 
        'Sclerophyllous-vegetation', 'Transitional-woodland-shrub', 'Beaches-sandy-plains', 
        'Intertidal-flats', 'Bare-areas', 'Burnt-areas', 'Inland-wetlands', 
        'Coastal-wetlands', 'Continental-water', 'Marine-water', 'Glaciers-and-perpetual-snow', 
        'Non-irrigated-arable-land', 'Permanently-irrigated-land', 'Rice-fields', 
        'Vineyards', 'Fruit-trees-and-berry-plantations', 'Olive-groves', 
        'Annual-crops-with-associated-fallow-lands', 
        'Land-principally-occupied-by-agriculture-with-significant-areas-of-natural-vegetation', 
        'Broad-leaved-forest-evergreen', 'Broad-leaved-forest-deciduous', 
        'Coniferous-forest-evergreen', 'Coniferous-forest-deciduous', 'Mixed-forest', 
        'Natural-grasslands', 'Sparsely-vegetated-areas', 'Salt-marshes', 
        'Bogs-and-peatlands', 'Water-bodies', 'Snow-and-ice', 'Urban-fabric', 
        'Industrial-or-commercial-units', 'Road-and-rail-networks-and-associated-land'
    ]
    
    # Placeholder Model Class
    class BigEarthNetv2_0_ImageClassifier(torch.nn.Module):
        """Placeholder model class for running without the external library."""
        def __init__(self, num_classes: int):
            super().__init__()
            self.num_classes = num_classes
            print("⚠️ Using a dummy torch.nn.Linear model as BigEarthNetv2_0_ImageClassifier placeholder.")
            # Use 12 bands for the dummy model size, assuming BANDS will be correctly set later
            input_size = 12 * 120 * 120 
            self.linear = torch.nn.Linear(input_size, num_classes) 

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Flatten C, H, W to a single vector for the linear layer
            return self.linear(x.flatten(start_dim=1))
            
    def stack_and_interpolate(bands_dict: Dict[str, np.ndarray]) -> np.ndarray:
        # Dummy implementation
        return np.stack(list(bands_dict.values()), axis=-1)

# ----------------------------------------------------------------------
# Consistent Label Color Map Generation
# ----------------------------------------------------------------------

LABEL_COLOR_MAP: Dict[str, np.ndarray] = {
    label: np.random.randint(0, 256, 3, dtype=np.uint8) 
    for label in NEW_LABELS
}
LABEL_COLOR_MAP[FALLBACK_LABEL_KEY] = np.array([128,128,128], dtype=np.uint8)


# ----------------------------------------------------------------------
# Visualization Functions
# ----------------------------------------------------------------------

def generate_low_res_preview(
    mask_data: np.ndarray, 
    output_path: Path, 
    save_preview: bool = True,
    downscale_factor: int = 10
):
    """
    Generates a low-resolution color PNG preview of the classification mask.
    """
    if not save_preview:
        return

    print(f"🎨 Generating low-res preview image (downscale={downscale_factor})...")

    if downscale_factor > 1:
        downscaled_mask = zoom(mask_data, 1.0 / downscale_factor, order=0)
    else:
        downscaled_mask = mask_data
    
    color_map_array = np.array([LABEL_COLOR_MAP[label] for label in NEW_LABELS], dtype=np.uint8)
    
    max_idx = len(NEW_LABELS) - 1
    safe_mask = np.clip(downscaled_mask, 0, max_idx)
    
    rgb_image = color_map_array[safe_mask]
    
    try:
        img_pil = Image.fromarray(rgb_image, 'RGB')
        img_pil.save(output_path, 'PNG')
        print(f"  Saved preview to {output_path.name}")
    except Exception as e:
        print(f"❌ Error saving preview image: {e}")

try:
    save_color_mask_preview
except NameError:
    save_color_mask_preview = generate_low_res_preview

# ----------------------------------------------------------------------
# Helper for GPU Inference
# ----------------------------------------------------------------------

def run_gpu_inference(
    patches: np.ndarray, 
    model: torch.nn.Module, 
    norm_m: torch.Tensor, 
    norm_s: torch.Tensor,
    device: torch.device,
    batch_size: int = 32,
    use_amp: bool = False
) -> np.ndarray:
    """
    Performs batch inference on the GPU for a NumPy array of patches.
    """
    start_time = time.time()
    all_probs = []
    
    autocast = get_autocast(device)
    
    # Ensure norms are on the correct device
    norm_m = norm_m.to(device)
    norm_s = norm_s.to(device)
    
    class InMemoryPatchDataset(torch.utils.data.Dataset):
        def __init__(self, patches: np.ndarray):
            self.patches = patches

        def __len__(self):
            return len(self.patches)

        def __getitem__(self, idx):
            return torch.as_tensor(self.patches[idx]).float()

    dataset = InMemoryPatchDataset(patches)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    first_batch = True
    for tensor_cpu in dataloader:
        try:
            tensor_gpu = tensor_cpu.to(device, non_blocking=True)
            tensor_gpu = (tensor_gpu - norm_m) / (norm_s + 1e-6)
            
            if first_batch:
                print(f"DEBUG: Normalized Batch Stats - Min: {tensor_gpu.min().item():.4f}, Max: {tensor_gpu.max().item():.4f}, Mean: {tensor_gpu.mean().item():.4f}")
                first_batch = False
            
            if use_amp and device.type == 'cuda':
                with autocast(device_type='cuda', dtype=torch.float16):
                    logits = model(tensor_gpu)
            else:
                logits = model(tensor_gpu)
                
            probs = torch.sigmoid(logits.float()).cpu().detach().numpy()
            all_probs.append(probs)
        except Exception as e:
            print(f"❌ GPU inference error on batch: {e}")
            probs = np.zeros((tensor_cpu.shape[0], len(NEW_LABELS)), dtype=np.float32)
            all_probs.append(probs)

    return np.concatenate(all_probs, axis=0)