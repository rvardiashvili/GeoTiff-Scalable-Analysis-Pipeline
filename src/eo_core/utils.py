"""
Utility functions and visualization tools.
"""
from typing import Dict, List, Any, Optional
from pathlib import Path
import numpy as np
import torch
from PIL import Image 
from scipy.ndimage import zoom
import matplotlib.pyplot as plt # Added
import matplotlib.cm as cm # Added 
import matplotlib.colors as colors # Added 

# ----------------------------------------------------------------------
# --- Constants & Helpers ---
# ----------------------------------------------------------------------

def get_device():
    """Returns the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# ----------------------------------------------------------------------
# --- Labels & Metadata ---
# ----------------------------------------------------------------------

FALLBACK_LABEL_KEY = 'No_Dominant_Class' 

# BigEarthNet v1 Labels (19 classes or 43 classes depending on version)
# Here we define the standard 43 classes as a fallback.
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
    downscale_factor: int = 10,
    labels: Optional[List[str]] = None,
    color_map: Optional[Dict[str, Any]] = None
):
    """
    Generates a low-resolution color PNG preview of the classification mask.
    """
    if not save_preview:
        return

    # print(f"🎨 Generating low-res preview image (downscale={downscale_factor})...")

    if downscale_factor > 1:
        downscaled_mask = zoom(mask_data, 1.0 / downscale_factor, order=0)
    else:
        downscaled_mask = mask_data
    
    # Use provided labels/colormap or fallback to globals
    target_labels = labels if labels is not None else NEW_LABELS
    target_color_map = color_map if color_map is not None else LABEL_COLOR_MAP
    
    # Ensure color map keys exist in labels (robustness)
    # Construct array based on index in target_labels
    color_map_array_list = []
    for label in target_labels:
        if label in target_color_map:
             c = target_color_map[label]
             # Handle list vs numpy array
             if isinstance(c, list):
                 c = np.array(c, dtype=np.uint8)
             color_map_array_list.append(c)
        else:
             color_map_array_list.append(np.array([128, 128, 128], dtype=np.uint8)) # Gray fallback
             
    color_map_array = np.array(color_map_array_list, dtype=np.uint8)
    
    max_idx = len(target_labels) - 1
    safe_mask = np.clip(downscaled_mask, 0, max_idx)
    
    rgb_image = color_map_array[safe_mask]
    
    try:
        img_pil = Image.fromarray(rgb_image, 'RGB')
        img_pil.save(output_path, 'PNG')
        # print(f"  Saved preview to {output_path.name}")
    except Exception as e:
        print(f"❌ Error saving preview image: {e}")

def generate_float_preview(
    data: np.ndarray,
    output_path: Path,
    save_preview: bool = True,
    downscale_factor: int = 10,
    cmap_name: str = 'magma_r', # Default colormap for continuous data
    title: Optional[str] = None, # Not used for image but could be useful for debugging
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    colorbar_path: Optional[Path] = None # New parameter for color bar output
):
    """
    Generates a low-resolution color PNG preview of continuous float data, and optionally a color bar.
    """
    if not save_preview:
        return

    if downscale_factor > 1:
        downscaled_data = zoom(data, 1.0 / downscale_factor, order=1) # order=1 for bilinear interpolation
    else:
        downscaled_data = data
    
    # Normalize data
    if vmin is None:
        vmin = np.min(downscaled_data)
    if vmax is None:
        vmax = np.max(downscaled_data)

    if vmax == vmin: # Handle constant data
        normalized_data = np.zeros_like(downscaled_data)
    else:
        normalized_data = (downscaled_data - vmin) / (vmax - vmin)
    
    # Apply colormap
    cmap = cm.get_cmap(cmap_name)
    rgb_image = cmap(normalized_data)[:, :, :3] # Take RGB channels, discard alpha
    
    # Convert to 8-bit image
    rgb_image = (rgb_image * 255).astype(np.uint8)

    try:
        img_pil = Image.fromarray(rgb_image, 'RGB')
        img_pil.save(output_path, 'PNG')
    except Exception as e:
        print(f"❌ Error saving float preview image: {e}")

    # Generate color bar if requested
    if colorbar_path:
        try:
            fig, ax = plt.subplots(figsize=(1.0, 4.0)) # Adjust size as needed
            cmap_obj = cm.get_cmap(cmap_name)
            norm = colors.Normalize(vmin=vmin, vmax=vmax)
            cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap_obj), cax=ax, orientation='vertical')
            cb.set_label(title if title else "") # Use title for colorbar label
            
            plt.tight_layout()
            plt.savefig(colorbar_path, bbox_inches='tight', dpi=100)
            plt.close(fig) # Close the figure to free memory
        except Exception as e:
            print(f"❌ Error saving color bar image: {e}")
