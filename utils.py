"""
Utility functions, placeholder logic for external libraries, and visualization tools.
"""
import base64
import io
from typing import Dict, List, Any

import torch
import numpy as np
import matplotlib.pyplot as plt
import rasterio

from config import DEVICE, PATCH_SIZE

# --- Placeholder Logic for Missing Libraries ---
# This section ensures the script can run even if the specific BENv2 libraries are not installed.
try:
    # Attempt to import the real libraries
    from configilm.extra.BENv2_utils import STANDARD_BANDS, NEW_LABELS, stack_and_interpolate, means, stds
    from BigEarthNetv2_0_ImageClassifier import BigEarthNetv2_0_ImageClassifier
    print("✅ Successfully loaded BENv2 libraries.")

    # Validate that the loaded libraries contain necessary data
    if 'STANDARD_BANDS' not in locals() or not STANDARD_BANDS:
        raise ImportError("BENv2 libraries loaded but are missing critical data (STANDARD_BANDS).")

except ImportError as e:
    print(f"⚠️ Warning: External BENv2 libraries not found or failed to load ({e}). Using robust placeholders.")
    
    # Define fallback data structures
    STANDARD_BANDS: Dict[int, List[str]] = {
        10: ['B02', 'B03', 'B04', 'B05', 'B06', 'O07', 'O08', 'O8A', 'B11', 'B12']
    }
    NEW_LABELS: List[str] = [f"Class_{i}" for i in range(43)]
    
    # Define fallback normalization constants
    means: Dict[str, Any] = {"120_nearest": {b: 0.0 for b in STANDARD_BANDS[10]}}
    stds: Dict[str, Any] = {"120_nearest": {b: 1.0 for b in STANDARD_BANDS[10]}}

    def stack_and_interpolate(data: Dict, order: List[str], img_size: int) -> torch.Tensor:
        """Placeholder function to create a zero tensor of the correct shape."""
        return torch.zeros(1, len(order), img_size, img_size)

    class MockModel(torch.nn.Module):
        """A mock model that mimics the input/output shape of the real model."""
        def __init__(self):
            super().__init__()
            num_bands = len(STANDARD_BANDS[10])
            num_labels = len(NEW_LABELS)
            self.linear = torch.nn.Linear(num_bands * PATCH_SIZE * PATCH_SIZE, num_labels)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear(x.flatten(1))

    class BigEarthNetv2_0_ImageClassifier:
        """A mock class to stand in for the real classifier."""
        @staticmethod
        def from_pretrained(repo_id: str) -> MockModel:
            print("Using Mock Model.")
            return MockModel()

# --- Normalization Tensor Calculation ---
def get_norm_tensors(bands: List[str]) -> (torch.Tensor, torch.Tensor):
    """Calculates and returns mean (m) and std dev (s) tensors on the appropriate device."""
    m = torch.tensor([means["120_nearest"][b] for b in bands], device=DEVICE).view(1, -1, 1, 1)
    s = torch.tensor([stds["120_nearest"][b] for b in bands], device=DEVICE).view(1, -1, 1, 1)
    return m, s

# --- Visualization Utility ---
def patch_to_base64_image(patch_tensor: torch.Tensor) -> str:
    """
    Converts a 10-band tensor into a Base64-encoded RGB image using bands 4, 3, 2.
    The input tensor is expected to be UN-NORMALIZED.
    """
    bands = STANDARD_BANDS[10]
    
    # Use CPU for normalization during visualization to avoid GPU<->CPU sync
    m_cpu = torch.tensor([means["120_nearest"][b] for b in bands]).view(1, -1, 1, 1)
    s_cpu = torch.tensor([stds["120_nearest"][b] for b in bands]).view(1, -1, 1, 1)
    
    # Normalize the patch for correct visualization
    normalized_patch = (patch_tensor.unsqueeze(0) - m_cpu) / (s_cpu + 1e-6)
    
    # RGB display uses B04 (Red, idx 2), B03 (Green, idx 1), B02 (Blue, idx 0)
    rgb_indices = [2, 1, 0]
    img_np = normalized_patch.squeeze(0)[rgb_indices, :, :].numpy().transpose(1, 2, 0)

    # Simple min-max scaling for visualization
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-6)
    img_np = np.clip(img_np, 0, 1)

    # Generate image using matplotlib
    fig, ax = plt.subplots(figsize=(1.5, 1.5))
    ax.imshow(img_np)
    ax.axis("off")
    
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    
    return base64.b64encode(buf.read()).decode("utf-8")

# --- Global Normalization Tensors ---
# Create normalization tensors once at startup for GPU offloading
NORM_M, NORM_S = get_norm_tensors(STANDARD_BANDS[10])
