import logging
log = logging.getLogger(__name__)

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Any, Dict, Tuple, List
import sys
import time # Import time for timing measurements
from huggingface_hub import hf_hub_download

# Add root to sys.path to import prithvi_mae
root_path = str(Path(__file__).parents[3])
if root_path not in sys.path:
    sys.path.append(root_path)

try:
    from ..adapters.prithvi_mae import PrithviViT
except ImportError:
    # Fallback or error if not found
    PrithviViT = None

from ..adapters.base import BaseAdapter
from ..adapters.wrappers import MetadataPassingWrapper
from ..data import _read_s2_bands_for_chunk, cut_into_patches
from ..memory_utils import estimate_optimal_batch_size

class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activate = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activate(x)
        return x

class CustomFCNHead(nn.Module):
    def __init__(self, in_channels, channels, num_classes):
        super().__init__()
        self.convs = nn.ModuleList([
            ConvModule(in_channels, channels, 3, 1)
        ])
        self.dropout = nn.Dropout(0.1)
        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward(self, x):
        # x is list of tensors, take last
        if isinstance(x, (list, tuple)):
            x = x[-1]
        x = self.convs[0](x)
        x = self.dropout(x)
        x = self.conv_seg(x)
        return x

class PrithviSegmentor(nn.Module):
    def __init__(self, backbone, decode_head):
        super().__init__()
        self.backbone = backbone
        self.decode_head = decode_head

    def forward(self, x):
        features = self.backbone.forward_features(x)
        features = self.backbone.prepare_features_for_image_model(features)
        out = self.decode_head(features)
        return out

class PrithviAdapter(BaseAdapter):
    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        if PrithviViT is None:
            raise ImportError("Could not import PrithviViT from prithvi_mae.py. Make sure it is in the project root.")

    def build_model(self) -> nn.Module:
        t_start = time.perf_counter()
        model_id = self.params.get('model_name_or_path', "ibm-nasa-geospatial/Prithvi-EO-1.0-100M-sen1floods11")
        filename = self.params.get('model_filename', "sen1floods11_Prithvi_100M.pth")

        log.debug(f"[{time.perf_counter()-t_start:.4f}s] Starting weight download...")
        local_checkpoint_path = self.params.get('local_checkpoint_path')
        if local_checkpoint_path:
            checkpoint_path = local_checkpoint_path
            if not Path(checkpoint_path).exists():
                raise FileNotFoundError(
                    f"Local checkpoint path '{checkpoint_path}' specified in config does not exist."
                )
            log.info(f"[{time.perf_counter()-t_start:.4f}s] Using local checkpoint from: {checkpoint_path}")
        else:
            try:
                checkpoint_path = hf_hub_download(repo_id=model_id, filename=filename)
                log.info(f"[{time.perf_counter()-t_start:.4f}s] Downloaded checkpoint to: {checkpoint_path}")
            except Exception as e:
                raise FileNotFoundError(
                    f"Could not download model checkpoint '{filename}' from Hugging Face repo '{model_id}'. "
                    f"Please check your internet connection or provide a 'local_checkpoint_path' in your config. "
                    f"Original error: {e}"
                )

        log.debug(f"[{time.perf_counter()-t_start:.4f}s] Building backbone...")
        # Build Backbone
        backbone_params = self.params.get('backbone_params', {})
        backbone = PrithviViT(
            img_size=backbone_params.get('img_size', 224),
            patch_size=backbone_params.get('patch_size', (1, 16, 16)),
            num_frames=backbone_params.get('num_frames', 1),
            in_chans=backbone_params.get('in_chans', self.num_bands),
            embed_dim=backbone_params.get('embed_dim', 768),
            depth=backbone_params.get('depth', 12),
            num_heads=backbone_params.get('num_heads', 12),
            mlp_ratio=backbone_params.get('mlp_ratio', 4.0),
        )

        log.debug(f"[{time.perf_counter()-t_start:.4f}s] Building head...")
        # Build Head
        head_params = self.params.get('head_params', {})
        head = CustomFCNHead(
            in_channels=head_params.get('in_channels', 768),
            channels=head_params.get('channels', 256),
            num_classes=head_params.get('num_classes', 2)
        )

        model = PrithviSegmentor(backbone, head)

        # Load weights
        device_str = self.params.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device(device_str)

        log.info(f"[{time.perf_counter()-t_start:.4f}s] Loading weights from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        # Filter strict=False for auxiliary head or other keys we implemented differently
        # Our CustomFCNHead should match 'decode_head' keys.
        # Backbone matches 'backbone' keys.

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            # It is expected to miss auxiliary_head
            log.debug(f"[{time.perf_counter()-t_start:.4f}s] Missing keys (expected if aux head missing): {len(missing)}")
            # print(missing[:5])

        model.to(device)

        means = self.params.get('means', [0.0] * 6)
        stds = self.params.get('stds', [1.0] * 6)

        norm_m = torch.tensor(means, dtype=torch.float32).view(len(means), 1, 1)
        norm_s = torch.tensor(stds, dtype=torch.float32).view(len(stds), 1, 1)

        batch_size = self.params.get('batch_size', 8)

        if batch_size == "auto":
            input_shape = (self.num_bands, self.patch_size, self.patch_size)
            batch_size = estimate_optimal_batch_size(model, input_shape, device)
            log.info(f"Auto-configured batch size: {batch_size}")
            print(f"Auto-configured batch size: {batch_size}")

        log.debug(f"[{time.perf_counter()-t_start:.4f}s] Model built and weights loaded.")
        return MetadataPassingWrapper(model, batch_size, norm_m, norm_s, device, activation='softmax')

    def preprocess(self, raw_input: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        t_start = time.perf_counter()
        tile_folder = Path(raw_input['tile_folder'])
        r = raw_input['r_start']
        c = raw_input['c_start']
        w = raw_input['w_read']
        h = raw_input['h_read']

        # Prithvi Bands: Blue, Green, Red, Narrow NIR (B8A), SWIR 1, SWIR 2
        bands_needed = self.params.get('bands', ['B02', 'B03', 'B04', 'B8A', 'B11', 'B12'])

        # Read Data
        s2_pattern = self.params.get('s2_file_pattern', "S2*.SAFE/**/*{band_name}*.jp2")
        s2_data, s2_crs, s2_transform, s2_size = _read_s2_bands_for_chunk(
            tile_folder, r, c, w, h, s2_pattern=s2_pattern, pad_if_needed=True, bands_list=bands_needed
        )

        # Check range. If max > 100, likely 0-10000.
        if s2_data.max() > 100:
             s2_data = s2_data.astype(np.float32) / 10000.0

        # Cut into patches
        patch_size = self.params.get('patch_size', 224)
        stride = self.params.get('stride', 112)

        patches, coords, H_crop, W_crop, _ = cut_into_patches(s2_data, patch_size, stride=stride)

        metadata = {
            'coords': coords,
            'H_crop': H_crop,
            'W_crop': W_crop,
            'original_r': r,
            'original_c': c,
            'shape': s2_data.shape
        }
        log.debug(f"[{time.perf_counter()-t_start:.4f}s] PrithviAdapter preprocess finished.")
        return patches, metadata

    def postprocess(self, model_output: Tuple[torch.Tensor, Dict[str, Any]]) -> Dict[str, Any]:
        t_start = time.perf_counter()
        probs_tensor, metadata = model_output

        # Upsample if needed to match patch size
        patch_size = self.params.get('patch_size', 224)

        if probs_tensor.shape[-1] != patch_size:
             probs_tensor = nn.functional.interpolate(
                 probs_tensor,
                 size=(patch_size, patch_size),
                 mode='bilinear',
                 align_corners=False
             )
        log.debug(f"[{time.perf_counter()-t_start:.4f}s] PrithviAdapter postprocess finished.")
        return {
            'probs_tensor': probs_tensor,
            'coords': metadata['coords'],
            'H_crop': metadata['H_crop'],
            'W_crop': metadata['W_crop'],
            'r_chunk': metadata['original_r'],
            'c_chunk': metadata['original_c']
        }

    @property
    def num_classes(self) -> int:
        return 2

    @property
    def num_bands(self) -> int:
        return len(self.params.get('bands', ['B02', 'B03', 'B04', 'B8A', 'B11', 'B12']))

    @property
    def patch_size(self) -> int:
        return self.params.get('patch_size', 224)

    @property
    def stride(self) -> int:
        return self.params.get('stride', 112)

    @property
    def is_segmentation(self) -> bool:
        return True

    @property
    def labels(self) -> List[str]:
        return ["Non-Flood", "Flood"]

    @property
    def color_map(self) -> Dict[str, Any]:
        return {
            "Non-Flood": [0, 0, 0],         # Black or dark color for non-flood
            "Flood": [0, 0, 255]            # Blue for flood
        }
