import logging
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Any, Dict, Tuple, List
import sys
import time
from huggingface_hub import hf_hub_download
from functools import partial

from ..adapters.base import BaseAdapter
from ..adapters.wrappers import MetadataPassingWrapper
from ..data import _read_s2_bands_for_chunk, cut_into_patches
from ..memory_utils import estimate_optimal_batch_size

log = logging.getLogger(__name__)

# --- 1. Internal Model Definition (Self-Contained) ---

class PatchEmbed(nn.Module):
    """ 3D Image to Patch Embedding for Prithvi (Spatio-Temporal) """
    def __init__(self, img_size=224, patch_size=16, in_chans=6, embed_dim=768, num_frames=1, tubelet_size=1):
        super().__init__()
        if isinstance(img_size, int): img_size = (img_size, img_size)
        if isinstance(patch_size, int): patch_size = (patch_size, patch_size)
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        
        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) * (num_frames // tubelet_size)
        
        self.proj = nn.Conv3d(
            in_chans, embed_dim, 
            kernel_size=(tubelet_size, patch_size[0], patch_size[1]), 
            stride=(tubelet_size, patch_size[0], patch_size[1])
        )

    def forward(self, x):
        # x: (B, C, H, W) -> need (B, C, T, H, W)
        if x.ndim == 4:
            x = x.unsqueeze(2)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class PrithviViT(nn.Module):
    """ Minimal implementation of Prithvi ViT Backbone """
    def __init__(self, img_size=224, patch_size=16, in_chans=6, embed_dim=768, depth=12, num_heads=12, 
                 mlp_ratio=4., num_frames=1, **kwargs):
        super().__init__()
        
        # Handle tuple patch size [1, 16, 16]
        if isinstance(patch_size, (list, tuple)) and len(patch_size) == 3:
            tubelet_size = patch_size[0]
            patch_size_2d = (patch_size[1], patch_size[2])
        else:
            tubelet_size = 1
            patch_size_2d = patch_size

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size_2d, in_chans=in_chans,
            embed_dim=embed_dim, num_frames=num_frames, tubelet_size=tubelet_size
        )
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        
        # Init weights
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed[:, :x.shape[1], :] # Simple slicing for pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    def prepare_features_for_image_model(self, x):
        x = x[:, 1:, :] # Remove CLS
        B, N, C = x.shape
        
        # Calculate spatial dimensions based on config
        # We assume square image and patches for now
        if isinstance(self.patch_embed.patch_size, (tuple, list)):
             p_h, p_w = self.patch_embed.patch_size[-2:]
        else:
             p_h = p_w = self.patch_embed.patch_size
             
        if isinstance(self.patch_embed.img_size, (tuple, list)):
             img_h, img_w = self.patch_embed.img_size
        else:
             img_h = img_w = self.patch_embed.img_size
             
        H_grid = img_h // p_h
        W_grid = img_w // p_w
        num_spatial_tokens = H_grid * W_grid
        
        # Check if we have temporal dimension
        if N > num_spatial_tokens:
            # Assume N = T * num_spatial_tokens
            T = N // num_spatial_tokens
            if N % num_spatial_tokens != 0:
                 # Fallback if shapes don't align perfectly (e.g. padding?)
                 # Use old method or error
                 log.warning(f"Token count {N} is not divisible by spatial grid {H_grid}x{W_grid}={num_spatial_tokens}. Resizing best effort.")
                 H = W = int(N**0.5)
                 x = x.permute(0, 2, 1).reshape(B, C, H, W)
                 return x
            
            # Reshape to (B, T, H*W, C)
            x = x.reshape(B, T, num_spatial_tokens, C)
            
            # Fuse Temporal Dimension: Average Pooling
            # Since inputs are replicated, Mean is safe.
            x = x.mean(dim=1) # (B, H*W, C)
            
        # Reshape to Image (B, C, H, W)
        x = x.permute(0, 2, 1).reshape(B, C, H_grid, W_grid)
        return x

class CustomFCNHead(nn.Module):
    def __init__(self, in_channels, channels, num_classes):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.dropout = nn.Dropout(0.1)
        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
        
    def forward(self, x):
        x = self.convs(x)
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

# --- 2. The Adapter Class ---

class PrithviAdapter(BaseAdapter):
    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
            
    def build_model(self) -> nn.Module:
        t_start = time.perf_counter()
        model_id = self.params.get('model_name_or_path', "ibm-nasa-geospatial/Prithvi-EO-1.0-100M-sen1floods11")
        filename = self.params.get('model_filename', "sen1floods11_Prithvi_100M.pth")
        
        # 1. Download/Load Weights
        local_checkpoint_path = self.params.get('local_checkpoint_path')
        if local_checkpoint_path and Path(local_checkpoint_path).exists():
            checkpoint_path = local_checkpoint_path
            log.info(f"Using local checkpoint: {checkpoint_path}")
        else:
            try:
                checkpoint_path = hf_hub_download(repo_id=model_id, filename=filename)
                log.info(f"Downloaded checkpoint to: {checkpoint_path}")
            except Exception as e:
                raise FileNotFoundError(f"Failed to download {filename} from {model_id}: {e}")
            
        # 2. Build Model Structure
        backbone_params = self.params.get('backbone_params', {})
        backbone = PrithviViT(
            img_size=backbone_params.get('img_size', 224),
            patch_size=backbone_params.get('patch_size', [1, 16, 16]),
            num_frames=backbone_params.get('num_frames', 1),
            in_chans=backbone_params.get('in_chans', self.num_bands),
            embed_dim=backbone_params.get('embed_dim', 768),
            depth=backbone_params.get('depth', 12),
            num_heads=backbone_params.get('num_heads', 12),
            mlp_ratio=backbone_params.get('mlp_ratio', 4.0),
        )
        
        head_params = self.params.get('head_params', {})
        head = CustomFCNHead(
            in_channels=head_params.get('in_channels', 768),
            channels=head_params.get('channels', 256),
            num_classes=head_params.get('num_classes', 2) # Will be 14 from config
        )
        
        model = PrithviSegmentor(backbone, head)
        
        # 3. Load State Dict
        device_str = self.params.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device(device_str)
        
        state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
            
        # --- Positional Embedding Resizing Logic ---
        pos_embed_key = 'backbone.pos_embed'
        if pos_embed_key in state_dict:
            ckpt_pos_embed = state_dict[pos_embed_key]
            model_pos_embed = model.backbone.pos_embed
            
            if ckpt_pos_embed.shape != model_pos_embed.shape:
                log.warning(f"Resizing positional embeddings from {ckpt_pos_embed.shape} to {model_pos_embed.shape}")
                # Assuming shape is (1, N_ckpt, D)
                # We simply slice the first N_model tokens. 
                # This works for T=3 -> T=1 if the first segment corresponds to T=0.
                
                # Check if it is just a length mismatch
                if ckpt_pos_embed.shape[2] == model_pos_embed.shape[2]:
                    # Slice
                    state_dict[pos_embed_key] = ckpt_pos_embed[:, :model_pos_embed.shape[1], :]
                else:
                    log.error("Embedding dimension mismatch! Cannot resize.")

        # Flexible Loading (Strict=False allows ignoring aux heads if they exist in weights but not our model)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            critical_missing = [k for k in missing if "patch_embed" in k or "backbone" in k]
            if critical_missing:
                log.warning(f"CRITICAL: Missing keys during load: {len(critical_missing)}. Examples: {critical_missing[:5]}")
                log.warning("If patch_embed is missing, the model will output random garbage (one color).")
            else:
                log.debug(f"Missing keys during load (expected for aux heads): {len(missing)}")
        
        model.to(device)
        
        # 4. Prepare Wrapper
        means = self.params.get('means', [0.0] * 6)
        stds = self.params.get('stds', [1.0] * 6)
        
        # Adjust normalization shape for Temporal (5D) vs Static (4D)
        # 5D: (B, C, T, H, W) -> Norm shape (C, 1, 1, 1)
        # 4D: (B, C, H, W)    -> Norm shape (C, 1, 1)
        
        num_frames = backbone_params.get('num_frames', 1)
        if num_frames > 1:
             norm_shape = (len(means), 1, 1, 1)
        else:
             norm_shape = (len(means), 1, 1)

        norm_m = torch.tensor(means, dtype=torch.float32).view(norm_shape)
        norm_s = torch.tensor(stds, dtype=torch.float32).view(norm_shape)
        
        batch_size = self.params.get('batch_size', 8)
        if batch_size == "auto":
            if num_frames > 1:
                input_shape = (self.num_bands, num_frames, self.patch_size, self.patch_size)
            else:
                input_shape = (self.num_bands, self.patch_size, self.patch_size)
                
            batch_size = estimate_optimal_batch_size(model, input_shape, device)
            print(f"Auto-configured batch size: {batch_size}")

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
        s2_pattern = self.params.get('s2_file_pattern', "S2*.SAFE/**/*{band_name}*.jp2")
        
        s2_data, _, _, _ = _read_s2_bands_for_chunk(
            tile_folder, r, c, w, h, s2_pattern=s2_pattern, pad_if_needed=True, bands_list=bands_needed
        )
        
        if s2_data.max() > 100:
             s2_data = s2_data.astype(np.float32) / 10000.0
        
        patches, coords, H_crop, W_crop, _ = cut_into_patches(s2_data, self.patch_size, stride=self.stride)
        
        # Handle Temporal Replication
        backbone_params = self.params.get('backbone_params', {})
        num_frames = backbone_params.get('num_frames', 1)
        
        if num_frames > 1:
            # patches is (N, C, H, W)
            # expand to (N, C, T, H, W)
            patches = np.expand_dims(patches, axis=2) # (N, C, 1, H, W)
            patches = np.repeat(patches, num_frames, axis=2) # (N, C, T, H, W)
        
        metadata = {
            'coords': coords, 'H_crop': H_crop, 'W_crop': W_crop,
            'original_r': r, 'original_c': c, 'shape': s2_data.shape
        }
        return patches, metadata

    def postprocess(self, model_output: Tuple[torch.Tensor, Dict[str, Any]]) -> Dict[str, Any]:
        probs_tensor, metadata = model_output
        
        # Resize output to match patch size if model outputs smaller feature map
        if probs_tensor.shape[-1] != self.patch_size:
             probs_tensor = nn.functional.interpolate(
                 probs_tensor, 
                 size=(self.patch_size, self.patch_size), 
                 mode='bilinear', 
                 align_corners=False
             )
        
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
        # distinct from head_params, or derived from it
        head_params = self.params.get('head_params', {})
        return head_params.get('num_classes', 2)

    @property
    def num_bands(self) -> int:
        return len(self.params.get('bands', ['B02', 'B03', 'B04', 'B8A', 'B11', 'B12']))

    @property
    def patch_size(self) -> int: return self.params.get('patch_size', 224)

    @property
    def stride(self) -> int: return self.params.get('stride', 112)

    @property
    def is_segmentation(self) -> bool: return True

    @property
    def labels(self) -> List[str]: 
        return self.params.get('labels', [f"Class_{i}" for i in range(self.num_classes)])

    @property
    def color_map(self) -> Dict[str, Any]:
        return self.params.get('color_map', {label: [np.random.randint(0,255) for _ in range(3)] for label in self.labels})