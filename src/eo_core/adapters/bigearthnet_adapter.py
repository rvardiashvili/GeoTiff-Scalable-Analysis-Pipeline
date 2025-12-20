import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Any, Dict, Tuple, List
import timm
import os
from huggingface_hub import hf_hub_download
import safetensors.torch
from ..adapters.base import BaseAdapter
from ..data import _read_s2_bands_for_chunk, _read_s1_bands_for_chunk, cut_into_patches
from .wrappers import MetadataPassingWrapper
from ..memory_utils import estimate_optimal_batch_size
from omegaconf import OmegaConf

# Try importing ConfigILM for BIFOLD compatibility
try:
    from configilm import ConfigILM
    from configilm.ConfigILM import ILMConfiguration, ILMType
    HAS_CONFIGILM = True
except ImportError:
    HAS_CONFIGILM = False

class BigEarthNetAdapter(BaseAdapter):
    def build_model(self) -> nn.Module:
        # Params must contain model configuration and normalization stats
        model_conf = self.params.get('model_config')
        if not model_conf:
            raise ValueError("BigEarthNetAdapter requires 'model_config' in params.")
            
        # 1. Determine Model Architecture and Weights Path
        if hasattr(model_conf, 'pretrained_model_name_or_path'):
            model_id = model_conf.pretrained_model_name_or_path
        elif isinstance(model_conf, dict) and 'pretrained_model_name_or_path' in model_conf:
            model_id = model_conf['pretrained_model_name_or_path']
        else:
            model_id = self.params.get('model_name', 'resnet50')

        print(f"DEBUG: Resolved model_id='{model_id}'")

        # 2. Build Model
        num_classes = self.params.get('num_classes', 19)
        in_channels = self.num_bands
        
        core_model = None
        is_configilm = False

        # Check if it's a BIFOLD model and we have ConfigILM
        if HAS_CONFIGILM and "bifold" in str(model_id).lower():
            print(f"🏗️ Building model with ConfigILM: {model_id}")
            try:
                # Map to TIMM names expected by ConfigILM
                timm_name = "resnet50"
                if "resnet50" in model_id.lower(): timm_name = "resnet50"
                elif "convnext" in model_id.lower(): timm_name = "convnextv2_base"
                elif "resnet101" in model_id.lower(): timm_name = "resnet101"
                
                config = ILMConfiguration(
                    timm_model_name=timm_name,
                    classes=num_classes,
                    network_type=ILMType.IMAGE_CLASSIFICATION,
                    channels=in_channels
                )
                # Monkeypatch/Warning suppression if needed (referenced from legacy code)
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    core_model = ConfigILM.ConfigILM(config)
                is_configilm = True
            except Exception as e:
                print(f"⚠️ ConfigILM instantiation failed ({e}). Falling back to direct TIMM.")

        if core_model is None:
            # Fallback to pure TIMM
            timm_name = "resnet50"
            if "resnet50" in model_id.lower(): timm_name = "resnet50"
            elif "convnext" in model_id.lower(): timm_name = "convnextv2_base"
            elif "resnet101" in model_id.lower(): timm_name = "resnet101"
            
            print(f"🏗️ Building model with TIMM: {timm_name} (In={in_channels}, Out={num_classes})")
            core_model = timm.create_model(
                timm_name, 
                pretrained=False, 
                num_classes=num_classes, 
                in_chans=in_channels
            )
        
        # 3. Load Custom Weights (Local or HF Hub)
        try:
            checkpoint_path = None
            if os.path.exists(model_id):
                checkpoint_path = model_id
            elif "/" in model_id: # Likely HF Hub ID
                try:
                    checkpoint_path = hf_hub_download(repo_id=model_id, filename="pytorch_model.bin")
                except:
                    try:
                        checkpoint_path = hf_hub_download(repo_id=model_id, filename="model.safetensors")
                    except:
                        print(f"⚠️ Could not find weights file for {model_id} on Hub. Using random init.")
            
            if checkpoint_path:
                print(f"📥 Loading weights from: {checkpoint_path}")
                
                if str(checkpoint_path).endswith('.safetensors'):
                    state_dict = safetensors.torch.load_file(checkpoint_path, device='cpu')
                else:
                    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                
                # Key Logic for ConfigILM vs TIMM
                # BIFOLD checkpoints might have keys like 'model.encoder...' or just 'model...'
                # We need to inspect keys.
                
                # 1. Unwrap 'state_dict' or 'model' wrapper in checkpoint file
                if 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                elif 'model' in state_dict:
                    state_dict = state_dict['model']
                
                # 2. Adjust keys
                new_state_dict = {}
                for k, v in state_dict.items():
                    # If ConfigILM, the model expects keys to match ConfigILM structure
                    # If TIMM, we need to strip prefixes often
                    
                    # Common prefix stripping
                    if k.startswith('model.'):
                        # If we ARE using ConfigILM (which wraps the model in .model attribute maybe? 
                        # No, ConfigILM *is* the module usually, but let's check if it has a .model attribute)
                        # The legacy code showed `self.model = ConfigILM.ConfigILM(config)` inside the LightningModule.
                        # So the LightningModule had a `.model` attribute which WAS the ConfigILM instance.
                        # BUT the ConfigILM instance itself might have internal structure.
                        
                        if is_configilm:
                            # If using ConfigILM directly, we might NOT want to strip 'model.' 
                            # because the checkpoint came from the LightningModule which had `self.model = ConfigILM(...)`
                            # So `model.conv1` in checkpoint -> `conv1` in ConfigILM.
                            new_state_dict[k[6:]] = v
                        else:
                            # For TIMM, strip 'model.'
                            new_state_dict[k[6:]] = v
                    else:
                        new_state_dict[k] = v
                        
                # Load
                missing, unexpected = core_model.load_state_dict(new_state_dict, strict=False)
                if missing:
                    print(f"   Missing keys: {len(missing)} (e.g. {missing[:3]}...)")
                else:
                    print("✅ Weights loaded successfully.")
        except Exception as e:
            print(f"⚠️ Failed to load weights: {e}. Proceeding with random initialization.")

        # Get Device (Engine will move this wrapper to device, but we need to know it for batching)
        device_str = self.params.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device(device_str)
        
        # Normalization Params
        means = self.params.get('means')
        stds = self.params.get('stds')
        if means is None or stds is None:
            raise ValueError("BigEarthNetAdapter requires 'means' and 'stds' for normalization.")
            
        norm_m = torch.tensor(means, dtype=torch.float32).view(1, len(means), 1, 1)
        norm_s = torch.tensor(stds, dtype=torch.float32).view(1, len(stds), 1, 1)
        
        # Try 'batch_size' first (model config), then 'gpu_batch_size' (pipeline config/legacy)
        batch_size = self.params.get('batch_size')
        if batch_size is None:
             batch_size = self.params.get('gpu_batch_size', 32)
        
        if batch_size == "auto":
            input_shape = (self.num_bands, self.patch_size, self.patch_size)
            # core_model is the model to be wrapped
            batch_size = estimate_optimal_batch_size(core_model, input_shape, device)
            print(f"Auto-configured batch size: {batch_size}")
        
        # Wrap it with activation on GPU
        return MetadataPassingWrapper(core_model, batch_size, norm_m, norm_s, device, activation='sigmoid')

    def preprocess(self, raw_input: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        raw_input: {
            'tile_folder': Path, 
            'r_start': int, 'c_start': int, 
            'w_read': int, 'h_read': int,
            'bands': List[str]
        }
        """
        tile_folder = Path(raw_input['tile_folder'])
        r = raw_input['r_start']
        c = raw_input['c_start']
        w = raw_input['w_read']
        h = raw_input['h_read']
        bands = raw_input['bands']
        
        # --- Logic adapted from process.py / ERFAwareInference.infer_tile_region ---
        
        use_s1 = any(b in ['VV', 'VH'] for b in bands)
        
        # S2 Data
        s2_bands = [b for b in bands if 'B' in b]
        s2_pattern = self.params.get('s2_file_pattern', "S2*.SAFE/**/*{band_name}*.jp2")
        s2_data, s2_crs, s2_transform, s2_size = _read_s2_bands_for_chunk(
            tile_folder, r, c, w, h, s2_base_pattern=s2_pattern, pad_if_needed=True, bands_list=s2_bands
        )
        
        # S1 Data
        if use_s1:
            s1_bands = [b for b in bands if b in ['VV', 'VH']]
            s1_pattern = self.params.get('s1_file_pattern', "S1*.SAFE/measurement/*{band_name}*.tiff")
            s1_data, _, _ = _read_s1_bands_for_chunk(
                tile_folder, r, c, w, h, 
                s1_pattern=s1_pattern,
                pad_if_needed=True, 
                bands_list=s1_bands,
                ref_crs=s2_crs,
                ref_transform=s2_transform,
                ref_size=s2_size
            )
            
            if s1_data.size > 0:
                # Clip S1
                s1_data = np.clip(s1_data, -50.0, 30.0) 
            
            # Concatenate S1 FIRST (if that's the convention in config)
            # Check if s1_data is empty (0 shape)
            if s1_data.size == 0:
                 input_data = s2_data
            else:
                 input_data = np.concatenate([s1_data, s2_data], axis=0)
        else:
            input_data = s2_data

        # Cut into patches
        patch_size = self.params.get('patch_size', 120)
        stride = self.params.get('stride', patch_size // 2)
        
        patches, coords, H_crop, W_crop, _ = cut_into_patches(input_data, patch_size, stride=stride)
        
        # patches is (N, C, P, P) numpy array
        # We return it as numpy so InferenceEngine doesn't auto-move it to GPU
        
        metadata = {
            'coords': coords, 
            'H_crop': H_crop, 
            'W_crop': W_crop,
            'original_r': r,
            'original_c': c
        }
        
        return patches, metadata

    def postprocess(self, model_output: Tuple[torch.Tensor, Dict[str, Any]]) -> Dict[str, Any]:
        probs, metadata = model_output
        
        # Sigmoid is now done on GPU in Wrapper
        
        return {
            'probs_tensor': probs, 
            'coords': metadata['coords'],
            'H_crop': metadata['H_crop'],
            'W_crop': metadata['W_crop'],
            'r_chunk': metadata['original_r'],
            'c_chunk': metadata['original_c']
        }

    @property
    def num_classes(self) -> int:
        return self.params.get('num_classes', 19)

    @property
    def num_bands(self) -> int:
        if 'bands' in self.params:
            return len(self.params['bands'])
        elif 'means' in self.params:
            return len(self.params['means'])
        else:
            return 12 # Default S2

    @property
    def patch_size(self) -> int:
        return self.params.get('patch_size', 120)

    @property
    def stride(self) -> int:
        return self.params.get('stride', 60)

    @property
    def labels(self) -> List[str]:
        return [
            'Agro-forestry areas', 
            'Arable land', 
            'Beaches, dunes, sands', 
            'Broad-leaved forest', 
            'Coastal wetlands', 
            'Complex cultivation patterns', 
            'Coniferous forest', 
            'Industrial or commercial units', 
            'Inland waters', 
            'Inland wetlands', 
            'Land principally occupied by agriculture, with significant areas of natural vegetation', 
            'Marine waters', 
            'Mixed forest', 
            'Moors, heathland and sclerophyllous vegetation', 
            'Natural grassland and sparsely vegetated areas', 
            'Pastures', 
            'Permanent crops', 
            'Transitional woodland, shrub', 
            'Urban fabric'
        ]

    @property
    def color_map(self) -> Dict[str, List[int]]:
        return {
            'Agro-forestry areas': [242, 166, 77],
            'Arable land': [255, 255, 168],
            'Beaches, dunes, sands': [230, 230, 230],
            'Broad-leaved forest': [128, 255, 0],
            'Coastal wetlands': [128, 204, 204],
            'Complex cultivation patterns': [255, 170, 0],
            'Coniferous forest': [0, 166, 0],
            'Industrial or commercial units': [204, 77, 243],
            'Inland waters': [0, 204, 242],
            'Inland wetlands': [204, 204, 204],
            'Land principally occupied by agriculture, with significant areas of natural vegetation': [230, 128, 0],
            'Marine waters': [0, 0, 255],
            'Mixed forest': [77, 255, 77],
            'Moors, heathland and sclerophyllous vegetation': [166, 230, 77],
            'Natural grassland and sparsely vegetated areas': [204, 242, 77],
            'Pastures': [230, 230, 77],
            'Permanent crops': [255, 255, 0],
            'Transitional woodland, shrub': [166, 242, 0],
            'Urban fabric': [255, 0, 0]
        }
