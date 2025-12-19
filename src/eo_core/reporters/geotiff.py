import rasterio
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, List
from .base import BaseReporter

log = logging.getLogger(__name__)

class GeoTIFFReporter(BaseReporter):
    """
    Reporter that writes inference results (Class map, Confidence, Entropy, etc.) to GeoTIFF files.
    """
    def __init__(self):
        self.dsts = {}
        self.out_paths = {}
        self.save_conf = False
        self.save_entr = False
        self.save_gap = False
        self.save_gradient = False
        self.n_classes = 1

    def on_start(self, context: Dict[str, Any]):
        output_path = context['output_path']
        tile_name = context['tile_name']
        profile = context['profile']
        config = context.get('config', {}) # Hydra config
        
        # Determine what to save from config
        output_cfg = config.get('pipeline', {}).get('output', {})
        self.save_conf = output_cfg.get('save_confidence', True)
        self.save_entr = output_cfg.get('save_entropy', True)
        self.save_gap = output_cfg.get('save_gap', True)
        self.save_gradient = output_cfg.get('save_gradient_preview', False)
        
        # Determine number of classes (needed for gradient/binary check)
        # Ideally this comes from adapter, but we can infer or it's passed in context
        adapter = context.get('adapter')
        self.n_classes = adapter.num_classes if adapter else 1
        
        # Prepare paths
        self.out_paths['class'] = output_path / f"{tile_name}_class.tif"
        if self.save_conf:
            self.out_paths['conf'] = output_path / f"{tile_name}_maxprob.tif"
        if self.save_entr:
            self.out_paths['entr'] = output_path / f"{tile_name}_entropy.tif"
        if self.save_gap:
            self.out_paths['gap'] = output_path / f"{tile_name}_gap.tif"
        if self.save_gradient and self.n_classes == 2:
             self.out_paths['gradient'] = output_path / f"{tile_name}_gradient.tif"

        # Open files
        for key, path in self.out_paths.items():
            p = profile.copy()
            if key == 'class':
                p.update(dtype='uint8', nodata=255)
            else:
                p.update(dtype='float32', nodata=None)
            
            if path.exists():
                try:
                    path.unlink()
                except OSError as e:
                    log.warning(f"Could not delete existing file {path}: {e}")
            
            self.dsts[key] = rasterio.open(path, 'w', **p)

    def on_chunk(self, data: Dict[str, Any]):
        valid_probs = data['valid_probs'] # (C, H, W)
        window = data['window']
        
        # Calculate Metrics
        # 1. Dominant Class (Argmax)
        dom = np.argmax(valid_probs, axis=0).astype(np.uint8)
        self.dsts['class'].write(dom[:window.height, :window.width], window=window, indexes=1)
        
        # 2. Confidence (Max Prob)
        if 'conf' in self.dsts:
            conf = np.max(valid_probs, axis=0).astype(np.float32)
            self.dsts['conf'].write(conf[:window.height, :window.width], window=window, indexes=1)
            
        # 3. Entropy
        if 'entr' in self.dsts:
            # -sum(p * log(p))
            entr = -np.sum(valid_probs * np.log(np.clip(valid_probs, 1e-6, 1.0)), axis=0).astype(np.float32)
            self.dsts['entr'].write(entr[:window.height, :window.width], window=window, indexes=1)
            
        # 4. Prediction Gap (Top1 - Top2)
        if 'gap' in self.dsts:
            if valid_probs.shape[0] >= 2:
                top2 = np.partition(valid_probs, -2, axis=0)[-2:]
                gap = (top2[1] - top2[0]).astype(np.float32)
                self.dsts['gap'].write(gap[:window.height, :window.width], window=window, indexes=1)
            else:
                # Fallback for binary or single channel if shaped (1, H, W) ?? 
                # Usually binary is (2, H, W) here.
                pass

        # 5. Gradient (Class 1 Prob for Binary)
        if 'gradient' in self.dsts and self.n_classes == 2:
            gradient = valid_probs[1].astype(np.float32)
            self.dsts['gradient'].write(gradient[:window.height, :window.width], window=window, indexes=1)

    def on_finish(self, context: Dict[str, Any]):
        for dst in self.dsts.values():
            dst.close()
