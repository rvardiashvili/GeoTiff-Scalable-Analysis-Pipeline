import rasterio
import logging
from pathlib import Path
from typing import Dict, Any
from .base import BaseReporter

log = logging.getLogger(__name__)

class ProbabilityReporter(BaseReporter):
    """
    Reporter that writes the full probability distribution (all classes) to a multi-band GeoTIFF.
    Each band corresponds to a class index.
    """
    def __init__(self):
        self.dst = None

    def on_start(self, context: Dict[str, Any]):
        output_path = context['output_path']
        tile_name = context['tile_name']
        profile = context['profile']
        adapter = context.get('adapter')
        
        num_classes = adapter.num_classes if adapter else 1
        
        out_path = output_path / f"{tile_name}_probs.tif"
        
        # Update profile for multi-band float32
        p = profile.copy()
        p.update(
            dtype='float32',
            count=num_classes,
            nodata=None
        )
        
        if out_path.exists():
            try:
                out_path.unlink()
            except OSError as e:
                log.warning(f"Could not delete existing file {out_path}: {e}")
        
        try:
            self.dst = rasterio.open(out_path, 'w', **p)
            
            # Set Band Descriptions (Class Names) if available
            if adapter and adapter.labels:
                for i, label in enumerate(adapter.labels):
                    # rasterio bands are 1-indexed
                    self.dst.set_band_description(i + 1, label)
                    
        except Exception as e:
            log.error(f"Failed to initialize ProbabilityReporter: {e}")

    def on_chunk(self, data: Dict[str, Any]):
        if self.dst is None:
            return

        valid_probs = data['valid_probs'] # (C, H, W)
        window = data['window']
        
        # valid_probs is (C, H, W), rasterio expects (C, H, W) for multi-band write
        # Crop valid_probs to match the window dimensions
        cropped_valid_probs = valid_probs[:, :window.height, :window.width]
        try:
            self.dst.write(cropped_valid_probs, window=window)
        except Exception as e:
            log.error(f"ProbabilityReporter write error: {e}")

    def on_finish(self, context: Dict[str, Any]):
        if self.dst:
            self.dst.close()
