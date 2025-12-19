import rasterio
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any
from .base import BaseReporter
from ..utils import generate_low_res_preview, generate_float_preview, NEW_LABELS, LABEL_COLOR_MAP

log = logging.getLogger(__name__)

class PreviewReporter(BaseReporter):
    """
    Reporter that generates a low-resolution PNG preview of the classification map.
    """
    def on_start(self, context: Dict[str, Any]):
        pass # Nothing to setup

    def on_chunk(self, data: Dict[str, Any]):
        pass # We don't process chunks incrementally yet (simplification)

    def on_finish(self, context: Dict[str, Any]):
        output_path = context['output_path']
        tile_name = context['tile_name']
        config = context.get('config', {})
        adapter = context.get('adapter')

        output_cfg = config.get('pipeline', {}).get('output', {})
        save_preview = output_cfg.get('save_preview', True)
        
        if not save_preview:
            return

        downscale = output_cfg.get("preview_downscale_factor", 10)
        class_path = output_path / f"{tile_name}_class.tif"
        
        # Determine labels/colormap
        if adapter and adapter.labels:
            labels = adapter.labels
            color_map = adapter.color_map
        else:
            labels = NEW_LABELS
            color_map = LABEL_COLOR_MAP

        # 1. Generate Class Map Preview
        if class_path.exists():
            try:
                log.info(f"Generating class map preview for {class_path}...")
                with rasterio.open(class_path) as src:
                    generate_low_res_preview(
                        src.read(1), 
                        output_path / "preview_class.png", # Renamed to specify type
                        save_preview=save_preview, 
                        downscale_factor=downscale,
                        labels=labels,
                        color_map=color_map
                    )
                log.info(f"Class map preview generated at {output_path / 'preview_class.png'}")
            except Exception as e:
                log.error(f"Failed to generate class map preview: {e}")
        else:
            log.warning(f"Class map file not found at {class_path}. Skipping class map preview generation.")

        # 2. Generate Previews for continuous metrics (Entropy, MaxProb, Gap)
        save_conf = output_cfg.get('save_confidence', True)
        save_entr = output_cfg.get('save_entropy', True)
        save_gap = output_cfg.get('save_gap', True)
        
        # MaxProb Preview
        if save_conf:
            maxprob_path = output_path / f"{tile_name}_maxprob.tif"
            if maxprob_path.exists():
                try:
                    log.info(f"Generating max probability preview for {maxprob_path}...")
                    with rasterio.open(maxprob_path) as src:
                        generate_float_preview(
                            src.read(1),
                            output_path / "preview_maxprob.png",
                            save_preview=save_preview,
                            downscale_factor=downscale,
                            cmap_name='viridis',
                            vmin=0.0,
                            vmax=1.0,
                            colorbar_path=output_path / "preview_maxprob_colorbar.png"
                        )
                    log.info(f"Max probability preview generated at {output_path / 'preview_maxprob.png'}")
                except Exception as e:
                    log.error(f"Failed to generate max probability preview: {e}")
            else:
                log.warning(f"Max probability file not found at {maxprob_path}. Skipping max probability preview generation.")

        # Entropy Preview
        if save_entr:
            entropy_path = output_path / f"{tile_name}_entropy.tif"
            if entropy_path.exists():
                try:
                    log.info(f"Generating entropy preview for {entropy_path}...")
                    with rasterio.open(entropy_path) as src:
                        # Assuming entropy is generally in a range like 0 to log(num_classes)
                        # For now, let's set vmax based on max possible entropy if adapter is available
                        max_entropy_val = np.log(adapter.num_classes) if adapter and adapter.num_classes > 1 else 1.0 # Default for 2 classes
                        
                        generate_float_preview(
                            src.read(1),
                            output_path / "preview_entropy.png",
                            save_preview=save_preview,
                            downscale_factor=downscale,
                            cmap_name='magma',
                            vmin=0.0,
                            vmax=max_entropy_val,
                            colorbar_path=output_path / "preview_entropy_colorbar.png"
                        )
                    log.info(f"Entropy preview generated at {output_path / 'preview_entropy.png'}")
                except Exception as e:
                    log.error(f"Failed to generate entropy preview: {e}")
            else:
                log.warning(f"Entropy file not found at {entropy_path}. Skipping entropy preview generation.")

        # Prediction Gap Preview
        if save_gap:
            gap_path = output_path / f"{tile_name}_gap.tif"
            if gap_path.exists():
                try:
                    log.info(f"Generating prediction gap preview for {gap_path}...")
                    with rasterio.open(gap_path) as src:
                        generate_float_preview(
                            src.read(1),
                            output_path / "preview_gap.png",
                            save_preview=save_preview,
                            downscale_factor=downscale,
                            cmap_name='plasma', # Or 'viridis_r'
                            vmin=0.0,
                            vmax=1.0, # Gap is also 0 to 1
                            colorbar_path=output_path / "preview_gap_colorbar.png"
                        )
                    log.info(f"Prediction gap preview generated at {output_path / 'preview_gap.png'}")
                except Exception as e:
                    log.error(f"Failed to generate prediction gap preview: {e}")
            else:
                log.warning(f"Prediction gap file not found at {gap_path}. Skipping prediction gap preview generation.")
