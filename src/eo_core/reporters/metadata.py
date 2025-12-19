import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any
from .base import BaseReporter
from ..utils import NEW_LABELS, LABEL_COLOR_MAP
from ..generate_viewer import generate_single_node_viewer

log = logging.getLogger(__name__)

class MetadataReporter(BaseReporter):
    """
    Reporter that generates metadata files (classmap.json) and viewer HTML.
    """
    def on_start(self, context: Dict[str, Any]):
        pass

    def on_chunk(self, data: Dict[str, Any]):
        pass

    def on_finish(self, context: Dict[str, Any]):
        output_path = context['output_path']
        tile_name = context['tile_name']
        adapter = context.get('adapter')
        
        # 1. Generate Class Map JSON
        if adapter and adapter.labels:
            labels = adapter.labels
            color_map = adapter.color_map
        else:
            labels = NEW_LABELS
            color_map = LABEL_COLOR_MAP

        class_map = {
            label: {
                "index": i, 
                "color": color_map.get(label, [128,128,128]) if isinstance(color_map.get(label), list) else color_map.get(label, np.array([128,128,128])).tolist()
            } 
            for i, label in enumerate(labels)
        }
        
        try:
            with open(output_path / f"{tile_name}_classmap.json", "w") as f:
                json.dump(class_map, f)
        except Exception as e:
            log.error(f"Failed to write classmap.json: {e}")
