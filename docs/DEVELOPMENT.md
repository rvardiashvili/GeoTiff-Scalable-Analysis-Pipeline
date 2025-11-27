# Development & Modification Guide

This document explains how to modify and extend the code, focusing on the new Adapter system.

## How to Add a New Model

Thanks to the Adapter pattern, you don't need to modify the core pipeline logic to add a new model type.

### 1. Create a New Adapter
Create a file in `src/eo_core/adapters/` (e.g., `my_custom_adapter.py`).
Inherit from `BaseAdapter` and implement the required methods and properties:

```python
from .base import BaseAdapter

class MyCustomAdapter(BaseAdapter):
    def build_model(self):
        # Initialize your PyTorch model here
        return MyModel()

    def preprocess(self, raw_input):
        # 1. Read data using raw_input['tile_folder']
        # 2. Cut into patches
        # 3. Return (patches_np, metadata)
        pass

    def postprocess(self, model_output):
        # Convert model logits to usable format (probs, masks)
        pass
    
    # Properties required for Memory Auto-Configuration
    @property
    def num_classes(self) -> int:
        return 10

    @property
    def num_bands(self) -> int:
        return 12

    @property
    def patch_size(self) -> int:
        return 120
        
    @property
    def stride(self) -> int:
        return 60
        
    @property
    def is_segmentation(self) -> bool:
        return False # Set True if model outputs a spatial map
```

### 2. Create a Configuration File
Create a YAML file in `configs/model/` (e.g., `my_model.yaml`). Point it to your new adapter:

```yaml
# @package _global_
model:
  adapter:
    path: "eo_core.adapters.my_custom_adapter"
    class: "MyCustomAdapter"
    params:
      some_param: 123
      bands: ['B02', 'B03', 'B04']
```

### 3. Run It
```bash
python src/main.py model=my_model input_path=...
```

## Modifying the Core Pipeline

If you need to change how tiling works (e.g., switching from squares to hexagons, or changing the writer logic):
*   **Tiling Logic:** Modify `src/eo_core/process.py` inside `main_hydra`.
*   **Writing Logic:** Modify `src/eo_core/process.py` inside `writer_process`.

## modifying Data Loading

*   **Sentinel-2 Reading:** `src/eo_core/data.py` -> `_read_s2_bands_for_chunk`
*   **Sentinel-1 Reading:** `src/eo_core/data.py` -> `_read_s1_bands_for_chunk`

## Git Workflow

The project uses standard Git practices.
1.  Check status: `git status`
2.  Review changes: `git diff`
3.  Commit often.
