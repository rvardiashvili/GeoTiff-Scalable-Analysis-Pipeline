# API Reference & Function Documentation

This document details the key functions and classes in the codebase, explaining their purpose, inputs, outputs, and logic.

---

## `src/eo_core/inference_engine.py`

### `class InferenceEngine`
**Reason:** Decouples the generic tiling pipeline from specific model architectures. It acts as a facade that standardizes `run()` calls.

#### `__init__(self, config: Dict[str, Any], device: Optional[str] = None)`
*   **Input:** Configuration dictionary containing adapter settings.
*   **Logic:** Dynamically imports and instantiates the specified Adapter class defined in the config. Builds the model and loads weights.

#### `run(self, raw_input: Any) -> Any`
*   **Input:** `raw_input` (Dictionary containing tile path, coordinates, etc.).
*   **Output:** Dictionary containing results (probabilities, metadata).
*   **Logic:**
    1.  Calls `adapter.preprocess(raw_input)`.
    2.  Moves data to GPU.
    3.  Calls `model(data)`.
    4.  Calls `adapter.postprocess(output)`.

---

## `src/eo_core/adapters/base.py`

### `class BaseAdapter(ABC)`
**Reason:** Defines the contract that all model adapters must follow.

*   `build_model(self) -> nn.Module`: Abstract method to construct the PyTorch model.
*   `preprocess(self, raw_input: Any) -> Any`: Abstract method to prepare data for the model.
*   `postprocess(self, model_output: Any) -> Any`: Abstract method to format model predictions.
*   `is_segmentation(self) -> bool`: Property returning `True` if model outputs a spatial map (N, C, H, W), `False` for vectors (N, C).
*   `num_classes(self) -> int`: Property returning the number of output classes.
*   `num_bands(self) -> int`: Property returning the number of input bands.
*   `patch_size(self) -> int`: Property returning the input patch size.
*   `stride(self) -> int`: Property returning the sliding window stride.

---

## `src/eo_core/process.py`

### `writer_process(...)`
**Reason:** A dedicated background process for CPU-intensive tasks to prevent blocking the GPU inference loop.
*   **Input:** Queue of raw probability tensors from the inference engine.
*   **Logic:**
    1.  **Reconstruction:** Stitches overlapping patches using Sinusoidal Weighting (soft-voting).
    2.  **Metrics:** Calculates Entropy and Prediction Gap per pixel.
    3.  **I/O:** Writes valid regions to the final GeoTIFF files using `rasterio`.

### `main_hydra(cfg: DictConfig)`
**Reason:** The main orchestration function.
*   **Logic:**
    1.  Reads configuration.
    2.  Instantiates `Adapter` first to retrieve model parameters (bands, classes).
    3.  Calls `memory_utils.resolve_zor` to dynamically calculate the safe chunk size based on RAM.
    4.  Initializes `InferenceEngine`.
    5.  Spawns the `writer_process`.
    6.  Loops over the large input tile in "Chunks".
    7.  For each chunk:
        *   Fetches batch (Synchronous or via Prefetcher Thread).
        *   Calls `engine.predict_raw()`.
        *   Pushes results to the writer queue.

---

## `src/eo_core/memory_utils.py`

### `calculate_optimal_zor(...)`
**Reason:** Determines the maximum safe chunk size (Zone of Responsibility) to prevent OOM crashes.

**Key Parameters:**
*   `num_bands`: Number of input channels (Float32).
*   `num_classes`: Number of output classes/channels (Float32).
*   `stride_ratio`: `stride / patch_size` (determines overlap factor).
*   `is_segmentation`: Boolean flag altering Logits memory calculation.
*   `prefetch_queue_size` & `writer_queue_size`: Control concurrency multipliers.
*   `save_conf`, `save_entr`, `save_gap`: Boolean flags for optional metrics.

**Memory Formula (Bytes Per Pixel):**
1.  **Input Patches:**
    $$ BPP_{patches} = 4 \cdot N_{bands} \cdot \left(\frac{1}{R_{stride}}\right)^2 \cdot (Q_{prefetch} + 1) $$
2.  **Logits (Model Output):**
    *   *Segmentation:* $$ BPP_{logits} = 4 \cdot N_{classes} \cdot \left(\frac{1}{R_{stride}}\right)^2 \cdot (Q_{writer} + 2) $$
    *   *Classification:* $$ BPP_{logits} = \frac{4 \cdot N_{classes}}{(P_{size} \cdot R_{stride})^2} \cdot (Q_{writer} + 2) $$
3.  **Reconstruction Buffer:**
    $$ BPP_{recon} = 4 \cdot (N_{classes} + 1) $$
4.  **Metrics Buffers:**
    $$ BPP_{metrics} = 1_{dom} + 4_{conf} + 4_{entr} + 12_{gap} $$
5.  **IO Overhead:**
    $$ BPP_{io} \approx 4 \cdot N_{bands} $$

**Total Footprint:** $$ BPP_{total} = BPP_{patches} + BPP_{logits} + BPP_{recon} + BPP_{metrics} + BPP_{io} + 200_{overhead} $$

---

## `src/eo_core/data.py`

### `class BigEarthNetAdapter(BaseAdapter)`
**Reason:** Implements the adapter for BigEarthNet-trained Image Classification models (ResNet, ConvNeXt, etc.).

#### `preprocess(self, raw_input)`
*   **Input:** Tile reading parameters (paths, ROI coordinates).
*   **Logic:**
    1.  Calls `data._read_s2_bands_for_chunk` (and S1 if needed).
    2.  Normalizes or clips data (e.g., S1 clipping).
    3.  Cuts the large chunk into overlapping patches (`cut_into_patches`).
*   **Output:** Tuple `(patches_numpy_array, metadata)`.

#### `build_model(self)`
*   **Logic:** Instantiates a `MetadataPassingWrapper` around the core model to handle batch-wise inference and normalization on the GPU.

---

## `src/eo_core/adapters/prithvi_adapter.py`

### `class PrithviAdapter(BaseAdapter)`
**Reason:** Implements the adapter for the Prithvi-100M Semantic Segmentation model.

#### `preprocess(self, raw_input)`
*   **Logic:** Reads the specific 6 bands required by Prithvi. Scales data if necessary. Cuts patches.

#### `postprocess(self, model_output)`
*   **Logic:**
    1.  Receives logits from the wrapper.
    2.  Upsamples logits to original patch size if necessary (Segformer often outputs 1/4 resolution).
    3.  Computes `argmax` to get class indices.
    4.  Returns segmentation mask chunks.

---

## `src/eo_core/process.py`

### `writer_process(...)`
**Reason:** A dedicated background process for CPU-intensive tasks to prevent blocking the GPU inference loop.
*   **Input:** Queue of raw probability tensors from the inference engine.
*   **Logic:**
    1.  **Reconstruction:** Stitches overlapping patches using Sinusoidal Weighting (soft-voting).
    2.  **Metrics:** Calculates Entropy and Prediction Gap per pixel.
    3.  **I/O:** Writes valid regions to the final GeoTIFF files using `rasterio`.

### `main_hydra(cfg: DictConfig)`
**Reason:** The main orchestration function.
*   **Logic:**
    1.  Reads configuration.
    2.  Initializes `InferenceEngine` with the correct adapter.
    3.  Spawns the `writer_process`.
    4.  Loops over the large input tile in "Chunks" (e.g., 4000x4000 pixels).
    5.  For each chunk:
        *   Calls `engine.run()`.
        *   Pushes results to the writer queue.

---

## `src/eo_core/data.py`

### `_read_s2_bands_for_chunk(...)`
**Reason:** Efficiently reads specific regions of Sentinel-2 images.
*   **Logic:** Uses `rasterio` Windowed Reading. Handles resampling of 20m bands to 10m using Nearest Neighbor.

### `_read_s1_bands_for_chunk(...)`
**Reason:** Reads Sentinel-1 data and aligns it to Sentinel-2.
*   **Logic:**
    1.  Creates a Virtual Raster (VRT) to reproject S1 to S2 CRS on the fly.
    2.  Reads the requested window.
    3.  Converts Amplitude to dB (`20 * log10(A) - 50`).

---

## `src/eo_core/adapters/wrappers.py`

### `class MetadataPassingWrapper(nn.Module)`
**Reason:** A utility wrapper to handle the mismatch between "Logical Batch" (Full Chunk) and "Physical Batch" (GPU limit).
*   **Logic:**
    1.  Receives a large numpy array of patches (e.g., 4000 patches).
    2.  Iterates in mini-batches (e.g., 32 patches).
    3.   moves mini-batch to GPU.
    4.  Applies normalization: `(x - mean) / std`.
    5.  Runs model.
    6.  Moves result back to CPU to free VRAM.
    7.  Concatenates all results.
