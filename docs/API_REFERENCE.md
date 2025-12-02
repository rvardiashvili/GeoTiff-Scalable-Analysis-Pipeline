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

### `resolve_zor(tiling_config: DictConfig, adapter: BaseAdapter, input_properties: Dict[str, Any], available_ram_gb: float) -> int`
**Reason:** Determines the optimal Zone of Responsibility (ZoR) size in pixels for a given input based on available system RAM and model/pipeline parameters. This ensures efficient processing of large images without exceeding CPU memory limits.

**Key Parameters:**
*   `tiling_config`: Configuration specific to tiling parameters, including `zone_of_responsibility_size` (which can be "auto").
*   `adapter`: The initialized model adapter, providing model-specific properties (`is_segmentation`, `num_classes`, `num_bands`, `patch_size`, `stride`).
*   `input_properties`: Dictionary containing properties of the input image, such as its data type.
*   `available_ram_gb`: The amount of system RAM (in GB) available for the process.

**Logic:**
1.  If `tiling_config.zone_of_responsibility_size` is explicitly set (not "auto"), it uses that value.
2.  If set to "auto", it calls `calculate_optimal_zor` from `src/eo_core/memory_utils.py`, passing detailed memory model parameters derived from the adapter and input properties.
3.  The result from `calculate_optimal_zor` is then adjusted (e.g., rounded down to a multiple of a certain value) to ensure practical tiling.

**Output:** An integer representing the optimal ZoR size in pixels.

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

### `estimate_optimal_batch_size(model: torch.nn.Module, input_shape: Tuple[int, ...], dtype: torch.dtype, max_batch_size: int = 128, min_batch_size: int = 1, device: str = "cuda") -> int`
**Reason:** Provides a heuristic to determine an optimal GPU batch size by measuring memory consumption for small batches. This helps in maximizing GPU utilization without encountering Out-Of-Memory (OOM) errors.

**Key Parameters:**
*   `model`: The PyTorch model for which to estimate batch size.
*   `input_shape`: A tuple representing the shape of a single input to the model (excluding the batch dimension), e.g., `(num_bands, patch_height, patch_width)`.
*   `dtype`: The data type of the input tensor (e.g., `torch.float32`).
*   `max_batch_size`: The upper limit for the batch size search.
*   `min_batch_size`: The lower limit for the batch size search.
*   `device`: The device on which the model will run (e.g., "cuda").

**Logic:** The function heuristically estimates the optimal batch size by probing the GPU's memory consumption. It performs a forward pass with a batch size of 1 and then a batch size of 4 (if successful) to measure peak memory usage. From these two points, it calculates the marginal memory cost per sample, allowing it to extrapolate and estimate the maximum batch size that fits within a safe fraction of the available GPU memory. This approach aims to accurately separate fixed model overheads from per-sample activation memory costs. The estimated batch size is capped at 4096.

**Output:** An integer representing the estimated optimal GPU batch size.

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

### `_find_band_path(tile_path: Path, band_name: str) -> Path`
**Reason:** Locates the specific band file within a Sentinel-2 tile directory structure. This function abstracts away the complex file organization of `.SAFE` archives.
*   **Input:**
    *   `tile_path`: Path to the root of the Sentinel `.SAFE` directory (e.g., `S2A_MSIL2A_20230101T100000_N0500_R022_T31TGM_20230101T100000.SAFE`).
    *   `band_name`: The name of the band to find (e.g., 'B02', 'B8A', 'TCI').
*   **Logic:** Searches for the band file based on common naming conventions and directory structures typical for Sentinel-2 L2A data within the provided `tile_path`. It specifically looks for `.jp2` files matching the band name.
*   **Output:** `pathlib.Path` object pointing to the found band file.

### `read_chunk_data(chunk_info: Dict[str, Any], s2_reader: Callable, s1_reader: Optional[Callable] = None, s1_config: Optional[Dict[str, Any]] = None) -> np.ndarray`
**Reason:** A wrapper function designed to combine Sentinel-2 and, optionally, Sentinel-1 data for a given processing chunk. It orchestrates the reading and alignment of multi-sensor data into a unified tensor.
*   **Input:**
    *   `chunk_info`: A dictionary containing metadata and parameters for the current chunk (e.g., coordinates, tile paths).
    *   `s2_reader`: A callable function (typically `_read_s2_bands_for_chunk`) to read Sentinel-2 data.
    *   `s1_reader`: An optional callable function (typically `_read_s1_bands_for_chunk`) to read Sentinel-1 data.
    *   `s1_config`: Optional dictionary with Sentinel-1 specific configuration.
*   **Logic:**
    1.  Invokes `s2_reader` to get the Sentinel-2 data for the chunk.
    2.  If `s1_reader` is provided, it's called to get Sentinel-1 data, which is then aligned spatially to the Sentinel-2 data.
    3.  The S1 and S2 data are stacked along the band dimension.
*   **Output:** A NumPy array (`np.ndarray`) containing the stacked and aligned input bands for the chunk.

### `cut_into_patches(chunk_data: np.ndarray, patch_size: int, stride: int) -> Tuple[np.ndarray, List[Dict[str, Any]]]`
**Reason:** Divides a large image chunk into smaller, overlapping patches that are suitable as input for neural network models, which typically expect fixed-size inputs.
*   **Input:**
    *   `chunk_data`: A 3D NumPy array representing the `(bands, height, width)` of the image chunk.
    *   `patch_size`: The desired height and width of each square patch.
    *   `stride`: The step size (in pixels) for the sliding window, determining the overlap between patches.
*   **Logic:**
    1.  Implements a sliding window approach across the `chunk_data`.
    2.  Extracts individual patches, ensuring that patches near the chunk boundaries are handled correctly (e.g., through implicit padding or partial patches).
    3.  Collects metadata for each patch, such as its original coordinates within the chunk.
*   **Output:** A tuple containing:
    *   A 4D NumPy array (`np.ndarray`) of shape `(num_patches, bands, patch_size, patch_size)`.
    *   A list of dictionaries, where each dictionary contains metadata (e.g., `x_offset`, `y_offset`) for the corresponding patch.

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
