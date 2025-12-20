# Technical Reference Manual: GeoSpatial Inference Pipeline (GSIP)

**Version:** 2.0.0
**Date:** November 2025
**System Architecture:** A Scalable and Multi-Modal Inference Engine

---

## 1. Introduction

### 1.1 System Overview
The GeoSpatial Inference Pipeline (GSIP) is a high-performance computing (HPC) framework designed for the semantic segmentation of gigapixel-scale satellite imagery. Unlike standard computer vision tasks that operate on fixed-size images (e.g., $512 \times 512$), Earth Observation (EO) data arrives in massive "tiles" (e.g., Sentinel-2 granules are $10,980 \times 10,980$ pixels).

This system addresses four critical engineering challenges in EO Deep Learning:
1.  **Memory Management:** Processing images exceeding GPU VRAM capacity (20GB+ uncompressed).
2.  **Spatial Continuity:** Eliminating "grid artifacts" caused by naive tiling strategies through a mathematically rigorous Overlap-Tile approach.
3.  **Multi-Modal Alignment:** Handling the asynchronous and spatially disjoint nature of Optical (Sentinel-2) and Radar (Sentinel-1) data.
4.  **Probabilistic Uncertainty:** Moving beyond simple class labels to provide pixel-wise uncertainty quantification (Entropy, Confidence).

### 1.2 Mathematical Notation
*   $\mathbf{I}$: Input Image Tensor.
*   $\Omega$: The spatial domain of the full tile.
*   $\mathcal{Z}$: Zone of Responsibility (valid output region).
*   $\mathcal{H}$: Halo region (context margin).
*   $f(\cdot)$: The neural network function (U-Net/ResNet).
*   $P$: Patch size (network input size).
*   $S$: Stride (step size for sliding window).

---

## 2. Data Ingestion & Signal Processing

The pipeline implements a robust Extract-Transform-Load (ETL) process that standardizes heterogeneous sensor data into a unified tensor representation.

### 2.1 Sentinel-2 (Optical) Pre-processing

Sentinel-2 L2A (Level-2A) data represents Bottom-of-Atmosphere (BOA) reflectance. The pipeline handles the complex directory structure of `.SAFE` archives.

#### 2.1.1 Resolution Unification
Sentinel-2 bands have varying spatial resolutions: 10m, 20m, and 60m.
*   **Reference Grid:** The 10m band `B02` (Blue) is selected as the geospatial anchor. All spatial transforms are calculated relative to `B02`'s affine transform matrix $\mathbf{M}_{ref}$.
*   **Resampling Algorithm:** Lower resolution bands (e.g., `B05` at 20m) are upsampled using **Nearest Neighbor Interpolation**.
    *   *Rationale:* Bilinear or Bicubic interpolation introduces synthetic values (smoothing) that physically do not exist, potentially confusing the spectral signature learned by the CNN. Nearest Neighbor preserves the original radiometric values.

#### 2.1.2 Radiometric Calibration
Digital Numbers (DN) are converted to physical reflectance $\rho$. The pipeline implements logic to handle ESA's Processing Baseline (PB) changes.

$$ \rho_{\lambda}(x, y) = \frac{\text{DN}_{\lambda}(x, y) + \Delta_{offset}}{Q_{value}} $$

*   $Q_{value} = 10000$ (Scaling factor).
*   $\Delta_{offset}$:
    *   For PB $< 04.00$ (Pre-2022): $\Delta_{offset} = 0$.
    *   For PB $\ge 04.00$ (Post-2022): $\Delta_{offset} = -1000$.

### 2.2 Sentinel-1 (SAR) Pre-processing

Sentinel-1 GRD (Ground Range Detected) data provides Synthetic Aperture Radar backscattering intensities in VV and VH polarizations.

#### 2.2.1 On-the-Fly Virtual Reprojection
S1 and S2 data are rarely pixel-aligned. S1 data often comes in a different Coordinate Reference System (CRS) or non-orthorectified grids.
*   **Method:** The pipeline utilizes `rasterio.vrt.WarpedVRT` to perform "lazy" reprojection.
*   **Algorithm:**
    1.  Read Ground Control Points (GCPs) from the S1 manifest.
    2.  Compute the thin-plate spline (TPS) or polynomial transform to map S1 pixels to the S2 CRS.
    3.  Resample pixels only when requested by the inference engine (Lazy Evaluation).

#### 2.2.2 Backscatter Conversion
Raw S1 Amplitude ($A$) is distributed exponentially and is unsuitable for CNNs. We convert it to the logarithmic decibel (dB) scale:

$$ \sigma^0_{dB} = 20 \cdot \log_{10}(A) - K_{calib} $$

*   $K_{calib}$: Calibration constant (set to **50.0** in `src/eo_core/data.py` to align mean statistics with BigEarthNet distributions).

#### 2.2.3 Dynamic Range Clipping
SAR data contains extreme outliers due to specular reflections (e.g., skyscrapers acting as corner reflectors).
*   **Clipping Range:** $[-50.0, 30.0]$ dB.
*   *Impact:* Values $<-50$ dB are treated as noise floor; values $>30$ dB are clamped to prevent gradient explosions in the network.

---

## 3. The ERF-Aware Inference Algorithm

A Convolutional Neural Network (CNN) has an **Effective Receptive Field (ERF)**—the region of input pixels that theoretically contributes to a specific output pixel. At the edges of an input image, the ERF is truncated (padding), leading to degraded prediction accuracy (boundary artifacts).

To solve this, we implement the **Overlap-Tile Strategy** with a strict Zone of Responsibility.

### 3.1 Theoretical Tiling Model

We decompose the massive image $\Omega$ into a set of processing chunks $C_k$. Each chunk consists of two regions:

1.  **Zone of Responsibility ($\mathcal{Z}$):** The central area where predictions are valid.
2.  **Halo ($\mathcal{H}$):** A context margin surrounding $\mathcal{Z}$.

The dimensions are related by:
$$ \text{Chunk}_{dim} = \text{ZoR}_{dim} + 2 \cdot \text{Halo}_{pixels} $$

*   **ZoR Size:** Dynamically calculated by default (`"auto"`) based on available RAM to maximize throughput without OOM errors.
*   **Halo Size:** Must be $\ge \frac{ERF}{2}$. For ResNet-50, typical ERF suggests a halo of ~128 pixels is safe.

#### 3.1.1 Memory Auto-Configuration
The system implements a rigorous memory model to calculate the safe ZoR size. This is managed by the `resolve_zor` function in `src/eo_core/process.py`, which leverages `calculate_optimal_zor` from `src/eo_core/memory_utils.py`. The formula accounts for the different memory footprints of classification vs. segmentation models, system RAM, and pipeline configuration.

$$ BPP_{total} = BPP_{patches} + BPP_{logits} + BPP_{recon} + BPP_{metrics} + BPP_{io} + 200_{overhead} $$

*   **Patches:** Weighted by the `prefetch_queue_size` and overlap factor.
*   **Logits:** The largest variable. For **Segmentation**, this is a full 4D map ($N \times C \times P \times P$), consuming significantly more RAM than **Classification** vectors ($N \times C$).
*   **Metrics:** Buffers for Entropy, Confidence, and Gap are only allocated if enabled in the config.

### 3.1.2 Optimal GPU Batch Size

While ZoR manages CPU memory for chunk processing, efficient GPU utilization requires optimizing the batch size for individual patches. The `estimate_optimal_batch_size` function (`src/eo_core/memory_utils.py`) provides a heuristic approach to determine the largest possible batch size that can be processed on the GPU without exceeding its VRAM capacity. This is achieved by iteratively testing small batch sizes and extrapolating the memory consumption to estimate the optimal value. Although primarily used by specific model adapters for internal optimization, this dynamic estimation contributes to robust and efficient GPU memory management.

### 3.2 Inference Workflow

For each tile coordinate $(r, c)$:

1.  **Read:** Extract window $[r - \text{Halo} : r + \text{ZoR} + \text{Halo}, \quad c - \text{Halo} : c + \text{ZoR} + \text{Halo}]$.
2.  **Pad:** If the window extends beyond the physical image boundaries, apply **Reflection Padding**:
    $$ \mathbf{I}_{pad}(x) = \mathbf{I}(|x|) \quad \text{for } x < 0 $$
    This mirrors the texture at the edge, minimizing high-frequency edge features that zero-padding would introduce.
3.  **Infer:** Pass the full padded chunk through the network.
4.  **Crop:** Extract the central $\mathcal{Z}$ region from the probability map $\mathbf{P}_{out}$.
    $$ \mathbf{P}_{valid} = \mathbf{P}_{out}[\text{Halo} : -\text{Halo}, \quad \text{Halo} : -\text{Halo}] $$
5.  **Write:** Save $\mathbf{P}_{valid}$ to the output mosaic at position $(r, c)$.

---

## 4. End-to-End Pipeline Workflow: Detailed Data Flow and Function Calls

This section provides a comprehensive walk-through of the GeoSpatial Inference Pipeline (GSIP), detailing the entire data flow from the initial configuration and entry point to the final output generation. It highlights the key functions and modules involved in each stage.

### 4.1 Entry Point and Configuration Initialization

The pipeline execution begins at `src/main.py`. This script leverages the `Hydra` configuration framework, indicated by the `@hydra.main` decorator, to load the primary configuration from `configs/config.yaml`.

*   **`configs/config.yaml`**: Serves as the root configuration file. It uses Hydra's `defaults` keyword to compose the final runtime configuration by integrating various sub-configurations from `configs/model/`, `configs/pipeline/`, and `configs/data_source/`. This modular approach allows for flexible and dynamic definition of models, data sources, and pipeline parameters.
*   **Mandatory Parameters**: Essential parameters such as `input_path` (specifying the path to the satellite imagery) and `output_path` (where results will be stored) are explicitly defined within the configuration.

Upon successful configuration loading, `src/main.py` invokes `main_hydra(cfg)` located in `src/eo_core/process.py`, which acts as the central orchestrator for the entire inference workflow.

### 4.2 Orchestration and System Initialization (`main_hydra`)

The `src/eo_core/process.py:main_hydra(cfg)` function sets up the inference environment and coordinates all subsequent steps:

1.  **Adapter Instantiation**: The system first instantiates the model-specific `Adapter` (inheriting from `BaseAdapter` in `src/eo_core/adapters/base.py`) as defined in the configuration. This step is crucial for retrieving model-specific parameters like expected input bands, number of classes, patch size, and stride.
2.  **Dynamic ZoR Calculation**: Utilizing the model parameters and available system resources, `main_hydra` calls `resolve_zor` (also in `src/eo_core/process.py`), which in turn uses `calculate_optimal_zor` from `src/eo_core/memory_utils.py`. This dynamically determines the optimal Zone of Responsibility (ZoR) size, ensuring that the processed chunks fit within the available CPU RAM without Out-Of-Memory (OOM) errors.
3.  **`InferenceEngine` Initialization**: An `InferenceEngine` instance (`src/eo_core/inference_engine.py`) is created. This engine is responsible for building the PyTorch model (via the adapter's `build_model` method) and loading any pre-trained weights.
4.  **Multiprocessing Setup**: To maximize throughput and decouple CPU-intensive tasks from GPU inference, `main_hydra` sets up a multiprocessing environment:
    *   **`writer_process`**: A dedicated background CPU process is spawned. This process is solely responsible for handling the post-inference tasks, including output reconstruction, metric calculation, and writing results to disk.
    *   **`prefetch_worker`**: An optional thread (`distributed.use_prefetcher: true`) is started to prefetch and preprocess data chunks in the background, ensuring a continuous supply of data to the GPU and preventing idle times.
5.  **Queue Management**: Two `multiprocessing.Queue` objects are established:
    *   `prefetch_queue`: Stores preprocessed input patches ready for GPU inference (managed by `prefetch_worker`).
    *   `writer_queue`: Collects raw model outputs (logits) from the `InferenceEngine`, which are then consumed by the `writer_process`.

### 4.3 Chunk Processing Loop

After initialization, `main_hydra` enters a main loop that iterates over the large input tile, processing it in dynamically sized "Chunks" determined by the calculated ZoR.

#### 4.3.1 Data Loading and Preprocessing (CPU-bound)

For each chunk, the following steps occur, often managed by the `prefetch_worker` thread:

1.  **Band Path Finding**: `_find_band_path` (`src/eo_core/data.py`) locates the specific band files within the complex directory structure of Sentinel `.SAFE` archives.
2.  **Raw Data Reading**: Functions like `_read_s2_bands_for_chunk` and `_read_s1_bands_for_chunk` (`src/eo_core/data.py`) are used to efficiently read specific regions of Sentinel-2 and Sentinel-1 imagery, respectively. This includes:
    *   Sentinel-2: Handling resolution unification (resampling to 10m using Nearest Neighbor interpolation) and radiometric calibration (DN to reflectance conversion).
    *   Sentinel-1: On-the-fly virtual reprojection to align with Sentinel-2, backscatter conversion to dB scale, and dynamic range clipping.
3.  **Chunk Data Assembly**: `read_chunk_data` (`src/eo_core/data.py`) acts as a wrapper, combining the pre-processed Sentinel-2 and (optionally) Sentinel-1 data for the current chunk, aligning and stacking the bands into a unified tensor.
4.  **Adapter Preprocessing**: The instantiated model adapter's `preprocess` method is invoked. This method:
    *   Receives the assembled chunk data.
    *   Further normalizes or scales the data as required by the specific model (e.g., Prithvi's normalization).
    *   Calls `cut_into_patches` (`src/eo_core/data.py`) to divide the large chunk into smaller, overlapping patches using a sliding window approach, preparing them for GPU inference.
5.  **Queueing**: The preprocessed patches and their metadata are then pushed into the `prefetch_queue` (if a prefetcher is active) or directly passed to the `InferenceEngine`.

#### 4.3.2 Model Inference (GPU-bound)

The `InferenceEngine` handles the core model prediction for the prepared patches:

1.  **Batching**: Patches are taken from the `prefetch_queue` (or directly) and batched for GPU processing.
2.  **`MetadataPassingWrapper`**: If configured, the `MetadataPassingWrapper` (`src/eo_core/adapters/wrappers.py`) manages the process of feeding patches to the GPU in mini-batches. It performs necessary operations like moving data to the GPU, applying model-specific normalization (`(x - mean) / std`), and running the model's forward pass.
3.  **Prediction**: The `InferenceEngine.predict_raw` method executes the neural network's forward pass within a `torch.no_grad()` context to obtain raw model outputs (logits) without computing gradients.
4.  **Output Queueing**: The raw logits and associated metadata are then pushed into the `writer_queue` for asynchronous post-processing by the `writer_process`.

### 4.4 Post-processing and Output Generation (CPU-bound - `writer_process`)

The `writer_process` operates independently in the background, consuming raw model outputs from the `writer_queue` and performing the final stages of data processing and output generation:

1.  **Adapter Post-processing**: The adapter's `postprocess` method is called to convert the raw logits into meaningful representations. For segmentation models, this typically involves converting logits to probabilities (e.g., softmax) and potentially upsampling them to the original patch resolution if the model has a lower-resolution output (e.g., Segformer).
2.  **Reconstruction (Sinusoidal Weighting)**: Overlapping patch predictions are meticulously stitched together to reconstruct the full-resolution output map for the chunk. This process uses **Sinusoidal Weighting** (soft-voting) to give higher confidence to predictions from the central regions of patches, thereby eliminating grid artifacts and ensuring spatial continuity.
3.  **Uncertainty Quantification**: If enabled in the configuration, the `writer_process` calculates per-pixel **Shannon Entropy** (to measure model confusion) and **Prediction Gap** (to quantify the margin between top contending classes), providing valuable reliability insights).
4.  **I/O Operations**: The processed results, including the probability maps and uncertainty metrics, are written to disk as GeoTIFF files using `rasterio`. This includes applying appropriate compression and geospatial metadata.
5.  **Final Output Generation**: Finally, auxiliary outputs like `generate_low_res_preview` (visual previews) and `generate_single_node_viewer` (web-based interactive viewer) are created, providing accessible ways to inspect the inference results.

---

## 5. Patch-Level Processing & Probabilistic Reconstruction

Ideally, we would feed the entire Chunk (e.g., $4256 \times 4256$) into the GPU. However, VRAM limitations necessitate breaking the Chunk into smaller **Patches**.

### 5.1 Sliding Window Generation
*   **Patch Size ($W_p$):** Defined by the Model Adapter (e.g., 120 for ResNet, 224 for Prithvi).
*   **Stride ($S_p$):** Defined by the Model Adapter (typically $W_p / 2$ for 50% overlap).

This dense overlapping ensures that every pixel in the Chunk is predicted by multiple passes of the CNN.

### 5.2 Soft-Voting (Sinusoidal Weighting)
Simple averaging of overlapping patches is suboptimal because CNN predictions are less accurate near the borders of a patch (padding effects). We employ a weighted average where pixels near the center of a patch contribute more.

**The 2D Weighting Window $\mathbf{W}$:**
Constructed as the outer product of two 1D Hann (squared sine) windows:

$$ w(n) = \sin^2\left(\frac{\pi n}{W_p - 1}\right), \quad 0 \le n < W_p $$
$$ \mathbf{W}(i, j) = w(i) \cdot w(j) $$

### 5.3 Aggregation Formula
Let $\mathcal{P}_k$ be the probability map for the $k$-th patch, and $\mathbf{W}_k$ be its weighting window placed at offset $(u_k, v_k)$. The aggregated probability map $\mathbf{M}$ for the chunk is:

$$ \mathbf{M}(x, y) = \frac{\sum_{k} \mathcal{P}_k(x-u_k, y-v_k) \cdot \mathbf{W}(x-u_k, y-v_k)}{\sum_{k} \mathbf{W}(x-u_k, y-v_k)} $$

This results in a seamless, artifact-free probability surface.

## 6. Uncertainty Quantification

The system produces auxiliary geospatial layers to assess model reliability.

### 6.1 Shannon Entropy (Uncertainty)
Measures the "confusion" of the model. High entropy implies the probability mass is spread across many classes.

$$ H(x, y) = -\sum_{c=1}^{C} p_c(x,y) \cdot \log_2(p_c(x,y) + \epsilon) $$
*   $C$: Number of classes (19).
*   $\epsilon = 1e^{-6}$: Numerical stability term.

### 6.2 Prediction Gap (Margin)
Measures the margin between the top two contending classes. A small gap indicates high ambiguity between two specific labels.

$$ \text{Gap}(x, y) = P_{(1)}(x, y) - P_{(2)}(x, y) $$
Where $P_{(n)}$ is the $n$-th highest probability.

---

## 7. Parallel Architecture (Producer-Consumer)

To hide I/O latency and CPU-heavy aggregation costs, the pipeline uses `torch.multiprocessing`.

1.  **Main Process (Producer):**
    *   Reads GeoTIFF chunks.
    *   Applies pre-processing/warping (optionally via a **Prefetcher Thread**).
    *   Batches patches to GPU.
    *   Executes Model Forward Pass (`model(x)`).
    *   Pushes raw logits and metadata to `Queue`.

2.  **Writer Process (Consumer):**
    *   Pulls raw data from `Queue`.
    *   Calls `adapter.postprocess()` (e.g., upsampling, argmax).
    *   Performs Sinusoidal Aggregation (CPU intensive).
    *   Calculates Metrics (Entropy, Gap).
    *   Writes to Disk (GeoTIFF compression).

---

## 8. Output Generation: The Reporter System

GSIP employs a flexible **Reporter** pattern to handle the generation of analysis results. This decouples the core inference loop from specific file formats, allowing users to "plug-and-play" different output modules without modifying the codebase.

### 8.1 The Reporter Interface (`BaseReporter`)
All reporters must inherit from `src.eo_core.reporters.base.BaseReporter` and implement three methods:
1.  **`on_start(context)`**: Initialization (e.g., creating file handles). `context` provides paths, profile metadata, and model info.
2.  **`on_chunk(data)`**: Called for each reconstructed output chunk. `data` contains the `valid_probs` (ZoR probabilities) and spatial `window`.
3.  **`on_finish(context)`**: Finalization (e.g., closing files, aggregating statistics).

### 8.2 Standard Reporters
*   **`GeoTIFFReporter`**: Writes standard geospatial rasters.
    *   *Outputs:* Class Map (`_class.tif`), Confidence (`_maxprob.tif`), Entropy (`_entropy.tif`), Prediction Gap (`_gap.tif`).
*   **`PreviewReporter`**: Generates a quick-look PNG preview of the classification map.
*   **`MetadataReporter`**: Generates metadata (`classmap.json`) and the interactive HTML viewer (`viewer.html`).
*   **`GlobalProbabilityReporter`**: Performs **Global Average Pooling** on the entire reconstructed tile to produce a single 1D probability vector (saved as `.npy` and `.json`). This provides a "whole-tile" classification summary.
*   **`ProbabilityReporter`**: (Optional) Writes the full probability distribution (multi-band GeoTIFF) for advanced analysis.

### 8.3 Configuration
Reporters are defined in the **Model Configuration** files (e.g., `configs/model/convnext_s2.yaml`) under `pipeline.reporters`. They use Hydra's `_target_` syntax for dynamic instantiation:

```yaml
pipeline:
  reporters:
    geotiff:
      _target_: eo_core.reporters.geotiff.GeoTIFFReporter
    preview:
      _target_: eo_core.reporters.preview.PreviewReporter
    # Custom reporters can be added here simply by pointing to their Python path
    my_stats:
      _target_: my_project.stats_reporter.JSONStatsReporter
```

---

## 9. Configuration System (Hydra)

The entire pipeline is controlled via hierarchical YAML configurations using **Hydra**.

**Root:** `configs/config.yaml`

### 9.1 Inference Parameters (`configs/pipeline/inference_params.yaml`)
| Parameter | Key | Definition |
| :--- | :--- | :--- |
| **Zone of Responsibility** | `tiling.zone_of_responsibility_size` | Unit of output writing. Can be set to `"auto"` for dynamic RAM optimization. |
| **Halo** | `tiling.halo_size_pixels` | The context margin (e.g., 128). Must be $\ge$ Half-Receptive-Field. |
| **Batch Size** | `distributed.gpu_batch_size` | Number of patches processed per CUDA call. |
| **Use Prefetcher** | `distributed.use_prefetcher` | `True`/`False`. Enables background data loading thread. |
| **Prefetch Queue** | `distributed.prefetch_queue_size` | Size of input buffer (chunks) when prefetcher is active. |
| **Writer Queue** | `distributed.writer_queue_size` | Size of output buffer (chunks) between Inference and Writer. |

### 8.2 Model Parameters (`configs/model/*.yaml`)
| Parameter | Key | Definition |
| :--- | :--- | :--- |
| **Adapter** | `adapter` | Configuration block for the model adapter (see below). |

### 8.3 Adapter Configuration
Adapters decouple the pipeline from specific model architectures.
*   `path`: Python import path to the adapter module.
*   `class`: Name of the adapter class.
*   `params`: Dictionary of arguments passed to the adapter's `__init__` (e.g., `bands`, `weights_path`).

---

## 10. Multi-Modal Deep Learning Fusion

For configurations enabling fusion (`use_sentinel_1: true`), the pipeline utilizes the `DeepLearningRegistrationPipeline` module (`src/eo_core/fusion.py`).

### 9.1 Spatial Transformer Network (STN) Flow
Instead of relying solely on geometric GCP alignment, this module allows for feature-based alignment:

1.  **Feature Extraction:** A shared encoder extracts keypoints from S1 (Source) and S2 (Target).
2.  **Regression:** A regressor predicts the affine transformation matrix $\theta$ (6 parameters).
3.  **Grid Generation:** $G = \texttt{affine_grid}(\theta, \text{size})$.
4.  **Sampling:** $S1_{aligned} = \texttt{grid_sample}(S1_{input}, G)$.
5.  **Concatenation:** $X_{input} = \text{Concat}(S1_{aligned}, S2)$.

*Note: The current implementation defaults to Identity transformation if no weights are provided, relying on the geometric alignment step in Section 2.2.1.*