# Documentation: BigEarthNet v2.0 Scalable Analysis Pipeline

This document provides a detailed, thesis-level description of the workflow, methodologies, and design principles of the BigEarthNet v2.0 Scalable Analysis Pipeline.

## 1. Introduction

### 1.1. Context and Motivation

The proliferation of Earth observation satellites has led to an unprecedented volume of high-resolution remote sensing data. Datasets like BigEarthNet, derived from the European Space Agency's (ESA) Sentinel-2 mission, offer vast potential for monitoring our planet's surface. However, the sheer scale of this data presents significant computational challenges. A single Sentinel-2 tile can be several gigabytes in size, making it infeasible to process as a single unit on standard hardware.

This project addresses the "Big Data" challenge in remote sensing by developing a scalable pipeline for land cover classification. The primary goal is to create a system that can efficiently process large Sentinel-2 tiles, producing accurate land cover maps and associated quality metrics. This work serves as a foundational component for large-area mapping and a baseline for future research in scalable geospatial analysis.

### 1.2. The BigEarthNet-S2 Dataset

The pipeline is designed to work with data structured like the BigEarthNet-S2 dataset. BigEarthNet-S2 is a large-scale, multi-label land cover classification dataset, containing 125 Sentinel-2 tiles acquired over 10 European countries. Each tile is annotated with multiple land cover classes from the CORINE Land Cover database. This project leverages a model pre-trained on this dataset, inheriting its rich semantic understanding of land cover types.

### 1.3. Problem Statement

The core problem is to perform semantic segmentation on a full, un-patched Sentinel-2 tile, which can be as large as 10980x10980 pixels. Given memory and computational constraints, the task is to design a workflow that:

1.  Processes the image without requiring the entire tile to be loaded into memory at once.
2.  Utilizes a pre-trained deep learning model for accurate, patch-based classification.
3.  Aggregates patch-level predictions into a seamless, tile-level classification map.
4.  Generates auxiliary data products to aid in the interpretation and quality assessment of the classification results.

### 2. Methodology and Workflow





The pipeline employs a "divide and conquer" strategy, processing the large input tile in a series of stages.







### 2.1. Data Acquisition





A dedicated module (`download_sentinel.py`) handles the acquisition of input data from the Copernicus Dataspace Ecosystem. It is designed to retrieve matching pairs of Sentinel-1 (SAR) and Sentinel-2 (Optical) imagery.







-   **Sentinel-2 Selection:** Filters for products with low cloud coverage (< 10%) within the specified time window and Area of Interest (AOI).



-   **Sentinel-1 Matching:** For each selected Sentinel-2 product, the system searches for a spatially overlapping Sentinel-1 GRD product acquired within a close temporal window (e.g., +/- 2 days).



-   **Pairing:** Valid pairs are downloaded and organized into a directory structure suitable for processing.







### 2.2. Input Data and Pre-processing

The pipeline ingests a single Sentinel-2 tile, represented as a folder of individual band files (e.g., `B01.jp2`, `B02.jp2`, etc.). These are typically Top-of-Atmosphere (L1C) or Bottom-of-Atmosphere (L2A) reflectance products.

#### 2.1.1. Band Selection

The model expects a specific set of input bands, as defined in `config.py`. The standard configuration uses either 10 or 12 of the 13 Sentinel-2 bands, depending on the pre-trained model's requirements.

#### 2.1.2. Normalization

Per-channel normalization is a critical pre-processing step for deep learning models. Before inference, each input patch is normalized using the mean and standard deviation statistics derived from the BigEarthNet training set. The formula for normalization is:

$$
\text{patch}_{\text{norm}} = \frac{\text{patch}_{\text{raw}} - \mu}{\sigma}
$$

Where $\mu$ and $\sigma$ are the per-band mean and standard deviation vectors, respectively. This ensures that the input data has a similar distribution to the data the model was trained on.

### 2.2. Scalable Processing: Chunking and Patching

To manage memory usage, the pipeline uses a two-level spatial division strategy.

#### 2.2.1. Chunking

The full Sentinel-2 tile is first divided into a grid of non-overlapping **chunks**. The size of these chunks (`CHUNK_SIZE`) is a key configuration parameter. A larger `CHUNK_SIZE` can improve processing efficiency by reducing the number of I/O operations, but it also increases the memory footprint. The optimal size is a trade-off between processing speed and available RAM. Each chunk is processed independently.

#### 2.2.2. Patching

Each chunk is then further subdivided into smaller, overlapping **patches**. These patches are the atomic units of processing for the neural network. The `PATCH_SIZE` parameter defines the dimensions of these patches.

The use of overlapping patches is a crucial design choice. It helps to mitigate "edge effects"—artifacts that can appear at the boundaries of predictions. By averaging the predictions from overlapping patches, the final output is smoother and more spatially consistent. The stride of the overlap is typically `PATCH_SIZE / 2`.

### 2.3. Deep Learning Model and Inference

#### 2.3.1. Model Architecture

The pipeline uses the `configilm` library, which provides a flexible framework for creating image and language models. The `BigEarthNetv2_0_ImageClassifier` is a `lightning.pytorch` wrapper around a `ConfigILM` model. This architecture is designed for multi-label image classification. The specific model configuration (e.g., backbone, number of layers) is loaded along with the pre-trained weights from the Hugging Face Hub, specified by the `REPO_ID`.

#### 2.3.2. Inference Process

For each patch within a chunk, the model performs a forward pass and outputs a vector of logits, one for each of the 19 land cover classes. These logits are then transformed into probabilities using the element-wise sigmoid function:

$$ P(\text{class}_i) = \sigma(\text{logit}_i) = \frac{1}{1 + e^{-\text{logit}_i}} $$

The sigmoid function is used because the problem is multi-label, meaning a single patch can contain multiple land cover classes.

### 2.4. Probability Aggregation and Smoothing

A key innovation of this pipeline is the method for reconstructing a smooth, chunk-level probability map from the overlapping patch predictions. A naive averaging would give undue weight to the centers of patches. Instead, a weighted averaging scheme is employed, where the influence of each patch's prediction is highest at its center and decays towards its edges.

This is achieved using a 2D sine window function as a weighting mask:

$$ W(x, y) = \sin^2\]\left(\frac{\pi x}{P_w}\right) \cdot \sin^2\]\left(\frac{\pi y}{P_h}\right) $$

Where $P_w$ and $P_h$ are the width and height of the patch (`PATCH_SIZE`).

The final aggregated probability for each class $c$ at each pixel $(x, y)$ in the chunk is calculated as:

$$ P_{\text{agg}, c}(x, y) = \frac{\sum_{i \in \text{patches}} P_{i, c}(x, y) \cdot W(x_i, y_i)}{\sum_{i \in \text{patches}} W(x_i, y_i)} $$

Where the sum is over all patches $i$ that cover the pixel $(x, y)$, and $(x_i, y_i)$ are the coordinates within patch $i$.

### 2.6. Extensibility: Extra Data Generators

From the final, aggregated probability map for each chunk, a suite of geospatial data products is generated. These are then mosaicked together to form the full, tile-level output files.

-   **Classification Map (`*_class.tif`):** The final land cover class for each pixel is determined by selecting the class with the highest probability (the `argmax` of the probability vector).

-   **Maximum Probability (`*_maxprob.tif`):** This is the probability of the winning class, representing the model's confidence in its prediction.

-   **Shannon Entropy (`*_entropy.tif`):** This is a measure of the uncertainty of the prediction. It is calculated from the probability vector $\mathbf{p}$ for each pixel:

    $$ H(\mathbf{p}) = -\sum_{i=1}^{N} p_i \log_2(p_i) $$

    High entropy values indicate that the model is uncertain, with probabilities spread across multiple classes. This can be useful for identifying areas of confusion or potential misclassification.

-   **Prediction Gap (`*_gap.tif`):** This is the difference between the highest and second-highest class probabilities. A small gap suggests that the model found it difficult to distinguish between the top two candidate classes.

-   **Full Probabilities (`*_probs.tif`):** An optional, multi-band GeoTIFF that stores the full probability vector for each pixel. This is a rich data product that allows for more advanced post-processing and analysis.







-   **Single Node Viewer (`viewer.html`):** An interactive HTML dashboard generated for each processed tile. It provides a user-friendly interface to view the classification preview, the legend, and potentially other metrics, facilitating immediate visual inspection without specialized GIS software.







All GeoTIFF outputs are created with tiling and LZW compression to optimize for storage and read performance.

### 3.1. Asynchronous Processing & Performance



To improve throughput, the pipeline uses a multi-threaded, producer-consumer architecture.



-   **Producer Thread:** The main thread reads the input data, creates chunks and patches, and performs the GPU-intensive model inference. It places the results (patch probabilities) into a queue.

-   **Consumer Thread (Mosaicking Worker):** A separate worker thread retrieves the results from the queue, performs the probability aggregation, calculates the output products, and writes the final chunk-level rasters to disk.



This design allows the GPU to be kept busy with inference while the CPU-bound aggregation and I/O operations happen in parallel.



A key performance optimization is the pre-calculation of the sinusoidal weighting masks used for smoothing. Instead of being wastefully regenerated for every processed chunk, these masks are created once at the beginning of the pipeline and reused, significantly reducing computational overhead.



### 3.2. Benchmark Mode and Viewer



The pipeline includes a benchmark mode to facilitate performance testing and analysis across multiple tiles. When run with the `--benchmark` flag, the script will:

1.  Iterate through all tile subdirectories in a given input directory.

2.  Process each tile sequentially, reusing the loaded model to avoid repeated setup costs.

3.  Record performance metrics for each tile, such as processing time, memory usage, and I/O sizes.

4.  Compile the results into a `benchmark_report.csv` file in the main output directory.

5.  Automatically generate an interactive `viewer.html` file that displays the benchmark report in a sortable table and shows a preview of each processed tile along with its color legend.



### 3.3. Configuration



The pipeline is highly configurable via the `src/ben_v2/config.py` file. Key parameters include:



-   `REPO_ID`: The Hugging Face Hub repository ID for the pre-trained model.

-   `CHUNK_SIZE`: The side dimension of a processing chunk.

-   `PATCH_SIZE`: The side dimension of a model input patch.

-   `GPU_BATCH_SIZE`: The number of patches in a batch for GPU inference.

-   `MODEL_NAME`: The name of the pre-trained model, used to dynamically load model-specific configurations (bands, means, stds).

-   `DEVICE`: The target device for computation (`"cuda"` or `"cpu"`).

-   `BANDS`: The list of Sentinel-2 bands to be used, dynamically determined by `MODEL_NAME`.



## 4. Future Work and Potential Enhancements

This pipeline provides a robust foundation for large-scale land cover classification. Several avenues for future work exist:



-   **Distributed Computing:** For processing massive collections of tiles, the pipeline could be integrated with a distributed computing framework like Dask or Apache Spark to parallelize processing across multiple nodes in a cluster.

-   **Alternative Model Architectures:** The `configilm` framework allows for experimentation with different model backbones (e.g., Vision Transformers, ConvNeXt) which may yield improved accuracy.

-   **Uncertainty Quantification:** The existing entropy and gap metrics could be supplemented with more advanced uncertainty quantification techniques, such as Monte Carlo dropout, to provide more reliable estimates of model uncertainty.

-   **Data Fusion:** The pipeline could be extended to incorporate data from other sensors (e.g., Sentinel-1 SAR data) to improve classification accuracy, particularly in areas with persistent cloud cover.

-   **Active Learning:** The uncertainty maps could be used to guide an active learning workflow, where the model requests human annotation for the most uncertain areas to improve its performance over time.
