# BigEarthNet v2.0 Scalable Analysis Pipeline

This project provides a scalable pipeline for land cover classification of BigEarthNet v2.0 Sentinel-2 imagery. It uses a deep learning model to perform semantic segmentation on large satellite images, processing them in chunks to handle memory constraints.

## Features



-   **Scalable Processing:** Analyzes large Sentinel-2 tiles by processing them in smaller chunks.

-   **Deep Learning-based:** Utilizes a pre-trained `configilm` model for accurate land cover classification.

-   **Benchmarking:** Built-in benchmark mode to process multiple tiles and generate a performance report.

-   **Multiple Outputs:** Generates not only the classification map but also confidence, entropy, and prediction gap maps.

-   **Extensible:** Allows for custom functions to generate additional data layers from the classification probabilities.

-   **Geospatial Outputs:** All outputs are saved as GeoTIFF files, preserving the original geospatial information.

-   **HTML Viewer:** Automatically generates an HTML viewer to inspect benchmark results and tile previews.



## Setup



### Dependencies



This project requires Python 3.8+ and the following packages:



-   `torch`

-   `lightning`

-   `configilm`

-   `huggingface_hub`

-   `rasterio`

-   `numpy`

-   `tqdm`

-   `pandas`

-   `psutil`



You can install them using pip:



```bash

pip install torch lightning configilm huggingface_hub rasterio numpy tqdm pandas psutil

```



### Project Installation



The project is structured as a Python package. To install it in editable mode, run the following command from the project's root directory:



```bash

pip install -e .

```



## Usage



The main entry point for the pipeline is `src/main.py`. It can be run in two modes:



### 1. Single Tile Processing



To process a single Sentinel-2 tile folder:



```bash

python src/main.py --tile_folder /path/to/your/S2A_MSIL1C_..._tile_folder --output_folder /path/to/your/output_folder

```



### 2. Benchmark Mode



To process multiple tiles and generate a performance report:



```bash

python src/main.py --benchmark --input_dir /path/to/tile_folders/ --output_folder /path/to/your/output_folder

```



This will process all tile subdirectories found in `--input_dir`, save the results for each, and generate a `benchmark_report.csv` and an interactive `viewer.html` in the output folder.



### 3. Data Acquisition (Sentinel-1 & Sentinel-2)



A utility script is provided to search for and download matching pairs of Sentinel-1 and Sentinel-2 imagery from the Copernicus Dataspace Ecosystem.



```bash

python src/ben_v2/download_sentinel.py

```



*Note: You will need to configure your Copernicus Dataspace credentials and Area of Interest (AOI) within `src/ben_v2/download_sentinel.py` before running.*



### Configuration

The pipeline's behavior is configured using [Hydra](https://hydra.cc/). The main configuration file is `configs/config.yaml`, which composes settings from:

-   `configs/model/`: Model-specific settings (bands, normalization).
-   `configs/pipeline/`: Inference parameters (tiling, batch size, output options).
-   `configs/data_source/`: Data loading settings.

Key parameters can be overridden via the command line. For example:

-   `pipeline.tiling.patch_size`: Size of inference patches.
-   `pipeline.distributed.gpu_batch_size`: Batch size for GPU inference.
-   `pipeline.output.save_preview`: Toggle preview image generation.

### Output Files



For each input tile, the pipeline generates the following files in the output directory:



-   `*_class.tif`: A single-band GeoTIFF where each pixel value represents the predicted land cover class ID.

-   `*_maxprob.tif`: A single-band GeoTIFF with the probability (confidence) of the predicted class for each pixel.

-   `*_entropy.tif`: A single-band GeoTIFF representing the prediction uncertainty. Higher values indicate higher uncertainty.

-   `*_gap.tif`: A single-band GeoTIFF showing the difference between the highest and second-highest class probabilities.

-   `*_classmap.json`: A JSON file that maps the class IDs in `*_class.tif` to human-readable class names.

-   `preview.png`: A downscaled PNG image of the classification map for a quick preview.

-   (Optional) `*_probs.tif`: A multi-band GeoTIFF containing the full probability distribution for each class for every pixel.



-   `viewer.html`: A standalone HTML viewer for the single tile, allowing for interactive inspection of the results and legend.



When running in benchmark mode, the following files are also generated in the root of the output directory:

- `benchmark_report.csv`: A CSV file containing performance metrics for each processed tile.

- `viewer.html`: An interactive HTML file to visualize the benchmark results and tile previews.
